# src/train_source.py

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

# Make `from src...` imports work whether run as `python -m src.train_source`
# or directly by file path (e.g. VS Code Run button).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.collator import RaiseAbsaCollator, IGNORE_INDEX
from src.models.raise_absa import RaiseAbsaModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZED_DATA_DIR = PROJECT_ROOT / "data" / "tokenized" / "deberta_laptop_max96"
TOKENIZER_DIR = PROJECT_ROOT / "models" / "deberta-v3-base-tokenizer"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "raise_absa_deberta_small"

MODEL_NAME = "microsoft/deberta-v3-small"
SEED = 42
BATCH_SIZE = 2            # lowered from 4 for VRAM headroom on the 4 GB RTX 3050
GRAD_ACCUM_STEPS = 8      # effective batch size stays 16
EPOCHS = 5
LEARNING_RATE = 3e-5
LAMBDA_SENTIMENT = 1.0
EMA_DECAY = 0.98          # lowered from 0.995 (only ~160 optimizer steps per epoch)
FREEZE_WORD_EMBEDDINGS = True  # embeddings are ~98M of 141M params; freezing saves ~1.2 GB VRAM
MIN_FREE_VRAM_MB = 2400


def decode_bio_spans(bio_preds, attention_mask):
    batch_spans = []
    for b_idx in range(bio_preds.shape[0]):
        preds = bio_preds[b_idx]
        mask = attention_mask[b_idx]
        spans = []
        start = None
        for t_idx in range(len(preds)):
            if mask[t_idx] == 0:
                break
            tag = int(preds[t_idx])
            if tag == 1:
                if start is not None:
                    spans.append((start, t_idx))
                start = t_idx
            elif tag == 2:
                if start is None:
                    start = t_idx
            else:
                if start is not None:
                    spans.append((start, t_idx))
                    start = None
        if start is not None:
            spans.append((start, len(preds)))
        batch_spans.append(spans)
    return batch_spans


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    gold_spans_total = 0
    pred_spans_total = 0
    exact_span_matches = 0
    exact_pair_matches = 0

    all_gold_sentiments = []
    all_pred_sentiments = []
    nonconformity_scores = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_batch_indices=batch["span_batch_indices"].to(device),
            span_token_starts=batch["span_token_starts"].to(device),
            span_token_ends=batch["span_token_ends"].to(device),
        )

        bio_preds = outputs["bio_logits"].argmax(dim=-1).cpu().numpy()
        decoded_spans = decode_bio_spans(bio_preds, attention_mask.cpu().numpy())

        if batch["span_sentiment_labels"].numel() > 0:
            sent_probs = torch.softmax(outputs["sentiment_logits"].float(), dim=-1).cpu().numpy()
            sent_preds = sent_probs.argmax(axis=-1)
            gold_sents = batch["span_sentiment_labels"].numpy()

            all_gold_sentiments.extend(gold_sents.tolist())
            all_pred_sentiments.extend(sent_preds.tolist())

            for probs, true_idx in zip(sent_probs, gold_sents):
                nonconformity_scores.append(float(1.0 - probs[true_idx]))

        for b_idx, gold_list in enumerate(batch["gold_aspects"]):
            gold_map = {(a["token_start"], a["token_end"]): a["sentiment_id"] for a in gold_list}
            pred_list = decoded_spans[b_idx]
            gold_spans_total += len(gold_map)
            pred_spans_total += len(pred_list)

            if pred_list:
                p_b_idx = torch.tensor([b_idx] * len(pred_list), dtype=torch.long, device=device)
                p_starts = torch.tensor([s for s, e in pred_list], dtype=torch.long, device=device)
                p_ends = torch.tensor([e for s, e in pred_list], dtype=torch.long, device=device)

                pred_sent_logits = model.classify_spans(
                    outputs["sequence_output"], p_b_idx, p_starts, p_ends
                )
                pred_sent_ids = pred_sent_logits.argmax(dim=-1).cpu().tolist()
            else:
                pred_sent_ids = []

            for (s, e), p_sent in zip(pred_list, pred_sent_ids):
                if (s, e) in gold_map:
                    exact_span_matches += 1
                    if gold_map[(s, e)] == p_sent:
                        exact_pair_matches += 1

    span_p = exact_span_matches / max(1, pred_spans_total)
    span_r = exact_span_matches / max(1, gold_spans_total)
    span_f1 = 2 * span_p * span_r / max(1e-8, span_p + span_r)

    pair_p = exact_pair_matches / max(1, pred_spans_total)
    pair_r = exact_pair_matches / max(1, gold_spans_total)
    pair_f1 = 2 * pair_p * pair_r / max(1e-8, pair_p + pair_r)

    sent_macro_f1 = (
        f1_score(all_gold_sentiments, all_pred_sentiments, average="macro", zero_division=0)
        if all_gold_sentiments
        else 0.0
    )

    return {
        "aspect_span_f1": round(span_f1, 4),
        "pair_micro_f1": round(pair_f1, 4),
        "sentiment_macro_f1": round(float(sent_macro_f1), 4),
        "conformal_q90": round(float(np.quantile(nonconformity_scores, 0.90)), 4)
        if nonconformity_scores
        else 1.0,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TRAINING RAISE-ABSA (DeBERTa-Small FP32-Safe) ON {device}")
    print("=" * 60)

    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        free_mb, total_mb = free_b / 1024**2, total_b / 1024**2
        print(f"GPU memory free: {free_mb:.0f} MB of {total_mb:.0f} MB")
        if free_mb < MIN_FREE_VRAM_MB:
            raise SystemExit(
                f"Only {free_mb:.0f} MB of GPU memory is free; training needs about {MIN_FREE_VRAM_MB} MB. "
                "Another process is holding VRAM. Run `nvidia-smi`, close it "
                "(stray python.exe, browser, Jupyter), then retry."
            )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    dataset = load_from_disk(str(TOKENIZED_DATA_DIR))
    collator = RaiseAbsaCollator(pad_token_id=tokenizer.pad_token_id, ignore_index=IGNORE_INDEX)

    train_loader = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
    )

    model = RaiseAbsaModel(MODEL_NAME, use_gradient_checkpointing=True).to(device).float()
    if FREEZE_WORD_EMBEDDINGS:
        model.encoder.embeddings.word_embeddings.weight.requires_grad = False
    ema_teacher = copy.deepcopy(model).to(device).float()
    for param in ema_teacher.parameters():
        param.requires_grad = False

    counts = torch.tensor([748.0, 400.0, 816.0, 40.0], device=device, dtype=torch.float32)
    class_weights = (counts.sum() / (4.0 * counts)).clamp(max=10.0)

    bio_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    sent_criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    print("Running pre-flight evaluation smoke test...")
    smoke_metrics = evaluate_model(ema_teacher, val_loader, device)
    print(f"Pre-flight evaluation smoke test: PASSED -> {smoke_metrics}\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    history = []
    best_pair_f1 = -1.0
    global_step = 0  # counts optimizer steps, used for EMA warm-up

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bio_labels = batch["bio_labels"].to(device)
            sent_labels = batch["span_sentiment_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_batch_indices=batch["span_batch_indices"].to(device),
                span_token_starts=batch["span_token_starts"].to(device),
                span_token_ends=batch["span_token_ends"].to(device),
            )
            loss_bio = bio_criterion(
                outputs["bio_logits"].view(-1, 3),
                bio_labels.view(-1),
            )
            if batch["span_sentiment_labels"].numel() > 0:
                loss_sent = sent_criterion(outputs["sentiment_logits"], sent_labels)
                loss = loss_bio + LAMBDA_SENTIMENT * loss_sent
            else:
                loss = loss_bio

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # EMA update with warm-up: the teacher tracks the student closely
                # at the start, then settles to EMA_DECAY.
                global_step += 1
                decay = min(EMA_DECAY, (1 + global_step) / (10 + global_step))
                with torch.no_grad():
                    for s_param, t_param in zip(model.parameters(), ema_teacher.parameters()):
                        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

        # Evaluate both student and EMA teacher
        student_metrics = evaluate_model(model, val_loader, device)
        teacher_metrics = evaluate_model(ema_teacher, val_loader, device)
        elapsed = round(time.time() - start_time, 1)
        peak_vram = (
            round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
            if torch.cuda.is_available()
            else 0.0
        )

        record = {
            "epoch": epoch,
            "train_loss": round(epoch_loss / len(train_loader), 4),
            "elapsed_sec": elapsed,
            "peak_vram_mb": peak_vram,
            "ema_decay_used": round(decay, 4),
            **teacher_metrics,  # keeps existing keys: teacher metrics (used by plots)
            "student_aspect_span_f1": student_metrics["aspect_span_f1"],
            "student_pair_micro_f1": student_metrics["pair_micro_f1"],
            "student_sentiment_macro_f1": student_metrics["sentiment_macro_f1"],
            "student_conformal_q90": student_metrics["conformal_q90"],
        }
        history.append(record)
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  student: {student_metrics}")
        print(f"  teacher: {teacher_metrics}")
        print(f"  loss={record['train_loss']} time={elapsed}s peak_vram={peak_vram}MB")

        # Checkpoint on whichever of student/teacher is better this epoch
        if student_metrics["pair_micro_f1"] > teacher_metrics["pair_micro_f1"]:
            epoch_best_source, epoch_best_metrics = "student", student_metrics
        else:
            epoch_best_source, epoch_best_metrics = "teacher", teacher_metrics

        if epoch_best_metrics["pair_micro_f1"] > best_pair_f1:
            best_pair_f1 = epoch_best_metrics["pair_micro_f1"]
            torch.save(
                {
                    "student_state_dict": model.state_dict(),
                    "teacher_state_dict": ema_teacher.state_dict(),
                    "best_source": epoch_best_source,
                    "val_metrics": epoch_best_metrics,
                    "student_val_metrics": student_metrics,
                    "teacher_val_metrics": teacher_metrics,
                    "epoch": epoch,
                },
                CHECKPOINT_DIR / "best_raise_absa_deberta.pt",
            )
            print(f"  -> saved checkpoint (best_source={epoch_best_source}, pair_f1={best_pair_f1:.4f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (CHECKPOINT_DIR / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Validation Pair Micro-F1: {best_pair_f1:.4f}")


if __name__ == "__main__":
    main()