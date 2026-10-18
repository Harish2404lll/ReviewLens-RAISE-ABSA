# src/evaluate_and_plot.py

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data.collator import RaiseAbsaCollator, IGNORE_INDEX
from src.models.raise_absa import RaiseAbsaModel
from src.train_source import TOKENIZED_DATA_DIR, TOKENIZER_DIR, CHECKPOINT_DIR, decode_bio_spans
from transformers import AutoTokenizer
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
SENTIMENT_NAMES = ["negative", "neutral", "positive", "conflict"]


@torch.no_grad()
def evaluate_and_plot():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("EVALUATING ON LOCKED TEST SET & GENERATING PLOTS")
    print("=" * 60)

    # Load Tokenizer & Test Data
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    dataset = load_from_disk(str(TOKENIZED_DATA_DIR))
    collator = RaiseAbsaCollator(pad_token_id=tokenizer.pad_token_id, ignore_index=IGNORE_INDEX)
    test_loader = DataLoader(dataset["test"], batch_size=4, shuffle=False, collate_fn=collator)

    # Load Best Checkpoint
    checkpoint_path = CHECKPOINT_DIR / "best_raise_absa_deberta.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RaiseAbsaModel("microsoft/deberta-v3-small", use_gradient_checkpointing=False).to(device).float()
    
    # Load teacher or student weights depending on saved format
    state_dict = checkpoint.get("teacher_state_dict", checkpoint.get("student_state_dict"))
    model.load_state_dict(state_dict)
    model.eval()

    gold_spans_total = 0
    pred_spans_total = 0
    exact_span_matches = 0
    exact_pair_matches = 0

    all_gold_sentiments = []
    all_pred_sentiments = []
    all_max_probs = []

    for batch in test_loader:
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
            all_max_probs.extend(sent_probs.max(axis=-1).tolist())

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

    test_metrics = {
        "aspect_span_f1": round(span_f1, 4),
        "aspect_span_precision": round(span_p, 4),
        "aspect_span_recall": round(span_r, 4),
        "pair_micro_f1": round(pair_f1, 4),
        "pair_micro_precision": round(pair_p, 4),
        "pair_micro_recall": round(pair_r, 4),
    }

    print("\nLocked Test Set Evaluation Results:")
    print(json.dumps(test_metrics, indent=2))

    # ==========================================================
    # GENERATE PLOTS & REPORTS
    # ==========================================================
    gold_arr = np.array(all_gold_sentiments)
    pred_arr = np.array(all_pred_sentiments)
    conf_arr = np.array(all_max_probs)

    # 1. Plot Training History Convergence
    history_file = CHECKPOINT_DIR / "training_history.json"
    if history_file.exists():
        with history_file.open("r", encoding="utf-8") as f:
            history = json.load(f)
        epochs = [h["epoch"] for h in history]
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [h["aspect_span_f1"] for h in history], marker="o", label="Aspect Span F1")
        plt.plot(epochs, [h["pair_micro_f1"] for h in history], marker="s", label="Pair Micro-F1 (Teacher)")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("RAISE-ABSA Source Validation Convergence")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "plot1_validation_convergence.png", dpi=300)
        plt.close()

    # 2. Plot Confusion Matrix
    cm = confusion_matrix(gold_arr, pred_arr, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SENTIMENT_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("RAISE-ABSA Locked Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "plot2_confusion_matrix.png", dpi=300)
    plt.close()

    # 3. Plot Selective Risk vs. Coverage Curve (Conformal Evaluation)
    if len(conf_arr) > 0:
        sorted_indices = np.argsort(-conf_arr)
        sorted_errors = (pred_arr[sorted_indices] != gold_arr[sorted_indices]).astype(float)
        coverages = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        selective_risks = np.cumsum(sorted_errors) / np.arange(1, len(sorted_errors) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(coverages, selective_risks, color="crimson", linewidth=2, label="RAISE-ABSA Selective Risk")
        plt.xlabel("Coverage (Fraction of Accepted Predictions)")
        plt.ylabel("Empirical Selective Error Rate (Risk)")
        plt.title("Conformal Selective Risk vs. Coverage Curve (Locked Test Set)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "plot3_selective_risk_coverage.png", dpi=300)
        plt.close()

    report_dict = classification_report(
        gold_arr, pred_arr, target_names=SENTIMENT_NAMES, output_dict=True, zero_division=0
    )
    with (REPORTS_DIR / "test_evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"test_metrics": test_metrics, "classification_report": report_dict}, f, indent=2)

    print(f"\nSaved evaluation metrics and figures to:\n{REPORTS_DIR}")


if __name__ == "__main__":
    evaluate_and_plot()