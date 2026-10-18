# src/conformal/calibrate_and_abstain.py

import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_from_disk

from src.models.raise_absa import RaiseAbsaModel
from src.data.collator import RaiseAbsaCollator, IGNORE_INDEX
from src.train_source import CHECKPOINT_DIR, TOKENIZED_DATA_DIR, TOKENIZER_DIR, decode_bio_spans
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFORMAL_OUTPUT_DIR = PROJECT_ROOT / "reports" / "conformal"


@torch.no_grad()
def calibrate_conformal_threshold():
    CONFORMAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PHASE 5: CONFORMAL CALIBRATION & SELECTIVE RISK CONTROL")
    print("=" * 60)

    # 1. Load Tokenizer & Model Checkpoint
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    checkpoint_path = CHECKPOINT_DIR / "best_raise_absa_deberta.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {checkpoint_path}. Train source model first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RaiseAbsaModel("microsoft/deberta-v3-small", use_gradient_checkpointing=False).to(device).float()
    
    state_dict = checkpoint.get("teacher_state_dict", checkpoint.get("student_state_dict"))
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load Validation Set for Calibration
    dataset = load_from_disk(str(TOKENIZED_DATA_DIR))
    collator = RaiseAbsaCollator(pad_token_id=tokenizer.pad_token_id, ignore_index=IGNORE_INDEX)
    val_loader = DataLoader(dataset["validation"], batch_size=4, shuffle=False, collate_fn=collator)

    print("Computing validation nonconformity scores for calibration...")
    nonconformity_scores = []
    correct_flags = []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_batch_indices=batch["span_batch_indices"].to(device),
            span_token_starts=batch["span_token_starts"].to(device),
            span_token_ends=batch["span_token_ends"].to(device),
        )

        if batch["span_sentiment_labels"].numel() > 0:
            sent_probs = torch.softmax(outputs["sentiment_logits"].float(), dim=-1).cpu().numpy()
            sent_preds = sent_probs.argmax(axis=-1)
            gold_sents = batch["span_sentiment_labels"].numpy()

            for probs, true_idx, pred_idx in zip(sent_probs, gold_sents, sent_preds):
                # Nonconformity score: 1.0 - probability assigned to the true label
                true_prob = float(probs[true_idx])
                nonconformity_scores.append(1.0 - true_prob)
                correct_flags.append(1 if pred_idx == true_idx else 0)

    if not nonconformity_scores:
        print("Warning: No sentiment labels found in validation batch for calibration.")
        return

    # 3. Compute Conformal Quantile (Target alpha = 0.10 -> 90% confidence / 10% error bound)
    alpha = 0.10
    q_level = np.ceil((len(nonconformity_scores) + 1) * (1 - alpha)) / len(nonconformity_scores)
    q_level = min(q_level, 1.0)
    conformal_threshold = float(np.quantile(nonconformity_scores, q_level))

    print(f"\nCalibration Results:")
    print(f"- Total Calibration Samples: {len(nonconformity_scores)}")
    print(f"- Target Error Rate (alpha): {alpha}")
    print(f"- Computed Conformal Threshold (q_90): {conformal_threshold:.4f}")

    # 4. Evaluate Selective Risk & Coverage on Test Set
    test_loader = DataLoader(dataset["test"], batch_size=4, shuffle=False, collate_fn=collator)
    test_scores = []
    test_correct = []

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

        if batch["span_sentiment_labels"].numel() > 0:
            sent_probs = torch.softmax(outputs["sentiment_logits"].float(), dim=-1).cpu().numpy()
            sent_preds = sent_probs.argmax(axis=-1)
            gold_sents = batch["span_sentiment_labels"].numpy()

            for probs, true_idx, pred_idx in zip(sent_probs, gold_sents, sent_preds):
                max_prob = float(probs.max())
                score = 1.0 - max_prob  # Uncertainty score based on prediction confidence
                test_scores.append(score)
                test_correct.append(1 if pred_idx == true_idx else 0)

    test_scores = np.array(test_scores)
    test_correct = np.array(test_correct)

    # Apply conformal abstention filter (accept only predictions with score <= conformal_threshold)
    accepted_mask = test_scores <= conformal_threshold
    coverage = float(accepted_mask.mean())
    selective_accuracy = float(test_correct[accepted_mask].mean()) if accepted_mask.sum() > 0 else 0.0
    selective_risk = 1.0 - selective_accuracy

    evaluation_summary = {
        "alpha": alpha,
        "conformal_threshold": conformal_threshold,
        "test_total_samples": len(test_scores),
        "test_coverage": round(coverage, 4),
        "test_selective_accuracy": round(selective_accuracy, 4),
        "test_selective_risk": round(selective_risk, 4),
    }

    output_path = CONFORMAL_OUTPUT_DIR / "conformal_calibration_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    print(f"\nLocked Test Set Conformal Evaluation:")
    print(json.dumps(evaluation_summary, indent=2))
    print(f"\nSaved conformal calibration summary to:\n{output_path}")


if __name__ == "__main__":
    calibrate_conformal_threshold()