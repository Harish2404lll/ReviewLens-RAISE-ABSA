# src/evaluation/evaluate_source_model.py

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import load_from_disk
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_DIRECTORY),
    )

from training.train_source_baseline import (  # noqa: E402
    BACKBONE_DIRECTORY,
    TOKENIZED_DATASET_DIRECTORY,
    BIO_LABELS,
    SENTIMENT_LABELS,
    NUMBER_OF_BIO_LABELS,
    NUMBER_OF_SENTIMENT_LABELS,
    JointABSACollator,
    JointDebertaABSA,
    calculate_class_weights,
    decode_bio_spans,
    model_forward_arguments,
    move_batch_to_device,
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen RAISE-ABSA source "
            "baseline on the held-out test split."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to the selected best_checkpoint "
            "directory."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Test batch size. Default: 8.",
    )

    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help=(
            "Allow repeated evaluation when a test "
            "marker already exists. Do not use this "
            "for test-driven model tuning."
        ),
    )

    return parser.parse_args()


# ============================================================
# FILE UTILITIES
# ============================================================

def save_json(data, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_jsonl(records, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# CHECKPOINT VALIDATION
# ============================================================

def validate_checkpoint(checkpoint_directory):
    checkpoint_directory = (
        checkpoint_directory.resolve()
    )

    required_paths = [
        checkpoint_directory
        / "lora_adapter",
        checkpoint_directory
        / "lora_adapter"
        / "adapter_config.json",
        checkpoint_directory
        / "task_heads.safetensors",
        checkpoint_directory
        / "checkpoint_metadata.json",
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        missing_text = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            "The selected checkpoint is incomplete.\n"
            f"Missing:\n{missing_text}"
        )

    adapter_safetensors = (
        checkpoint_directory
        / "lora_adapter"
        / "adapter_model.safetensors"
    )

    adapter_binary = (
        checkpoint_directory
        / "lora_adapter"
        / "adapter_model.bin"
    )

    if not (
        adapter_safetensors.exists()
        or adapter_binary.exists()
    ):
        raise FileNotFoundError(
            "No LoRA adapter weights were found in:\n"
            f"{checkpoint_directory / 'lora_adapter'}"
        )

    return checkpoint_directory


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint_model(
    checkpoint_directory,
    device,
):
    metadata_path = (
        checkpoint_directory
        / "checkpoint_metadata.json"
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    config = metadata.get(
        "configuration",
        {},
    )

    required_config_keys = [
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "classifier_dropout",
    ]

    missing_config = [
        key
        for key in required_config_keys
        if key not in config
    ]

    if missing_config:
        raise KeyError(
            "Checkpoint metadata is missing "
            f"configuration values: {missing_config}"
        )

    print("Creating model architecture...")

    model = JointDebertaABSA(
        backbone_directory=(
            BACKBONE_DIRECTORY
        ),
        bio_classes=(
            NUMBER_OF_BIO_LABELS
        ),
        sentiment_classes=(
            NUMBER_OF_SENTIMENT_LABELS
        ),
        lora_rank=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config[
            "lora_dropout"
        ],
        classifier_dropout=config[
            "classifier_dropout"
        ],
    )

    adapter_directory = (
        checkpoint_directory
        / "lora_adapter"
    )

    adapter_safetensors = (
        adapter_directory
        / "adapter_model.safetensors"
    )

    adapter_binary = (
        adapter_directory
        / "adapter_model.bin"
    )

    print("Loading LoRA adapter weights...")

    if adapter_safetensors.exists():
        adapter_state = load_file(
            str(adapter_safetensors),
            device="cpu",
        )
    elif adapter_binary.exists():
        adapter_state = torch.load(
            adapter_binary,
            map_location="cpu",
            weights_only=True,
        )
    else:
        raise FileNotFoundError(
            "LoRA adapter weights are missing."
        )

    adapter_result = (
        set_peft_model_state_dict(
            model.encoder,
            adapter_state,
            adapter_name="default",
        )
    )

    unexpected_adapter_keys = list(
        getattr(
            adapter_result,
            "unexpected_keys",
            [],
        )
    )

    if unexpected_adapter_keys:
        raise RuntimeError(
            "Unexpected LoRA adapter keys:\n"
            + "\n".join(
                unexpected_adapter_keys
            )
        )

    print("Loading task-head weights...")

    head_state = load_file(
        str(
            checkpoint_directory
            / "task_heads.safetensors"
        ),
        device="cpu",
    )

    bio_prefix = "bio_classifier."
    sentiment_prefix = (
        "sentiment_classifier."
    )

    bio_state = {
        key[len(bio_prefix):]: value
        for key, value in head_state.items()
        if key.startswith(bio_prefix)
    }

    sentiment_state = {
        key[len(sentiment_prefix):]: value
        for key, value in head_state.items()
        if key.startswith(sentiment_prefix)
    }

    if not bio_state:
        raise RuntimeError(
            "BIO classifier weights were not "
            "found in task_heads.safetensors."
        )

    if not sentiment_state:
        raise RuntimeError(
            "Sentiment classifier weights were not "
            "found in task_heads.safetensors."
        )

    model.bio_classifier.load_state_dict(
        bio_state,
        strict=True,
    )

    model.sentiment_classifier.load_state_dict(
        sentiment_state,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, metadata


# ============================================================
# PREDICTION HELPERS
# ============================================================

def decode_term(
    tokenizer,
    input_ids,
    start,
    end,
):
    term = tokenizer.decode(
        input_ids[start:end],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    return term


def calculate_prf(
    correct,
    predicted,
    gold,
):
    precision = safe_divide(
        correct,
        predicted,
    )

    recall = safe_divide(
        correct,
        gold,
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return precision, recall, f1


# ============================================================
# TEST EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate_test_split(
    model,
    test_loader,
    test_dataset,
    tokenizer,
    device,
    bio_weights,
    sentiment_weights,
):
    model.eval()

    total_loss = 0.0
    total_bio_loss = 0.0
    total_sentiment_loss = 0.0
    processed_batches = 0

    gold_span_count = 0
    predicted_span_count = 0
    correct_span_count = 0

    gold_triplet_count = 0
    predicted_triplet_count = 0
    correct_triplet_count = 0

    oracle_gold_sentiments = []
    oracle_predicted_sentiments = []

    prediction_records = []
    error_records = []

    dataset_cursor = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for batch in tqdm(
        test_loader,
        desc="Testing",
    ):
        batch_size = batch[
            "input_ids"
        ].shape[0]

        original_records = [
            test_dataset[
                dataset_cursor + offset
            ]
            for offset in range(batch_size)
        ]

        dataset_cursor += batch_size

        batch = move_batch_to_device(
            batch,
            device,
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(
                **model_forward_arguments(batch),
                bio_class_weights=bio_weights,
                sentiment_class_weights=(
                    sentiment_weights
                ),
            )

        processed_batches += 1

        if outputs["loss"] is not None:
            loss_value = (
                outputs["loss"]
                .detach()
                .float()
                .item()
            )

            if not np.isfinite(loss_value):
                raise RuntimeError(
                    "Non-finite test loss detected."
                )

            total_loss += loss_value

        if outputs["bio_loss"] is not None:
            total_bio_loss += (
                outputs["bio_loss"]
                .detach()
                .float()
                .item()
            )

        if (
            outputs["sentiment_loss"]
            is not None
        ):
            total_sentiment_loss += (
                outputs["sentiment_loss"]
                .detach()
                .float()
                .item()
            )

        predicted_bio = (
            outputs["bio_logits"]
            .argmax(dim=-1)
        )

        gold_bio = batch["bio_labels"]

        predicted_spans_per_record = []

        for record_index in range(
            batch_size
        ):
            predicted_labels = (
                predicted_bio[record_index]
                .detach()
                .cpu()
                .tolist()
            )

            gold_labels = (
                gold_bio[record_index]
                .detach()
                .cpu()
                .tolist()
            )

            masked_predictions = [
                (
                    predicted_label
                    if gold_label != -100
                    else -100
                )
                for predicted_label, gold_label
                in zip(
                    predicted_labels,
                    gold_labels,
                )
            ]

            predicted_spans = (
                decode_bio_spans(
                    masked_predictions
                )
            )

            predicted_spans_per_record.append(
                predicted_spans
            )

        # ----------------------------------------------------
        # Gold aspects and oracle sentiment predictions
        # ----------------------------------------------------

        gold_batch_indices = (
            batch["aspect_batch_indices"]
            .detach()
            .cpu()
            .tolist()
        )

        gold_starts = (
            batch["aspect_token_starts"]
            .detach()
            .cpu()
            .tolist()
        )

        gold_ends = (
            batch["aspect_token_ends"]
            .detach()
            .cpu()
            .tolist()
        )

        gold_sentiments = (
            batch["aspect_sentiment_ids"]
            .detach()
            .cpu()
            .tolist()
        )

        gold_triplets_per_record = [
            set()
            for _ in range(batch_size)
        ]

        gold_items_per_record = [
            []
            for _ in range(batch_size)
        ]

        if gold_sentiments:
            oracle_predictions = (
                outputs["sentiment_logits"]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .tolist()
            )
        else:
            oracle_predictions = []

        for (
            record_index,
            start,
            end,
            sentiment,
            oracle_prediction,
        ) in zip(
            gold_batch_indices,
            gold_starts,
            gold_ends,
            gold_sentiments,
            oracle_predictions,
        ):
            gold_triplet = (
                int(start),
                int(end),
                int(sentiment),
            )

            gold_triplets_per_record[
                record_index
            ].add(gold_triplet)

            term = decode_term(
                tokenizer=tokenizer,
                input_ids=(
                    batch["input_ids"][
                        record_index
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                ),
                start=start,
                end=end,
            )

            gold_items_per_record[
                record_index
            ].append(
                {
                    "term": term,
                    "token_start": int(start),
                    "token_end": int(end),
                    "sentiment_id": int(
                        sentiment
                    ),
                    "sentiment": (
                        SENTIMENT_LABELS[
                            int(sentiment)
                        ]
                    ),
                    "oracle_prediction": (
                        SENTIMENT_LABELS[
                            int(
                                oracle_prediction
                            )
                        ]
                    ),
                    "oracle_correct": (
                        int(sentiment)
                        == int(
                            oracle_prediction
                        )
                    ),
                }
            )

            oracle_gold_sentiments.append(
                int(sentiment)
            )

            oracle_predicted_sentiments.append(
                int(oracle_prediction)
            )

        # ----------------------------------------------------
        # Sentiment classification for predicted spans
        # ----------------------------------------------------

        predicted_batch_indices = []
        predicted_starts = []
        predicted_ends = []

        for record_index, spans in enumerate(
            predicted_spans_per_record
        ):
            for start, end in spans:
                predicted_batch_indices.append(
                    record_index
                )
                predicted_starts.append(start)
                predicted_ends.append(end)

        if predicted_batch_indices:
            predicted_batch_tensor = torch.tensor(
                predicted_batch_indices,
                dtype=torch.long,
                device=device,
            )

            predicted_start_tensor = torch.tensor(
                predicted_starts,
                dtype=torch.long,
                device=device,
            )

            predicted_end_tensor = torch.tensor(
                predicted_ends,
                dtype=torch.long,
                device=device,
            )

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                predicted_sentiment_logits = (
                    model.classify_spans(
                        hidden_states=(
                            outputs[
                                "hidden_states"
                            ]
                        ),
                        aspect_batch_indices=(
                            predicted_batch_tensor
                        ),
                        aspect_token_starts=(
                            predicted_start_tensor
                        ),
                        aspect_token_ends=(
                            predicted_end_tensor
                        ),
                    )
                )

            probabilities = torch.softmax(
                predicted_sentiment_logits.float(),
                dim=-1,
            )

            predicted_sentiment_ids = (
                probabilities.argmax(dim=-1)
            )

            predicted_confidences = (
                probabilities.max(dim=-1).values
            )

            predicted_sentiment_ids = (
                predicted_sentiment_ids
                .detach()
                .cpu()
                .tolist()
            )

            predicted_confidences = (
                predicted_confidences
                .detach()
                .cpu()
                .tolist()
            )
        else:
            predicted_sentiment_ids = []
            predicted_confidences = []

        predicted_triplets_per_record = [
            set()
            for _ in range(batch_size)
        ]

        predicted_items_per_record = [
            []
            for _ in range(batch_size)
        ]

        for (
            record_index,
            start,
            end,
            sentiment_id,
            confidence,
        ) in zip(
            predicted_batch_indices,
            predicted_starts,
            predicted_ends,
            predicted_sentiment_ids,
            predicted_confidences,
        ):
            predicted_triplets_per_record[
                record_index
            ].add(
                (
                    int(start),
                    int(end),
                    int(sentiment_id),
                )
            )

            term = decode_term(
                tokenizer=tokenizer,
                input_ids=(
                    batch["input_ids"][
                        record_index
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                ),
                start=start,
                end=end,
            )

            predicted_items_per_record[
                record_index
            ].append(
                {
                    "term": term,
                    "token_start": int(start),
                    "token_end": int(end),
                    "sentiment_id": int(
                        sentiment_id
                    ),
                    "sentiment": (
                        SENTIMENT_LABELS[
                            int(sentiment_id)
                        ]
                    ),
                    "confidence": float(
                        confidence
                    ),
                }
            )

        # ----------------------------------------------------
        # Record-level metrics and saved predictions
        # ----------------------------------------------------

        for record_index in range(
            batch_size
        ):
            gold_triplets = (
                gold_triplets_per_record[
                    record_index
                ]
            )

            predicted_triplets = (
                predicted_triplets_per_record[
                    record_index
                ]
            )

            gold_spans = {
                (start, end)
                for start, end, _
                in gold_triplets
            }

            predicted_spans = {
                (start, end)
                for start, end, _
                in predicted_triplets
            }

            gold_span_count += len(
                gold_spans
            )

            predicted_span_count += len(
                predicted_spans
            )

            correct_span_count += len(
                gold_spans
                & predicted_spans
            )

            gold_triplet_count += len(
                gold_triplets
            )

            predicted_triplet_count += len(
                predicted_triplets
            )

            correct_triplet_count += len(
                gold_triplets
                & predicted_triplets
            )

            source_record = (
                original_records[
                    record_index
                ]
            )

            prediction_record = {
                "id": str(
                    source_record.get(
                        "id",
                        batch["record_ids"][
                            record_index
                        ],
                    )
                ),
                "text": source_record.get(
                    "text"
                ),
                "gold_aspects": (
                    gold_items_per_record[
                        record_index
                    ]
                ),
                "predicted_aspects": (
                    predicted_items_per_record[
                        record_index
                    ]
                ),
                "exact_span_match": (
                    gold_spans
                    == predicted_spans
                ),
                "exact_triplet_match": (
                    gold_triplets
                    == predicted_triplets
                ),
            }

            prediction_records.append(
                prediction_record
            )

            if (
                gold_triplets
                != predicted_triplets
            ):
                error_record = {
                    **prediction_record,
                    "missing_triplets": [
                        {
                            "token_start": start,
                            "token_end": end,
                            "sentiment": (
                                SENTIMENT_LABELS[
                                    sentiment
                                ]
                            ),
                        }
                        for start, end, sentiment
                        in sorted(
                            gold_triplets
                            - predicted_triplets
                        )
                    ],
                    "extra_triplets": [
                        {
                            "token_start": start,
                            "token_end": end,
                            "sentiment": (
                                SENTIMENT_LABELS[
                                    sentiment
                                ]
                            ),
                        }
                        for start, end, sentiment
                        in sorted(
                            predicted_triplets
                            - gold_triplets
                        )
                    ],
                }

                error_records.append(
                    error_record
                )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    if processed_batches == 0:
        raise RuntimeError(
            "No test batches were evaluated."
        )

    span_precision, span_recall, span_f1 = (
        calculate_prf(
            correct=correct_span_count,
            predicted=predicted_span_count,
            gold=gold_span_count,
        )
    )

    (
        triplet_precision,
        triplet_recall,
        triplet_f1,
    ) = calculate_prf(
        correct=correct_triplet_count,
        predicted=predicted_triplet_count,
        gold=gold_triplet_count,
    )

    oracle_accuracy = accuracy_score(
        oracle_gold_sentiments,
        oracle_predicted_sentiments,
    )

    oracle_macro_f1 = f1_score(
        oracle_gold_sentiments,
        oracle_predicted_sentiments,
        labels=list(
            range(
                NUMBER_OF_SENTIMENT_LABELS
            )
        ),
        average="macro",
        zero_division=0,
    )

    (
        class_precision,
        class_recall,
        class_f1,
        class_support,
    ) = precision_recall_fscore_support(
        oracle_gold_sentiments,
        oracle_predicted_sentiments,
        labels=list(
            range(
                NUMBER_OF_SENTIMENT_LABELS
            )
        ),
        zero_division=0,
    )

    sentiment_confusion = confusion_matrix(
        oracle_gold_sentiments,
        oracle_predicted_sentiments,
        labels=list(
            range(
                NUMBER_OF_SENTIMENT_LABELS
            )
        ),
    )

    per_class_metrics = {}

    for class_id, class_name in (
        SENTIMENT_LABELS.items()
    ):
        per_class_metrics[class_name] = {
            "precision": float(
                class_precision[class_id]
            ),
            "recall": float(
                class_recall[class_id]
            ),
            "f1": float(
                class_f1[class_id]
            ),
            "support": int(
                class_support[class_id]
            ),
        }

    peak_allocated_memory = 0.0
    peak_reserved_memory = 0.0

    if device.type == "cuda":
        peak_allocated_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        peak_reserved_memory = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )

    number_of_sentences = len(
        test_dataset
    )

    metrics = {
        "test_sentences": (
            number_of_sentences
        ),
        "test_aspects": (
            gold_triplet_count
        ),
        "average_test_loss": (
            total_loss
            / processed_batches
        ),
        "average_bio_loss": (
            total_bio_loss
            / processed_batches
        ),
        "average_sentiment_loss": (
            total_sentiment_loss
            / processed_batches
        ),
        "aspect_span": {
            "precision": span_precision,
            "recall": span_recall,
            "f1": span_f1,
            "gold_count": gold_span_count,
            "predicted_count": (
                predicted_span_count
            ),
            "correct_count": (
                correct_span_count
            ),
        },
        "oracle_sentiment": {
            "accuracy": float(
                oracle_accuracy
            ),
            "macro_f1": float(
                oracle_macro_f1
            ),
            "per_class": (
                per_class_metrics
            ),
            "confusion_matrix": (
                sentiment_confusion.tolist()
            ),
        },
        "end_to_end_triplet": {
            "precision": (
                triplet_precision
            ),
            "recall": triplet_recall,
            "f1": triplet_f1,
            "gold_count": (
                gold_triplet_count
            ),
            "predicted_count": (
                predicted_triplet_count
            ),
            "correct_count": (
                correct_triplet_count
            ),
        },
        "execution": {
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "milliseconds_per_sentence": (
                1000
                * elapsed_seconds
                / number_of_sentences
            ),
            "sentences_per_second": (
                number_of_sentences
                / elapsed_seconds
            ),
            "peak_allocated_gpu_memory_gb": (
                peak_allocated_memory
            ),
            "peak_reserved_gpu_memory_gb": (
                peak_reserved_memory
            ),
        },
        "records_with_triplet_errors": (
            len(error_records)
        ),
    }

    return (
        metrics,
        prediction_records,
        error_records,
    )


# ============================================================
# CONFUSION-MATRIX CHART
# ============================================================

def create_confusion_matrix_chart(
    confusion,
    output_path,
):
    matrix = np.asarray(
        confusion,
        dtype=int,
    )

    class_names = [
        SENTIMENT_LABELS[index]
        for index in range(
            NUMBER_OF_SENTIMENT_LABELS
        )
    ]

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(
        matrix,
        cmap="Blues",
        interpolation="nearest",
    )

    figure.colorbar(image, ax=axis)

    axis.set_xticks(
        np.arange(len(class_names))
    )

    axis.set_yticks(
        np.arange(len(class_names))
    )

    axis.set_xticklabels(
        class_names,
        rotation=30,
        ha="right",
    )

    axis.set_yticklabels(class_names)

    axis.set_xlabel(
        "Predicted sentiment"
    )

    axis.set_ylabel(
        "True sentiment"
    )

    axis.set_title(
        "Held-Out Test Sentiment "
        "Confusion Matrix"
    )

    threshold = (
        matrix.max() / 2
        if matrix.size
        else 0
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, column]
                    > threshold
                    else "black"
                ),
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# MAIN
# ============================================================

def main():
    arguments = parse_arguments()

    print("=" * 70)
    print("RAISE-ABSA HELD-OUT SOURCE TEST EVALUATION")
    print("=" * 70)

    checkpoint_directory = (
        validate_checkpoint(
            arguments.checkpoint
        )
    )

    evaluation_marker = (
        checkpoint_directory
        / "TEST_EVALUATED.json"
    )

    if (
        evaluation_marker.exists()
        and not arguments.allow_repeat
    ):
        raise FileExistsError(
            "This checkpoint has already been "
            "evaluated on the test split:\n"
            f"{evaluation_marker}\n\n"
            "Do not repeatedly inspect test results "
            "while modifying the model. Use "
            "--allow-repeat only to reproduce the "
            "same frozen evaluation."
        )

    if not TOKENIZED_DATASET_DIRECTORY.exists():
        raise FileNotFoundError(
            "Tokenized dataset not found:\n"
            f"{TOKENIZED_DATASET_DIRECTORY}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:")
    print(device)

    if device.type == "cuda":
        print("\nGPU:")
        print(
            torch.cuda.get_device_name(0)
        )
    else:
        print(
            "\nWarning: CUDA is unavailable. "
            "Evaluation will run on CPU."
        )

    print("\nCheckpoint:")
    print(checkpoint_directory)

    metadata_path = (
        checkpoint_directory
        / "checkpoint_metadata.json"
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        checkpoint_metadata = json.load(file)

    selected_epoch = (
        checkpoint_metadata.get("epoch")
    )

    validation_metrics = (
        checkpoint_metadata.get(
            "validation_metrics",
            {},
        )
    )

    print("\nSelected epoch:")
    print(selected_epoch)

    print(
        "\nCheckpoint validation triplet F1:"
    )

    print(
        validation_metrics.get(
            "triplet_f1",
            "Not recorded",
        )
    )

    tokenizer_path = (
        checkpoint_directory
        / "tokenizer"
    )

    if not tokenizer_path.exists():
        tokenizer_path = (
            PROJECT_ROOT
            / "models"
            / "deberta-v3-base-tokenizer"
        )

    print("\nLoading tokenizer...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            use_fast=True,
        )
    )

    print("Loading tokenized dataset...")

    dataset = load_from_disk(
        str(
            TOKENIZED_DATASET_DIRECTORY
        )
    )

    if "test" not in dataset:
        raise KeyError(
            "The tokenized dataset does not "
            "contain a test split."
        )

    if "train" not in dataset:
        raise KeyError(
            "The train split is required only "
            "to reconstruct the original class "
            "weights used during training."
        )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print(
        "Test sentences:",
        len(test_dataset),
    )

    (
        _,
        _,
        bio_weights,
        sentiment_weights,
    ) = calculate_class_weights(
        train_dataset
    )

    bio_weights = bio_weights.to(device)

    sentiment_weights = (
        sentiment_weights.to(device)
    )

    collator = JointABSACollator(
        tokenizer=tokenizer,
        padding_multiple=8,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model, loaded_metadata = (
        load_checkpoint_model(
            checkpoint_directory=(
                checkpoint_directory
            ),
            device=device,
        )
    )

    print("\nEvaluating frozen model...")

    (
        metrics,
        predictions,
        errors,
    ) = evaluate_test_split(
        model=model,
        test_loader=test_loader,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        bio_weights=bio_weights,
        sentiment_weights=(
            sentiment_weights
        ),
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_directory = (
        PROJECT_ROOT
        / "reports"
        / "source_baseline"
        / f"test_{timestamp}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    metrics["checkpoint"] = str(
        checkpoint_directory
    )

    metrics["selected_epoch"] = (
        loaded_metadata.get("epoch")
    )

    metrics["validation_metrics"] = (
        loaded_metadata.get(
            "validation_metrics",
            {},
        )
    )

    save_json(
        metrics,
        output_directory
        / "test_metrics.json",
    )

    save_jsonl(
        predictions,
        output_directory
        / "test_predictions.jsonl",
    )

    save_jsonl(
        errors,
        output_directory
        / "test_errors.jsonl",
    )

    create_confusion_matrix_chart(
        confusion=metrics[
            "oracle_sentiment"
        ]["confusion_matrix"],
        output_path=(
            output_directory
            / "test_confusion_matrix.png"
        ),
    )

    marker_data = {
        "evaluated_at": timestamp,
        "checkpoint": str(
            checkpoint_directory
        ),
        "output_directory": str(
            output_directory
        ),
        "test_metrics_file": str(
            output_directory
            / "test_metrics.json"
        ),
        "warning": (
            "Do not tune model hyperparameters "
            "using held-out test results."
        ),
    }

    save_json(
        marker_data,
        evaluation_marker,
    )

    print("\n" + "=" * 70)
    print("HELD-OUT TEST EVALUATION COMPLETED")
    print("=" * 70)

    print("\nTest loss:")
    print(
        f"{metrics['average_test_loss']:.4f}"
    )

    print("\nAspect span P/R/F1:")
    print(
        f"{metrics['aspect_span']['precision']:.4f} / "
        f"{metrics['aspect_span']['recall']:.4f} / "
        f"{metrics['aspect_span']['f1']:.4f}"
    )

    print("\nOracle sentiment accuracy:")
    print(
        f"{metrics['oracle_sentiment']['accuracy']:.4f}"
    )

    print("\nOracle sentiment Macro-F1:")
    print(
        f"{metrics['oracle_sentiment']['macro_f1']:.4f}"
    )

    print("\nEnd-to-end triplet P/R/F1:")
    print(
        f"{metrics['end_to_end_triplet']['precision']:.4f} / "
        f"{metrics['end_to_end_triplet']['recall']:.4f} / "
        f"{metrics['end_to_end_triplet']['f1']:.4f}"
    )

    print("\nInference speed:")
    print(
        f"{metrics['execution']['milliseconds_per_sentence']:.3f} "
        "ms/sentence"
    )

    print("\nPeak allocated GPU memory:")
    print(
        f"{metrics['execution']['peak_allocated_gpu_memory_gb']:.3f} GB"
    )

    print("\nResults saved to:")
    print(output_directory)

    print(
        "\nDo not modify the source model based "
        "on the held-out test result."
    )


if __name__ == "__main__":
    main()