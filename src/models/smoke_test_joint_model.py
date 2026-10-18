import math
import sys
import traceback

from collections import Counter
from pathlib import Path

import torch

from datasets import load_from_disk
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIRECTORY),
    )

from data.joint_absa_collator import (
    JointABSACollator,
)

from models.joint_absa_model import (
    JointDebertaABSA,
)


TOKENIZED_DATASET_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "deberta_v3_laptop_max96"
)

TOKENIZER_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "deberta-v3-base-tokenizer"
)

BIO_CLASS_COUNT = 3
SENTIMENT_CLASS_COUNT = 4

IGNORE_INDEX = -100


def calculate_inverse_sqrt_weights(
    counts,
    number_of_classes,
):
    """
    Inverse square-root weighting is less extreme than direct
    inverse-frequency weighting.
    """

    raw_weights = []

    for class_id in range(
        number_of_classes
    ):
        count = counts.get(
            class_id,
            0,
        )

        if count <= 0:
            raise ValueError(
                f"Class {class_id} has no examples."
            )

        raw_weights.append(
            1.0 / math.sqrt(count)
        )

    mean_weight = sum(raw_weights) / len(
        raw_weights
    )

    return [
        weight / mean_weight
        for weight in raw_weights
    ]


def calculate_class_weights(
    train_dataset,
):
    bio_counts = Counter()
    sentiment_counts = Counter()

    for record in train_dataset:
        for label_id in record["bio_labels"]:
            if label_id != IGNORE_INDEX:
                bio_counts[int(label_id)] += 1

        for sentiment_id in record[
            "aspect_sentiment_ids"
        ]:
            sentiment_counts[
                int(sentiment_id)
            ] += 1

    bio_weights = (
        calculate_inverse_sqrt_weights(
            bio_counts,
            BIO_CLASS_COUNT,
        )
    )

    sentiment_weights = (
        calculate_inverse_sqrt_weights(
            sentiment_counts,
            SENTIMENT_CLASS_COUNT,
        )
    )

    return (
        bio_counts,
        sentiment_counts,
        bio_weights,
        sentiment_weights,
    )


def choose_smoke_test_records(
    train_dataset,
):
    """
    Select one long sentence containing multiple aspects and
    one long sentence without aspects.
    """

    multi_aspect_records = [
        record
        for record in train_dataset
        if len(record["aspect_terms"]) >= 2
    ]

    no_aspect_records = [
        record
        for record in train_dataset
        if len(record["aspect_terms"]) == 0
    ]

    if not multi_aspect_records:
        raise ValueError(
            "No multi-aspect training record found."
        )

    if not no_aspect_records:
        raise ValueError(
            "No no-aspect training record found."
        )

    multi_aspect_record = max(
        multi_aspect_records,
        key=lambda record: record[
            "sequence_length"
        ],
    )

    no_aspect_record = max(
        no_aspect_records,
        key=lambda record: record[
            "sequence_length"
        ],
    )

    return [
        multi_aspect_record,
        no_aspect_record,
    ]


def move_batch_to_device(
    batch,
    device,
):
    return {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value in batch.items()
    }


def main():
    print("=" * 60)
    print("JOINT ABSA GPU SMOKE TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. The smoke test "
            "requires the NVIDIA GPU."
        )

    device = torch.device("cuda")

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIRECTORY,
        local_files_only=True,
        use_fast=True,
    )

    print("Loading tokenized dataset...")

    dataset = load_from_disk(
        str(TOKENIZED_DATASET_DIRECTORY)
    )

    train_dataset = dataset["train"]

    (
        bio_counts,
        sentiment_counts,
        bio_weights,
        sentiment_weights,
    ) = calculate_class_weights(
        train_dataset
    )

    print("\nBIO counts:")
    print(dict(sorted(bio_counts.items())))

    print("BIO weights:")
    print(
        [
            round(weight, 4)
            for weight in bio_weights
        ]
    )

    print("\nSentiment counts:")
    print(
        dict(
            sorted(
                sentiment_counts.items()
            )
        )
    )

    print("Sentiment weights:")
    print(
        [
            round(weight, 4)
            for weight in sentiment_weights
        ]
    )

    selected_records = (
        choose_smoke_test_records(
            train_dataset
        )
    )

    print("\nSelected smoke-test records:")

    for record in selected_records:
        print(
            f"ID={record['id']}, "
            f"tokens={record['sequence_length']}, "
            f"aspects={len(record['aspect_terms'])}"
        )

    collator = JointABSACollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    batch = collator(
        selected_records
    )

    print("\nBatch shapes:")

    for key, value in batch.items():
        print(
            f"{key}: {tuple(value.shape)}"
        )

    print("\nCreating joint model...")

    model = JointDebertaABSA(
        bio_class_weights=bio_weights,
        sentiment_class_weights=(
            sentiment_weights
        ),
        bio_loss_weight=1.0,
        sentiment_loss_weight=1.0,
        enable_gradient_checkpointing=True,
    )

    parameter_summary = (
        model.trainable_parameter_summary()
    )

    print("\nTrainable parameters:")
    print(
        f"{parameter_summary['trainable_parameters']:,}"
    )

    print("Trainable percentage:")
    print(
        f"{parameter_summary['trainable_percentage']:.4f}%"
    )

    print("\nMoving model and batch to GPU...")

    model.to(device)

    batch = move_batch_to_device(
        batch,
        device,
    )

    model.train()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\nRunning FP16 forward pass...")

    try:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            output = model(**batch)

        loss = output["loss"]

        if loss is None:
            raise ValueError(
                "The model did not return a loss."
            )

        if not torch.isfinite(loss):
            raise ValueError(
                f"Non-finite loss detected: {loss}"
            )

        print("Total loss:", loss.item())

        print(
            "BIO loss:",
            output["bio_loss"].item(),
        )

        print(
            "Sentiment loss:",
            output[
                "sentiment_loss"
            ].item(),
        )

        print(
            "BIO logits shape:",
            tuple(
                output["bio_logits"].shape
            ),
        )

        print(
            "Sentiment logits shape:",
            tuple(
                output[
                    "sentiment_logits"
                ].shape
            ),
        )

        print("\nRunning backward pass...")

        loss.backward()

        trainable_parameters_with_gradients = sum(
            1
            for parameter in model.parameters()
            if (
                parameter.requires_grad
                and parameter.grad is not None
            )
        )

        if (
            trainable_parameters_with_gradients
            == 0
        ):
            raise ValueError(
                "No trainable parameter received "
                "a gradient."
            )

        peak_allocated_gb = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        peak_reserved_gb = (
            torch.cuda.max_memory_reserved()
            / 1024**3
        )

        print(
            "Trainable tensors with gradients:",
            trainable_parameters_with_gradients,
        )

        print(
            "Peak allocated GPU memory:",
            f"{peak_allocated_gb:.3f} GB",
        )

        print(
            "Peak reserved GPU memory:",
            f"{peak_reserved_gb:.3f} GB",
        )

        print()
        print("=" * 60)
        print("GPU SMOKE TEST PASSED")
        print("=" * 60)

    except torch.cuda.OutOfMemoryError:
        print()
        print("=" * 60)
        print("GPU OUT OF MEMORY")
        print("=" * 60)

        print(
            "Batch size 2 does not fit safely. "
            "Use batch size 1 during training."
        )

        torch.cuda.empty_cache()

        raise

    except Exception:
        print()
        print("=" * 60)
        print("GPU SMOKE TEST FAILED")
        print("=" * 60)

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()