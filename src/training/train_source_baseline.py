# src/training/train_source_baseline.py

import argparse
import json
import math
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "deberta-v3-base-tokenizer"
)

BACKBONE_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "deberta-v3-base-backbone"
)

TOKENIZED_DATASET_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "deberta_v3_laptop_max96"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "source_baseline"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "source_baseline"
)


# ============================================================
# LABEL DEFINITIONS
# ============================================================

BIO_LABELS = {
    0: "O",
    1: "B-ASP",
    2: "I-ASP",
}

SENTIMENT_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
    3: "conflict",
}

NUMBER_OF_BIO_LABELS = len(BIO_LABELS)
NUMBER_OF_SENTIMENT_LABELS = len(SENTIMENT_LABELS)

IGNORE_INDEX = -100


# ============================================================
# DEFAULT TRAINING CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    "seed": 42,
    "maximum_length": 96,
    "train_batch_size": 4,
    "validation_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "maximum_epochs": 8,
    "early_stopping_patience": 2,
    "lora_learning_rate": 2e-4,
    "head_learning_rate": 5e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "maximum_gradient_norm": 1.0,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.10,
    "classifier_dropout": 0.20,
    "bio_loss_weight": 1.0,
    "sentiment_loss_weight": 1.0,
    "padding_multiple": 8,
    "num_workers": 0,
}

PILOT_TRAIN_BATCHES = 20
PILOT_VALIDATION_BATCHES = 10


# ============================================================
# GENERAL UTILITIES
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train the source-only joint DeBERTa ABSA baseline."
        )
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Run a short one-epoch smoke-training experiment."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_required_paths():
    required_paths = [
        TOKENIZER_DIRECTORY,
        BACKBONE_DIRECTORY,
        TOKENIZED_DATASET_DIRECTORY,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        message = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Required project paths are missing:\n"
            f"{message}"
        )


def safely_create_run_directory(pilot):
    mode = "pilot" if pilot else "full"

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_directory = (
        CHECKPOINT_ROOT
        / f"{mode}_{timestamp}"
    )

    report_directory = (
        REPORT_ROOT
        / f"{mode}_{timestamp}"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_directory, report_directory


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


# ============================================================
# DATA COLLATOR
# ============================================================

class JointABSACollator:
    def __init__(
        self,
        tokenizer,
        padding_multiple=8,
    ):
        self.tokenizer = tokenizer
        self.padding_multiple = padding_multiple

    def __call__(self, records):
        maximum_length = max(
            len(record["input_ids"])
            for record in records
        )

        if self.padding_multiple:
            maximum_length = (
                math.ceil(
                    maximum_length
                    / self.padding_multiple
                )
                * self.padding_multiple
            )

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "bio_labels": [],
        }

        has_token_type_ids = (
            "token_type_ids" in records[0]
        )

        if has_token_type_ids:
            batch["token_type_ids"] = []

        aspect_batch_indices = []
        aspect_token_starts = []
        aspect_token_ends = []
        aspect_sentiment_ids = []

        record_ids = []

        for batch_index, record in enumerate(records):
            sequence_length = len(
                record["input_ids"]
            )

            padding_length = (
                maximum_length - sequence_length
            )

            input_ids = (
                list(record["input_ids"])
                + [
                    self.tokenizer.pad_token_id
                ] * padding_length
            )

            attention_mask = (
                list(record["attention_mask"])
                + [0] * padding_length
            )

            bio_labels = (
                list(record["bio_labels"])
                + [IGNORE_INDEX] * padding_length
            )

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(
                attention_mask
            )
            batch["bio_labels"].append(bio_labels)

            if has_token_type_ids:
                token_type_ids = (
                    list(record["token_type_ids"])
                    + [0] * padding_length
                )

                batch["token_type_ids"].append(
                    token_type_ids
                )

            starts = record.get(
                "aspect_token_starts",
                [],
            )

            ends = record.get(
                "aspect_token_ends",
                [],
            )

            sentiments = record.get(
                "aspect_sentiment_ids",
                [],
            )

            if not (
                len(starts)
                == len(ends)
                == len(sentiments)
            ):
                raise ValueError(
                    "Aspect arrays have inconsistent "
                    f"lengths in record: {record.get('id')}"
                )

            for start, end, sentiment in zip(
                starts,
                ends,
                sentiments,
            ):
                aspect_batch_indices.append(
                    batch_index
                )
                aspect_token_starts.append(
                    int(start)
                )
                aspect_token_ends.append(
                    int(end)
                )
                aspect_sentiment_ids.append(
                    int(sentiment)
                )

            record_ids.append(
                str(
                    record.get(
                        "id",
                        f"record_{batch_index}",
                    )
                )
            )

        tensor_batch = {
            "input_ids": torch.tensor(
                batch["input_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                batch["attention_mask"],
                dtype=torch.long,
            ),
            "bio_labels": torch.tensor(
                batch["bio_labels"],
                dtype=torch.long,
            ),
            "aspect_batch_indices": torch.tensor(
                aspect_batch_indices,
                dtype=torch.long,
            ),
            "aspect_token_starts": torch.tensor(
                aspect_token_starts,
                dtype=torch.long,
            ),
            "aspect_token_ends": torch.tensor(
                aspect_token_ends,
                dtype=torch.long,
            ),
            "aspect_sentiment_ids": torch.tensor(
                aspect_sentiment_ids,
                dtype=torch.long,
            ),
            "record_ids": record_ids,
        }

        if has_token_type_ids:
            tensor_batch["token_type_ids"] = (
                torch.tensor(
                    batch["token_type_ids"],
                    dtype=torch.long,
                )
            )

        return tensor_batch


# ============================================================
# JOINT DEBERTA ABSA MODEL
# ============================================================

class JointDebertaABSA(nn.Module):
    def __init__(
        self,
        backbone_directory,
        bio_classes,
        sentiment_classes,
        lora_rank,
        lora_alpha,
        lora_dropout,
        classifier_dropout,
    ):
        super().__init__()

        backbone = AutoModel.from_pretrained(
            backbone_directory,
            local_files_only=True,
        )

        backbone.config.use_cache = False

        try:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False
                }
            )
        except TypeError:
            backbone.gradient_checkpointing_enable()

        if hasattr(
            backbone,
            "enable_input_require_grads",
        ):
            backbone.enable_input_require_grads()

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "query_proj",
                "value_proj",
            ],
            bias="none",
        )

        self.encoder = get_peft_model(
            backbone,
            lora_config,
        )

        self.hidden_size = (
            self.encoder.config.hidden_size
        )

        self.dropout = nn.Dropout(
            classifier_dropout
        )

        self.bio_classifier = nn.Linear(
            self.hidden_size,
            bio_classes,
        )

        sentiment_input_size = (
            self.hidden_size * 4
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(
                sentiment_input_size,
                self.hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(
                self.hidden_size,
                sentiment_classes,
            ),
        )

    def pool_aspect_spans(
        self,
        hidden_states,
        aspect_batch_indices,
        aspect_token_starts,
        aspect_token_ends,
    ):
        if aspect_batch_indices.numel() == 0:
            return hidden_states.new_empty(
                (0, self.hidden_size * 4)
            )

        representations = []

        for batch_index, start, end in zip(
            aspect_batch_indices.tolist(),
            aspect_token_starts.tolist(),
            aspect_token_ends.tolist(),
        ):
            sequence_length = hidden_states.shape[1]

            start = max(
                0,
                min(start, sequence_length - 1),
            )

            end = max(
                start + 1,
                min(end, sequence_length),
            )

            aspect_vector = hidden_states[
                batch_index,
                start:end,
                :,
            ].mean(dim=0)

            cls_vector = hidden_states[
                batch_index,
                0,
                :,
            ]

            interaction = (
                aspect_vector * cls_vector
            )

            difference = torch.abs(
                aspect_vector - cls_vector
            )

            combined = torch.cat(
                [
                    aspect_vector,
                    cls_vector,
                    interaction,
                    difference,
                ],
                dim=-1,
            )

            representations.append(combined)

        return torch.stack(
            representations,
            dim=0,
        )

    def classify_spans(
        self,
        hidden_states,
        aspect_batch_indices,
        aspect_token_starts,
        aspect_token_ends,
    ):
        pooled = self.pool_aspect_spans(
            hidden_states=hidden_states,
            aspect_batch_indices=(
                aspect_batch_indices
            ),
            aspect_token_starts=(
                aspect_token_starts
            ),
            aspect_token_ends=(
                aspect_token_ends
            ),
        )

        if pooled.shape[0] == 0:
            return hidden_states.new_empty(
                (
                    0,
                    NUMBER_OF_SENTIMENT_LABELS,
                )
            )

        return self.sentiment_classifier(
            self.dropout(pooled)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        bio_labels=None,
        aspect_batch_indices=None,
        aspect_token_starts=None,
        aspect_token_ends=None,
        aspect_sentiment_ids=None,
        bio_class_weights=None,
        sentiment_class_weights=None,
    ):
        encoder_arguments = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }

        if token_type_ids is not None:
            encoder_arguments[
                "token_type_ids"
            ] = token_type_ids

        encoder_output = self.encoder(
            **encoder_arguments
        )

        hidden_states = (
            encoder_output.last_hidden_state
        )

        bio_logits = self.bio_classifier(
            self.dropout(hidden_states)
        )

        if aspect_batch_indices is None:
            aspect_batch_indices = torch.empty(
                0,
                dtype=torch.long,
                device=input_ids.device,
            )

        if aspect_token_starts is None:
            aspect_token_starts = torch.empty(
                0,
                dtype=torch.long,
                device=input_ids.device,
            )

        if aspect_token_ends is None:
            aspect_token_ends = torch.empty(
                0,
                dtype=torch.long,
                device=input_ids.device,
            )

        sentiment_logits = self.classify_spans(
            hidden_states=hidden_states,
            aspect_batch_indices=(
                aspect_batch_indices
            ),
            aspect_token_starts=(
                aspect_token_starts
            ),
            aspect_token_ends=(
                aspect_token_ends
            ),
        )

        bio_loss = None
        sentiment_loss = None
        total_loss = None

        if bio_labels is not None:
            bio_loss_function = nn.CrossEntropyLoss(
                weight=bio_class_weights,
                ignore_index=IGNORE_INDEX,
            )

            bio_loss = bio_loss_function(
                bio_logits.reshape(
                    -1,
                    bio_logits.shape[-1],
                ),
                bio_labels.reshape(-1),
            )

        if (
            aspect_sentiment_ids is not None
            and aspect_sentiment_ids.numel() > 0
        ):
            sentiment_loss_function = (
                nn.CrossEntropyLoss(
                    weight=sentiment_class_weights
                )
            )

            sentiment_loss = (
                sentiment_loss_function(
                    sentiment_logits,
                    aspect_sentiment_ids,
                )
            )

        if bio_loss is not None:
            total_loss = bio_loss

        if sentiment_loss is not None:
            if total_loss is None:
                total_loss = sentiment_loss
            else:
                total_loss = (
                    DEFAULT_CONFIG[
                        "bio_loss_weight"
                    ] * bio_loss
                    + DEFAULT_CONFIG[
                        "sentiment_loss_weight"
                    ] * sentiment_loss
                )

        return {
            "loss": total_loss,
            "bio_loss": bio_loss,
            "sentiment_loss": sentiment_loss,
            "bio_logits": bio_logits,
            "sentiment_logits": (
                sentiment_logits
            ),
            "hidden_states": hidden_states,
        }


# ============================================================
# CLASS WEIGHTS
# ============================================================

def inverse_square_root_weights(
    counts,
    number_of_classes,
):
    count_array = np.array(
        [
            max(counts.get(class_id, 0), 1)
            for class_id in range(
                number_of_classes
            )
        ],
        dtype=np.float64,
    )

    weights = 1.0 / np.sqrt(count_array)

    weights = (
        weights
        / weights.mean()
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def calculate_class_weights(train_dataset):
    bio_counts = Counter()
    sentiment_counts = Counter()

    for record in train_dataset:
        for label in record["bio_labels"]:
            if label != IGNORE_INDEX:
                bio_counts[int(label)] += 1

        for label in record.get(
            "aspect_sentiment_ids",
            [],
        ):
            sentiment_counts[int(label)] += 1

    bio_weights = inverse_square_root_weights(
        bio_counts,
        NUMBER_OF_BIO_LABELS,
    )

    sentiment_weights = (
        inverse_square_root_weights(
            sentiment_counts,
            NUMBER_OF_SENTIMENT_LABELS,
        )
    )

    return (
        bio_counts,
        sentiment_counts,
        bio_weights,
        sentiment_weights,
    )


# ============================================================
# BIO SPAN DECODING
# ============================================================

def decode_bio_spans(labels):
    spans = []
    current_start = None

    for token_index, label in enumerate(labels):
        if label == IGNORE_INDEX:
            if current_start is not None:
                spans.append(
                    (
                        current_start,
                        token_index,
                    )
                )
                current_start = None

            continue

        if label == 1:
            if current_start is not None:
                spans.append(
                    (
                        current_start,
                        token_index,
                    )
                )

            current_start = token_index

        elif label == 2:
            if current_start is None:
                current_start = token_index

        else:
            if current_start is not None:
                spans.append(
                    (
                        current_start,
                        token_index,
                    )
                )
                current_start = None

    if current_start is not None:
        spans.append(
            (
                current_start,
                len(labels),
            )
        )

    return spans


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# DEVICE HANDLING
# ============================================================

def move_batch_to_device(batch, device):
    output = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(
                device,
                non_blocking=True,
            )
        else:
            output[key] = value

    return output


def model_forward_arguments(batch):
    arguments = {
        "input_ids": batch["input_ids"],
        "attention_mask": (
            batch["attention_mask"]
        ),
        "bio_labels": batch["bio_labels"],
        "aspect_batch_indices": (
            batch["aspect_batch_indices"]
        ),
        "aspect_token_starts": (
            batch["aspect_token_starts"]
        ),
        "aspect_token_ends": (
            batch["aspect_token_ends"]
        ),
        "aspect_sentiment_ids": (
            batch["aspect_sentiment_ids"]
        ),
    }

    if "token_type_ids" in batch:
        arguments["token_type_ids"] = (
            batch["token_type_ids"]
        )

    return arguments


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    bio_weights,
    sentiment_weights,
    gradient_accumulation_steps,
    maximum_gradient_norm,
    maximum_batches=None,
):
    model.train()

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_bio_loss = 0.0
    total_sentiment_loss = 0.0
    processed_batches = 0

    effective_batches = len(train_loader)

    if maximum_batches is not None:
        effective_batches = min(
            effective_batches,
            maximum_batches,
        )

    progress_bar = tqdm(
        train_loader,
        total=effective_batches,
        desc="Training",
        leave=False,
    )

    for batch_index, batch in enumerate(
        progress_bar
    ):
        if (
            maximum_batches is not None
            and batch_index >= maximum_batches
        ):
            break

        batch = move_batch_to_device(
            batch,
            device,
        )

        with torch.amp.autocast(
            device_type="cuda",
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

            raw_loss = outputs["loss"]

            loss = (
                raw_loss
                / gradient_accumulation_steps
            )

        scaler.scale(loss).backward()

        processed_batches += 1

        total_loss += raw_loss.detach().item()

        if outputs["bio_loss"] is not None:
            total_bio_loss += (
                outputs["bio_loss"]
                .detach()
                .item()
            )

        if (
            outputs["sentiment_loss"]
            is not None
        ):
            total_sentiment_loss += (
                outputs["sentiment_loss"]
                .detach()
                .item()
            )

        accumulation_complete = (
            processed_batches
            % gradient_accumulation_steps
            == 0
        )

        final_available_batch = (
            processed_batches
            == effective_batches
        )

        if (
            accumulation_complete
            or final_available_batch
        ):
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                maximum_gradient_norm,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            scheduler.step()

        progress_bar.set_postfix(
            loss=f"{raw_loss.item():.4f}"
        )

    if processed_batches == 0:
        raise RuntimeError(
            "No training batches were processed."
        )

    return {
        "loss": (
            total_loss / processed_batches
        ),
        "bio_loss": (
            total_bio_loss
            / processed_batches
        ),
        "sentiment_loss": (
            total_sentiment_loss
            / processed_batches
        ),
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    validation_loader,
    device,
    bio_weights,
    sentiment_weights,
    maximum_batches=None,
):
    model.eval()

    total_loss = 0.0
    total_bio_loss = 0.0
    total_sentiment_loss = 0.0
    processed_batches = 0

    gold_span_total = 0
    predicted_span_total = 0
    correct_span_total = 0

    gold_triplet_total = 0
    predicted_triplet_total = 0
    correct_triplet_total = 0

    oracle_gold_sentiments = []
    oracle_predicted_sentiments = []

    effective_batches = len(
        validation_loader
    )

    if maximum_batches is not None:
        effective_batches = min(
            effective_batches,
            maximum_batches,
        )

    progress_bar = tqdm(
        validation_loader,
        total=effective_batches,
        desc="Validation",
        leave=False,
    )

    for batch_index, batch in enumerate(
        progress_bar
    ):
        if (
            maximum_batches is not None
            and batch_index >= maximum_batches
        ):
            break

        batch = move_batch_to_device(
            batch,
            device,
        )

        with torch.amp.autocast(
            device_type="cuda",
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

        total_loss += (
            outputs["loss"].detach().item()
        )

        if outputs["bio_loss"] is not None:
            total_bio_loss += (
                outputs["bio_loss"]
                .detach()
                .item()
            )

        if (
            outputs["sentiment_loss"]
            is not None
        ):
            total_sentiment_loss += (
                outputs["sentiment_loss"]
                .detach()
                .item()
            )

        predicted_bio = (
            outputs["bio_logits"]
            .argmax(dim=-1)
        )

        gold_bio = batch["bio_labels"]

        batch_size = gold_bio.shape[0]

        gold_spans_per_record = []
        predicted_spans_per_record = []

        for record_index in range(
            batch_size
        ):
            gold_labels = (
                gold_bio[record_index]
                .detach()
                .cpu()
                .tolist()
            )

            predicted_labels = (
                predicted_bio[record_index]
                .detach()
                .cpu()
                .tolist()
            )

            masked_predictions = [
                (
                    predicted_label
                    if gold_label != IGNORE_INDEX
                    else IGNORE_INDEX
                )
                for predicted_label, gold_label
                in zip(
                    predicted_labels,
                    gold_labels,
                )
            ]

            gold_spans = decode_bio_spans(
                gold_labels
            )

            predicted_spans = (
                decode_bio_spans(
                    masked_predictions
                )
            )

            gold_spans_per_record.append(
                gold_spans
            )

            predicted_spans_per_record.append(
                predicted_spans
            )

            gold_span_set = set(gold_spans)
            predicted_span_set = set(
                predicted_spans
            )

            gold_span_total += len(
                gold_span_set
            )

            predicted_span_total += len(
                predicted_span_set
            )

            correct_span_total += len(
                gold_span_set
                & predicted_span_set
            )

        gold_sentiments = (
            batch["aspect_sentiment_ids"]
        )

        if gold_sentiments.numel() > 0:
            oracle_predictions = (
                outputs["sentiment_logits"]
                .argmax(dim=-1)
            )

            oracle_gold_sentiments.extend(
                gold_sentiments
                .detach()
                .cpu()
                .tolist()
            )

            oracle_predicted_sentiments.extend(
                oracle_predictions
                .detach()
                .cpu()
                .tolist()
            )

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
                device_type="cuda",
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

            predicted_sentiment_ids = (
                predicted_sentiment_logits
                .argmax(dim=-1)
                .detach()
                .cpu()
                .tolist()
            )
        else:
            predicted_sentiment_ids = []

        gold_triplets_per_record = [
            set()
            for _ in range(batch_size)
        ]

        predicted_triplets_per_record = [
            set()
            for _ in range(batch_size)
        ]

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

        gold_sentiment_list = (
            batch["aspect_sentiment_ids"]
            .detach()
            .cpu()
            .tolist()
        )

        for (
            record_index,
            start,
            end,
            sentiment,
        ) in zip(
            gold_batch_indices,
            gold_starts,
            gold_ends,
            gold_sentiment_list,
        ):
            gold_triplets_per_record[
                record_index
            ].add(
                (
                    start,
                    end,
                    sentiment,
                )
            )

        for (
            record_index,
            start,
            end,
            sentiment,
        ) in zip(
            predicted_batch_indices,
            predicted_starts,
            predicted_ends,
            predicted_sentiment_ids,
        ):
            predicted_triplets_per_record[
                record_index
            ].add(
                (
                    start,
                    end,
                    sentiment,
                )
            )

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

            gold_triplet_total += len(
                gold_triplets
            )

            predicted_triplet_total += len(
                predicted_triplets
            )

            correct_triplet_total += len(
                gold_triplets
                & predicted_triplets
            )

    if processed_batches == 0:
        raise RuntimeError(
            "No validation batches were processed."
        )

    span_precision = safe_divide(
        correct_span_total,
        predicted_span_total,
    )

    span_recall = safe_divide(
        correct_span_total,
        gold_span_total,
    )

    span_f1 = safe_divide(
        2 * span_precision * span_recall,
        span_precision + span_recall,
    )

    triplet_precision = safe_divide(
        correct_triplet_total,
        predicted_triplet_total,
    )

    triplet_recall = safe_divide(
        correct_triplet_total,
        gold_triplet_total,
    )

    triplet_f1 = safe_divide(
        2
        * triplet_precision
        * triplet_recall,
        triplet_precision
        + triplet_recall,
    )

    if oracle_gold_sentiments:
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

        confusion = confusion_matrix(
            oracle_gold_sentiments,
            oracle_predicted_sentiments,
            labels=list(
                range(
                    NUMBER_OF_SENTIMENT_LABELS
                )
            ),
        )
    else:
        oracle_accuracy = 0.0
        oracle_macro_f1 = 0.0

        class_precision = np.zeros(
            NUMBER_OF_SENTIMENT_LABELS
        )

        class_recall = np.zeros(
            NUMBER_OF_SENTIMENT_LABELS
        )

        class_f1 = np.zeros(
            NUMBER_OF_SENTIMENT_LABELS
        )

        class_support = np.zeros(
            NUMBER_OF_SENTIMENT_LABELS
        )

        confusion = np.zeros(
            (
                NUMBER_OF_SENTIMENT_LABELS,
                NUMBER_OF_SENTIMENT_LABELS,
            ),
            dtype=int,
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

    return {
        "loss": (
            total_loss / processed_batches
        ),
        "bio_loss": (
            total_bio_loss
            / processed_batches
        ),
        "sentiment_loss": (
            total_sentiment_loss
            / processed_batches
        ),
        "span_precision": span_precision,
        "span_recall": span_recall,
        "span_f1": span_f1,
        "oracle_sentiment_accuracy": (
            oracle_accuracy
        ),
        "oracle_sentiment_macro_f1": (
            oracle_macro_f1
        ),
        "triplet_precision": (
            triplet_precision
        ),
        "triplet_recall": triplet_recall,
        "triplet_f1": triplet_f1,
        "per_class_sentiment": (
            per_class_metrics
        ),
        "confusion_matrix": (
            confusion.tolist()
        ),
    }


# ============================================================
# MODEL SAVING
# ============================================================

def save_checkpoint(
    model,
    tokenizer,
    output_directory,
    epoch,
    metrics,
    config,
):
    if output_directory.exists():
        shutil.rmtree(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    adapter_directory = (
        output_directory
        / "lora_adapter"
    )

    tokenizer_directory = (
        output_directory
        / "tokenizer"
    )

    model.encoder.save_pretrained(
        adapter_directory
    )

    tokenizer.save_pretrained(
        tokenizer_directory
    )

    head_state = {}

    for name, tensor in (
        model.bio_classifier.state_dict().items()
    ):
        head_state[
            f"bio_classifier.{name}"
        ] = (
            tensor.detach()
            .cpu()
            .contiguous()
        )

    for name, tensor in (
        model.sentiment_classifier
        .state_dict()
        .items()
    ):
        head_state[
            f"sentiment_classifier.{name}"
        ] = (
            tensor.detach()
            .cpu()
            .contiguous()
        )

    save_file(
        head_state,
        str(
            output_directory
            / "task_heads.safetensors"
        ),
    )

    metadata = {
        "epoch": epoch,
        "validation_metrics": metrics,
        "configuration": config,
        "bio_labels": BIO_LABELS,
        "sentiment_labels": (
            SENTIMENT_LABELS
        ),
        "backbone_directory": str(
            BACKBONE_DIRECTORY
        ),
        "tokenizer_directory": str(
            TOKENIZER_DIRECTORY
        ),
    }

    save_json(
        metadata,
        output_directory
        / "checkpoint_metadata.json",
    )


# ============================================================
# CHART GENERATION
# ============================================================

def create_training_chart(
    history,
    output_path,
):
    epochs = [
        record["epoch"]
        for record in history
    ]

    train_losses = [
        record["train"]["loss"]
        for record in history
    ]

    validation_losses = [
        record["validation"]["loss"]
        for record in history
    ]

    span_f1_values = [
        record["validation"]["span_f1"]
        for record in history
    ]

    sentiment_f1_values = [
        record["validation"][
            "oracle_sentiment_macro_f1"
        ]
        for record in history
    ]

    triplet_f1_values = [
        record["validation"]["triplet_f1"]
        for record in history
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    axes[0].plot(
        epochs,
        train_losses,
        marker="o",
        linewidth=2,
        label="Training loss",
    )

    axes[0].plot(
        epochs,
        validation_losses,
        marker="o",
        linewidth=2,
        label="Validation loss",
    )

    axes[0].set_title(
        "Training and Validation Loss"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        span_f1_values,
        marker="o",
        linewidth=2,
        label="Aspect span F1",
    )

    axes[1].plot(
        epochs,
        sentiment_f1_values,
        marker="o",
        linewidth=2,
        label="Oracle sentiment Macro-F1",
    )

    axes[1].plot(
        epochs,
        triplet_f1_values,
        marker="o",
        linewidth=2,
        label="End-to-end triplet F1",
    )

    axes[1].set_title(
        "Validation Performance"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle(
        "RAISE-ABSA Source Baseline",
        fontsize=15,
        fontweight="bold",
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_confusion_matrix_chart(
    confusion,
    output_path,
):
    matrix = np.asarray(confusion)

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
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(
            len(class_names)
        ),
        yticks=np.arange(
            len(class_names)
        ),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted sentiment",
        ylabel="True sentiment",
        title=(
            "Oracle Aspect Sentiment "
            "Confusion Matrix"
        ),
    )

    threshold = (
        matrix.max() / 2
        if matrix.size > 0
        else 0
    )

    for row in range(matrix.shape[0]):
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
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    arguments = parse_arguments()

    config = DEFAULT_CONFIG.copy()

    if arguments.epochs is not None:
        if arguments.epochs < 1:
            raise ValueError(
                "--epochs must be at least 1."
            )

        config["maximum_epochs"] = (
            arguments.epochs
        )

    if arguments.pilot:
        config["maximum_epochs"] = 1

    print("=" * 70)
    print("RAISE-ABSA SOURCE-ONLY BASELINE TRAINING")
    print("=" * 70)

    check_required_paths()
    set_seed(config["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This training "
            "configuration expects the RTX 3050 GPU."
        )

    device = torch.device("cuda")

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

    print("\nMode:")
    print(
        "PILOT"
        if arguments.pilot
        else "FULL TRAINING"
    )

    run_directory, report_directory = (
        safely_create_run_directory(
            arguments.pilot
        )
    )

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

    if "train" not in dataset:
        raise KeyError(
            "Tokenized dataset has no train split."
        )

    if "validation" not in dataset:
        raise KeyError(
            "Tokenized dataset has no "
            "validation split."
        )

    train_dataset = dataset["train"]
    validation_dataset = (
        dataset["validation"]
    )

    print(
        f"Train records: {len(train_dataset)}"
    )

    print(
        "Validation records:",
        len(validation_dataset),
    )

    collator = JointABSACollator(
        tokenizer=tokenizer,
        padding_multiple=config[
            "padding_multiple"
        ],
    )

    generator = torch.Generator()
    generator.manual_seed(config["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config[
            "train_batch_size"
        ],
        shuffle=True,
        collate_fn=collator,
        num_workers=config["num_workers"],
        pin_memory=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config[
            "validation_batch_size"
        ],
        shuffle=False,
        collate_fn=collator,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    (
        bio_counts,
        sentiment_counts,
        bio_weights,
        sentiment_weights,
    ) = calculate_class_weights(
        train_dataset
    )

    bio_weights = bio_weights.to(device)
    sentiment_weights = (
        sentiment_weights.to(device)
    )

    print("\nBIO counts:")
    print(dict(bio_counts))

    print("BIO weights:")
    print(
        [
            round(value, 4)
            for value in (
                bio_weights
                .detach()
                .cpu()
                .tolist()
            )
        ]
    )

    print("\nSentiment counts:")
    print(dict(sentiment_counts))

    print("Sentiment weights:")
    print(
        [
            round(value, 4)
            for value in (
                sentiment_weights
                .detach()
                .cpu()
                .tolist()
            )
        ]
    )

    print("\nCreating joint model...")

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

    model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    print(
        "Trainable percentage:",
        f"{100 * trainable_parameters / total_parameters:.4f}%",
    )

    lora_parameters = []
    head_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if "lora_" in name:
            lora_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    if not lora_parameters:
        raise RuntimeError(
            "No trainable LoRA parameters found."
        )

    if not head_parameters:
        raise RuntimeError(
            "No trainable task-head parameters found."
        )

    optimizer = AdamW(
        [
            {
                "params": lora_parameters,
                "lr": config[
                    "lora_learning_rate"
                ],
                "weight_decay": config[
                    "weight_decay"
                ],
            },
            {
                "params": head_parameters,
                "lr": config[
                    "head_learning_rate"
                ],
                "weight_decay": config[
                    "weight_decay"
                ],
            },
        ]
    )

    if arguments.pilot:
        train_batches_per_epoch = min(
            len(train_loader),
            PILOT_TRAIN_BATCHES,
        )
    else:
        train_batches_per_epoch = len(
            train_loader
        )

    update_steps_per_epoch = math.ceil(
        train_batches_per_epoch
        / config[
            "gradient_accumulation_steps"
        ]
    )

    total_update_steps = (
        update_steps_per_epoch
        * config["maximum_epochs"]
    )

    warmup_steps = int(
        total_update_steps
        * config["warmup_ratio"]
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(
                total_update_steps
            ),
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )

    print("\nOptimizer update steps:")
    print(total_update_steps)

    print("Warmup steps:")
    print(warmup_steps)

    history = []

    best_triplet_f1 = -1.0
    epochs_without_improvement = 0

    best_checkpoint_directory = (
        run_directory
        / "best_checkpoint"
    )

    torch.cuda.reset_peak_memory_stats()

    for epoch in range(
        1,
        config["maximum_epochs"] + 1,
    ):
        print("\n" + "=" * 70)
        print(
            f"EPOCH {epoch}/"
            f"{config['maximum_epochs']}"
        )
        print("=" * 70)

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            bio_weights=bio_weights,
            sentiment_weights=(
                sentiment_weights
            ),
            gradient_accumulation_steps=(
                config[
                    "gradient_accumulation_steps"
                ]
            ),
            maximum_gradient_norm=(
                config[
                    "maximum_gradient_norm"
                ]
            ),
            maximum_batches=(
                PILOT_TRAIN_BATCHES
                if arguments.pilot
                else None
            ),
        )

        validation_metrics = evaluate(
            model=model,
            validation_loader=(
                validation_loader
            ),
            device=device,
            bio_weights=bio_weights,
            sentiment_weights=(
                sentiment_weights
            ),
            maximum_batches=(
                PILOT_VALIDATION_BATCHES
                if arguments.pilot
                else None
            ),
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": (
                validation_metrics
            ),
        }

        history.append(epoch_record)

        print(
            f"Train loss: "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            "Span P/R/F1: "
            f"{validation_metrics['span_precision']:.4f} / "
            f"{validation_metrics['span_recall']:.4f} / "
            f"{validation_metrics['span_f1']:.4f}"
        )

        print(
            "Oracle sentiment accuracy: "
            f"{validation_metrics['oracle_sentiment_accuracy']:.4f}"
        )

        print(
            "Oracle sentiment Macro-F1: "
            f"{validation_metrics['oracle_sentiment_macro_f1']:.4f}"
        )

        print(
            "End-to-end triplet P/R/F1: "
            f"{validation_metrics['triplet_precision']:.4f} / "
            f"{validation_metrics['triplet_recall']:.4f} / "
            f"{validation_metrics['triplet_f1']:.4f}"
        )

        current_score = (
            validation_metrics[
                "triplet_f1"
            ]
        )

        if current_score > best_triplet_f1:
            best_triplet_f1 = current_score
            epochs_without_improvement = 0

            print(
                "New best validation "
                "triplet F1. Saving checkpoint..."
            )

            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_directory=(
                    best_checkpoint_directory
                ),
                epoch=epoch,
                metrics=validation_metrics,
                config=config,
            )
        else:
            epochs_without_improvement += 1

            print(
                "No validation improvement. "
                f"Patience: "
                f"{epochs_without_improvement}/"
                f"{config['early_stopping_patience']}"
            )

        save_json(
            history,
            report_directory
            / "training_history.json",
        )

        create_training_chart(
            history=history,
            output_path=(
                report_directory
                / "training_curves.png"
            ),
        )

        create_confusion_matrix_chart(
            confusion=validation_metrics[
                "confusion_matrix"
            ],
            output_path=(
                report_directory
                / (
                    "latest_validation_"
                    "confusion_matrix.png"
                )
            ),
        )

        if (
            not arguments.pilot
            and epochs_without_improvement
            >= config[
                "early_stopping_patience"
            ]
        ):
            print(
                "\nEarly stopping activated."
            )
            break

    peak_allocated_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    peak_reserved_memory = (
        torch.cuda.max_memory_reserved()
        / (1024 ** 3)
    )

    final_summary = {
        "mode": (
            "pilot"
            if arguments.pilot
            else "full"
        ),
        "completed_epochs": len(history),
        "best_validation_triplet_f1": (
            best_triplet_f1
        ),
        "best_checkpoint": str(
            best_checkpoint_directory
        ),
        "training_chart": str(
            report_directory
            / "training_curves.png"
        ),
        "confusion_matrix_chart": str(
            report_directory
            / (
                "latest_validation_"
                "confusion_matrix.png"
            )
        ),
        "peak_allocated_gpu_memory_gb": (
            peak_allocated_memory
        ),
        "peak_reserved_gpu_memory_gb": (
            peak_reserved_memory
        ),
        "test_split_evaluated": False,
    }

    save_json(
        final_summary,
        report_directory
        / "training_summary.json",
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        "\nBest validation triplet F1:",
        f"{best_triplet_f1:.4f}",
    )

    print(
        "Peak allocated GPU memory:",
        f"{peak_allocated_memory:.3f} GB",
    )

    print(
        "Peak reserved GPU memory:",
        f"{peak_reserved_memory:.3f} GB",
    )

    print("\nBest checkpoint:")
    print(best_checkpoint_directory)

    print("\nTraining chart:")
    print(
        report_directory
        / "training_curves.png"
    )

    print("\nConfusion-matrix chart:")
    print(
        report_directory
        / (
            "latest_validation_"
            "confusion_matrix.png"
        )
    )

    print(
        "\nThe test split was not evaluated. "
        "Keep it untouched until the final "
        "model configuration is selected."
    )


if __name__ == "__main__":
    main()