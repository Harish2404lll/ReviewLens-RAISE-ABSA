from pathlib import Path

import torch
import torch.nn.functional as F

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from torch import nn
from transformers import AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKBONE_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "deberta-v3-base-backbone"
)

NUMBER_OF_BIO_LABELS = 3
NUMBER_OF_SENTIMENT_LABELS = 4

IGNORE_INDEX = -100

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.10

DROPOUT_RATE = 0.10

LORA_TARGET_MODULES = [
    "query_proj",
    "value_proj",
]


# ============================================================
# JOINT ABSA MODEL
# ============================================================

class JointDebertaABSA(nn.Module):
    """
    Joint aspect-based sentiment analysis model.

    Components:
    1. Shared DeBERTa-v3 encoder with LoRA adapters.
    2. BIO token-classification head for aspect extraction.
    3. Aspect-span mean pooling.
    4. Sentence-context representation using the CLS token.
    5. Aspect-conditioned four-class sentiment classifier.
    """

    def __init__(
        self,
        backbone_directory=BACKBONE_DIRECTORY,
        number_of_bio_labels=NUMBER_OF_BIO_LABELS,
        number_of_sentiment_labels=(
            NUMBER_OF_SENTIMENT_LABELS
        ),
        bio_class_weights=None,
        sentiment_class_weights=None,
        bio_loss_weight=1.0,
        sentiment_loss_weight=1.0,
        enable_gradient_checkpointing=True,
    ):
        super().__init__()

        self.number_of_bio_labels = (
            number_of_bio_labels
        )

        self.number_of_sentiment_labels = (
            number_of_sentiment_labels
        )

        self.bio_loss_weight = bio_loss_weight

        self.sentiment_loss_weight = (
            sentiment_loss_weight
        )

        if not Path(backbone_directory).exists():
            raise FileNotFoundError(
                f"Backbone directory does not exist:\n"
                f"{backbone_directory}"
            )

        backbone = AutoModel.from_pretrained(
            backbone_directory,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

        self.hidden_size = (
            backbone.config.hidden_size
        )

        if enable_gradient_checkpointing:
            if hasattr(
                backbone,
                "gradient_checkpointing_enable",
            ):
                backbone.gradient_checkpointing_enable()

            if hasattr(
                backbone,
                "enable_input_require_grads",
            ):
                backbone.enable_input_require_grads()

        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False

        lora_configuration = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )

        self.encoder = get_peft_model(
            backbone,
            lora_configuration,
        )

        self.dropout = nn.Dropout(
            DROPOUT_RATE
        )

        # Token-level aspect extraction head.
        self.bio_classifier = nn.Linear(
            self.hidden_size,
            self.number_of_bio_labels,
        )

        # The sentiment representation concatenates:
        # 1. Aspect-span representation
        # 2. CLS sentence representation
        # 3. Element-wise interaction
        # 4. Absolute difference
        sentiment_input_size = (
            self.hidden_size * 4
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(
                sentiment_input_size,
                self.hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(
                self.hidden_size,
                self.number_of_sentiment_labels,
            ),
        )

        if bio_class_weights is None:
            bio_class_weights = torch.ones(
                self.number_of_bio_labels,
                dtype=torch.float32,
            )
        else:
            bio_class_weights = torch.as_tensor(
                bio_class_weights,
                dtype=torch.float32,
            )

        if (
            len(bio_class_weights)
            != self.number_of_bio_labels
        ):
            raise ValueError(
                "BIO class-weight count does not match "
                "the number of BIO labels."
            )

        if sentiment_class_weights is None:
            sentiment_class_weights = torch.ones(
                self.number_of_sentiment_labels,
                dtype=torch.float32,
            )
        else:
            sentiment_class_weights = (
                torch.as_tensor(
                    sentiment_class_weights,
                    dtype=torch.float32,
                )
            )

        if (
            len(sentiment_class_weights)
            != self.number_of_sentiment_labels
        ):
            raise ValueError(
                "Sentiment class-weight count does not "
                "match the number of sentiment labels."
            )

        self.register_buffer(
            "bio_class_weights",
            bio_class_weights,
        )

        self.register_buffer(
            "sentiment_class_weights",
            sentiment_class_weights,
        )

    def pool_aspect_spans(
        self,
        sequence_output,
        aspect_batch_indices,
        aspect_token_starts,
        aspect_token_ends,
    ):
        """
        Mean-pool each aspect span and combine it with the
        corresponding sentence's CLS representation.

        Token-end indices are exclusive.
        """

        if aspect_batch_indices is None:
            return sequence_output.new_empty(
                (
                    0,
                    self.hidden_size * 4,
                )
            )

        number_of_aspects = (
            aspect_batch_indices.numel()
        )

        if number_of_aspects == 0:
            return sequence_output.new_empty(
                (
                    0,
                    self.hidden_size * 4,
                )
            )

        if not (
            number_of_aspects
            == aspect_token_starts.numel()
            == aspect_token_ends.numel()
        ):
            raise ValueError(
                "Aspect span tensors have "
                "different lengths."
            )

        batch_size = sequence_output.size(0)
        sequence_length = sequence_output.size(1)

        aspect_features = []

        for aspect_index in range(
            number_of_aspects
        ):
            batch_index = int(
                aspect_batch_indices[
                    aspect_index
                ].item()
            )

            token_start = int(
                aspect_token_starts[
                    aspect_index
                ].item()
            )

            token_end = int(
                aspect_token_ends[
                    aspect_index
                ].item()
            )

            if not (
                0 <= batch_index < batch_size
            ):
                raise ValueError(
                    f"Invalid aspect batch index: "
                    f"{batch_index}"
                )

            if not (
                0 <= token_start
                < token_end
                <= sequence_length
            ):
                raise ValueError(
                    "Invalid aspect token span: "
                    f"[{token_start}, {token_end})"
                )

            span_hidden_states = (
                sequence_output[
                    batch_index,
                    token_start:token_end,
                    :,
                ]
            )

            aspect_representation = (
                span_hidden_states.mean(dim=0)
            )

            sentence_representation = (
                sequence_output[
                    batch_index,
                    0,
                    :,
                ]
            )

            interaction = (
                aspect_representation
                * sentence_representation
            )

            absolute_difference = torch.abs(
                aspect_representation
                - sentence_representation
            )

            combined_representation = torch.cat(
                [
                    aspect_representation,
                    sentence_representation,
                    interaction,
                    absolute_difference,
                ],
                dim=-1,
            )

            aspect_features.append(
                combined_representation
            )

        return torch.stack(
            aspect_features,
            dim=0,
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

        sequence_output = (
            encoder_output.last_hidden_state
        )

        dropped_sequence_output = self.dropout(
            sequence_output
        )

        bio_logits = self.bio_classifier(
            dropped_sequence_output
        )

        aspect_features = self.pool_aspect_spans(
            sequence_output=sequence_output,
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

        if aspect_features.size(0) > 0:
            sentiment_logits = (
                self.sentiment_classifier(
                    self.dropout(
                        aspect_features
                    )
                )
            )
        else:
            sentiment_logits = (
                sequence_output.new_empty(
                    (
                        0,
                        self.number_of_sentiment_labels,
                    )
                )
            )

        bio_loss = None
        sentiment_loss = None
        total_loss = None

        if bio_labels is not None:
            if bio_labels.shape != input_ids.shape:
                raise ValueError(
                    "BIO labels must have the same "
                    "shape as input IDs."
                )

            bio_loss = F.cross_entropy(
                bio_logits.float().reshape(
                    -1,
                    self.number_of_bio_labels,
                ),
                bio_labels.reshape(-1),
                weight=self.bio_class_weights.float(),
                ignore_index=IGNORE_INDEX,
            )

        if aspect_sentiment_ids is not None:
            if (
                sentiment_logits.size(0)
                != aspect_sentiment_ids.numel()
            ):
                raise ValueError(
                    "The number of sentiment logits "
                    "does not match the number of "
                    "sentiment labels."
                )

            if aspect_sentiment_ids.numel() > 0:
                sentiment_loss = F.cross_entropy(
                    sentiment_logits.float(),
                    aspect_sentiment_ids.reshape(-1),
                    weight=(
                        self.sentiment_class_weights.float()
                    ),
                )

        if (
            bio_loss is not None
            and sentiment_loss is not None
        ):
            total_loss = (
                self.bio_loss_weight * bio_loss
                + self.sentiment_loss_weight
                * sentiment_loss
            )
        elif bio_loss is not None:
            total_loss = (
                self.bio_loss_weight
                * bio_loss
            )
        elif sentiment_loss is not None:
            total_loss = (
                self.sentiment_loss_weight
                * sentiment_loss
            )

        return {
            "loss": total_loss,
            "bio_loss": bio_loss,
            "sentiment_loss": sentiment_loss,
            "bio_logits": bio_logits,
            "sentiment_logits": (
                sentiment_logits
            ),
            "last_hidden_state": (
                sequence_output
            ),
        }

    def trainable_parameter_summary(self):
        total_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        percentage = (
            100.0
            * trainable_parameters
            / total_parameters
        )

        return {
            "total_parameters": total_parameters,
            "trainable_parameters": (
                trainable_parameters
            ),
            "trainable_percentage": percentage,
        }


# ============================================================
# ARCHITECTURE CHECK
# ============================================================

def main():
    print("=" * 60)
    print("JOINT DEBERTA ABSA MODEL CHECK")
    print("=" * 60)

    model = JointDebertaABSA()

    summary = (
        model.trainable_parameter_summary()
    )

    print("\nBackbone:")
    print(BACKBONE_DIRECTORY)

    print("\nHidden size:")
    print(model.hidden_size)

    print("\nBIO classes:")
    print(model.number_of_bio_labels)

    print("\nSentiment classes:")
    print(model.number_of_sentiment_labels)

    print("\nLoRA targets:")
    print(LORA_TARGET_MODULES)

    print("\nTotal parameters:")
    print(
        f"{summary['total_parameters']:,}"
    )

    print("Trainable parameters:")
    print(
        f"{summary['trainable_parameters']:,}"
    )

    print("Trainable percentage:")
    print(
        f"{summary['trainable_percentage']:.4f}%"
    )

    if summary["trainable_percentage"] >= 10:
        raise ValueError(
            "Too many parameters are trainable. "
            "LoRA freezing may not be working."
        )

    print()
    print("=" * 60)
    print("JOINT MODEL ARCHITECTURE CHECK PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()