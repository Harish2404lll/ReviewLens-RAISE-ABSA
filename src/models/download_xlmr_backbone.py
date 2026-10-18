from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "FacebookAI/xlm-roberta-base"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_OUTPUT = (
    PROJECT_ROOT
    / "models"
    / "xlm-roberta-base-tokenizer"
)

BACKBONE_OUTPUT = (
    PROJECT_ROOT
    / "models"
    / "xlm-roberta-base-backbone"
)


def main():
    print("=" * 70)
    print("XLM-ROBERTA-BASE DOWNLOAD")
    print("=" * 70)

    TOKENIZER_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    BACKBONE_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nDownloading tokenizer:")
    print(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    tokenizer.save_pretrained(
        TOKENIZER_OUTPUT
    )

    print("\nTokenizer saved to:")
    print(TOKENIZER_OUTPUT)

    print("\nDownloading backbone:")
    print(MODEL_NAME)

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )

    model.save_pretrained(
        BACKBONE_OUTPUT,
        safe_serialization=True,
    )

    print("\nBackbone saved to:")
    print(BACKBONE_OUTPUT)

    print("\nTesting offline tokenizer reload...")

    offline_tokenizer = (
        AutoTokenizer.from_pretrained(
            TOKENIZER_OUTPUT,
            local_files_only=True,
            use_fast=True,
        )
    )

    print("Offline tokenizer reload: PASSED")

    print("\nTesting offline model reload...")

    offline_model = AutoModel.from_pretrained(
        BACKBONE_OUTPUT,
        local_files_only=True,
        torch_dtype=torch.float16,
    )

    print("Offline model reload: PASSED")

    sample = (
        "The battery is excellent, "
        "ஆனால் keyboard மிகவும் uncomfortable."
    )

    encoded = offline_tokenizer(
        sample,
        return_offsets_mapping=True,
        truncation=False,
    )

    tokens = offline_tokenizer.convert_ids_to_tokens(
        encoded["input_ids"]
    )

    print("\nMultilingual tokenization test:")

    for token, offset in zip(
        tokens,
        encoded["offset_mapping"],
    ):
        print(token, offset)

    total_parameters = sum(
        parameter.numel()
        for parameter in offline_model.parameters()
    )

    print("\nModel class:")
    print(
        offline_model.__class__.__name__
    )

    print("\nHidden size:")
    print(offline_model.config.hidden_size)

    print("\nEncoder layers:")
    print(
        offline_model.config.num_hidden_layers
    )

    print("\nVocabulary size:")
    print(
        offline_model.config.vocab_size
    )

    print("\nTotal parameters:")
    print(f"{total_parameters:,}")

    print("\n" + "=" * 70)
    print("XLM-ROBERTA PREPARATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()