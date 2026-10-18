from collections import Counter
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "deberta-v3-base-backbone"
)


def main():
    print("=" * 60)
    print("DEBERTA BACKBONE INSPECTION")
    print("=" * 60)

    model = AutoModel.from_pretrained(
        MODEL_DIRECTORY,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    dtype_distribution = Counter(
        str(parameter.dtype)
        for parameter in model.parameters()
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    linear_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    attention_projection_modules = [
        name
        for name in linear_modules
        if any(
            target in name
            for target in (
                "query_proj",
                "key_proj",
                "value_proj",
            )
        )
    ]

    print("\nModel class:")
    print(model.__class__.__name__)

    print("\nTotal parameters:")
    print(f"{total_parameters:,}")

    print("\nParameter dtypes:")
    for dtype, count in dtype_distribution.items():
        print(f"{dtype}: {count} tensors")

    print("\nAttention projection modules:")
    for name in attention_projection_modules:
        print(name)

    print(
        "\nAttention projection count:",
        len(attention_projection_modules),
    )

    query_count = sum(
        name.endswith("query_proj")
        for name in attention_projection_modules
    )

    key_count = sum(
        name.endswith("key_proj")
        for name in attention_projection_modules
    )

    value_count = sum(
        name.endswith("value_proj")
        for name in attention_projection_modules
    )

    print("\nProjection counts:")
    print("query_proj:", query_count)
    print("key_proj:", key_count)
    print("value_proj:", value_count)

    if query_count != model.config.num_hidden_layers:
        raise ValueError(
            "Unexpected query projection count."
        )

    if value_count != model.config.num_hidden_layers:
        raise ValueError(
            "Unexpected value projection count."
        )

    print("\nLoRA target recommendation:")
    print('["query_proj", "value_proj"]')

    print("\nCUDA available:")
    print(torch.cuda.is_available())

    print()
    print("=" * 60)
    print("BACKBONE INSPECTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()