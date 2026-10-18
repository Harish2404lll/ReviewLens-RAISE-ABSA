import gc
import json

from datetime import datetime
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "microsoft/deberta-v3-base"

# Pin the model revision for reproducibility.
MODEL_REVISION = (
    "e4647970a4c145e8a616c83caeaec8493e8aaf17"
)

OUTPUT_DIRECTORY = (
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

MANIFEST_FILE = (
    PROJECT_ROOT
    / "models"
    / "deberta-v3-base-backbone-manifest.json"
)


def backup_existing_directory(directory):
    if not directory.exists():
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = directory.with_name(
        f"{directory.name}_backup_{timestamp}"
    )

    counter = 1

    while backup.exists():
        backup = directory.with_name(
            f"{directory.name}_backup_"
            f"{timestamp}_{counter}"
        )

        counter += 1

    directory.rename(backup)

    return backup


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def find_largest_input_id(dataset):
    largest_input_id = 0

    for split_name in dataset.keys():
        for input_ids in dataset[
            split_name
        ]["input_ids"]:
            if input_ids:
                largest_input_id = max(
                    largest_input_id,
                    max(input_ids),
                )

    return largest_input_id


def main():
    print("=" * 60)
    print("DEBERTA-V3-BASE BACKBONE DOWNLOAD")
    print("=" * 60)

    if not TOKENIZED_DATASET_DIRECTORY.exists():
        raise FileNotFoundError(
            "Tokenized dataset does not exist:\n"
            f"{TOKENIZED_DATASET_DIRECTORY}"
        )

    OUTPUT_DIRECTORY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup = backup_existing_directory(
        OUTPUT_DIRECTORY
    )

    if backup is not None:
        print("\nExisting model directory moved to:")
        print(backup)

    print("\nDownloading model:")
    print(MODEL_NAME)

    print("\nPinned revision:")
    print(MODEL_REVISION)

    # The model is loaded on CPU. Do not move it to the GPU
    # during this download and verification stage.
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        low_cpu_mem_usage=True,
    )

    total_parameters = count_parameters(model)

    print("\nModel class:")
    print(model.__class__.__name__)

    print("Hidden size:")
    print(model.config.hidden_size)

    print("Encoder layers:")
    print(model.config.num_hidden_layers)

    print("Vocabulary size:")
    print(model.config.vocab_size)

    print("Total parameters:")
    print(f"{total_parameters:,}")

    print("\nSaving model for offline use...")

    model.save_pretrained(
        OUTPUT_DIRECTORY,
        safe_serialization=True,
    )

    model_vocabulary_size = (
        model.config.vocab_size
    )

    del model
    gc.collect()

    print("\nLoading tokenized dataset...")

    dataset = load_from_disk(
        str(TOKENIZED_DATASET_DIRECTORY)
    )

    largest_input_id = find_largest_input_id(
        dataset
    )

    print("Largest token ID in dataset:")
    print(largest_input_id)

    if largest_input_id >= model_vocabulary_size:
        raise ValueError(
            "Dataset contains a token ID outside the "
            "model vocabulary:"
            f"\nLargest token ID: {largest_input_id}"
            f"\nModel vocabulary size: "
            f"{model_vocabulary_size}"
        )

    print("Vocabulary compatibility: PASSED")

    print("\nTesting offline model reload...")

    offline_model = AutoModel.from_pretrained(
        OUTPUT_DIRECTORY,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    offline_parameter_count = count_parameters(
        offline_model
    )

    if offline_parameter_count != total_parameters:
        raise ValueError(
            "Parameter count changed after offline reload."
        )

    print("Offline model reload: PASSED")

    manifest = {
        "model_name": MODEL_NAME,
        "revision": MODEL_REVISION,
        "local_directory": str(
            OUTPUT_DIRECTORY
        ),
        "model_class": (
            offline_model.__class__.__name__
        ),
        "hidden_size": (
            offline_model.config.hidden_size
        ),
        "encoder_layers": (
            offline_model.config.num_hidden_layers
        ),
        "vocabulary_size": (
            offline_model.config.vocab_size
        ),
        "largest_dataset_input_id": (
            largest_input_id
        ),
        "total_parameters": (
            offline_parameter_count
        ),
        "offline_reload_verified": True,
        "previous_output_backup": (
            str(backup)
            if backup is not None
            else None
        ),
    }

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=True,
        )

        file.write("\n")

    print("\nModel saved to:")
    print(OUTPUT_DIRECTORY)

    print("\nManifest saved to:")
    print(MANIFEST_FILE)

    print()
    print("=" * 60)
    print("BACKBONE PREPARATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()