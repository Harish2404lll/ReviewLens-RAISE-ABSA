from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "microsoft/deberta-v3-base"

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "deberta-v3-base-tokenizer"
)


def main():
    print("=" * 60)
    print("DEBERTA-V3 TOKENIZER DOWNLOAD")
    print("=" * 60)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nDownloading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required because the project "
            "needs character-to-token offset mappings."
        )

    tokenizer.save_pretrained(
        OUTPUT_DIRECTORY
    )

    print("\nTokenizer saved to:")
    print(OUTPUT_DIRECTORY)

    print("\nTokenizer details:")
    print("Class:", tokenizer.__class__.__name__)
    print("Fast tokenizer:", tokenizer.is_fast)
    print("Vocabulary size:", tokenizer.vocab_size)
    print(
        "Model maximum length:",
        tokenizer.model_max_length,
    )

    test_text = (
        "The battery life is excellent, "
        "but the keyboard is uncomfortable."
    )

    encoded = tokenizer(
        test_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    print("\nTokenization test:")
    print(
        list(
            zip(
                tokenizer.convert_ids_to_tokens(
                    encoded["input_ids"]
                ),
                encoded["offset_mapping"],
            )
        )
    )

    print("\nTesting offline reload...")

    offline_tokenizer = (
        AutoTokenizer.from_pretrained(
            OUTPUT_DIRECTORY,
            use_fast=True,
            local_files_only=True,
        )
    )

    if not offline_tokenizer.is_fast:
        raise RuntimeError(
            "Offline tokenizer did not reload as a "
            "fast tokenizer."
        )

    print("Offline reload: PASSED")
    print("\nTOKENIZER PREPARATION COMPLETED")


if __name__ == "__main__":
    main()