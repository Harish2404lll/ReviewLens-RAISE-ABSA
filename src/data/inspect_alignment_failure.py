import json

from pathlib import Path
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "official_laptop"
)

TOKENIZER_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "deberta-v3-base-tokenizer"
)

SPLIT_FILES = {
    "train": DATA_DIRECTORY / "laptop_train.jsonl",
    "validation": (
        DATA_DIRECTORY / "laptop_validation.jsonl"
    ),
    "test": DATA_DIRECTORY / "laptop_test.jsonl",
}


def load_jsonl(file_path):
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def aspect_is_covered(aspect, offsets):
    start = aspect["start"]
    end = aspect["end"]

    overlapping = [
        (token_start, token_end)
        for token_start, token_end in offsets
        if token_end > token_start
        and token_start < end
        and token_end > start
    ]

    if not overlapping:
        return False

    covered_start = min(
        token_start
        for token_start, _ in overlapping
    )

    covered_end = max(
        token_end
        for _, token_end in overlapping
    )

    return (
        covered_start <= start
        and covered_end >= end
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIRECTORY,
        use_fast=True,
        local_files_only=True,
    )

    total_failures = 0

    for split_name, file_path in SPLIT_FILES.items():
        records = load_jsonl(file_path)

        print("\n" + "=" * 70)
        print(f"SPLIT: {split_name.upper()}")
        print("=" * 70)

        split_failures = 0

        for record in records:
            encoding = tokenizer(
                record["text"],
                add_special_tokens=True,
                truncation=False,
                return_offsets_mapping=True,
            )

            tokens = tokenizer.convert_ids_to_tokens(
                encoding["input_ids"]
            )

            offsets = encoding["offset_mapping"]

            for aspect in record["aspects"]:
                if aspect_is_covered(
                    aspect,
                    offsets,
                ):
                    continue

                total_failures += 1
                split_failures += 1

                print("\nALIGNMENT FAILURE")
                print("-" * 70)
                print("Record ID:", record["id"])
                print("Source ID:", record["source_id"])
                print("Text:", repr(record["text"]))
                print("Aspect:", json.dumps(
                    aspect,
                    ensure_ascii=False,
                ))

                print(
                    "Expected slice:",
                    repr(
                        record["text"][
                            aspect["start"]:
                            aspect["end"]
                        ]
                    ),
                )

                print("\nTokens and offsets:")

                for index, (
                    token,
                    offset,
                ) in enumerate(
                    zip(tokens, offsets)
                ):
                    token_start, token_end = offset

                    if token_end > token_start:
                        original_slice = (
                            record["text"][
                                token_start:token_end
                            ]
                        )
                    else:
                        original_slice = ""

                    near_aspect = (
                        token_end >= aspect["start"] - 10
                        and token_start <= aspect["end"] + 10
                    )

                    marker = (
                        " <-- NEAR ASPECT"
                        if near_aspect
                        else ""
                    )

                    print(
                        f"{index:03d} "
                        f"{token!r:25} "
                        f"{offset!s:12} "
                        f"{original_slice!r}"
                        f"{marker}"
                    )

        print(
            f"\nFailures in {split_name}: "
            f"{split_failures}"
        )

    print("\n" + "=" * 70)
    print("TOTAL ALIGNMENT FAILURES:", total_failures)
    print("=" * 70)


if __name__ == "__main__":
    main()