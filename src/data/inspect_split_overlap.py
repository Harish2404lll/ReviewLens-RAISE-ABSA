import json
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "official_laptop"
)


def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        text.strip().casefold(),
    )


def load_jsonl(file_path):
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path}, "
                    f"line {line_number}"
                ) from error

    return records


def create_index(records):
    index = defaultdict(list)

    for record in records:
        index[normalize_text(record["text"])].append(record)

    return index


def main():
    train_file = DATA_DIRECTORY / "laptop_train.jsonl"
    validation_file = (
        DATA_DIRECTORY / "laptop_validation.jsonl"
    )
    test_file = DATA_DIRECTORY / "laptop_test.jsonl"

    train_records = load_jsonl(train_file)
    validation_records = load_jsonl(validation_file)
    test_records = load_jsonl(test_file)

    train_index = create_index(train_records)
    validation_index = create_index(validation_records)
    test_index = create_index(test_records)

    comparisons = {
        "TRAIN vs VALIDATION": (
            train_index,
            validation_index,
        ),
        "TRAIN vs TEST": (
            train_index,
            test_index,
        ),
        "VALIDATION vs TEST": (
            validation_index,
            test_index,
        ),
    }

    for comparison_name, (left, right) in comparisons.items():
        overlaps = sorted(set(left) & set(right))

        print("\n" + "=" * 70)
        print(comparison_name)
        print("=" * 70)
        print("Overlapping normalized sentences:", len(overlaps))

        for overlap_number, normalized_text in enumerate(
            overlaps,
            start=1,
        ):
            print(f"\nOverlap {overlap_number}")
            print("-" * 70)
            print("Normalized text:")
            print(normalized_text)

            print("\nLeft-side records:")
            for record in left[normalized_text]:
                print(
                    json.dumps(
                        record,
                        indent=2,
                        ensure_ascii=False,
                    )
                )

            print("\nRight-side records:")
            for record in right[normalized_text]:
                print(
                    json.dumps(
                        record,
                        indent=2,
                        ensure_ascii=False,
                    )
                )


if __name__ == "__main__":
    main()