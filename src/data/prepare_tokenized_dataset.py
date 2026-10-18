import json

from collections import Counter
from datetime import datetime
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIRECTORY = (
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

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "deberta_v3_laptop_max96"
)

AUDIT_FILE = (
    INPUT_DIRECTORY
    / "tokenization_audit_max96.json"
)

SPLIT_FILES = {
    "train": (
        INPUT_DIRECTORY
        / "laptop_train.jsonl"
    ),
    "validation": (
        INPUT_DIRECTORY
        / "laptop_validation.jsonl"
    ),
    "test": (
        INPUT_DIRECTORY
        / "laptop_test.jsonl"
    ),
}

MAX_LENGTH = 96
IGNORE_INDEX = -100

BIO_LABEL_TO_ID = {
    "O": 0,
    "B-ASP": 1,
    "I-ASP": 2,
}

BIO_ID_TO_LABEL = {
    label_id: label
    for label, label_id
    in BIO_LABEL_TO_ID.items()
}

SENTIMENT_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "conflict": 3,
}


# ============================================================
# FILE UTILITIES
# ============================================================

def require_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file does not exist:\n"
            f"{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Expected a file but found something else:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"Required file is empty:\n"
            f"{file_path}"
        )


def load_jsonl(file_path):
    require_file(file_path)

    records = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path}, "
                    f"line {line_number}"
                ) from error

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found in:\n{file_path}"
        )

    return records


def backup_existing_output(output_directory):
    """
    Move an existing generated directory to a timestamped
    backup. Nothing is deleted.
    """

    if not output_directory.exists():
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_directory = output_directory.with_name(
        f"{output_directory.name}_backup_{timestamp}"
    )

    counter = 1

    while backup_directory.exists():
        backup_directory = (
            output_directory.with_name(
                f"{output_directory.name}_backup_"
                f"{timestamp}_{counter}"
            )
        )

        counter += 1

    output_directory.rename(
        backup_directory
    )

    return backup_directory


# ============================================================
# ASPECT NORMALIZATION
# ============================================================

def normalize_aspect_boundary(
    text,
    aspect,
):
    """
    Trim only leading and trailing Unicode whitespace from an
    aspect span.

    Internal whitespace is preserved. The original JSONL files
    are not modified.
    """

    original_start = aspect["start"]
    original_end = aspect["end"]
    original_term = aspect["term"]

    corrected_start = original_start
    corrected_end = original_end

    while (
        corrected_start < corrected_end
        and text[corrected_start].isspace()
    ):
        corrected_start += 1

    while (
        corrected_end > corrected_start
        and text[corrected_end - 1].isspace()
    ):
        corrected_end -= 1

    if corrected_start >= corrected_end:
        raise ValueError(
            "Aspect became empty after whitespace "
            f"normalization: {aspect}"
        )

    corrected_term = text[
        corrected_start:corrected_end
    ]

    corrected_aspect = {
        "term": corrected_term,
        "start": corrected_start,
        "end": corrected_end,
        "sentiment": aspect["sentiment"],
    }

    changed = (
        corrected_start != original_start
        or corrected_end != original_end
        or corrected_term != original_term
    )

    correction = None

    if changed:
        correction = {
            "original": {
                "term": original_term,
                "start": original_start,
                "end": original_end,
            },
            "corrected": corrected_aspect,
            "reason": (
                "leading_or_trailing_unicode_whitespace"
            ),
        }

    return corrected_aspect, correction


# ============================================================
# TOKEN ALIGNMENT
# ============================================================

def find_aspect_token_indices(
    aspect,
    offsets,
):
    """
    Find every token that overlaps the aspect character span.
    """

    aspect_start = aspect["start"]
    aspect_end = aspect["end"]

    token_indices = []

    for token_index, offset in enumerate(offsets):
        token_start, token_end = offset

        # Special tokens normally use (0, 0).
        if token_end <= token_start:
            continue

        overlaps = (
            token_start < aspect_end
            and token_end > aspect_start
        )

        if overlaps:
            token_indices.append(token_index)

    if not token_indices:
        return []

    covered_start = min(
        offsets[index][0]
        for index in token_indices
    )

    covered_end = max(
        offsets[index][1]
        for index in token_indices
    )

    fully_covered = (
        covered_start <= aspect_start
        and covered_end >= aspect_end
    )

    if not fully_covered:
        return []

    expected_indices = list(
        range(
            token_indices[0],
            token_indices[-1] + 1,
        )
    )

    if token_indices != expected_indices:
        raise ValueError(
            "Non-contiguous tokens found for aspect "
            f"{aspect}. Token indices: {token_indices}"
        )

    return token_indices


# ============================================================
# RECORD TOKENIZATION
# ============================================================

def tokenize_record(
    record,
    tokenizer,
    split_name,
):
    required_fields = {
        "id",
        "source_id",
        "domain",
        "text",
        "aspects",
    }

    missing_fields = (
        required_fields - set(record)
    )

    if missing_fields:
        raise ValueError(
            f"Record is missing fields "
            f"{sorted(missing_fields)}: {record}"
        )

    text = record["text"]

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_offsets_mapping=True,
    )

    input_ids = list(
        encoding["input_ids"]
    )

    attention_mask = list(
        encoding["attention_mask"]
    )

    offsets = [
        [int(start), int(end)]
        for start, end
        in encoding["offset_mapping"]
    ]

    sequence_length = len(input_ids)

    if sequence_length > MAX_LENGTH:
        raise ValueError(
            f"Record {record['id']} has "
            f"{sequence_length} tokens, exceeding "
            f"MAX_LENGTH={MAX_LENGTH}"
        )

    if not (
        sequence_length
        == len(attention_mask)
        == len(offsets)
    ):
        raise ValueError(
            f"Tokenizer field-length mismatch for "
            f"record {record['id']}"
        )

    bio_labels = []

    for token_start, token_end in offsets:
        if token_start == token_end:
            bio_labels.append(
                IGNORE_INDEX
            )
        else:
            bio_labels.append(
                BIO_LABEL_TO_ID["O"]
            )

    aspect_terms = []
    aspect_sentiments = []
    aspect_sentiment_ids = []

    aspect_char_starts = []
    aspect_char_ends = []

    aspect_token_starts = []
    aspect_token_ends = []

    corrections = []

    for aspect_number, raw_aspect in enumerate(
        record["aspects"],
        start=1,
    ):
        normalized_aspect, correction = (
            normalize_aspect_boundary(
                text,
                raw_aspect,
            )
        )

        sentiment = normalized_aspect[
            "sentiment"
        ]

        if sentiment not in SENTIMENT_TO_ID:
            raise ValueError(
                f"Unknown sentiment '{sentiment}' "
                f"in record {record['id']}"
            )

        if correction is not None:
            correction.update(
                {
                    "split": split_name,
                    "record_id": record["id"],
                    "source_id": (
                        record["source_id"]
                    ),
                    "aspect_number": (
                        aspect_number
                    ),
                    "text": text,
                }
            )

            corrections.append(correction)

        token_indices = (
            find_aspect_token_indices(
                normalized_aspect,
                offsets,
            )
        )

        if not token_indices:
            raise ValueError(
                "\nAspect could not be aligned:"
                f"\nSplit: {split_name}"
                f"\nRecord: {record['id']}"
                f"\nText: {text!r}"
                f"\nAspect: {normalized_aspect}"
                f"\nOffsets: {offsets}"
            )

        # Prevent overlapping annotations from silently
        # overwriting each other's BIO labels.
        for token_index in token_indices:
            current_label = bio_labels[
                token_index
            ]

            if current_label != BIO_LABEL_TO_ID["O"]:
                raise ValueError(
                    "\nOverlapping aspect annotations:"
                    f"\nSplit: {split_name}"
                    f"\nRecord: {record['id']}"
                    f"\nAspect: {normalized_aspect}"
                    f"\nToken index: {token_index}"
                    f"\nExisting BIO label: "
                    f"{current_label}"
                )

        first_token_index = token_indices[0]

        bio_labels[first_token_index] = (
            BIO_LABEL_TO_ID["B-ASP"]
        )

        for token_index in token_indices[1:]:
            bio_labels[token_index] = (
                BIO_LABEL_TO_ID["I-ASP"]
            )

        aspect_terms.append(
            normalized_aspect["term"]
        )

        aspect_sentiments.append(
            sentiment
        )

        aspect_sentiment_ids.append(
            SENTIMENT_TO_ID[sentiment]
        )

        aspect_char_starts.append(
            normalized_aspect["start"]
        )

        aspect_char_ends.append(
            normalized_aspect["end"]
        )

        aspect_token_starts.append(
            token_indices[0]
        )

        # Store the token end as an exclusive index.
        aspect_token_ends.append(
            token_indices[-1] + 1
        )

    tokenized_record = {
        "id": record["id"],
        "source_id": record["source_id"],
        "domain": record["domain"],
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offsets,
        "bio_labels": bio_labels,
        "sequence_length": sequence_length,
        "aspect_terms": aspect_terms,
        "aspect_sentiments": aspect_sentiments,
        "aspect_sentiment_ids": (
            aspect_sentiment_ids
        ),
        "aspect_char_starts": (
            aspect_char_starts
        ),
        "aspect_char_ends": aspect_char_ends,
        "aspect_token_starts": (
            aspect_token_starts
        ),
        "aspect_token_ends": (
            aspect_token_ends
        ),
    }

    if "token_type_ids" in encoding:
        tokenized_record["token_type_ids"] = list(
            encoding["token_type_ids"]
        )

    return tokenized_record, corrections


# ============================================================
# SPLIT PROCESSING
# ============================================================

def process_split(
    split_name,
    records,
    tokenizer,
):
    tokenized_records = []
    all_corrections = []

    sentiment_distribution = Counter()
    bio_distribution = Counter()

    sentences_with_aspects = 0

    for record in records:
        tokenized_record, corrections = (
            tokenize_record(
                record,
                tokenizer,
                split_name,
            )
        )

        tokenized_records.append(
            tokenized_record
        )

        all_corrections.extend(
            corrections
        )

        if tokenized_record["aspect_terms"]:
            sentences_with_aspects += 1

        sentiment_distribution.update(
            tokenized_record[
                "aspect_sentiments"
            ]
        )

        for label_id in tokenized_record[
            "bio_labels"
        ]:
            if label_id == IGNORE_INDEX:
                continue

            label_name = BIO_ID_TO_LABEL[
                label_id
            ]

            bio_distribution[
                label_name
            ] += 1

    sequence_lengths = [
        record["sequence_length"]
        for record in tokenized_records
    ]

    audit = {
        "sentences": len(tokenized_records),
        "sentences_with_aspects": (
            sentences_with_aspects
        ),
        "sentences_without_aspects": (
            len(tokenized_records)
            - sentences_with_aspects
        ),
        "total_aspects": sum(
            len(record["aspect_terms"])
            for record in tokenized_records
        ),
        "minimum_sequence_length": min(
            sequence_lengths
        ),
        "maximum_sequence_length": max(
            sequence_lengths
        ),
        "sentiment_distribution": dict(
            sorted(
                sentiment_distribution.items()
            )
        ),
        "bio_label_distribution": dict(
            sorted(
                bio_distribution.items()
            )
        ),
        "annotation_correction_count": len(
            all_corrections
        ),
        "annotation_corrections": (
            all_corrections
        ),
    }

    return tokenized_records, audit


# ============================================================
# VALIDATION
# ============================================================

def validate_tokenized_split(
    split_name,
    records,
):
    for record in records:
        expected_length = len(
            record["input_ids"]
        )

        if expected_length == 0:
            raise ValueError(
                f"Empty token sequence in "
                f"{split_name}, record {record['id']}"
            )

        if expected_length > MAX_LENGTH:
            raise ValueError(
                f"Sequence exceeds MAX_LENGTH in "
                f"{split_name}, record {record['id']}"
            )

        sequence_fields = {
            "attention_mask": (
                record["attention_mask"]
            ),
            "offset_mapping": (
                record["offset_mapping"]
            ),
            "bio_labels": (
                record["bio_labels"]
            ),
        }

        if "token_type_ids" in record:
            sequence_fields[
                "token_type_ids"
            ] = record["token_type_ids"]

        for field_name, field_value in (
            sequence_fields.items()
        ):
            if len(field_value) != expected_length:
                raise ValueError(
                    f"Length mismatch in {split_name}, "
                    f"record {record['id']}, "
                    f"field {field_name}"
                )

        aspect_count = len(
            record["aspect_terms"]
        )

        aspect_fields = [
            "aspect_sentiments",
            "aspect_sentiment_ids",
            "aspect_char_starts",
            "aspect_char_ends",
            "aspect_token_starts",
            "aspect_token_ends",
        ]

        for field_name in aspect_fields:
            if (
                len(record[field_name])
                != aspect_count
            ):
                raise ValueError(
                    f"Aspect-field mismatch in "
                    f"{split_name}, "
                    f"record {record['id']}, "
                    f"field {field_name}"
                )

        for (
            token_start,
            token_end,
            sentiment_id,
        ) in zip(
            record["aspect_token_starts"],
            record["aspect_token_ends"],
            record["aspect_sentiment_ids"],
        ):
            if not (
                0 <= token_start
                < token_end
                <= expected_length
            ):
                raise ValueError(
                    f"Invalid aspect token span in "
                    f"{split_name}, "
                    f"record {record['id']}"
                )

            if sentiment_id not in (
                SENTIMENT_TO_ID.values()
            ):
                raise ValueError(
                    f"Invalid sentiment ID in "
                    f"{split_name}, "
                    f"record {record['id']}"
                )

            if (
                record["bio_labels"][token_start]
                != BIO_LABEL_TO_ID["B-ASP"]
            ):
                raise ValueError(
                    f"Aspect token span does not begin "
                    f"with B-ASP in {split_name}, "
                    f"record {record['id']}"
                )


# ============================================================
# OFFLINE RELOAD CHECK
# ============================================================

def validate_reloaded_dataset(
    original_dataset,
    reloaded_dataset,
):
    expected_splits = set(
        SPLIT_FILES.keys()
    )

    if set(reloaded_dataset.keys()) != expected_splits:
        raise ValueError(
            "Reloaded dataset has unexpected splits: "
            f"{list(reloaded_dataset.keys())}"
        )

    for split_name in expected_splits:
        original_count = len(
            original_dataset[split_name]
        )

        reloaded_count = len(
            reloaded_dataset[split_name]
        )

        if original_count != reloaded_count:
            raise ValueError(
                f"Reload count mismatch for "
                f"{split_name}: "
                f"{original_count} != {reloaded_count}"
            )

        if reloaded_count == 0:
            raise ValueError(
                f"Reloaded split '{split_name}' is empty."
            )

        sample = reloaded_dataset[
            split_name
        ][0]

        required_fields = {
            "input_ids",
            "attention_mask",
            "bio_labels",
            "aspect_sentiment_ids",
        }

        missing_fields = (
            required_fields - set(sample)
        )

        if missing_fields:
            raise ValueError(
                f"Reloaded split '{split_name}' "
                f"is missing fields: "
                f"{sorted(missing_fields)}"
            )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("DEBERTA BIO DATASET PREPARATION")
    print("=" * 60)

    if not TOKENIZER_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Tokenizer directory not found:\n"
            f"{TOKENIZER_DIRECTORY}"
        )

    for input_file in SPLIT_FILES.values():
        require_file(input_file)

    print("\nLoading tokenizer from local storage...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIRECTORY,
        use_fast=True,
        local_files_only=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for "
            "offset mapping."
        )

    print(
        "Tokenizer class:",
        tokenizer.__class__.__name__,
    )

    print(
        "Tokenizer vocabulary size:",
        tokenizer.vocab_size,
    )

    print(
        "Configured maximum length:",
        MAX_LENGTH,
    )

    print("\nChecking existing output directory...")

    backup_directory = backup_existing_output(
        OUTPUT_DIRECTORY
    )

    if backup_directory is None:
        print("No existing output directory found.")
    else:
        print("Existing output moved safely to:")
        print(backup_directory)

    dataset_splits = {}
    audit_splits = {}

    for split_name, input_file in (
        SPLIT_FILES.items()
    ):
        print()
        print("-" * 60)
        print(f"Processing split: {split_name}")
        print("-" * 60)

        records = load_jsonl(
            input_file
        )

        tokenized_records, split_audit = (
            process_split(
                split_name,
                records,
                tokenizer,
            )
        )

        validate_tokenized_split(
            split_name,
            tokenized_records,
        )

        dataset_splits[split_name] = (
            Dataset.from_list(
                tokenized_records
            )
        )

        audit_splits[split_name] = (
            split_audit
        )

        print(
            "Sentences:",
            split_audit["sentences"],
        )

        print(
            "Sentences with aspects:",
            split_audit[
                "sentences_with_aspects"
            ],
        )

        print(
            "Sentences without aspects:",
            split_audit[
                "sentences_without_aspects"
            ],
        )

        print(
            "Total aspects:",
            split_audit["total_aspects"],
        )

        print(
            "Maximum sequence length:",
            split_audit[
                "maximum_sequence_length"
            ],
        )

        print(
            "Annotation corrections:",
            split_audit[
                "annotation_correction_count"
            ],
        )

        print(
            "Sentiment distribution:",
            split_audit[
                "sentiment_distribution"
            ],
        )

        print(
            "BIO distribution:",
            split_audit[
                "bio_label_distribution"
            ],
        )

    tokenized_dataset = DatasetDict(
        dataset_splits
    )

    OUTPUT_DIRECTORY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSaving tokenized dataset...")

    tokenized_dataset.save_to_disk(
        str(OUTPUT_DIRECTORY)
    )

    audit_report = {
        "tokenizer_directory": str(
            TOKENIZER_DIRECTORY
        ),
        "tokenizer_class": (
            tokenizer.__class__.__name__
        ),
        "tokenizer_vocabulary_size": (
            tokenizer.vocab_size
        ),
        "maximum_length": MAX_LENGTH,
        "padding_policy": (
            "Dynamic padding during batching"
        ),
        "truncation_enabled": True,
        "bio_label_to_id": BIO_LABEL_TO_ID,
        "bio_id_to_label": {
            str(label_id): label
            for label_id, label
            in BIO_ID_TO_LABEL.items()
        },
        "sentiment_to_id": SENTIMENT_TO_ID,
        "ignore_index": IGNORE_INDEX,
        "aspect_boundary_policy": (
            "Leading and trailing Unicode whitespace "
            "is trimmed during token alignment. "
            "Original JSONL files remain unchanged."
        ),
        "previous_output_backup": (
            str(backup_directory)
            if backup_directory is not None
            else None
        ),
        "splits": audit_splits,
    }

    with AUDIT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit_report,
            file,
            indent=2,
            ensure_ascii=True,
        )

        file.write("\n")

    print("\nReloading dataset from disk...")

    reloaded_dataset = load_from_disk(
        str(OUTPUT_DIRECTORY)
    )

    validate_reloaded_dataset(
        tokenized_dataset,
        reloaded_dataset,
    )

    print("Offline reload verification: PASSED")

    print("\nTokenized dataset saved to:")
    print(OUTPUT_DIRECTORY)

    print("\nAudit report saved to:")
    print(AUDIT_FILE)

    if backup_directory is not None:
        print("\nPrevious output backup:")
        print(backup_directory)

    print()
    print("=" * 60)
    print("TOKENIZED DATASET PREPARATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()