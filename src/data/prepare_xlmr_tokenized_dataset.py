# src/data/prepare_xlmr_tokenized_dataset.py

import json
import shutil

from collections import Counter
from datetime import datetime
from pathlib import Path

from datasets import (
    Dataset,
    DatasetDict,
    load_from_disk,
)
from transformers import AutoTokenizer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "official_laptop"
)

TOKENIZER_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "xlm-roberta-base-tokenizer"
)

TOKEN_LENGTH_AUDIT_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "xlmr_token_length_audit.json"
)

SPLIT_FILES = {
    "train": (
        PROCESSED_DATA_DIRECTORY
        / "laptop_train.jsonl"
    ),
    "validation": (
        PROCESSED_DATA_DIRECTORY
        / "laptop_validation.jsonl"
    ),
    "test": (
        PROCESSED_DATA_DIRECTORY
        / "laptop_test.jsonl"
    ),
}


# ============================================================
# LABEL MAPPINGS
# ============================================================

BIO_LABEL_TO_ID = {
    "O": 0,
    "B-ASP": 1,
    "I-ASP": 2,
}

BIO_ID_TO_LABEL = {
    value: key
    for key, value in BIO_LABEL_TO_ID.items()
}

SENTIMENT_LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "conflict": 3,
}

SENTIMENT_ID_TO_LABEL = {
    value: key
    for key, value
    in SENTIMENT_LABEL_TO_ID.items()
}

IGNORE_INDEX = -100


# ============================================================
# JSON UTILITIES
# ============================================================

def load_json(file_path):
    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(data, file_path):
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=True,
        )

        file.write("\n")


def load_jsonl(file_path):
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
                    f"line {line_number}."
                ) from error

            records.append(record)

    return records


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_required_files():
    required_paths = [
        TOKENIZER_DIRECTORY,
        TOKEN_LENGTH_AUDIT_PATH,
        *SPLIT_FILES.values(),
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        missing_text = "\n".join(
            str(path)
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Required files or directories are missing:\n"
            f"{missing_text}"
        )


# ============================================================
# AUDIT CONFIGURATION
# ============================================================

def load_verified_maximum_length():
    audit = load_json(
        TOKEN_LENGTH_AUDIT_PATH
    )

    model_family = audit.get(
        "model_family"
    )

    tokenizer_path = str(
        audit.get(
            "tokenizer",
            "",
        )
    ).lower()

    tokenizer_class = str(
        audit.get(
            "tokenizer_class",
            "",
        )
    )

    total_failures = audit.get(
        "total_alignment_failures"
    )

    maximum_length = audit.get(
        "recommended_max_length"
    )

    if model_family not in {
        None,
        "xlm-roberta-base",
    }:
        raise ValueError(
            "The token-length audit identifies an "
            "unexpected model family.\n"
            f"Found: {model_family}"
        )

    if model_family is None:
        print(
            "Warning: model_family is absent from the "
            "audit. XLM-R identity will be verified "
            "using tokenizer path and tokenizer class."
        )

    if "xlm-roberta-base-tokenizer" not in (
        tokenizer_path
    ):
        raise ValueError(
            "The token-length audit was not generated "
            "with the XLM-R tokenizer.\n"
            f"Tokenizer: {audit.get('tokenizer')}"
        )

    if "XLMRoberta" not in tokenizer_class:
        raise ValueError(
            "Unexpected tokenizer class in audit:\n"
            f"{tokenizer_class}"
        )

    if total_failures != 0:
        raise ValueError(
            "The XLM-R token-length audit contains "
            f"{total_failures} alignment failures."
        )

    if maximum_length is None:
        raise ValueError(
            "The audit did not recommend a maximum "
            "sequence length."
        )

    try:
        maximum_length = int(
            maximum_length
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The recommended maximum length is not "
            f"a valid integer: {maximum_length}"
        ) from error

    if maximum_length < 4:
        raise ValueError(
            "The recommended maximum length is "
            f"invalid: {maximum_length}"
        )

    return maximum_length, audit


# ============================================================
# ASPECT NORMALIZATION
# ============================================================

def normalize_aspect_boundary(
    text,
    aspect,
):
    """
    Trim only leading/trailing Unicode whitespace
    from an annotated aspect span.

    The original JSONL file is not modified.
    """

    required_keys = {
        "term",
        "start",
        "end",
        "sentiment",
    }

    missing_keys = (
        required_keys - set(aspect)
    )

    if missing_keys:
        raise KeyError(
            "Aspect annotation is missing keys: "
            f"{sorted(missing_keys)}"
        )

    original_start = int(
        aspect["start"]
    )

    original_end = int(
        aspect["end"]
    )

    original_term = str(
        aspect["term"]
    )

    sentiment = str(
        aspect["sentiment"]
    ).strip().lower()

    if sentiment not in (
        SENTIMENT_LABEL_TO_ID
    ):
        raise ValueError(
            f"Unsupported sentiment: {sentiment}"
        )

    if not (
        0
        <= original_start
        < original_end
        <= len(text)
    ):
        raise ValueError(
            "Invalid aspect character span.\n"
            f"Text length: {len(text)}\n"
            f"Aspect: {aspect}"
        )

    corrected_start = original_start
    corrected_end = original_end

    while (
        corrected_start < corrected_end
        and text[
            corrected_start
        ].isspace()
    ):
        corrected_start += 1

    while (
        corrected_end > corrected_start
        and text[
            corrected_end - 1
        ].isspace()
    ):
        corrected_end -= 1

    if corrected_start >= corrected_end:
        raise ValueError(
            "Aspect became empty after Unicode "
            f"whitespace normalization: {aspect}"
        )

    corrected_term = text[
        corrected_start:corrected_end
    ]

    corrected_aspect = {
        "term": corrected_term,
        "start": corrected_start,
        "end": corrected_end,
        "sentiment": sentiment,
    }

    was_corrected = (
        corrected_start != original_start
        or corrected_end != original_end
        or corrected_term != original_term
    )

    correction_record = None

    if was_corrected:
        correction_record = {
            "original": {
                "term": original_term,
                "start": original_start,
                "end": original_end,
                "sentiment": sentiment,
            },
            "corrected": corrected_aspect,
            "reason": (
                "leading_or_trailing_unicode_"
                "whitespace"
            ),
        }

    return (
        corrected_aspect,
        correction_record,
    )


# ============================================================
# TOKEN-SPAN ALIGNMENT
# ============================================================

def find_aspect_token_span(
    aspect,
    offset_mapping,
):
    aspect_start = aspect["start"]
    aspect_end = aspect["end"]

    overlapping_token_indexes = []

    for token_index, offset in enumerate(
        offset_mapping
    ):
        token_start = int(offset[0])
        token_end = int(offset[1])

        # Ignore special tokens such as <s> and </s>.
        if token_end <= token_start:
            continue

        overlaps = (
            token_start < aspect_end
            and token_end > aspect_start
        )

        if overlaps:
            overlapping_token_indexes.append(
                token_index
            )

    if not overlapping_token_indexes:
        return None

    token_start_index = min(
        overlapping_token_indexes
    )

    token_end_index = (
        max(overlapping_token_indexes) + 1
    )

    covered_start = min(
        int(
            offset_mapping[index][0]
        )
        for index
        in overlapping_token_indexes
    )

    covered_end = max(
        int(
            offset_mapping[index][1]
        )
        for index
        in overlapping_token_indexes
    )

    completely_covered = (
        covered_start <= aspect_start
        and covered_end >= aspect_end
    )

    if not completely_covered:
        return None

    return {
        "token_start": (
            token_start_index
        ),
        "token_end": token_end_index,
        "covered_start": covered_start,
        "covered_end": covered_end,
    }


# ============================================================
# RECORD TOKENIZATION
# ============================================================

def tokenize_record(
    record,
    tokenizer,
    maximum_length,
    split_name,
):
    required_record_keys = {
        "id",
        "text",
        "aspects",
    }

    missing_keys = (
        required_record_keys - set(record)
    )

    if missing_keys:
        raise KeyError(
            f"Record is missing keys "
            f"{sorted(missing_keys)}:\n"
            f"{record}"
        )

    record_id = str(record["id"])
    text = str(record["text"])

    if not text.strip():
        raise ValueError(
            f"Empty text in record {record_id}."
        )

    normalized_aspects = []
    correction_records = []

    for aspect_index, aspect in enumerate(
        record["aspects"]
    ):
        (
            normalized_aspect,
            correction,
        ) = normalize_aspect_boundary(
            text=text,
            aspect=aspect,
        )

        normalized_aspects.append(
            normalized_aspect
        )

        if correction is not None:
            correction["record_id"] = (
                record_id
            )

            correction["split"] = (
                split_name
            )

            correction["aspect_index"] = (
                aspect_index
            )

            correction["text"] = text

            correction_records.append(
                correction
            )

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=maximum_length,
        return_attention_mask=True,
        return_offsets_mapping=True,
        padding=False,
    )

    input_ids = list(
        encoding["input_ids"]
    )

    attention_mask = list(
        encoding["attention_mask"]
    )

    offset_mapping = [
        [int(start), int(end)]
        for start, end
        in encoding["offset_mapping"]
    ]

    if len(input_ids) > maximum_length:
        raise RuntimeError(
            "Tokenizer produced a sequence longer "
            "than the configured maximum length.\n"
            f"Record: {record_id}\n"
            f"Produced: {len(input_ids)}\n"
            f"Maximum: {maximum_length}"
        )

    if not (
        len(input_ids)
        == len(attention_mask)
        == len(offset_mapping)
    ):
        raise RuntimeError(
            "Tokenization arrays have inconsistent "
            f"lengths in record {record_id}."
        )

    bio_labels = []

    for token_start, token_end in (
        offset_mapping
    ):
        if token_end <= token_start:
            bio_labels.append(
                IGNORE_INDEX
            )
        else:
            bio_labels.append(
                BIO_LABEL_TO_ID["O"]
            )

    aspect_token_starts = []
    aspect_token_ends = []
    aspect_sentiment_ids = []
    aspect_terms = []
    aspect_character_starts = []
    aspect_character_ends = []

    occupied_token_indexes = set()

    for aspect in normalized_aspects:
        token_span = (
            find_aspect_token_span(
                aspect=aspect,
                offset_mapping=(
                    offset_mapping
                ),
            )
        )

        if token_span is None:
            raise ValueError(
                "Aspect could not be aligned after "
                "XLM-R tokenization.\n"
                f"Split: {split_name}\n"
                f"Record: {record_id}\n"
                f"Text: {text!r}\n"
                f"Aspect: {aspect}\n"
                f"Maximum length: {maximum_length}"
            )

        token_start = token_span[
            "token_start"
        ]

        token_end = token_span[
            "token_end"
        ]

        current_token_indexes = set(
            range(
                token_start,
                token_end,
            )
        )

        overlap = (
            occupied_token_indexes
            & current_token_indexes
        )

        if overlap:
            raise ValueError(
                "Overlapping aspect annotations "
                "cannot be represented using simple "
                "BIO labels.\n"
                f"Split: {split_name}\n"
                f"Record: {record_id}\n"
                f"Overlapping token indexes: "
                f"{sorted(overlap)}"
            )

        occupied_token_indexes.update(
            current_token_indexes
        )

        bio_labels[token_start] = (
            BIO_LABEL_TO_ID["B-ASP"]
        )

        for token_index in range(
            token_start + 1,
            token_end,
        ):
            bio_labels[token_index] = (
                BIO_LABEL_TO_ID["I-ASP"]
            )

        sentiment_id = (
            SENTIMENT_LABEL_TO_ID[
                aspect["sentiment"]
            ]
        )

        aspect_token_starts.append(
            token_start
        )

        aspect_token_ends.append(
            token_end
        )

        aspect_sentiment_ids.append(
            sentiment_id
        )

        aspect_terms.append(
            aspect["term"]
        )

        aspect_character_starts.append(
            aspect["start"]
        )

        aspect_character_ends.append(
            aspect["end"]
        )

    output_record = {
        "id": record_id,
        "source_id": str(
            record.get(
                "source_id",
                record_id,
            )
        ),
        "domain": str(
            record.get(
                "domain",
                "laptop",
            )
        ),
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bio_labels": bio_labels,
        "aspect_token_starts": (
            aspect_token_starts
        ),
        "aspect_token_ends": (
            aspect_token_ends
        ),
        "aspect_sentiment_ids": (
            aspect_sentiment_ids
        ),
        "aspect_terms": aspect_terms,
        "aspect_character_starts": (
            aspect_character_starts
        ),
        "aspect_character_ends": (
            aspect_character_ends
        ),
    }

    if "token_type_ids" in encoding:
        output_record["token_type_ids"] = list(
            encoding["token_type_ids"]
        )

    return (
        output_record,
        correction_records,
    )


# ============================================================
# SPLIT PROCESSING
# ============================================================

def process_split(
    split_name,
    input_file,
    tokenizer,
    maximum_length,
):
    records = load_jsonl(input_file)

    processed_records = []
    all_corrections = []

    sentiment_distribution = Counter()
    bio_distribution = Counter()

    sentence_count_with_aspects = 0
    maximum_observed_length = 0
    total_aspects = 0
    largest_token_id = -1

    for record in records:
        (
            processed_record,
            corrections,
        ) = tokenize_record(
            record=record,
            tokenizer=tokenizer,
            maximum_length=maximum_length,
            split_name=split_name,
        )

        processed_records.append(
            processed_record
        )

        all_corrections.extend(
            corrections
        )

        sequence_length = len(
            processed_record["input_ids"]
        )

        maximum_observed_length = max(
            maximum_observed_length,
            sequence_length,
        )

        if processed_record["input_ids"]:
            largest_token_id = max(
                largest_token_id,
                max(
                    processed_record[
                        "input_ids"
                    ]
                ),
            )

        aspect_count = len(
            processed_record[
                "aspect_sentiment_ids"
            ]
        )

        total_aspects += aspect_count

        if aspect_count > 0:
            sentence_count_with_aspects += 1

        for sentiment_id in (
            processed_record[
                "aspect_sentiment_ids"
            ]
        ):
            sentiment_distribution[
                SENTIMENT_ID_TO_LABEL[
                    sentiment_id
                ]
            ] += 1

        for bio_id in (
            processed_record[
                "bio_labels"
            ]
        ):
            if bio_id == IGNORE_INDEX:
                continue

            bio_distribution[
                BIO_ID_TO_LABEL[bio_id]
            ] += 1

    sentence_count = len(
        processed_records
    )

    statistics = {
        "split": split_name,
        "sentences": sentence_count,
        "sentences_with_aspects": (
            sentence_count_with_aspects
        ),
        "sentences_without_aspects": (
            sentence_count
            - sentence_count_with_aspects
        ),
        "total_aspects": total_aspects,
        "maximum_sequence_length": (
            maximum_observed_length
        ),
        "configured_maximum_length": (
            maximum_length
        ),
        "largest_token_id": (
            largest_token_id
        ),
        "annotation_corrections": (
            all_corrections
        ),
        "annotation_correction_count": (
            len(all_corrections)
        ),
        "sentiment_distribution": dict(
            sorted(
                sentiment_distribution.items()
            )
        ),
        "bio_distribution": dict(
            sorted(
                bio_distribution.items()
            )
        ),
    }

    return (
        processed_records,
        statistics,
    )


# ============================================================
# OUTPUT BACKUP
# ============================================================

def generate_backup_path(
    output_directory,
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    base_backup_path = (
        output_directory.parent
        / (
            f"{output_directory.name}"
            f"_backup_{timestamp}"
        )
    )

    backup_path = base_backup_path
    counter = 1

    while backup_path.exists():
        backup_path = (
            output_directory.parent
            / (
                f"{base_backup_path.name}"
                f"_{counter}"
            )
        )

        counter += 1

    return backup_path


# ============================================================
# DATASET VERIFICATION
# ============================================================

def verify_saved_dataset(
    output_directory,
    expected_statistics,
    maximum_length,
):
    reloaded_dataset = load_from_disk(
        str(output_directory)
    )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        reloaded_dataset.keys()
    )

    if actual_splits != expected_splits:
        raise RuntimeError(
            "Reloaded dataset contains unexpected "
            "splits.\n"
            f"Expected: {sorted(expected_splits)}\n"
            f"Actual: {sorted(actual_splits)}"
        )

    for split_name in sorted(
        expected_splits
    ):
        split_data = reloaded_dataset[
            split_name
        ]

        expected_sentences = (
            expected_statistics[
                split_name
            ]["sentences"]
        )

        if (
            len(split_data)
            != expected_sentences
        ):
            raise RuntimeError(
                "Reloaded sentence count mismatch "
                f"for split {split_name}."
            )

        for record_index, record in enumerate(
            split_data
        ):
            input_length = len(
                record["input_ids"]
            )

            if input_length > maximum_length:
                raise RuntimeError(
                    "Reloaded record exceeds maximum "
                    "length.\n"
                    f"Split: {split_name}\n"
                    f"Record index: {record_index}\n"
                    f"Length: {input_length}\n"
                    f"Maximum: {maximum_length}"
                )

            expected_length = input_length

            sequence_fields = [
                "attention_mask",
                "bio_labels",
            ]

            if "token_type_ids" in record:
                sequence_fields.append(
                    "token_type_ids"
                )

            for field in sequence_fields:
                if (
                    len(record[field])
                    != expected_length
                ):
                    raise RuntimeError(
                        "Sequence-field length mismatch.\n"
                        f"Split: {split_name}\n"
                        f"Record: {record.get('id')}\n"
                        f"Field: {field}"
                    )

            aspect_lengths = [
                len(
                    record[
                        "aspect_token_starts"
                    ]
                ),
                len(
                    record[
                        "aspect_token_ends"
                    ]
                ),
                len(
                    record[
                        "aspect_sentiment_ids"
                    ]
                ),
                len(
                    record[
                        "aspect_terms"
                    ]
                ),
                len(
                    record[
                        "aspect_character_starts"
                    ]
                ),
                len(
                    record[
                        "aspect_character_ends"
                    ]
                ),
            ]

            if len(set(aspect_lengths)) != 1:
                raise RuntimeError(
                    "Reloaded aspect arrays have "
                    "inconsistent lengths.\n"
                    f"Split: {split_name}\n"
                    f"Record: {record.get('id')}"
                )

            for start, end in zip(
                record[
                    "aspect_token_starts"
                ],
                record[
                    "aspect_token_ends"
                ],
            ):
                if not (
                    0
                    <= start
                    < end
                    <= input_length
                ):
                    raise RuntimeError(
                        "Invalid aspect token span "
                        "after reload.\n"
                        f"Split: {split_name}\n"
                        f"Record: {record.get('id')}\n"
                        f"Span: ({start}, {end})"
                    )

    return reloaded_dataset


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("XLM-ROBERTA BIO DATASET PREPARATION")
    print("=" * 70)

    validate_required_files()

    (
        maximum_length,
        token_length_audit,
    ) = load_verified_maximum_length()

    output_directory = (
        PROJECT_ROOT
        / "data"
        / "tokenized"
        / f"xlmr_laptop_max{maximum_length}"
    )

    tokenization_audit_path = (
        PROCESSED_DATA_DIRECTORY
        / (
            "xlmr_tokenization_audit_"
            f"max{maximum_length}.json"
        )
    )

    print("\nLoading tokenizer from local storage...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIRECTORY,
        local_files_only=True,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for "
            "offset-based BIO alignment."
        )

    if (
        "XLMRoberta"
        not in tokenizer.__class__.__name__
    ):
        raise RuntimeError(
            "Unexpected tokenizer class:\n"
            f"{tokenizer.__class__.__name__}"
        )

    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "The tokenizer does not define a "
            "padding token ID."
        )

    print(
        "Tokenizer class:",
        tokenizer.__class__.__name__,
    )

    print(
        "Tokenizer vocabulary size:",
        len(tokenizer),
    )

    print(
        "Configured maximum length:",
        maximum_length,
    )

    print("\nProcessing official Laptop splits...")

    processed_splits = {}
    split_statistics = {}

    for split_name, input_file in (
        SPLIT_FILES.items()
    ):
        print("\n" + "-" * 60)
        print(
            f"Processing split: {split_name}"
        )
        print("-" * 60)

        (
            processed_records,
            statistics,
        ) = process_split(
            split_name=split_name,
            input_file=input_file,
            tokenizer=tokenizer,
            maximum_length=maximum_length,
        )

        processed_splits[split_name] = (
            Dataset.from_list(
                processed_records
            )
        )

        split_statistics[split_name] = (
            statistics
        )

        print(
            "Sentences:",
            statistics["sentences"],
        )

        print(
            "Sentences with aspects:",
            statistics[
                "sentences_with_aspects"
            ],
        )

        print(
            "Sentences without aspects:",
            statistics[
                "sentences_without_aspects"
            ],
        )

        print(
            "Total aspects:",
            statistics["total_aspects"],
        )

        print(
            "Maximum sequence length:",
            statistics[
                "maximum_sequence_length"
            ],
        )

        print(
            "Largest token ID:",
            statistics[
                "largest_token_id"
            ],
        )

        print(
            "Annotation corrections:",
            statistics[
                "annotation_correction_count"
            ],
        )

        print(
            "Sentiment distribution:",
            statistics[
                "sentiment_distribution"
            ],
        )

        print(
            "BIO distribution:",
            statistics[
                "bio_distribution"
            ],
        )

    expected_counts = {
        "train": {
            "sentences": 2587,
            "aspects": 2004,
        },
        "validation": {
            "sentences": 457,
            "aspects": 353,
        },
        "test": {
            "sentences": 800,
            "aspects": 654,
        },
    }

    for split_name, expected in (
        expected_counts.items()
    ):
        actual = split_statistics[
            split_name
        ]

        if (
            actual["sentences"]
            != expected["sentences"]
        ):
            raise RuntimeError(
                f"Unexpected sentence count for "
                f"{split_name}.\n"
                f"Expected: {expected['sentences']}\n"
                f"Actual: {actual['sentences']}"
            )

        if (
            actual["total_aspects"]
            != expected["aspects"]
        ):
            raise RuntimeError(
                f"Unexpected aspect count for "
                f"{split_name}.\n"
                f"Expected: {expected['aspects']}\n"
                f"Actual: {actual['total_aspects']}"
            )

    total_corrections = sum(
        statistics[
            "annotation_correction_count"
        ]
        for statistics
        in split_statistics.values()
    )

    if total_corrections != 1:
        raise RuntimeError(
            "Unexpected number of annotation "
            f"corrections: {total_corrections}. "
            "Expected exactly 1."
        )

    dataset = DatasetDict(
        processed_splits
    )

    backup_directory = None

    if output_directory.exists():
        backup_directory = (
            generate_backup_path(
                output_directory
            )
        )

        print(
            "\nExisting output moved safely to:"
        )

        print(backup_directory)

        shutil.move(
            str(output_directory),
            str(backup_directory),
        )

    output_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSaving tokenized dataset...")

    dataset.save_to_disk(
        str(output_directory)
    )

    print("\nReloading dataset from disk...")

    reloaded_dataset = verify_saved_dataset(
        output_directory=output_directory,
        expected_statistics=(
            split_statistics
        ),
        maximum_length=maximum_length,
    )

    print("Offline dataset reload: PASSED")

    largest_dataset_token_id = max(
        statistics["largest_token_id"]
        for statistics
        in split_statistics.values()
    )

    if (
        largest_dataset_token_id
        >= len(tokenizer)
    ):
        raise RuntimeError(
            "Dataset contains a token ID outside "
            "the tokenizer vocabulary.\n"
            f"Largest token ID: "
            f"{largest_dataset_token_id}\n"
            f"Tokenizer size: {len(tokenizer)}"
        )

    tokenization_audit = {
        "model_family": (
            "xlm-roberta-base"
        ),
        "tokenizer_directory": str(
            TOKENIZER_DIRECTORY
        ),
        "tokenizer_class": (
            tokenizer.__class__.__name__
        ),
        "tokenizer_vocabulary_size": len(
            tokenizer
        ),
        "maximum_length": (
            maximum_length
        ),
        "maximum_length_source": str(
            TOKEN_LENGTH_AUDIT_PATH
        ),
        "output_directory": str(
            output_directory
        ),
        "bio_label_to_id": (
            BIO_LABEL_TO_ID
        ),
        "sentiment_label_to_id": (
            SENTIMENT_LABEL_TO_ID
        ),
        "ignore_index": IGNORE_INDEX,
        "padding_strategy": (
            "Dynamic padding will be applied "
            "by the training collator."
        ),
        "normalization_policy": (
            "Trim only leading and trailing "
            "Unicode whitespace from aspect "
            "boundaries during alignment."
        ),
        "total_annotation_corrections": (
            total_corrections
        ),
        "largest_token_id": (
            largest_dataset_token_id
        ),
        "splits": split_statistics,
        "offline_reload_verified": True,
        "reloaded_split_sizes": {
            split_name: len(
                reloaded_dataset[split_name]
            )
            for split_name
            in reloaded_dataset.keys()
        },
        "source_token_length_audit": (
            token_length_audit
        ),
    }

    save_json(
        tokenization_audit,
        tokenization_audit_path,
    )

    print("\nTokenized dataset saved to:")
    print(output_directory)

    print("\nTokenization audit saved to:")
    print(tokenization_audit_path)

    if backup_directory is not None:
        print("\nPrevious output backup:")
        print(backup_directory)

    print("\n" + "=" * 70)
    print(
        "XLM-ROBERTA TOKENIZED DATASET "
        "PREPARATION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
