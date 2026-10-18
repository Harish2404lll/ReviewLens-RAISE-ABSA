import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET

from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "official"
    / "semeval2014_laptop"
)

TRAIN_XML = (
    OFFICIAL_DIRECTORY
    / "Laptop_Train_v2.xml"
)

PHASE_B_XML = (
    OFFICIAL_DIRECTORY
    / "Laptops_Test_Data_PhaseB.xml"
)

GOLD_TEST_XML = (
    OFFICIAL_DIRECTORY
    / "Laptops_Test_Gold.xml"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "official_laptop"
)

VALID_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
    "conflict",
}

VALIDATION_RATIO = 0.15
RANDOM_SEED = 42


# ============================================================
# GENERAL UTILITIES
# ============================================================

def require_file(file_path):
    """
    Confirm that a required source file exists and is not empty.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file does not exist:\n{file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Expected a file but found something else:\n"
            f"{file_path}"
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"Required file is empty:\n{file_path}"
        )


def calculate_sha256(file_path):
    """
    Calculate a SHA-256 checksum for reproducibility.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(
            lambda: file.read(65536),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


def normalize_sentence(text):
    """
    Normalize text only for duplicate detection.

    The original sentence is never modified in the output.
    """

    return re.sub(
        r"\s+",
        " ",
        text.strip().casefold(),
    )


# ============================================================
# XML PARSING
# ============================================================

def parse_xml(
    file_path,
    split_prefix,
    require_polarity=True,
):
    """
    Parse a SemEval-2014 Laptop XML file.

    Important:
    - Sentences without aspect terms are preserved.
    - Aspect offsets are validated against the sentence.
    - Unknown or missing labels cause an error.
    - Malformed annotations are never silently skipped.
    """

    try:
        tree = ET.parse(file_path)
    except ET.ParseError as error:
        raise ValueError(
            f"Invalid XML file:\n{file_path}\n{error}"
        ) from error

    root = tree.getroot()

    records = []
    sentence_ids = set()

    for row_number, sentence_element in enumerate(
        root.iter("sentence"),
        start=1,
    ):
        source_id = sentence_element.get("id")

        if source_id is None or not source_id.strip():
            raise ValueError(
                f"Missing sentence ID in {file_path}, "
                f"row {row_number}"
            )

        source_id = source_id.strip()

        if source_id in sentence_ids:
            raise ValueError(
                f"Duplicate sentence ID '{source_id}' "
                f"in {file_path}"
            )

        sentence_ids.add(source_id)

        text_element = sentence_element.find("text")

        if text_element is None:
            raise ValueError(
                f"Missing text element for sentence "
                f"{source_id}"
            )

        if text_element.text is None:
            raise ValueError(
                f"Empty sentence text for ID {source_id}"
            )

        text = text_element.text
        aspects = []

        for aspect_number, aspect_element in enumerate(
            sentence_element.iter("aspectTerm"),
            start=1,
        ):
            term = aspect_element.get("term")
            polarity = aspect_element.get("polarity")
            start_value = aspect_element.get("from")
            end_value = aspect_element.get("to")

            if term is None or not term:
                raise ValueError(
                    f"Missing aspect term in sentence "
                    f"{source_id}"
                )

            if start_value is None or end_value is None:
                raise ValueError(
                    f"Missing offsets for aspect '{term}' "
                    f"in sentence {source_id}"
                )

            if require_polarity:
                if polarity is None:
                    raise ValueError(
                        f"Missing polarity for aspect '{term}' "
                        f"in sentence {source_id}"
                    )

                polarity = polarity.strip().lower()

                if polarity not in VALID_SENTIMENTS:
                    raise ValueError(
                        f"Unknown polarity '{polarity}' "
                        f"for aspect '{term}' "
                        f"in sentence {source_id}"
                    )

            try:
                start = int(start_value)
                end = int(end_value)
            except ValueError as error:
                raise ValueError(
                    f"Non-integer offsets for aspect '{term}' "
                    f"in sentence {source_id}: "
                    f"from={start_value}, to={end_value}"
                ) from error

            if start < 0:
                raise ValueError(
                    f"Negative start offset for aspect "
                    f"'{term}' in sentence {source_id}"
                )

            if end <= start:
                raise ValueError(
                    f"Invalid offset order for aspect "
                    f"'{term}' in sentence {source_id}: "
                    f"start={start}, end={end}"
                )

            if end > len(text):
                raise ValueError(
                    f"Aspect end offset exceeds sentence "
                    f"length in sentence {source_id}: "
                    f"end={end}, text_length={len(text)}"
                )

            extracted_term = text[start:end]

            if extracted_term != term:
                raise ValueError(
                    "\nAspect offset mismatch:"
                    f"\nFile: {file_path}"
                    f"\nSentence ID: {source_id}"
                    f"\nAspect number: {aspect_number}"
                    f"\nExpected term: {term!r}"
                    f"\nExtracted term: {extracted_term!r}"
                    f"\nOffsets: [{start}, {end}]"
                    f"\nSentence: {text!r}"
                )

            aspect_record = {
                "term": term,
                "start": start,
                "end": end,
            }

            if require_polarity:
                aspect_record["sentiment"] = polarity

            aspects.append(aspect_record)

        record = {
            "id": f"{split_prefix}_{source_id}",
            "source_id": source_id,
            "domain": "laptop",
            "text": text,
            "aspects": aspects,
        }

        records.append(record)

    if not records:
        raise ValueError(
            f"No sentences were found in:\n{file_path}"
        )

    return records


# ============================================================
# PHASE B AND GOLD TEST VALIDATION
# ============================================================

def aspect_signature(record):
    """
    Create a comparable aspect signature without polarity.
    """

    signature = []

    for aspect in record["aspects"]:
        signature.append(
            (
                aspect["term"],
                aspect["start"],
                aspect["end"],
            )
        )

    return sorted(signature)


def validate_phase_b_against_gold(
    phase_b_records,
    gold_records,
):
    """
    Verify that Phase B and the gold test file contain the same
    sentences, aspect terms and character offsets.
    """

    phase_b_by_id = {
        record["source_id"]: record
        for record in phase_b_records
    }

    gold_by_id = {
        record["source_id"]: record
        for record in gold_records
    }

    phase_b_ids = set(phase_b_by_id)
    gold_ids = set(gold_by_id)

    if phase_b_ids != gold_ids:
        missing_in_gold = sorted(
            phase_b_ids - gold_ids
        )

        missing_in_phase_b = sorted(
            gold_ids - phase_b_ids
        )

        raise ValueError(
            "Phase B and gold test sentence IDs do not match."
            f"\nMissing in gold: {missing_in_gold[:10]}"
            f"\nMissing in Phase B: "
            f"{missing_in_phase_b[:10]}"
        )

    for source_id, phase_b_record in (
        phase_b_by_id.items()
    ):
        gold_record = gold_by_id[source_id]

        if phase_b_record["text"] != gold_record["text"]:
            raise ValueError(
                f"Sentence text mismatch between Phase B "
                f"and gold test for ID {source_id}"
            )

        phase_b_aspects = aspect_signature(
            phase_b_record
        )

        gold_aspects = aspect_signature(
            gold_record
        )

        if phase_b_aspects != gold_aspects:
            raise ValueError(
                f"Aspect annotation mismatch between "
                f"Phase B and gold test for sentence "
                f"{source_id}"
            )


# ============================================================
# TRAIN-TEST DECONTAMINATION
# ============================================================

def remove_test_overlap(
    training_records,
    test_records,
):
    """
    Remove training records whose normalized sentence text
    appears in the official test set.

    Removed records are saved separately for transparency.
    """

    test_texts = {
        normalize_sentence(record["text"])
        for record in test_records
    }

    cleaned_training_records = []
    removed_records = []

    for record in training_records:
        normalized_text = normalize_sentence(
            record["text"]
        )

        if normalized_text in test_texts:
            removed_records.append(record)
        else:
            cleaned_training_records.append(record)

    return cleaned_training_records, removed_records


# ============================================================
# GROUPED TRAIN-VALIDATION SPLIT
# ============================================================

def create_grouped_split(records):
    """
    Create a deterministic train-validation split.

    Duplicate normalized sentences are kept together so that
    the same sentence cannot occur in both splits.
    """

    groups = defaultdict(list)

    for record in records:
        normalized_text = normalize_sentence(
            record["text"]
        )

        groups[normalized_text].append(record)

    group_keys = sorted(groups.keys())

    random_generator = random.Random(
        RANDOM_SEED
    )

    random_generator.shuffle(group_keys)

    target_validation_size = round(
        len(records) * VALIDATION_RATIO
    )

    validation_keys = set()
    validation_size = 0

    for key in group_keys:
        if validation_size >= target_validation_size:
            break

        validation_keys.add(key)
        validation_size += len(groups[key])

    train_records = []
    validation_records = []

    for record in records:
        normalized_text = normalize_sentence(
            record["text"]
        )

        if normalized_text in validation_keys:
            validation_records.append(record)
        else:
            train_records.append(record)

    if not train_records:
        raise ValueError(
            "The training split is empty."
        )

    if not validation_records:
        raise ValueError(
            "The validation split is empty."
        )

    return train_records, validation_records


# ============================================================
# DATASET AUDITING
# ============================================================

def summarize_records(records):
    """
    Produce sentence, aspect, sentiment and duplicate counts.
    """

    sentiment_distribution = Counter()
    normalized_text_counts = Counter()

    sentences_with_aspects = 0
    total_aspects = 0

    for record in records:
        normalized_text = normalize_sentence(
            record["text"]
        )

        normalized_text_counts[normalized_text] += 1

        if record["aspects"]:
            sentences_with_aspects += 1

        for aspect in record["aspects"]:
            total_aspects += 1

            if "sentiment" in aspect:
                sentiment_distribution[
                    aspect["sentiment"]
                ] += 1

    duplicate_sentence_groups = sum(
        1
        for count in normalized_text_counts.values()
        if count > 1
    )

    duplicate_extra_rows = sum(
        count - 1
        for count in normalized_text_counts.values()
        if count > 1
    )

    return {
        "sentences": len(records),
        "sentences_with_aspects": (
            sentences_with_aspects
        ),
        "sentences_without_aspects": (
            len(records) - sentences_with_aspects
        ),
        "total_aspects": total_aspects,
        "sentiment_distribution": dict(
            sorted(sentiment_distribution.items())
        ),
        "duplicate_sentence_groups": (
            duplicate_sentence_groups
        ),
        "duplicate_extra_rows": (
            duplicate_extra_rows
        ),
    }


def normalized_text_set(records):
    return {
        normalize_sentence(record["text"])
        for record in records
    }


def check_split_leakage(
    train_records,
    validation_records,
    test_records,
):
    """
    Confirm that normalized sentence text does not appear across
    any pair of splits.
    """

    train_texts = normalized_text_set(
        train_records
    )

    validation_texts = normalized_text_set(
        validation_records
    )

    test_texts = normalized_text_set(
        test_records
    )

    leakage_report = {
        "train_validation_overlap": len(
            train_texts & validation_texts
        ),
        "train_test_overlap": len(
            train_texts & test_texts
        ),
        "validation_test_overlap": len(
            validation_texts & test_texts
        ),
    }

    if any(
        value != 0
        for value in leakage_report.values()
    ):
        raise ValueError(
            "Cross-split duplicate leakage detected: "
            f"{leakage_report}"
        )

    return leakage_report


def validate_class_coverage(
    train_records,
    validation_records,
    test_records,
):
    """
    Verify that each expected sentiment appears in each split.
    """

    split_records = {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }

    coverage_report = {}

    for split_name, records in split_records.items():
        present_sentiments = {
            aspect["sentiment"]
            for record in records
            for aspect in record["aspects"]
            if "sentiment" in aspect
        }

        missing_sentiments = (
            VALID_SENTIMENTS - present_sentiments
        )

        coverage_report[split_name] = {
            "present": sorted(present_sentiments),
            "missing": sorted(missing_sentiments),
        }

        if missing_sentiments:
            raise ValueError(
                f"Split '{split_name}' is missing "
                f"sentiment classes: "
                f"{sorted(missing_sentiments)}"
            )

    return coverage_report


# ============================================================
# OUTPUT FUNCTIONS
# ============================================================

def write_jsonl(records, output_file):
    """
    Write records using UTF-8 JSON Lines format.
    """

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_summary(split_name, summary):
    print("-" * 60)
    print(f"Split: {split_name}")
    print("-" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("OFFICIAL SEMEVAL-2014 LAPTOP DATA PREPARATION")
    print("=" * 60)

    print("\nChecking required source files...")

    require_file(TRAIN_XML)
    require_file(PHASE_B_XML)
    require_file(GOLD_TEST_XML)

    print("All required source files are available.")

    print("\nParsing official training data...")

    complete_training_records = parse_xml(
        file_path=TRAIN_XML,
        split_prefix="laptop_train",
        require_polarity=True,
    )

    print("Parsing unlabeled Phase B test data...")

    phase_b_records = parse_xml(
        file_path=PHASE_B_XML,
        split_prefix="laptop_phase_b",
        require_polarity=False,
    )

    print("Parsing official gold test data...")

    test_records = parse_xml(
        file_path=GOLD_TEST_XML,
        split_prefix="laptop_test",
        require_polarity=True,
    )

    print(
        "Cross-validating Phase B against "
        "the gold test file..."
    )

    validate_phase_b_against_gold(
        phase_b_records=phase_b_records,
        gold_records=test_records,
    )

    print("Phase B and gold test alignment: PASSED")

    original_training_summary = summarize_records(
        complete_training_records
    )

    print("\nRemoving train-test duplicate sentences...")

    cleaned_training_pool, removed_overlap_records = (
        remove_test_overlap(
            training_records=complete_training_records,
            test_records=test_records,
        )
    )

    print(
        "Removed train-test overlap records:",
        len(removed_overlap_records),
    )

    for record in removed_overlap_records:
        print(
            f"Removed source ID: "
            f"{record['source_id']}"
        )

        print(
            f"Removed sentence: "
            f"{record['text']}"
        )

    if len(removed_overlap_records) == 0:
        print(
            "No train-test duplicate sentences "
            "were detected."
        )

    cleaned_training_summary = summarize_records(
        cleaned_training_pool
    )

    print(
        "\nCreating grouped train-validation split..."
    )

    train_records, validation_records = (
        create_grouped_split(
            cleaned_training_pool
        )
    )

    train_summary = summarize_records(
        train_records
    )

    validation_summary = summarize_records(
        validation_records
    )

    test_summary = summarize_records(
        test_records
    )

    print(
        "Checking sentiment-class coverage..."
    )

    class_coverage_report = validate_class_coverage(
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
    )

    print(
        "Checking cross-split duplicate leakage..."
    )

    leakage_report = check_split_leakage(
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_output = (
        OUTPUT_DIRECTORY
        / "laptop_train.jsonl"
    )

    validation_output = (
        OUTPUT_DIRECTORY
        / "laptop_validation.jsonl"
    )

    test_output = (
        OUTPUT_DIRECTORY
        / "laptop_test.jsonl"
    )

    removed_output = (
        OUTPUT_DIRECTORY
        / "removed_train_test_overlap.jsonl"
    )

    audit_output = (
        OUTPUT_DIRECTORY
        / "laptop_dataset_audit.json"
    )

    write_jsonl(
        train_records,
        train_output,
    )

    write_jsonl(
        validation_records,
        validation_output,
    )

    write_jsonl(
        test_records,
        test_output,
    )

    write_jsonl(
        removed_overlap_records,
        removed_output,
    )

    audit_report = {
        "dataset": "SemEval-2014 Task 4 Laptop",
        "domain": "laptop",
        "validation_ratio": VALIDATION_RATIO,
        "random_seed": RANDOM_SEED,
        "sentiment_classes": sorted(
            VALID_SENTIMENTS
        ),
        "phase_b_gold_validation": "passed",
        "source_files": {
            "train": {
                "path": str(TRAIN_XML),
                "size_bytes": TRAIN_XML.stat().st_size,
                "sha256": calculate_sha256(
                    TRAIN_XML
                ),
            },
            "phase_b": {
                "path": str(PHASE_B_XML),
                "size_bytes": (
                    PHASE_B_XML.stat().st_size
                ),
                "sha256": calculate_sha256(
                    PHASE_B_XML
                ),
            },
            "gold_test": {
                "path": str(GOLD_TEST_XML),
                "size_bytes": (
                    GOLD_TEST_XML.stat().st_size
                ),
                "sha256": calculate_sha256(
                    GOLD_TEST_XML
                ),
            },
        },
        "decontamination": {
            "method": (
                "normalized exact-text overlap removal"
            ),
            "original_training_pool": (
                original_training_summary
            ),
            "cleaned_training_pool": (
                cleaned_training_summary
            ),
            "removed_training_records": len(
                removed_overlap_records
            ),
            "removed_source_ids": [
                record["source_id"]
                for record in removed_overlap_records
            ],
            "removed_records_file": str(
                removed_output
            ),
        },
        "splits": {
            "train": train_summary,
            "validation": validation_summary,
            "test": test_summary,
        },
        "class_coverage": (
            class_coverage_report
        ),
        "cross_split_duplicate_check": (
            leakage_report
        ),
    }

    with audit_output.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            audit_report,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    print()
    print_summary(
        "TRAIN",
        train_summary,
    )

    print()
    print_summary(
        "VALIDATION",
        validation_summary,
    )

    print()
    print_summary(
        "TEST",
        test_summary,
    )

    print("\nClass coverage:")

    for split_name, coverage in (
        class_coverage_report.items()
    ):
        print(
            f"{split_name}: "
            f"present={coverage['present']}, "
            f"missing={coverage['missing']}"
        )

    print("\nCross-split duplicate check:")

    for key, value in leakage_report.items():
        print(f"{key}: {value}")

    print("\nCreated files:")
    print(train_output)
    print(validation_output)
    print(test_output)
    print(removed_output)
    print(audit_output)

    print()
    print("=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()