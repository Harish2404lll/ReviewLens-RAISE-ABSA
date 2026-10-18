import json
import math
import statistics
from pathlib import Path
from transformers import AutoTokenizer

# ============================================================
# PROJECT PATHS & CONFIGURATION
# ============================================================

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
    / "xlm-roberta-base-tokenizer"
)

AUDIT_OUTPUT = (
    DATA_DIRECTORY
    / "xlmr_token_length_audit.json"
)

SPLIT_FILES = {
    "train": DATA_DIRECTORY / "laptop_train.jsonl",
    "validation": (
        DATA_DIRECTORY / "laptop_validation.jsonl"
    ),
    "test": DATA_DIRECTORY / "laptop_test.jsonl",
}

CANDIDATE_LENGTHS = [
    64,
    96,
    128,
    160,
    192,
    256,
    384,
    512,
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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


def percentile(values, percentage):
    sorted_values = sorted(values)
    index = (
        math.ceil(percentage * len(sorted_values)) - 1
    )
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def normalize_aspect_boundary(text, aspect):
    """
    Trim leading and trailing Unicode whitespace from an
    annotated aspect span without modifying the JSONL files.
    """
    original_start = int(aspect["start"])
    original_end = int(aspect["end"])
    original_term = str(aspect["term"])
    sentiment = str(aspect["sentiment"]).strip().lower()

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
            "Aspect became empty after Unicode "
            f"whitespace normalization: {aspect}"
        )

    corrected_term = text[corrected_start:corrected_end]

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

    return corrected_aspect, was_corrected


def aspect_is_covered(aspect, offsets):
    """
    Determine whether the retained tokens fully cover an
    annotated character span.
    """
    start = aspect["start"]
    end = aspect["end"]

    overlapping_offsets = [
        (int(token_start), int(token_end))
        for token_start, token_end in offsets
        if token_end > token_start
        and token_start < end
        and token_end > start
    ]

    if not overlapping_offsets:
        return False

    covered_start = min(
        token_start
        for token_start, _ in overlapping_offsets
    )
    covered_end = max(
        token_end
        for _, token_end in overlapping_offsets
    )

    return (
        covered_start <= start
        and covered_end >= end
    )


# ============================================================
# SPLIT AUDITING
# ============================================================

def audit_split(split_name, records, tokenizer):
    token_lengths = []
    longest_records = []
    full_alignment_failures = []
    boundary_corrections = 0

    threshold_results = {
        str(length): {
            "truncated_sentences": 0,
            "lost_aspects": 0,
            "affected_sentence_ids": [],
        }
        for length in CANDIDATE_LENGTHS
    }

    for record in records:
        text = record["text"]
        normalized_aspects = []

        for aspect in record["aspects"]:
            norm_aspect, was_corrected = (
                normalize_aspect_boundary(text, aspect)
            )
            normalized_aspects.append(norm_aspect)
            if was_corrected:
                boundary_corrections += 1

        full_encoding = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )

        full_length = len(full_encoding["input_ids"])
        token_lengths.append(full_length)

        longest_records.append(
            {
                "id": record["id"],
                "source_id": record["source_id"],
                "tokens": full_length,
                "characters": len(text),
                "text": text,
            }
        )

        for aspect in normalized_aspects:
            if not aspect_is_covered(
                aspect,
                full_encoding["offset_mapping"],
            ):
                full_alignment_failures.append(
                    {
                        "id": record["id"],
                        "aspect": aspect,
                        "text": text,
                    }
                )

        for maximum_length in CANDIDATE_LENGTHS:
            truncated_encoding = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=maximum_length,
                return_offsets_mapping=True,
            )

            if full_length > maximum_length:
                threshold_results[
                    str(maximum_length)
                ]["truncated_sentences"] += 1

            lost_aspects = 0
            for aspect in normalized_aspects:
                if not aspect_is_covered(
                    aspect,
                    truncated_encoding["offset_mapping"],
                ):
                    lost_aspects += 1

            if lost_aspects > 0:
                threshold_results[
                    str(maximum_length)
                ]["lost_aspects"] += lost_aspects
                threshold_results[
                    str(maximum_length)
                ]["affected_sentence_ids"].append(
                    record["id"]
                )

    longest_records.sort(
        key=lambda item: item["tokens"],
        reverse=True,
    )

    return {
        "split": split_name,
        "sentences": len(records),
        "aspects": sum(
            len(record["aspects"]) for record in records
        ),
        "boundary_corrections": boundary_corrections,
        "token_length_statistics": {
            "minimum": min(token_lengths),
            "mean": round(
                statistics.mean(token_lengths),
                2,
            ),
            "median": statistics.median(token_lengths),
            "p95": percentile(token_lengths, 0.95),
            "p99": percentile(token_lengths, 0.99),
            "maximum": max(token_lengths),
        },
        "full_alignment_failures": full_alignment_failures,
        "candidate_lengths": threshold_results,
        "five_longest_sentences": longest_records[:5],
    }


def recommend_length(audit):
    """
    Select the smallest candidate length that truncates 0
    sentences and loses 0 annotated aspects across all splits.
    """
    for candidate in CANDIDATE_LENGTHS:
        candidate_key = str(candidate)
        safe_for_all_splits = all(
            split_audit["candidate_lengths"][
                candidate_key
            ]["truncated_sentences"] == 0
            and split_audit["candidate_lengths"][
                candidate_key
            ]["lost_aspects"] == 0
            for split_audit in audit["splits"].values()
        )
        if safe_for_all_splits:
            return candidate
    return None


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("XLM-ROBERTA TOKEN-LENGTH AND SPAN AUDIT")
    print("=" * 60)

    if not TOKENIZER_DIRECTORY.exists():
        raise FileNotFoundError(
            "XLM-RoBERTa tokenizer directory not found:\n"
            f"{TOKENIZER_DIRECTORY}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIRECTORY,
        use_fast=True,
        local_files_only=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "Offset mapping requires a fast tokenizer."
        )

    if "XLMRoberta" not in tokenizer.__class__.__name__:
        raise RuntimeError(
            "Unexpected tokenizer class:\n"
            f"{tokenizer.__class__.__name__}"
        )

    audit = {
        "model_family": "xlm-roberta-base",
        "tokenizer": str(TOKENIZER_DIRECTORY),
        "tokenizer_class": tokenizer.__class__.__name__,
        "candidate_lengths": CANDIDATE_LENGTHS,
        "splits": {},
    }

    for split_name, file_path in SPLIT_FILES.items():
        print(f"\nAuditing split: {split_name}")
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing split file:\n{file_path}"
            )

        records = load_jsonl(file_path)
        split_audit = audit_split(
            split_name,
            records,
            tokenizer,
        )
        audit["splits"][split_name] = split_audit

        stats = split_audit["token_length_statistics"]
        print("Sentences:", split_audit["sentences"])
        print("Aspects:", split_audit["aspects"])
        print(
            "Boundary corrections:",
            split_audit["boundary_corrections"],
        )
        print("Mean tokens:", stats["mean"])
        print("P95 tokens:", stats["p95"])
        print("P99 tokens:", stats["p99"])
        print("Maximum tokens:", stats["maximum"])
        print(
            "Full alignment failures:",
            len(split_audit["full_alignment_failures"]),
        )

        for candidate in CANDIDATE_LENGTHS:
            result = split_audit["candidate_lengths"][
                str(candidate)
            ]
            print(
                f"Length {candidate}: "
                f"truncated_sentences="
                f"{result['truncated_sentences']}, "
                f"lost_aspects="
                f"{result['lost_aspects']}"
            )

    recommended_length = recommend_length(audit)
    audit["recommended_max_length"] = recommended_length

    all_alignment_failures = sum(
        len(split_audit["full_alignment_failures"])
        for split_audit in audit["splits"].values()
    )

    # Save both keys for compatibility across scripts
    audit["total_alignment_failures"] = (
        all_alignment_failures
    )
    audit["total_full_alignment_failures"] = (
        all_alignment_failures
    )

    if all_alignment_failures != 0:
        raise ValueError(
            f"Found {all_alignment_failures} aspects "
            "that could not be aligned."
        )

    if recommended_length is None:
        raise ValueError(
            "No candidate length preserved all sentences "
            "and aspects."
        )

    print(
        "\nRecommended maximum length:",
        recommended_length,
    )

    with AUDIT_OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(
            audit,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    print("\nAudit saved to:")
    print(AUDIT_OUTPUT)
    print("\nXLM-ROBERTA TOKEN-LENGTH AUDIT COMPLETED")


if __name__ == "__main__":
    main()