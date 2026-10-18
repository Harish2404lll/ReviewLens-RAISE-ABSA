import ast
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk


# =========================================================
# PROJECT PATHS
# =========================================================

# File location:
# RAISE-ABSA/src/data/prepare_source.py
#
# parents[2] points to:
# RAISE-ABSA/

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATASET_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "semeval2014_hf"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# =========================================================
# VALID LABELS
# =========================================================

VALID_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
    "conflict"
}


# =========================================================
# SENTENCE EXTRACTION
# =========================================================

def extract_sentence(value):
    """
    Convert the dataset input into one clean sentence.

    Supported formats:

    ["The battery is excellent."]
    "The battery is excellent."
    """

    if value is None:
        raise ValueError(
            "Sentence value is missing."
        )

    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError(
                "Sentence list is empty."
            )

        if len(value) != 1:
            raise ValueError(
                "Expected exactly one sentence, "
                f"but received {len(value)}: {value}"
            )

        sentence = str(value[0]).strip()

    else:
        sentence = str(value).strip()

    if not sentence:
        raise ValueError(
            "Sentence is empty."
        )

    return sentence


# =========================================================
# DICTIONARY VALUE HELPER
# =========================================================

def get_first_value(dictionary, keys):
    """
    Return the value from the first matching key.
    """

    for key in keys:
        if key in dictionary:
            return dictionary[key]

    return None


# =========================================================
# ANNOTATION-PAIR CHECK
# =========================================================

def is_annotation_pair(value):
    """
    Check whether value looks like:

    ["screen", "positive"]
    """

    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and value[0] is not None
        and isinstance(value[1], str)
        and value[1].strip().lower()
        in VALID_SENTIMENTS
    )


# =========================================================
# NULL ANNOTATION CHECK
# =========================================================

def is_empty_annotation(value):
    """
    Check whether an annotation value is empty.
    """

    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() in {
            "",
            "null",
            "none",
            "[]",
            "{}"
        }

    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0

    return False


# =========================================================
# ASPECT-PAIR PARSER
# =========================================================

def parse_aspect_pairs(value):
    """
    Convert different annotation formats into:

    [
        {
            "aspect": "screen",
            "sentiment": "positive"
        },
        {
            "aspect": "battery",
            "sentiment": "negative"
        }
    ]

    Only explicit aspect terms are processed.

    Aspect categories are ignored because a category may
    not appear directly in the original sentence.
    """

    if is_empty_annotation(value):
        return []

    # -----------------------------------------------------
    # Convert string representation to a Python object
    # -----------------------------------------------------

    if isinstance(value, str):
        text = value.strip()

        try:
            value = ast.literal_eval(text)

        except (ValueError, SyntaxError) as error:
            raise ValueError(
                f"Could not parse annotation: {text}"
            ) from error

    raw_pairs = []

    # -----------------------------------------------------
    # FORMAT 1: Dictionary
    # -----------------------------------------------------

    if isinstance(value, dict):

        # Example:
        #
        # {
        #   "aspect_term": [
        #       ["bar", "positive"],
        #       ["tables", "negative"]
        #   ],
        #   "aspect_category": [...]
        # }

        if "aspect_term" in value:
            aspect_term_data = value[
                "aspect_term"
            ]

            if is_empty_annotation(
                aspect_term_data
            ):
                return []

            # One direct pair:
            # ["screen", "positive"]
            if is_annotation_pair(
                aspect_term_data
            ):
                raw_pairs = [
                    aspect_term_data
                ]

            # Multiple pairs:
            # [
            #   ["screen", "positive"],
            #   ["battery", "negative"]
            # ]
            elif isinstance(
                aspect_term_data,
                (list, tuple)
            ):
                raw_pairs = list(
                    aspect_term_data
                )

            # Separate dictionary fields:
            # {
            #   "aspect_term": "screen",
            #   "polarity": "positive"
            # }
            else:
                sentiment = get_first_value(
                    value,
                    [
                        "polarity",
                        "sentiment",
                        "sentiment_polarity",
                        "aspect_polarity"
                    ]
                )

                if sentiment is None:
                    raise ValueError(
                        "Aspect term is present, but "
                        "sentiment is missing: "
                        f"{value}"
                    )

                raw_pairs = [
                    [
                        aspect_term_data,
                        sentiment
                    ]
                ]

        # Alternative dictionary structure:
        #
        # {
        #   "aspect": "screen",
        #   "sentiment": "positive"
        # }
        else:
            aspects = get_first_value(
                value,
                [
                    "aspect",
                    "term",
                    "target",
                    "aspect_terms"
                ]
            )

            sentiments = get_first_value(
                value,
                [
                    "polarity",
                    "sentiment",
                    "sentiments",
                    "sentiment_polarity",
                    "aspect_polarity"
                ]
            )

            # Category-only records cannot be used for
            # explicit aspect-span extraction.
            if aspects is None:
                return []

            if sentiments is None:
                raise ValueError(
                    "Aspect is present, but sentiment "
                    f"is missing: {value}"
                )

            if not isinstance(
                aspects,
                (list, tuple)
            ):
                aspects = [aspects]

            if not isinstance(
                sentiments,
                (list, tuple)
            ):
                sentiments = [sentiments]

            if len(aspects) != len(sentiments):
                raise ValueError(
                    "Aspect and sentiment counts do "
                    f"not match: {value}"
                )

            raw_pairs = list(
                zip(aspects, sentiments)
            )

    # -----------------------------------------------------
    # FORMAT 2: List or tuple
    # -----------------------------------------------------

    elif isinstance(value, (list, tuple)):

        if len(value) == 0:
            return []

        # One direct pair:
        # ["screen", "positive"]
        if is_annotation_pair(value):
            raw_pairs = [value]

        # Multiple pairs
        else:
            raw_pairs = list(value)

    else:
        raise ValueError(
            "Unsupported annotation type: "
            f"{type(value).__name__}: {value}"
        )

    # -----------------------------------------------------
    # Validate every pair
    # -----------------------------------------------------

    validated_pairs = []

    for pair in raw_pairs:

        # Dictionary inside a list
        if isinstance(pair, dict):
            aspect = get_first_value(
                pair,
                [
                    "aspect_term",
                    "aspect",
                    "term",
                    "target"
                ]
            )

            sentiment = get_first_value(
                pair,
                [
                    "polarity",
                    "sentiment",
                    "sentiment_polarity",
                    "aspect_polarity"
                ]
            )

        # Standard pair:
        # ["screen", "positive"]
        elif is_annotation_pair(pair):
            aspect = pair[0]
            sentiment = pair[1]

        else:
            raise ValueError(
                f"Invalid aspect pair: {pair}"
            )

        if aspect is None:
            raise ValueError(
                f"Aspect term is missing: {pair}"
            )

        if sentiment is None:
            raise ValueError(
                f"Sentiment label is missing: {pair}"
            )

        aspect = str(aspect).strip()

        sentiment = (
            str(sentiment)
            .strip()
            .lower()
        )

        if not aspect:
            raise ValueError(
                f"Aspect term is empty: {pair}"
            )

        if sentiment not in VALID_SENTIMENTS:
            raise ValueError(
                f"Unknown sentiment '{sentiment}' "
                f"for aspect '{aspect}'"
            )

        validated_pairs.append({
            "aspect": aspect,
            "sentiment": sentiment
        })

    return validated_pairs


# =========================================================
# ASPECT-SPAN SEARCH
# =========================================================

def find_aspect_span(sentence, aspect):
    """
    Find the first case-insensitive occurrence of an
    aspect phrase in the original sentence.

    Returns character offsets.
    """

    sentence_lower = sentence.lower()
    aspect_lower = aspect.lower()

    start = sentence_lower.find(
        aspect_lower
    )

    if start == -1:
        return None

    end = start + len(aspect)

    return {
        "start": start,
        "end": end
    }


# =========================================================
# SPAN VALIDATION
# =========================================================

def validate_span(sentence, aspect, span):
    """
    Verify that the character offsets produce the
    expected aspect text.
    """

    if span is None:
        return False

    extracted_text = sentence[
        span["start"]:span["end"]
    ]

    return (
        extracted_text.lower()
        == aspect.lower()
    )


# =========================================================
# OUTPUT-FORMAT DETECTION
# =========================================================

def detect_annotation_structure(value):
    """
    Identify the broad annotation structure for auditing.
    """

    if isinstance(value, str):
        text = value.strip()

        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return "unparseable_string"

    if isinstance(value, dict):
        if (
            "aspect_term" in value
            and "aspect_category" in value
        ):
            return "term_and_category"

        if "aspect_term" in value:
            return "aspect_term_dictionary"

        if "aspect_category" in value:
            return "category_only"

        return "other_dictionary"

    if isinstance(value, (list, tuple)):
        return "list_or_tuple"

    return type(value).__name__


# =========================================================
# PREPARE ONE DATASET SPLIT
# =========================================================

def prepare_split(
    split_name,
    split_data,
    output_directory
):
    """
    Process and save one dataset split.
    """

    output_file = (
        output_directory
        / f"source_{split_name}.jsonl"
    )

    statistics = {
        "rows": 0,
        "rows_with_aspects": 0,
        "empty_aspect_rows": 0,
        "total_aspects": 0,
        "supported_aspects": 0,
        "unsupported_aspects": 0,
        "duplicate_sentences": 0
    }

    sentiment_counts = Counter()
    structure_counts = Counter()
    seen_sentences = set()

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as output_handle:

        for row_id, row in enumerate(split_data):
            statistics["rows"] += 1

            # -----------------------------------------
            # Extract sentence
            # -----------------------------------------

            try:
                sentence = extract_sentence(
                    row["input"]
                )

            except Exception as error:
                raise ValueError(
                    "\nFailed to extract sentence.\n"
                    f"Split: {split_name}\n"
                    f"Row: {row_id}\n"
                    f"Raw input: "
                    f"{row.get('input')!r}\n"
                    f"Reason: {error}"
                ) from error

            # -----------------------------------------
            # Record annotation structure
            # -----------------------------------------

            annotation_structure = (
                detect_annotation_structure(
                    row["output"]
                )
            )

            structure_counts[
                annotation_structure
            ] += 1

            # -----------------------------------------
            # Check duplicate sentence
            # -----------------------------------------

            normalized_sentence = " ".join(
                sentence.lower().split()
            )

            is_duplicate = (
                normalized_sentence
                in seen_sentences
            )

            if is_duplicate:
                statistics[
                    "duplicate_sentences"
                ] += 1
            else:
                seen_sentences.add(
                    normalized_sentence
                )

            # -----------------------------------------
            # Parse annotations
            # -----------------------------------------

            try:
                pairs = parse_aspect_pairs(
                    row["output"]
                )

            except Exception as error:
                raise ValueError(
                    "\nFailed to process annotation.\n"
                    f"Split: {split_name}\n"
                    f"Row: {row_id}\n"
                    f"Sentence: {sentence!r}\n"
                    f"Raw output: "
                    f"{row.get('output')!r}\n"
                    f"Reason: {error}"
                ) from error

            if len(pairs) == 0:
                statistics[
                    "empty_aspect_rows"
                ] += 1
            else:
                statistics[
                    "rows_with_aspects"
                ] += 1

            # -----------------------------------------
            # Locate aspect spans
            # -----------------------------------------

            processed_aspects = []

            for pair in pairs:
                aspect = pair["aspect"]
                sentiment = pair["sentiment"]

                statistics[
                    "total_aspects"
                ] += 1

                sentiment_counts[
                    sentiment
                ] += 1

                span = find_aspect_span(
                    sentence,
                    aspect
                )

                span_supported = validate_span(
                    sentence,
                    aspect,
                    span
                )

                if span_supported:
                    statistics[
                        "supported_aspects"
                    ] += 1

                    start = span["start"]
                    end = span["end"]

                else:
                    statistics[
                        "unsupported_aspects"
                    ] += 1

                    start = None
                    end = None

                processed_aspects.append({
                    "term": aspect,
                    "sentiment": sentiment,
                    "start": start,
                    "end": end,
                    "span_supported": span_supported
                })

            # -----------------------------------------
            # Build normalized record
            # -----------------------------------------

            record = {
                "id": f"{split_name}_{row_id}",
                "source": "NEUDM/semeval-2014",
                "domain": "semeval2014_mixed",
                "split": split_name,
                "text": sentence,
                "annotation_structure": (
                    annotation_structure
                ),
                "is_duplicate": is_duplicate,
                "aspects": processed_aspects
            }

            output_handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    statistics[
        "sentiment_distribution"
    ] = dict(sentiment_counts)

    statistics[
        "annotation_structures"
    ] = dict(structure_counts)

    return output_file, statistics


# =========================================================
# SAVE AUDIT REPORT
# =========================================================

def save_audit_report(
    audit_results,
    output_directory
):
    """
    Save dataset statistics in JSON format.
    """

    audit_file = (
        output_directory
        / "source_dataset_audit.json"
    )

    with audit_file.open(
        "w",
        encoding="utf-8"
    ) as output_handle:
        json.dump(
            audit_results,
            output_handle,
            ensure_ascii=False,
            indent=4
        )

    return audit_file


# =========================================================
# DISPLAY SPLIT STATISTICS
# =========================================================

def display_split_statistics(
    split_name,
    output_file,
    statistics
):
    """
    Print dataset audit statistics.
    """

    total_aspects = statistics[
        "total_aspects"
    ]

    unsupported_aspects = statistics[
        "unsupported_aspects"
    ]

    if total_aspects > 0:
        unsupported_rate = (
            unsupported_aspects
            / total_aspects
        )
    else:
        unsupported_rate = 0.0

    print(f"\nCompleted split: {split_name}")
    print(f"Created: {output_file}")

    print(
        "Rows:",
        statistics["rows"]
    )

    print(
        "Rows with aspects:",
        statistics["rows_with_aspects"]
    )

    print(
        "Empty-aspect rows:",
        statistics["empty_aspect_rows"]
    )

    print(
        "Total aspects:",
        total_aspects
    )

    print(
        "Supported aspects:",
        statistics["supported_aspects"]
    )

    print(
        "Unsupported aspects:",
        unsupported_aspects
    )

    print(
        "Unsupported rate:",
        f"{unsupported_rate:.2%}"
    )

    print(
        "Duplicate sentences:",
        statistics["duplicate_sentences"]
    )

    print(
        "Sentiment distribution:",
        statistics["sentiment_distribution"]
    )

    print(
        "Annotation structures:",
        statistics["annotation_structures"]
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():
    print("=" * 60)
    print("SEMEVAL-2014 SOURCE DATA PREPARATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Confirm downloaded dataset exists
    # -----------------------------------------------------

    if not SOURCE_DATASET_DIRECTORY.exists():
        raise FileNotFoundError(
            "\nDownloaded dataset was not found.\n\n"
            f"Expected location:\n"
            f"{SOURCE_DATASET_DIRECTORY}\n\n"
            "Run download_source.py first."
        )

    # -----------------------------------------------------
    # Load dataset from local storage
    # -----------------------------------------------------

    print("\nLoading dataset from:")
    print(SOURCE_DATASET_DIRECTORY)

    dataset = load_from_disk(
        str(SOURCE_DATASET_DIRECTORY)
    )

    print("\nDataset loaded successfully.")
    print(dataset)

    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    audit_results = {
        "dataset": "NEUDM/semeval-2014",
        "declared_domain": "mixed_or_unverified",
        "source_directory": str(
            SOURCE_DATASET_DIRECTORY
        ),
        "output_directory": str(
            OUTPUT_DIRECTORY
        ),
        "valid_sentiments": sorted(
            VALID_SENTIMENTS
        ),
        "splits": {}
    }

    # -----------------------------------------------------
    # Process all dataset splits
    # -----------------------------------------------------

    for split_name, split_data in dataset.items():
        print("\n" + "-" * 60)
        print(f"Processing split: {split_name}")
        print("-" * 60)

        output_file, statistics = prepare_split(
            split_name=split_name,
            split_data=split_data,
            output_directory=OUTPUT_DIRECTORY
        )

        audit_results["splits"][
            split_name
        ] = statistics

        display_split_statistics(
            split_name=split_name,
            output_file=output_file,
            statistics=statistics
        )

    # -----------------------------------------------------
    # Save audit report
    # -----------------------------------------------------

    audit_file = save_audit_report(
        audit_results,
        OUTPUT_DIRECTORY
    )

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)

    print("\nProcessed dataset location:")
    print(OUTPUT_DIRECTORY)

    print("\nAudit report:")
    print(audit_file)

    print(
        "\nWarning: this Hugging Face copy contains "
        "mixed or unverified domains."
    )

    print(
        "Do not begin final RAISE-ABSA training "
        "until the source domain is isolated."
    )

    print(
        "Use the official SemEval-2014 Laptop XML "
        "for the final research experiment."
    )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()