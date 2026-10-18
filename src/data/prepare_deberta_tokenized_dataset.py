# src/data/prepare_deberta_tokenized_dataset.py

import json
from collections import Counter
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "official_laptop"
TOKENIZER_DIRECTORY = PROJECT_ROOT / "models" / "deberta-v3-base-tokenizer"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "tokenized" / "deberta_laptop_max96"

MAX_LENGTH = 96
IGNORE_INDEX = -100

BIO_LABEL_TO_ID = {"O": 0, "B-ASP": 1, "I-ASP": 2}
SENTIMENT_LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2, "conflict": 3}

SPLIT_FILES = {
    "train": PROCESSED_DATA_DIRECTORY / "laptop_train.jsonl",
    "validation": PROCESSED_DATA_DIRECTORY / "laptop_validation.jsonl",
    "test": PROCESSED_DATA_DIRECTORY / "laptop_test.jsonl",
}

def load_jsonl(file_path):
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def normalize_aspect_boundary(text, aspect):
    start = int(aspect["start"])
    end = int(aspect["end"])
    term = str(aspect["term"])
    sentiment = str(aspect["sentiment"]).strip().lower()

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    return {
        "term": text[start:end],
        "start": start,
        "end": end,
        "sentiment": sentiment,
    }

def find_aspect_token_span(aspect, offset_mapping):
    aspect_start = aspect["start"]
    aspect_end = aspect["end"]
    overlapping = []

    for idx, (t_start, t_end) in enumerate(offset_mapping):
        t_start, t_end = int(t_start), int(t_end)
        if t_end <= t_start:
            continue
        if t_start < aspect_end and t_end > aspect_start:
            overlapping.append(idx)

    if not overlapping:
        return None

    t_start_idx = min(overlapping)
    t_end_idx = max(overlapping) + 1
    
    covered_start = min(int(offset_mapping[i][0]) for i in overlapping)
    covered_end = max(int(offset_mapping[i][1]) for i in overlapping)

    if covered_start <= aspect_start and covered_end >= aspect_end:
        return {"token_start": t_start_idx, "token_end": t_end_idx}
    return None

def process_split(input_file, tokenizer, max_length):
    records = load_jsonl(input_file)
    processed = []

    for record in records:
        text = record["text"]
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )

        input_ids = list(encoding["input_ids"])
        attention_mask = list(encoding["attention_mask"])
        offset_mapping = [[int(s), int(e)] for s, e in encoding["offset_mapping"]]

        bio_labels = [
            IGNORE_INDEX if e <= s else BIO_LABEL_TO_ID["O"]
            for s, e in offset_mapping
        ]

        starts, ends, sentiments, terms, c_starts, c_ends = [], [], [], [], [], []
        occupied = set()

        for raw_aspect in record["aspects"]:
            aspect = normalize_aspect_boundary(text, raw_aspect)
            span = find_aspect_token_span(aspect, offset_mapping)
            if span is None:
                raise ValueError(f"Alignment failed for aspect: {aspect} in text: {text}")

            t_start, t_end = span["token_start"], span["token_end"]
            current_tokens = set(range(t_start, t_end))
            if occupied & current_tokens:
                raise ValueError(f"Overlapping aspect tokens in record {record['id']}")
            occupied.update(current_tokens)

            bio_labels[t_start] = BIO_LABEL_TO_ID["B-ASP"]
            for idx in range(t_start + 1, t_end):
                bio_labels[idx] = BIO_LABEL_TO_ID["I-ASP"]

            starts.append(t_start)
            ends.append(t_end)
            sentiments.append(SENTIMENT_LABEL_TO_ID[aspect["sentiment"]])
            terms.append(aspect["term"])
            c_starts.append(aspect["start"])
            c_ends.append(aspect["end"])

        processed.append({
            "id": record["id"],
            "source_id": str(record.get("source_id", record["id"])),
            "domain": str(record.get("domain", "laptop")),
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bio_labels": bio_labels,
            "aspect_token_starts": starts,
            "aspect_token_ends": ends,
            "aspect_sentiment_ids": sentiments,
            "aspect_terms": terms,
            "aspect_character_starts": c_starts,
            "aspect_character_ends": c_ends,
        })
    return processed

def main():
    print("=" * 60)
    print("PREPARING DEBERTA TOKENIZED DATASET (max_length=96)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIRECTORY, use_fast=True, local_files_only=True)
    splits = {}

    for split_name, file_path in SPLIT_FILES.items():
        print(f"Processing {split_name}...")
        records = process_split(file_path, tokenizer, MAX_LENGTH)
        splits[split_name] = Dataset.from_list(records)

    dataset_dict = DatasetDict(splits)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(OUTPUT_DIRECTORY))
    print(f"\nSuccessfully saved DeBERTa tokenized dataset to:\n{OUTPUT_DIRECTORY}")

if __name__ == "__main__":
    main()