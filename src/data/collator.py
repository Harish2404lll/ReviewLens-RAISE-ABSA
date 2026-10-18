# src/data/collator.py

from pathlib import Path
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# ============================================================
# PROJECT PATHS & CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BIO_LABEL_TO_ID = {
    "O": 0,
    "B-ASP": 1,
    "I-ASP": 2,
}

SENTIMENT_LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "conflict": 3,
}

IGNORE_INDEX = -100


# ============================================================
# DYNAMIC PADDING COLLATOR
# ============================================================

class RaiseAbsaCollator:
    """
    Dynamically pads variable-length tokenized ABSA records to the
    maximum sequence length present in the current batch.
    """

    def __init__(
        self,
        pad_token_id: int,
        ignore_index: int = IGNORE_INDEX,
    ):
        if pad_token_id is None or pad_token_id < 0:
            raise ValueError(
                f"Invalid pad_token_id: {pad_token_id}"
            )
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)

    def __call__(self, batch: list[dict]) -> dict:
        if not batch:
            raise ValueError("Collator received an empty batch.")

        batch_size = len(batch)
        max_batch_length = max(
            len(item["input_ids"]) for item in batch
        )

        padded_input_ids = torch.full(
            (batch_size, max_batch_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        padded_attention_mask = torch.zeros(
            (batch_size, max_batch_length),
            dtype=torch.long,
        )
        padded_bio_labels = torch.full(
            (batch_size, max_batch_length),
            fill_value=self.ignore_index,
            dtype=torch.long,
        )

        has_token_type_ids = "token_type_ids" in batch[0]
        padded_token_type_ids = (
            torch.zeros(
                (batch_size, max_batch_length),
                dtype=torch.long,
            )
            if has_token_type_ids
            else None
        )

        # Flattened span tensors for vectorized/looped span pooling
        span_batch_indices = []
        span_token_starts = []
        span_token_ends = []
        span_sentiment_labels = []

        # Per-record metadata for exact span-pair evaluation
        record_ids = []
        texts = []
        gold_aspects = []

        for batch_index, item in enumerate(batch):
            seq_len = len(item["input_ids"])

            padded_input_ids[batch_index, :seq_len] = (
                torch.tensor(item["input_ids"], dtype=torch.long)
            )
            padded_attention_mask[batch_index, :seq_len] = (
                torch.tensor(item["attention_mask"], dtype=torch.long)
            )
            padded_bio_labels[batch_index, :seq_len] = (
                torch.tensor(item["bio_labels"], dtype=torch.long)
            )

            if has_token_type_ids:
                padded_token_type_ids[batch_index, :seq_len] = (
                    torch.tensor(
                        item["token_type_ids"],
                        dtype=torch.long,
                    )
                )

            starts = item["aspect_token_starts"]
            ends = item["aspect_token_ends"]
            sentiments = item["aspect_sentiment_ids"]
            terms = item["aspect_terms"]
            char_starts = item["aspect_character_starts"]
            char_ends = item["aspect_character_ends"]

            record_gold = []
            for s_tok, e_tok, sent_id, term, c_start, c_end in zip(
                starts,
                ends,
                sentiments,
                terms,
                char_starts,
                char_ends,
            ):
                if not (0 <= s_tok < e_tok <= seq_len):
                    raise ValueError(
                        f"Invalid token span ({s_tok}, {e_tok}) "
                        f"in record {item['id']} of length {seq_len}."
                    )

                span_batch_indices.append(batch_index)
                span_token_starts.append(int(s_tok))
                span_token_ends.append(int(e_tok))
                span_sentiment_labels.append(int(sent_id))

                record_gold.append(
                    {
                        "term": term,
                        "token_start": int(s_tok),
                        "token_end": int(e_tok),
                        "char_start": int(c_start),
                        "char_end": int(c_end),
                        "sentiment_id": int(sent_id),
                    }
                )

            record_ids.append(item["id"])
            texts.append(item["text"])
            gold_aspects.append(record_gold)

        output = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "bio_labels": padded_bio_labels,
            "span_batch_indices": torch.tensor(
                span_batch_indices, dtype=torch.long
            ),
            "span_token_starts": torch.tensor(
                span_token_starts, dtype=torch.long
            ),
            "span_token_ends": torch.tensor(
                span_token_ends, dtype=torch.long
            ),
            "span_sentiment_labels": torch.tensor(
                span_sentiment_labels, dtype=torch.long
            ),
            "record_ids": record_ids,
            "texts": texts,
            "gold_aspects": gold_aspects,
        }

        if padded_token_type_ids is not None:
            output["token_type_ids"] = padded_token_type_ids

        return output


# ============================================================
# SELF-VERIFICATION ENTRY POINT
# ============================================================

def verify_dataset_collation(
    dataset_directory: Path,
    tokenizer_directory: Path,
    name: str,
):
    print("\n" + "-" * 60)
    print(f"Verifying collator for: {name}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory,
        local_files_only=True,
        use_fast=True,
    )
    dataset = load_from_disk(str(dataset_directory))

    collator = RaiseAbsaCollator(
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=IGNORE_INDEX,
    )

    for split_name in ["train", "validation", "test"]:
        loader = DataLoader(
            dataset[split_name],
            batch_size=8,
            shuffle=False,
            collate_fn=collator,
        )

        total_batches = 0
        total_spans = 0

        for batch in loader:
            total_batches += 1
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            bio_labels = batch["bio_labels"]

            assert input_ids.shape == attention_mask.shape == bio_labels.shape

            # Verify every span aligns with B-ASP / I-ASP in bio_labels
            for b_idx, t_start, t_end in zip(
                batch["span_batch_indices"].tolist(),
                batch["span_token_starts"].tolist(),
                batch["span_token_ends"].tolist(),
            ):
                total_spans += 1
                span_bio = bio_labels[b_idx, t_start:t_end].tolist()
                if span_bio[0] != BIO_LABEL_TO_ID["B-ASP"]:
                    raise RuntimeError(
                        f"Span start is not B-ASP in {split_name}: {span_bio}"
                    )
                if any(
                    label != BIO_LABEL_TO_ID["I-ASP"]
                    for label in span_bio[1:]
                ):
                    raise RuntimeError(
                        f"Span continuation is not I-ASP in {split_name}: {span_bio}"
                    )

        print(
            f"Split '{split_name}': {total_batches} batches | "
            f"{total_spans} verified spans | PASSED"
        )


def main():
    print("=" * 70)
    print("DYNAMIC COLLATOR & SPAN-SLICE VERIFICATION")
    print("=" * 70)

    xlmr_dataset_dir = (
        PROJECT_ROOT / "data" / "tokenized" / "xlmr_laptop_max128"
    )
    xlmr_tokenizer_dir = (
        PROJECT_ROOT / "models" / "xlm-roberta-base-tokenizer"
    )

    if xlmr_dataset_dir.exists() and xlmr_tokenizer_dir.exists():
        verify_dataset_collation(
            dataset_directory=xlmr_dataset_dir,
            tokenizer_directory=xlmr_tokenizer_dir,
            name="XLM-RoBERTa (xlmr_laptop_max128)",
        )

    deberta_dataset_dir = (
        PROJECT_ROOT / "data" / "tokenized" / "deberta_laptop_max96"
    )
    deberta_tokenizer_dir = (
        PROJECT_ROOT / "models" / "deberta-v3-base-tokenizer"
    )

    if deberta_dataset_dir.exists() and deberta_tokenizer_dir.exists():
        verify_dataset_collation(
            dataset_directory=deberta_dataset_dir,
            tokenizer_directory=deberta_tokenizer_dir,
            name="DeBERTa-v3 (deberta_laptop_max96)",
        )

    print("\n" + "=" * 70)
    print("ALL COLLATOR CHECKS PASSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()