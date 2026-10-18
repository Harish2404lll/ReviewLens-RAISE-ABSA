import torch


IGNORE_INDEX = -100


class JointABSACollator:
    """
    Dynamically pad token sequences and flatten aspect spans
    across a batch.

    Aspect token positions remain valid because right-side
    padding is used.
    """

    def __init__(
        self,
        tokenizer,
        pad_to_multiple_of=8,
    ):
        if tokenizer.padding_side != "right":
            raise ValueError(
                "This collator requires right-side padding."
            )

        self.tokenizer = tokenizer

        self.pad_to_multiple_of = (
            pad_to_multiple_of
        )

    def __call__(self, features):
        if not features:
            raise ValueError(
                "Cannot collate an empty batch."
            )

        tokenizer_features = []

        for feature in features:
            tokenizer_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": (
                    feature["attention_mask"]
                ),
            }

            if "token_type_ids" in feature:
                tokenizer_feature[
                    "token_type_ids"
                ] = feature["token_type_ids"]

            tokenizer_features.append(
                tokenizer_feature
            )

        batch = self.tokenizer.pad(
            tokenizer_features,
            padding=True,
            pad_to_multiple_of=(
                self.pad_to_multiple_of
            ),
            return_tensors="pt",
        )

        padded_sequence_length = (
            batch["input_ids"].shape[1]
        )

        padded_bio_labels = []

        aspect_batch_indices = []
        aspect_token_starts = []
        aspect_token_ends = []
        aspect_sentiment_ids = []

        for batch_index, feature in enumerate(
            features
        ):
            original_labels = list(
                feature["bio_labels"]
            )

            padding_length = (
                padded_sequence_length
                - len(original_labels)
            )

            if padding_length < 0:
                raise ValueError(
                    "Padded length is shorter than "
                    "the original BIO-label length."
                )

            padded_labels = (
                original_labels
                + [IGNORE_INDEX] * padding_length
            )

            padded_bio_labels.append(
                padded_labels
            )

            starts = feature[
                "aspect_token_starts"
            ]

            ends = feature[
                "aspect_token_ends"
            ]

            sentiments = feature[
                "aspect_sentiment_ids"
            ]

            if not (
                len(starts)
                == len(ends)
                == len(sentiments)
            ):
                raise ValueError(
                    "Aspect fields have different "
                    f"lengths in record {feature['id']}."
                )

            for start, end, sentiment in zip(
                starts,
                ends,
                sentiments,
            ):
                if not (
                    0 <= start
                    < end
                    <= len(original_labels)
                ):
                    raise ValueError(
                        "Invalid aspect token span "
                        f"[{start}, {end}) "
                        f"in record {feature['id']}."
                    )

                aspect_batch_indices.append(
                    batch_index
                )

                aspect_token_starts.append(
                    start
                )

                aspect_token_ends.append(
                    end
                )

                aspect_sentiment_ids.append(
                    sentiment
                )

        batch["bio_labels"] = torch.tensor(
            padded_bio_labels,
            dtype=torch.long,
        )

        batch["aspect_batch_indices"] = (
            torch.tensor(
                aspect_batch_indices,
                dtype=torch.long,
            )
        )

        batch["aspect_token_starts"] = (
            torch.tensor(
                aspect_token_starts,
                dtype=torch.long,
            )
        )

        batch["aspect_token_ends"] = (
            torch.tensor(
                aspect_token_ends,
                dtype=torch.long,
            )
        )

        batch["aspect_sentiment_ids"] = (
            torch.tensor(
                aspect_sentiment_ids,
                dtype=torch.long,
            )
        )

        return batch