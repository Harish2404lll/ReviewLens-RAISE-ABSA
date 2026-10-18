# src/models/raise_absa.py

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class RaiseAbsaModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-v3-small",
        num_bio_labels: int = 3,
        num_sentiment_labels: int = 4,
        dropout_rate: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config,
        ).float()

        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        self.bio_classifier = nn.Linear(hidden_size, num_bio_labels)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_sentiment_labels),
        )
        self.float()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = self.dropout(outputs.last_hidden_state.float())
        return hidden_states

    def classify_spans(
        self,
        sequence_output: torch.Tensor,
        batch_indices: torch.Tensor,
        token_starts: torch.Tensor,
        token_ends: torch.Tensor,
    ) -> torch.Tensor:
        sequence_output = sequence_output.float()
        num_spans = batch_indices.shape[0]
        if num_spans == 0:
            return torch.empty(
                0,
                self.sentiment_classifier[-1].out_features,
                device=sequence_output.device,
                dtype=torch.float32,
            )

        # Vectorized Span Pooling: [CLS] + MeanPool + MaxPool
        cls_reps = sequence_output[batch_indices, 0, :]
        
        mean_reps = []
        max_reps = []
        for i in range(num_spans):
            b_idx = int(batch_indices[i].item())
            start = int(token_starts[i].item())
            end = int(token_ends[i].item())
            span_tokens = sequence_output[b_idx, start:end, :]
            if span_tokens.shape[0] == 0:
                span_tokens = sequence_output[b_idx, start : start + 1, :]
            mean_reps.append(span_tokens.mean(dim=0))
            max_reps.append(span_tokens.max(dim=0).values)

        mean_tensor = torch.stack(mean_reps, dim=0)
        max_tensor = torch.stack(max_reps, dim=0)

        combined = torch.cat([cls_reps, mean_tensor, max_tensor], dim=-1)
        return self.sentiment_classifier(combined)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_batch_indices: torch.Tensor = None,
        span_token_starts: torch.Tensor = None,
        span_token_ends: torch.Tensor = None,
    ) -> dict:
        sequence_output = self.encode(input_ids, attention_mask)
        bio_logits = self.bio_classifier(sequence_output)

        sentiment_logits = None
        if span_batch_indices is not None and span_batch_indices.numel() > 0:
            sentiment_logits = self.classify_spans(
                sequence_output=sequence_output,
                batch_indices=span_batch_indices,
                token_starts=span_token_starts,
                token_ends=span_token_ends,
            )

        return {
            "bio_logits": bio_logits,
            "sentiment_logits": sentiment_logits,
            "sequence_output": sequence_output,
        }