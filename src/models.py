from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ProteinMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class NeighborAttentionResidual(nn.Module):
    def __init__(
        self,
        baseline: ProteinMLP,
        feature_dim: int,
        output_dim: int,
        attention_hidden_dim: int = 128,
        correction_hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.baseline = copy.deepcopy(baseline)
        for parameter in self.baseline.parameters():
            parameter.requires_grad = False

        self.attention = nn.Sequential(
            nn.Linear(2 * feature_dim + 2, attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.value_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.eye_(self.value_projection.weight)
        self.correction = nn.Sequential(
            nn.Linear(2 * feature_dim, correction_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(correction_hidden_dim, output_dim),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(
        self,
        query_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        edge_features: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            baseline_logits = self.baseline(query_features)
        expanded_query = query_features[:, None, :].expand_as(neighbor_features)
        attention_input = torch.cat([expanded_query, neighbor_features, edge_features], dim=-1)
        scores = self.attention(attention_input).squeeze(-1)
        scores = scores.masked_fill(~neighbor_mask, -1e9)
        weights = torch.softmax(scores, dim=1) * neighbor_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        summary = torch.sum(
            weights[:, :, None] * self.value_projection(neighbor_features), dim=1
        )
        correction = self.correction(torch.cat([query_features, summary], dim=-1))
        gate = neighbor_mask.any(dim=1, keepdim=True).float()
        return baseline_logits + gate * correction, baseline_logits
