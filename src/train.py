from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import make_feature_loader
from .evaluate import multilabel_metrics
from .models import NeighborAttentionResidual, ProteinMLP
from .utils import clone_state_dict, seeded_generator, set_seed


@dataclass
class FitResult:
    model: nn.Module
    best_epoch: int
    best_validation_macro_auroc: float
    history: list[dict]


class NeighborDataset(Dataset):
    def __init__(
        self,
        query_indices: np.ndarray,
        labels: np.ndarray,
        neighbor_indices: np.ndarray,
        edge_features: np.ndarray,
        neighbor_mask: np.ndarray,
    ):
        self.query_indices = torch.as_tensor(query_indices, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.neighbor_indices = torch.as_tensor(neighbor_indices, dtype=torch.long)
        self.edge_features = torch.as_tensor(edge_features, dtype=torch.float32)
        self.neighbor_mask = torch.as_tensor(neighbor_mask, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.query_indices)

    def __getitem__(self, item: int):
        query_index = self.query_indices[item]
        return (
            query_index,
            self.labels[query_index],
            self.neighbor_indices[query_index],
            self.edge_features[query_index],
            self.neighbor_mask[query_index],
        )


def make_neighbor_loader(
    indices: np.ndarray,
    labels: np.ndarray,
    neighbor_indices: np.ndarray,
    edge_features: np.ndarray,
    neighbor_mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        NeighborDataset(indices, labels, neighbor_indices, edge_features, neighbor_mask),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=seeded_generator(seed),
    )


@torch.inference_mode()
def predict_baseline(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_feature_loader(features, labels, indices, batch_size, False, 0)
    model.eval()
    targets, probabilities = [], []
    for batch_features, batch_targets in loader:
        logits = model(batch_features.to(device))
        targets.append(batch_targets.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def fit_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
    device: torch.device,
    epochs: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    train_batch_size: int,
    evaluation_batch_size: int,
) -> FitResult:
    set_seed(seed)
    model = ProteinMLP(features.shape[1], hidden_dim, labels.shape[1], dropout).to(device)
    loader = make_feature_loader(
        features, labels, train_indices, train_batch_size, True, seed + 10_000
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_epoch, best_score = 0, -np.inf
    best_state = clone_state_dict(model)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features.to(device)), batch_targets.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_targets)
        targets, probabilities = predict_baseline(
            model, features, labels, validation_indices, device, evaluation_batch_size
        )
        score = multilabel_metrics(targets, probabilities)["macro_auroc"]
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(train_indices),
            "validation_macro_auroc": score,
        })
        if score > best_score:
            best_epoch, best_score, best_state = epoch, score, clone_state_dict(model)
    model.load_state_dict(best_state)
    return FitResult(model, best_epoch, float(best_score), history)


@torch.inference_mode()
def predict_graph(
    model: NeighborAttentionResidual,
    feature_bank: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    indices, targets, probabilities, baseline_probabilities = [], [], [], []
    for query_index, batch_targets, neighbor_index, edge, mask in loader:
        query_device = query_index.to(device)
        neighbor_device = neighbor_index.to(device)
        logits, baseline_logits = model(
            feature_bank[query_device],
            feature_bank[neighbor_device],
            edge.to(device),
            mask.to(device),
        )
        indices.append(query_index.numpy())
        targets.append(batch_targets.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        baseline_probabilities.append(torch.sigmoid(baseline_logits).cpu().numpy())
    return (
        np.concatenate(indices),
        np.concatenate(targets),
        np.concatenate(probabilities),
        np.concatenate(baseline_probabilities),
    )


def fit_graph(
    baseline: ProteinMLP,
    features: np.ndarray,
    labels: np.ndarray,
    neighbor_indices: np.ndarray,
    edge_features: np.ndarray,
    neighbor_mask: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
    device: torch.device,
    epochs: int,
    attention_hidden_dim: int,
    correction_hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
) -> tuple[FitResult, dict]:
    connected_train = train_indices[neighbor_mask[train_indices].any(axis=1)]
    connected_validation = validation_indices[neighbor_mask[validation_indices].any(axis=1)]
    train_loader = make_neighbor_loader(
        connected_train, labels, neighbor_indices, edge_features, neighbor_mask,
        batch_size, True, seed + 20_000,
    )
    validation_loader = make_neighbor_loader(
        connected_validation, labels, neighbor_indices, edge_features, neighbor_mask,
        batch_size, False, seed + 30_000,
    )
    set_seed(seed + 100_000)
    model = NeighborAttentionResidual(
        baseline, features.shape[1], labels.shape[1], attention_hidden_dim,
        correction_hidden_dim, dropout,
    ).to(device)
    bank = torch.as_tensor(features, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    _, epoch0_targets, epoch0_probabilities, epoch0_baseline = predict_graph(
        model, bank, validation_loader, device
    )
    epoch0_error = float(np.max(np.abs(epoch0_probabilities - epoch0_baseline)))
    if epoch0_error != 0.0:
        raise RuntimeError(f"Epoch-0 graph output differs from baseline: {epoch0_error}")
    best_score = multilabel_metrics(epoch0_targets, epoch0_probabilities)["macro_auroc"]
    best_epoch, best_state = 0, clone_state_dict(model)
    history = [{"epoch": 0, "train_loss": np.nan, "validation_macro_auroc": best_score}]

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for query_index, batch_targets, neighbor_index, edge, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            query_device = query_index.to(device)
            neighbor_device = neighbor_index.to(device)
            logits, _ = model(
                bank[query_device], bank[neighbor_device], edge.to(device), mask.to(device)
            )
            loss = criterion(logits, batch_targets.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_targets)
        _, validation_targets, validation_probabilities, _ = predict_graph(
            model, bank, validation_loader, device
        )
        score = multilabel_metrics(validation_targets, validation_probabilities)["macro_auroc"]
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(connected_train),
            "validation_macro_auroc": score,
        })
        if score > best_score:
            best_epoch, best_score, best_state = epoch, score, clone_state_dict(model)
    model.load_state_dict(best_state)
    diagnostics = {
        "connected_train": int(len(connected_train)),
        "connected_validation": int(len(connected_validation)),
        "epoch0_max_abs_error": epoch0_error,
    }
    return FitResult(model, best_epoch, float(best_score), history), diagnostics
