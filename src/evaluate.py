from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .data import LABEL_COLUMNS


RANKING_METRICS = ["macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc"]
F1_METRICS = ["macro_f1", "micro_f1"]
METRIC_COLUMNS = RANKING_METRICS + F1_METRICS


def tune_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    start: float = 0.05,
    stop: float = 0.95,
    steps: int = 19,
) -> np.ndarray:
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    for label in range(targets.shape[1]):
        best_score = -1.0
        for threshold in np.linspace(start, stop, steps):
            score = f1_score(
                targets[:, label], probabilities[:, label] >= threshold, zero_division=0
            )
            if score > best_score:
                best_score = score
                thresholds[label] = threshold
    return thresholds


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
    predictions: np.ndarray | None = None,
) -> dict[str, float | int]:
    valid = [index for index in range(targets.shape[1]) if np.unique(targets[:, index]).size == 2]
    if predictions is None:
        if thresholds is None:
            thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
        predictions = probabilities >= thresholds[None, :]
    return {
        "n": int(len(targets)),
        "valid_labels": int(len(valid)),
        "macro_auroc": float(np.mean([
            roc_auc_score(targets[:, index], probabilities[:, index]) for index in valid
        ])) if valid else float("nan"),
        "micro_auroc": float(roc_auc_score(targets.ravel(), probabilities.ravel())),
        "macro_auprc": float(np.mean([
            average_precision_score(targets[:, index], probabilities[:, index]) for index in valid
        ])) if valid else float("nan"),
        "micro_auprc": float(average_precision_score(targets.ravel(), probabilities.ravel())),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
    }


def per_label_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    predictions: np.ndarray | None = None,
) -> pd.DataFrame:
    if predictions is None:
        predictions = probabilities >= thresholds[None, :]
    rows = []
    for index, label in enumerate(LABEL_COLUMNS):
        valid = np.unique(targets[:, index]).size == 2
        rows.append({
            "label": label,
            "threshold": float(thresholds[index]),
            "prevalence": float(targets[:, index].mean()),
            "auroc": float(roc_auc_score(targets[:, index], probabilities[:, index])) if valid else np.nan,
            "auprc": float(average_precision_score(targets[:, index], probabilities[:, index])) if valid else np.nan,
            "f1": float(f1_score(targets[:, index], predictions[:, index], zero_division=0)),
        })
    return pd.DataFrame(rows)


def paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "representation", "subset"]
    baseline = metrics.loc[~metrics["graph"], keys + ["n", *METRIC_COLUMNS]]
    graph = metrics.loc[metrics["graph"], keys + METRIC_COLUMNS]
    merged = baseline.merge(graph, on=keys, suffixes=("_baseline", "_graph"))
    for metric in METRIC_COLUMNS:
        merged[f"delta_{metric}"] = merged[f"{metric}_graph"] - merged[f"{metric}_baseline"]
    return merged


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["representation", "graph", "model", "subset"]
    for values, group in metrics.groupby(keys, sort=False):
        row = dict(zip(keys, values))
        row.update({"n": int(group["n"].iloc[0]), "seeds": int(group["seed"].nunique())})
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for values, group in deltas.groupby(["representation", "subset"], sort=False):
        row = {
            "representation": values[0],
            "subset": values[1],
            "n": int(group["n"].iloc[0]),
            "seeds": int(group["seed"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            column = f"delta_{metric}"
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)

