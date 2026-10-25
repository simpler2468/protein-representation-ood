from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"

COLORS = {"AAC": "#D97706", "ESM2": "#2563EB", "AAC+Graph": "#F59E0B", "ESM2+Graph": "#60A5FA"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_both(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.png", facecolor="white")
    figure.savefig(FIGURES / f"{stem}.pdf", facecolor="white")
    plt.close(figure)


def main_graph_delta(delta: pd.DataFrame) -> None:
    connected = delta[delta["subset"] == "has_train_neighbor"].set_index("representation")
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for axis, metric, title in zip(
        axes,
        ["macro_auroc", "macro_auprc"],
        ["Connected-test Macro AUROC", "Connected-test Macro AUPRC"],
    ):
        names = ["AAC", "ESM2"]
        means = [connected.loc[name, f"delta_{metric}_mean"] for name in names]
        errors = [connected.loc[name, f"delta_{metric}_std"] for name in names]
        axis.bar(names, means, yerr=errors, capsize=4, color=[COLORS[name] for name in names], width=0.62)
        axis.axhline(0, color="#374151", linewidth=0.9)
        axis.set_ylabel("Graph delta (mean +/- SD)")
        axis.set_title(title)
        axis.text(0.98, 0.96, "n = 940", transform=axis.transAxes, ha="right", va="top", color="#4B5563")
        upper = max(value + error for value, error in zip(means, errors)) * 1.25
        lower = min(-0.002, min(value - error for value, error in zip(means, errors)) * 1.20)
        axis.set_ylim(lower, upper)
        for position, (value, error) in enumerate(zip(means, errors)):
            offset = 0.025 * (upper - lower)
            y = value + error + offset if value >= 0 else value - error - offset
            axis.text(position, y, f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top")
    figure.suptitle("Explicit homology helps AAC but adds little beyond ESM2", fontweight="bold")
    figure.tight_layout()
    save_both(figure, "figure1_connected_graph_delta")


def overall_comparison(summary: pd.DataFrame) -> None:
    overall = summary[summary["subset"] == "overall"].set_index("model")
    models = ["AAC", "AAC+Graph", "ESM2", "ESM2+Graph"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for axis, metric, title in zip(
        axes, ["macro_auroc", "macro_auprc"], ["Whole-test Macro AUROC", "Whole-test Macro AUPRC"]
    ):
        values = [overall.loc[model, f"{metric}_mean"] for model in models]
        errors = [overall.loc[model, f"{metric}_std"] for model in models]
        axis.bar(
            np.arange(len(models)), values, yerr=errors, capsize=3,
            color=[COLORS[model] for model in models], width=0.70,
        )
        axis.set_xticks(np.arange(len(models)), models, rotation=18, ha="right")
        axis.set_ylabel("Mean +/- SD")
        axis.set_title(title)
        lower = max(0, min(values) - 0.08)
        axis.set_ylim(lower, min(1.0, max(values) + 0.08))
        for position, value in enumerate(values):
            axis.text(position, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
    figure.tight_layout()
    save_both(figure, "figure2_overall_representation_comparison")


def similarity_bins(delta: pd.DataFrame) -> None:
    order = ["fident_0.30-0.40", "fident_0.40-0.60", "fident_>=0.60"]
    labels = ["0.30-0.40\n(n=776)", "0.40-0.60\n(n=112)", ">=0.60\n(n=52)"]
    aac = delta[(delta["representation"] == "AAC") & delta["subset"].isin(order)].set_index("subset")
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for axis, metric, title in zip(
        axes, ["macro_auroc", "macro_auprc"], ["AAC delta Macro AUROC", "AAC delta Macro AUPRC"]
    ):
        means = [aac.loc[name, f"delta_{metric}_mean"] for name in order]
        errors = [aac.loc[name, f"delta_{metric}_std"] for name in order]
        axis.errorbar(
            np.arange(3), means, yerr=errors, fmt="o-", color=COLORS["AAC"],
            linewidth=2, markersize=7, capsize=4,
        )
        axis.axhline(0, color="#374151", linewidth=0.9)
        axis.set_xticks(np.arange(3), labels)
        axis.set_xlabel("Strongest training-neighbor identity")
        axis.set_ylabel("Graph delta (mean +/- SD)")
        axis.set_title(title)
    figure.suptitle("AAC graph gain is not monotonic across similarity bins", fontweight="bold")
    figure.tight_layout()
    save_both(figure, "figure3_aac_similarity_bins")


def homophily_figure() -> None:
    payload = json.loads((METRICS / "graph_homophily.json").read_text(encoding="utf-8"))
    values = [payload["sequence_edge_shared_label_rate"], payload["random_pair_shared_label_rate"]]
    figure, axis = plt.subplots(figsize=(4.6, 3.5))
    bars = axis.bar(["Sequence edges", "Random pairs"], values, color=["#059669", "#9CA3AF"], width=0.62)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Share at least one label")
    axis.set_title("Training-graph label homophily")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    figure.tight_layout()
    save_both(figure, "figure4_graph_homophily")


if __name__ == "__main__":
    style()
    summary = pd.read_csv(TABLES / "main_metrics_summary.csv")
    delta = pd.read_csv(TABLES / "paired_graph_delta_summary.csv")
    main_graph_delta(delta)
    overall_comparison(summary)
    similarity_bins(delta)
    homophily_figure()
    print(f"Figures written to {FIGURES}")
