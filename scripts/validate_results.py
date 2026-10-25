from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


if __name__ == "__main__":
    metrics = pd.read_csv(METRICS / "metrics_by_seed.csv")
    deltas = pd.read_csv(METRICS / "paired_graph_deltas_by_seed.csv")
    diagnostics = pd.read_csv(METRICS / "training_diagnostics.csv")
    metadata = json.loads((METRICS / "run_metadata.json").read_text(encoding="utf-8"))

    require(set(metadata["seeds"]) == {42, 43, 44}, "seeds 42/43/44 completed")
    require(metadata["epochs"] == 50, "full 50-epoch budget recorded")
    require(set(metrics["model"]) == {"AAC", "AAC+Graph", "ESM2", "ESM2+Graph"}, "all four models present")
    require(
        metrics.loc[metrics["subset"] == "overall"].groupby("model")["seed"].nunique().eq(3).all(),
        "three seeds available for every whole-test model",
    )
    require((diagnostics["epoch0_max_abs_error"] == 0).all(), "epoch-0 graph output equals baseline exactly")
    require((diagnostics["no_neighbor_max_abs_error"] == 0).all(), "no-neighbor graph output equals baseline exactly")
    require((diagnostics["base_alignment_max_abs_error"] <= 1e-6).all(), "frozen baseline remains aligned")
    no_neighbor = deltas[deltas["subset"] == "no_train_neighbor"]
    delta_columns = [column for column in deltas if column.startswith("delta_")]
    require(np.abs(no_neighbor[delta_columns].to_numpy()).max() == 0, "all no-neighbor paired deltas are exactly zero")
    require(metadata["test_has_train_neighbor"] == 940, "connected-test count is 940")
    require(metadata["test_no_train_neighbor"] == 4522, "no-neighbor test count is 4522")
    require((TABLES / "main_metrics_summary.csv").exists(), "main summary table exists")
    require((TABLES / "paired_graph_delta_summary.csv").exists(), "paired-delta summary exists")
    expected_figures = [
        "figure1_connected_graph_delta",
        "figure2_overall_representation_comparison",
        "figure3_aac_similarity_bins",
        "figure4_graph_homophily",
    ]
    require(
        all((FIGURES / f"{name}.{extension}").stat().st_size > 0 for name in expected_figures for extension in ("png", "pdf")),
        "all PNG and PDF figures exist and are non-empty",
    )
    print("Final result validation passed.")
