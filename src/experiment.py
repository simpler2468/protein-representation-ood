from __future__ import annotations

import gc
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from .data import LABEL_COLUMNS, labels_array, load_dataset, split_indices
from .embeddings import load_esm_embeddings, make_aac
from .evaluate import (
    METRIC_COLUMNS,
    multilabel_metrics,
    paired_deltas,
    per_label_metrics,
    summarize_deltas,
    summarize_metrics,
    tune_thresholds,
)
from .graph import build_inductive_graph, graph_homophily, load_mmseqs
from .train import (
    fit_baseline,
    fit_graph,
    make_neighbor_loader,
    predict_baseline,
    predict_graph,
)
from .utils import resolve_device, save_json


def _metric_row(seed, representation, graph, subset, targets, probabilities, predictions):
    row = {
        "seed": seed,
        "representation": representation,
        "graph": graph,
        "model": representation + ("+Graph" if graph else ""),
        "subset": subset,
    }
    row.update(multilabel_metrics(targets, probabilities, predictions=predictions))
    return row


def _checkpoint(path: Path, model, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def run_experiment(
    project_root: Path,
    config: dict,
    seeds: list[int] | None = None,
    epochs: int | None = None,
    device_name: str | None = None,
) -> Path:
    seeds = seeds or list(config["project"]["seed_list"])
    training = config["training"]
    epochs = epochs or int(training["epochs"])
    device = resolve_device(device_name or training["device"])
    torch.set_num_threads(int(training["num_threads"]))

    data_cfg = config["data"]
    frame = load_dataset(project_root / data_cfg["table"])
    labels = labels_array(frame)
    splits = split_indices(frame, data_cfg["split_column"])
    representations = {
        "AAC": make_aac(frame["Sequence"]),
        "ESM2": load_esm_embeddings(project_root / data_cfg["esm_embeddings"], frame["ACC"]),
    }
    hits = load_mmseqs(project_root / data_cfg["mmseqs_hits"])
    graph_cfg = config["graph"]
    graph_data = build_inductive_graph(
        frame,
        hits,
        float(graph_cfg["min_fident"]),
        float(graph_cfg["min_coverage"]),
        int(graph_cfg["top_k"]),
    )

    results_root = project_root / config["output"]["root"]
    tables_dir = results_root / "tables"
    metrics_dir = results_root / "metrics"
    checkpoints_dir = results_root / "checkpoints"
    for directory in (tables_dir, metrics_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    homophily = graph_homophily(frame, graph_data.filtered_edges)
    save_json(homophily, metrics_dir / "graph_homophily.json")

    test_indices = splits["test"]
    test_connected = graph_data.neighbor_mask[test_indices].any(axis=1)
    strongest = graph_data.strongest_fident[test_indices]
    test_subsets = {
        "overall": np.ones(len(test_indices), dtype=bool),
        "has_train_neighbor": test_connected,
        "no_train_neighbor": ~test_connected,
        "fident_0.30-0.40": test_connected & (strongest >= 0.30) & (strongest < 0.40),
        "fident_0.40-0.60": test_connected & (strongest >= 0.40) & (strongest < 0.60),
        "fident_>=0.60": test_connected & (strongest >= 0.60),
    }

    metric_rows, threshold_rows, per_label_rows = [], [], []
    diagnostics, history_rows = [], []
    model_cfg = config["model"]

    for seed in seeds:
        for representation, features in representations.items():
            print(f"seed={seed} {representation}: baseline ({epochs} epochs)", flush=True)
            baseline_fit = fit_baseline(
                features=features,
                labels=labels,
                train_indices=splits["train"],
                validation_indices=splits["val"],
                seed=seed,
                device=device,
                epochs=epochs,
                hidden_dim=int(model_cfg["baseline_hidden_dim"]),
                dropout=float(model_cfg["dropout"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                train_batch_size=int(training["baseline_batch_size"]),
                evaluation_batch_size=int(training["evaluation_batch_size"]),
            )
            for item in baseline_fit.history:
                history_rows.append({
                    "seed": seed, "representation": representation, "graph": False, **item
                })
            validation_targets, validation_baseline = predict_baseline(
                baseline_fit.model, features, labels, splits["val"], device,
                int(training["evaluation_batch_size"]),
            )
            baseline_thresholds = tune_thresholds(
                validation_targets,
                validation_baseline,
                float(training["threshold_grid_start"]),
                float(training["threshold_grid_stop"]),
                int(training["threshold_grid_steps"]),
            )
            for label, threshold in zip(LABEL_COLUMNS, baseline_thresholds):
                threshold_rows.append({
                    "seed": seed, "representation": representation, "graph": False,
                    "model": representation, "label": label, "threshold": float(threshold),
                    "selected_on": "validation_overall",
                })
            test_targets, test_baseline = predict_baseline(
                baseline_fit.model, features, labels, test_indices, device,
                int(training["evaluation_batch_size"]),
            )
            baseline_validation_predictions = validation_baseline >= baseline_thresholds[None, :]
            baseline_test_predictions = test_baseline >= baseline_thresholds[None, :]

            metric_rows.append(_metric_row(
                seed, representation, False, "validation_overall", validation_targets,
                validation_baseline, baseline_validation_predictions,
            ))
            for subset, mask in test_subsets.items():
                metric_rows.append(_metric_row(
                    seed, representation, False, subset, test_targets[mask],
                    test_baseline[mask], baseline_test_predictions[mask],
                ))
            label_frame = per_label_metrics(
                test_targets, test_baseline, baseline_thresholds, baseline_test_predictions
            )
            label_frame.insert(0, "model", representation)
            label_frame.insert(0, "graph", False)
            label_frame.insert(0, "representation", representation)
            label_frame.insert(0, "seed", seed)
            per_label_rows.append(label_frame)

            baseline_metadata = {
                "seed": seed,
                "representation": representation,
                "graph": False,
                "best_epoch": baseline_fit.best_epoch,
                "validation_macro_auroc": baseline_fit.best_validation_macro_auroc,
                "epochs_run": epochs,
                "thresholds": baseline_thresholds.tolist(),
            }
            _checkpoint(
                checkpoints_dir / f"{representation.lower()}_seed{seed}_baseline.pt",
                baseline_fit.model,
                baseline_metadata,
            )

            print(f"seed={seed} {representation}: graph ({epochs} epochs)", flush=True)
            graph_fit, graph_diagnostics = fit_graph(
                baseline=baseline_fit.model,
                features=features,
                labels=labels,
                neighbor_indices=graph_data.neighbor_indices,
                edge_features=graph_data.edge_features,
                neighbor_mask=graph_data.neighbor_mask,
                train_indices=splits["train"],
                validation_indices=splits["val"],
                seed=seed,
                device=device,
                epochs=epochs,
                attention_hidden_dim=int(model_cfg["attention_hidden_dim"]),
                correction_hidden_dim=int(model_cfg["correction_hidden_dim"]),
                dropout=float(model_cfg["dropout"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                batch_size=int(training["graph_batch_size"]),
            )
            for item in graph_fit.history:
                history_rows.append({
                    "seed": seed, "representation": representation, "graph": True, **item
                })
            bank = torch.as_tensor(features, dtype=torch.float32, device=device)
            validation_loader = make_neighbor_loader(
                splits["val"], labels, graph_data.neighbor_indices, graph_data.edge_features,
                graph_data.neighbor_mask, int(training["graph_batch_size"]), False, seed + 40_000,
            )
            validation_query, validation_graph_targets, validation_graph, validation_internal_base = predict_graph(
                graph_fit.model, bank, validation_loader, device
            )
            validation_connected = graph_data.neighbor_mask[validation_query].any(axis=1)
            graph_thresholds = tune_thresholds(
                validation_graph_targets[validation_connected],
                validation_graph[validation_connected],
                float(training["threshold_grid_start"]),
                float(training["threshold_grid_stop"]),
                int(training["threshold_grid_steps"]),
            )
            for label, threshold in zip(LABEL_COLUMNS, graph_thresholds):
                threshold_rows.append({
                    "seed": seed, "representation": representation, "graph": True,
                    "model": representation + "+Graph", "label": label,
                    "threshold": float(threshold), "selected_on": "validation_connected",
                })
            test_loader = make_neighbor_loader(
                test_indices, labels, graph_data.neighbor_indices, graph_data.edge_features,
                graph_data.neighbor_mask, int(training["graph_batch_size"]), False, seed + 50_000,
            )
            test_query, test_graph_targets, test_graph, test_internal_base = predict_graph(
                graph_fit.model, bank, test_loader, device
            )
            if not np.array_equal(test_query, test_indices) or not np.array_equal(test_targets, test_graph_targets):
                raise RuntimeError("Baseline and graph test rows are misaligned")
            base_alignment_error = float(np.max(np.abs(test_internal_base - test_baseline)))
            no_neighbor_error = float(np.max(np.abs(
                test_graph[~test_connected] - test_internal_base[~test_connected]
            )))
            if base_alignment_error > 1e-6 or no_neighbor_error != 0.0:
                raise RuntimeError(
                    f"Strict fallback check failed: base={base_alignment_error}, no-neighbor={no_neighbor_error}"
                )

            # Store the strict fallback using the baseline arrays themselves. The
            # internal equality was checked above; this also prevents harmless
            # batch-size floating-point differences from appearing as a delta.
            validation_graph[~validation_connected] = validation_baseline[~validation_connected]
            test_graph[~test_connected] = test_baseline[~test_connected]
            graph_validation_predictions = validation_graph >= graph_thresholds[None, :]
            graph_validation_predictions[~validation_connected] = (
                validation_internal_base[~validation_connected] >= baseline_thresholds[None, :]
            )
            graph_test_predictions = test_graph >= graph_thresholds[None, :]
            graph_test_predictions[~test_connected] = baseline_test_predictions[~test_connected]
            metric_rows.append(_metric_row(
                seed, representation, True, "validation_overall", validation_graph_targets,
                validation_graph, graph_validation_predictions,
            ))
            for subset, mask in test_subsets.items():
                metric_rows.append(_metric_row(
                    seed, representation, True, subset, test_graph_targets[mask],
                    test_graph[mask], graph_test_predictions[mask],
                ))
            label_frame = per_label_metrics(
                test_graph_targets, test_graph, graph_thresholds, graph_test_predictions
            )
            label_frame.insert(0, "model", representation + "+Graph")
            label_frame.insert(0, "graph", True)
            label_frame.insert(0, "representation", representation)
            label_frame.insert(0, "seed", seed)
            per_label_rows.append(label_frame)

            graph_metadata = {
                "seed": seed,
                "representation": representation,
                "graph": True,
                "best_epoch": graph_fit.best_epoch,
                "validation_connected_macro_auroc": graph_fit.best_validation_macro_auroc,
                "epochs_run": epochs,
                "thresholds": graph_thresholds.tolist(),
                **graph_diagnostics,
            }
            _checkpoint(
                checkpoints_dir / f"{representation.lower()}_seed{seed}_graph.pt",
                graph_fit.model,
                graph_metadata,
            )
            diagnostics.append({
                **baseline_metadata,
                "baseline_best_epoch": baseline_fit.best_epoch,
                "baseline_validation_macro_auroc": baseline_fit.best_validation_macro_auroc,
                "graph_best_epoch": graph_fit.best_epoch,
                "graph_connected_validation_macro_auroc": graph_fit.best_validation_macro_auroc,
                "epoch0_max_abs_error": graph_diagnostics["epoch0_max_abs_error"],
                "base_alignment_max_abs_error": base_alignment_error,
                "no_neighbor_max_abs_error": no_neighbor_error,
            })
            baseline_auc = multilabel_metrics(
                test_targets, test_baseline, predictions=baseline_test_predictions
            )["macro_auroc"]
            graph_auc = multilabel_metrics(
                test_graph_targets, test_graph, predictions=graph_test_predictions
            )["macro_auroc"]
            print(
                f"seed={seed} {representation}: Macro AUROC {baseline_auc:.4f} -> "
                f"{graph_auc:.4f} ({graph_auc - baseline_auc:+.4f}); "
                f"best epochs {baseline_fit.best_epoch}/{graph_fit.best_epoch}",
                flush=True,
            )
            del baseline_fit, graph_fit, bank
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    test_metrics = metrics[~metrics["subset"].str.startswith("validation_")].copy()
    deltas = paired_deltas(test_metrics)
    summary = summarize_metrics(test_metrics)
    delta_summary = summarize_deltas(deltas)
    metrics.to_csv(metrics_dir / "metrics_by_seed.csv", index=False)
    summary.to_csv(tables_dir / "main_metrics_summary.csv", index=False)
    deltas.to_csv(metrics_dir / "paired_graph_deltas_by_seed.csv", index=False)
    delta_summary.to_csv(tables_dir / "paired_graph_delta_summary.csv", index=False)
    pd.concat(per_label_rows, ignore_index=True).to_csv(metrics_dir / "per_label_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(metrics_dir / "validation_thresholds.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(metrics_dir / "training_diagnostics.csv", index=False)
    pd.DataFrame(history_rows).to_csv(metrics_dir / "training_history.csv", index=False)

    run_metadata = {
        "seeds": seeds,
        "epochs": epochs,
        "device": str(device),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "split_sizes": {name: int(len(index)) for name, index in splits.items()},
        "test_has_train_neighbor": int(test_connected.sum()),
        "test_no_train_neighbor": int((~test_connected).sum()),
        "similarity_bin_sizes": {name: int(mask.sum()) for name, mask in test_subsets.items() if name.startswith("fident_")},
        "config": config,
    }
    save_json(run_metadata, metrics_dir / "run_metadata.json")
    return results_root

