from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABEL_COLUMNS


MMSEQS_COLUMNS = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"]


@dataclass
class GraphData:
    neighbor_indices: np.ndarray
    edge_features: np.ndarray
    neighbor_mask: np.ndarray
    strongest_fident: np.ndarray
    filtered_edges: pd.DataFrame


def load_mmseqs(path: str | Path) -> pd.DataFrame:
    hits = pd.read_csv(path, sep="\t", header=None, names=MMSEQS_COLUMNS)
    hits["query"] = hits["query"].astype(str)
    hits["target"] = hits["target"].astype(str)
    for column in ("fident", "qcov", "tcov"):
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    hits["min_cov"] = hits[["qcov", "tcov"]].min(axis=1).clip(upper=1.0)
    return hits


def build_inductive_graph(
    frame: pd.DataFrame,
    hits: pd.DataFrame,
    min_fident: float,
    min_coverage: float,
    top_k: int,
) -> GraphData:
    accessions = frame["ACC"].astype(str)
    accession_to_index = {accession: index for index, accession in enumerate(accessions)}
    known = set(accession_to_index)
    filtered = hits.loc[
        (hits["query"] != hits["target"])
        & hits["query"].isin(known)
        & hits["target"].isin(known)
        & (hits["fident"] >= min_fident)
        & (hits["min_cov"] >= min_coverage),
        ["query", "target", "fident", "min_cov"],
    ].copy()
    filtered.drop_duplicates(["query", "target"], keep="first", inplace=True)

    train_accessions = set(frame.loc[frame["split_homology"] == "train", "ACC"].astype(str))
    to_train = filtered[filtered["target"].isin(train_accessions)].copy()
    to_train["rank_score"] = to_train["fident"] * to_train["min_cov"]
    to_train.sort_values(
        ["query", "rank_score", "fident", "min_cov"],
        ascending=[True, False, False, False],
        inplace=True,
    )
    top_edges = to_train.groupby("query", sort=False).head(top_k)

    neighbor_indices = np.zeros((len(frame), top_k), dtype=np.int64)
    edge_features = np.zeros((len(frame), top_k, 2), dtype=np.float32)
    neighbor_mask = np.zeros((len(frame), top_k), dtype=bool)
    for query, group in top_edges.groupby("query", sort=False):
        query_index = accession_to_index[query]
        for position, row in enumerate(group.itertuples(index=False)):
            neighbor_indices[query_index, position] = accession_to_index[row.target]
            edge_features[query_index, position] = [row.fident, row.min_cov]
            neighbor_mask[query_index, position] = True

    strongest = to_train.groupby("query")["fident"].max()
    strongest_fident = accessions.map(strongest).to_numpy(dtype=np.float32)
    return GraphData(neighbor_indices, edge_features, neighbor_mask, strongest_fident, filtered)


def graph_homophily(frame: pd.DataFrame, filtered_edges: pd.DataFrame, seed: int = 42) -> dict:
    train_accessions = frame.loc[
        frame["split_homology"] == "train", "ACC"
    ].astype(str).to_numpy()
    train = set(train_accessions)
    edges = filtered_edges[
        filtered_edges["query"].isin(train) & filtered_edges["target"].isin(train)
    ]
    labels = frame[LABEL_COLUMNS].to_numpy(dtype=np.int8)
    location = {accession: index for index, accession in enumerate(frame["ACC"].astype(str))}
    query_index = np.fromiter((location[value] for value in edges["query"]), dtype=np.int64)
    target_index = np.fromiter((location[value] for value in edges["target"]), dtype=np.int64)
    edge_shared = ((labels[query_index] & labels[target_index]).any(axis=1)).mean()

    # Preserve dataset row order so the seeded random-pair control is stable
    # across Python processes (set iteration order is intentionally unstable).
    train_index = np.fromiter((location[value] for value in train_accessions), dtype=np.int64)
    rng = np.random.default_rng(seed)
    random_query = rng.choice(train_index, size=len(edges), replace=True)
    random_target = rng.choice(train_index, size=len(edges), replace=True)
    random_shared = ((labels[random_query] & labels[random_target]).any(axis=1)).mean()
    return {
        "train_train_edges": int(len(edges)),
        "sequence_edge_shared_label_rate": float(edge_shared),
        "random_pair_shared_label_rate": float(random_shared),
        "definition": "Fraction of pairs sharing at least one positive localization label",
    }

