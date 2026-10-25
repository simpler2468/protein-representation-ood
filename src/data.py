from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from .utils import seeded_generator


LABEL_COLUMNS = [
    "Membrane",
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"ACC", "Sequence", "Partition", "split_homology", *LABEL_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if frame["ACC"].duplicated().any():
        raise ValueError("Protein accessions must be unique")
    return frame


def labels_array(frame: pd.DataFrame) -> np.ndarray:
    return frame[LABEL_COLUMNS].to_numpy(dtype=np.float32)


def split_indices(frame: pd.DataFrame, column: str = "split_homology") -> dict[str, np.ndarray]:
    split = frame[column].astype(str).to_numpy()
    indices = {name: np.flatnonzero(split == name) for name in ("train", "val", "test")}
    if any(len(indices[name]) == 0 for name in indices):
        raise ValueError(f"Incomplete split in column {column}: { {k: len(v) for k, v in indices.items()} }")
    return indices


def make_feature_loader(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(features[indices], dtype=torch.float32),
        torch.as_tensor(labels[indices], dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=seeded_generator(seed),
    )


def prepare_deeploc(raw_path: str | Path, output_path: str | Path, seed: int = 42) -> pd.DataFrame:
    frame = pd.read_csv(raw_path).drop(columns=["Unnamed: 0"], errors="ignore")
    frame["sequence_length"] = frame["Sequence"].str.len()
    frame["num_labels"] = frame[LABEL_COLUMNS].sum(axis=1).astype(int)
    targets = frame[LABEL_COLUMNS].to_numpy()
    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_index, temporary_index = next(first.split(np.zeros(len(frame)), targets))
    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    validation_local, test_local = next(
        second.split(np.zeros(len(temporary_index)), targets[temporary_index])
    )
    random_split = np.empty(len(frame), dtype=object)
    random_split[train_index] = "train"
    random_split[temporary_index[validation_local]] = "val"
    random_split[temporary_index[test_local]] = "test"
    frame["split_random"] = random_split
    frame["split_homology"] = frame["Partition"].map(
        {0: "train", 1: "train", 2: "train", 3: "val", 4: "test"}
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def export_fasta(frame: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for accession, sequence in frame[["ACC", "Sequence"]].itertuples(index=False):
            handle.write(f">{accession}\n{sequence}\n")

