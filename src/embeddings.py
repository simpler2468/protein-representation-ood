from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def make_aac(sequences: pd.Series) -> np.ndarray:
    features = np.zeros((len(sequences), len(STANDARD_AMINO_ACIDS)), dtype=np.float32)
    for row, sequence in enumerate(sequences.astype(str).str.upper()):
        if not sequence:
            raise ValueError(f"Empty protein sequence at row {row}")
        features[row] = np.fromiter(
            (sequence.count(amino_acid) / len(sequence) for amino_acid in STANDARD_AMINO_ACIDS),
            dtype=np.float32,
        )
    if not np.isfinite(features).all():
        raise ValueError("AAC features contain non-finite values")
    return features


def load_esm_embeddings(path: str | Path, accessions: pd.Series) -> np.ndarray:
    archive = np.load(path, allow_pickle=True)
    embedding_key = next(
        (key for key in ("embeddings", "embedding", "features", "x") if key in archive.files),
        None,
    )
    accession_key = next(
        (key for key in ("accessions", "accession", "acc", "ids", "id") if key in archive.files),
        None,
    )
    if embedding_key is None or accession_key is None:
        raise ValueError(f"Unexpected embedding archive keys: {archive.files}")
    embeddings = archive[embedding_key].astype(np.float32)
    stored_accessions = archive[accession_key].astype(str)
    expected = accessions.astype(str).to_numpy()
    if not np.array_equal(stored_accessions, expected):
        location = {accession: index for index, accession in enumerate(stored_accessions)}
        try:
            embeddings = embeddings[[location[accession] for accession in expected]]
        except KeyError as error:
            raise ValueError(f"Missing ESM2 embedding for {error.args[0]}") from error
    if embeddings.shape != (len(expected), 320):
        raise ValueError(f"Expected {(len(expected), 320)} ESM2 matrix, found {embeddings.shape}")
    return embeddings


def extract_esm2_embeddings(
    frame: pd.DataFrame,
    output_path: str | Path,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    max_residues: int = 1022,
) -> np.ndarray:
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    protein_embeddings = []
    with torch.inference_mode():
        for sequence in tqdm(frame["Sequence"].astype(str), desc="ESM2 embeddings"):
            chunks = [sequence[start:start + max_residues] for start in range(0, len(sequence), max_residues)]
            vectors, lengths = [], []
            for chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt", add_special_tokens=True)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                hidden = model(**inputs).last_hidden_state[:, 1:-1, :]
                vectors.append(hidden.mean(dim=1).squeeze(0))
                lengths.append(len(chunk))
            weights = torch.as_tensor(lengths, dtype=vectors[0].dtype, device=device)
            weights = weights / weights.sum()
            protein_embeddings.append(
                (torch.stack(vectors) * weights[:, None]).sum(dim=0).cpu().numpy()
            )
    embeddings = np.stack(protein_embeddings).astype(np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, embeddings=embeddings, acc=frame["ACC"].to_numpy())
    return embeddings

