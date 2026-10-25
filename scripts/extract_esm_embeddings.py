from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_dataset
from src.embeddings import extract_esm2_embeddings


if __name__ == "__main__":
    frame = load_dataset(ROOT / "data/processed/deeploc_multilabel.csv")
    output = ROOT / "data/processed/esm2_t6_embeddings.npz"
    embeddings = extract_esm2_embeddings(frame, output)
    print(f"Saved {embeddings.shape} embeddings to {output}")

