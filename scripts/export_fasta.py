from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import export_fasta, load_dataset


if __name__ == "__main__":
    frame = load_dataset(ROOT / "data/processed/deeploc_multilabel.csv")
    output = ROOT / "data/processed/deeploc_sequences.fasta"
    export_fasta(frame, output)
    print(f"Wrote {len(frame):,} sequences to {output}")

