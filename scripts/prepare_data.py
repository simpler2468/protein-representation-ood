from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_deeploc


if __name__ == "__main__":
    frame = prepare_deeploc(
        ROOT / "data/raw/multisub_5_partitions_unique.csv",
        ROOT / "data/processed/deeploc_multilabel.csv",
    )
    print(frame["split_homology"].value_counts().to_string())

