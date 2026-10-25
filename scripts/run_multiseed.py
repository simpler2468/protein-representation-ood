from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment import run_experiment
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unified four-model multi-seed experiment.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output = run_experiment(PROJECT_ROOT, config, args.seeds, args.epochs, args.device)
    print(f"Final results written to {output}")


if __name__ == "__main__":
    main()
