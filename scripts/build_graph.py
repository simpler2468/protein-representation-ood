from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_dataset
from src.graph import build_inductive_graph, graph_homophily, load_mmseqs
from src.utils import load_config, save_json


if __name__ == "__main__":
    config = load_config(ROOT / "configs/default.yaml")
    frame = load_dataset(ROOT / config["data"]["table"])
    hits = load_mmseqs(ROOT / config["data"]["mmseqs_hits"])
    graph_config = config["graph"]
    graph = build_inductive_graph(
        frame, hits, graph_config["min_fident"], graph_config["min_coverage"], graph_config["top_k"]
    )
    summary = graph_homophily(frame, graph.filtered_edges)
    summary["proteins_with_training_neighbor"] = int(graph.neighbor_mask.any(axis=1).sum())
    output = ROOT / "results/metrics/graph_homophily.json"
    save_json(summary, output)
    print(summary)
