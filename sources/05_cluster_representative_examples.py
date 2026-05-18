from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTERED = PROJECT_ROOT / "результаты" / "sample_clustered_texts.csv"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "данные" / "sample_embeddings.npy"


def shorten(text: str, limit: int = 260) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def representative_examples(df: pd.DataFrame, embeddings: np.ndarray, examples_per_cluster: int) -> pd.DataFrame:
    rows = []
    for cluster_id, group in df.groupby("cluster", sort=True):
        group_indices = group.index.to_numpy()
        cluster_embeddings = embeddings[group_indices]
        centroid = cluster_embeddings.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        distances = 1 - np.dot(cluster_embeddings, centroid)
        order = np.argsort(distances)[:examples_per_cluster]

        for rank, local_idx in enumerate(order, start=1):
            source_row = group.iloc[int(local_idx)]
            rows.append(
                {
                    "cluster": int(cluster_id),
                    "rank": rank,
                    "id": int(source_row["id"]),
                    "src": source_row["src"],
                    "sentiment": source_row["sentiment"],
                    "sentiment_score": float(source_row["sentiment_score"]),
                    "cluster_name": source_row["cluster_name"],
                    "distance_to_centroid": float(distances[int(local_idx)]),
                    "text_excerpt": shorten(source_row["clean_text"]),
                    "clean_text": source_row["clean_text"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustered", type=Path, default=DEFAULT_CLUSTERED)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--examples-per-cluster", type=int, default=6)
    parser.add_argument("--output-prefix", type=str, default="sample_")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "результаты"
    df = pd.read_csv(args.clustered, encoding="utf-8-sig")
    embeddings = np.load(args.embeddings)
    if len(df) != len(embeddings):
        raise ValueError(f"Rows and embeddings mismatch: {len(df)} != {len(embeddings)}")

    examples = representative_examples(df, embeddings, args.examples_per_cluster)
    examples.to_csv(results_dir / f"{args.output_prefix}cluster_representative_examples.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
