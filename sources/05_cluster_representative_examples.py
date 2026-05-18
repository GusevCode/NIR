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
    return clean[: limit - 1].rstrip() + "…"


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


def write_markdown(path: Path, examples: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = ["# Репрезентативные примеры текстов по кластерам", ""]
    lines.append(
        "Примеры выбраны по близости эмбеддинга текста к центру соответствующего кластера. "
        "Это позволяет проверить не отдельные случайные тексты, а типичное содержание каждой группы."
    )
    lines.append("")

    summary_map = summary.set_index("cluster").to_dict(orient="index")
    for cluster_id, group in examples.groupby("cluster", sort=True):
        info = summary_map.get(cluster_id, {})
        lines.append(f"## Кластер {cluster_id}. {info.get('cluster_name', '')}")
        lines.append("")
        lines.append(f"Ключевые слова: {info.get('cluster_keywords', '')}")
        lines.append("")
        lines.append(
            f"Размер кластера: {info.get('count', '')} текстов; доля: {info.get('share', '')}; "
            f"преобладающая тональность: {info.get('main_sentiment', '')}."
        )
        lines.append("")
        lines.append("| Ранг | ID | Источник | Тональность | Фрагмент текста |")
        lines.append("|---:|---:|---|---|---|")
        for row in group.itertuples(index=False):
            excerpt = str(row.text_excerpt).replace("|", "\\|")
            lines.append(f"| {row.rank} | {row.id} | {row.src} | {row.sentiment} | {excerpt} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select representative texts for each cluster")
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
    summary = pd.read_csv(results_dir / f"{args.output_prefix}cluster_summary.csv", encoding="utf-8-sig")

    csv_path = results_dir / f"{args.output_prefix}cluster_representative_examples.csv"
    md_path = results_dir / f"{args.output_prefix}cluster_representative_examples.md"
    examples.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_markdown(md_path, examples, summary)

    print("Representative examples saved")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(examples[["cluster", "rank", "id", "src", "sentiment", "cluster_name", "text_excerpt"]].to_string(index=False))


if __name__ == "__main__":
    main()
