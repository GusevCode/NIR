from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "данные" / "clean_sample.csv"
DEFAULT_MODEL = "seara/rubert-tiny2-russian-sentiment"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512
EXPECTED_LABELS = ["neutral", "positive", "negative"]


def batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalize_id2label(raw: dict[int, str]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for key, value in raw.items():
        idx = int(key)
        label = str(value).lower().strip()
        if label.startswith("label_"):
            label_idx = int(label.split("_", 1)[1])
            label = EXPECTED_LABELS[label_idx]
        normalized[idx] = label
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "результаты"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    required_columns = {"id", "src", "clean_text"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if args.limit is not None:
        df = df.head(args.limit).copy()

    texts = df["clean_text"].fillna("").astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    id2label = normalize_id2label(model.config.id2label)
    label_to_idx = {label: idx for idx, label in id2label.items()}
    predictions: list[str] = []
    scores: list[float] = []
    probs_by_label = {label: [] for label in EXPECTED_LABELS}

    with torch.no_grad():
        total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
        for batch in tqdm(batches(texts, args.batch_size), total=total_batches):
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu()
            best_scores, best_indices = probabilities.max(dim=-1)

            for row_probs, best_idx, best_score in zip(probabilities, best_indices, best_scores):
                label = id2label[int(best_idx)]
                predictions.append(label)
                scores.append(float(best_score))
                for expected_label in EXPECTED_LABELS:
                    idx = label_to_idx.get(expected_label)
                    probs_by_label[expected_label].append(float(row_probs[idx]) if idx is not None else 0.0)

    out = df.copy()
    out["sentiment"] = predictions
    out["sentiment_score"] = scores
    out["prob_neutral"] = probs_by_label["neutral"]
    out["prob_positive"] = probs_by_label["positive"]
    out["prob_negative"] = probs_by_label["negative"]

    prefix = args.output_prefix
    out.to_csv(results_dir / f"{prefix}sentiment_results.csv", index=False, encoding="utf-8-sig")

    distribution = (
        out.groupby("sentiment", as_index=False)
        .agg(count=("id", "count"), avg_score=("sentiment_score", "mean"))
        .sort_values("sentiment")
    )
    distribution["share"] = distribution["count"].map(lambda value: f"{value / len(out) * 100:.2f}%")
    distribution = distribution[["sentiment", "count", "share", "avg_score"]]
    distribution.to_csv(results_dir / f"{prefix}step2_sentiment_distribution.csv", index=False, encoding="utf-8-sig")

    source_table = pd.crosstab(out["src"], out["sentiment"])
    for label in EXPECTED_LABELS:
        if label not in source_table.columns:
            source_table[label] = 0
    source_table = source_table[["negative", "neutral", "positive"]]
    source_table["total"] = source_table.sum(axis=1)
    source_table.sort_values("total", ascending=False).reset_index().to_csv(
        results_dir / f"{prefix}step2_sentiment_by_source.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
