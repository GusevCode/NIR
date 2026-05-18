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
            try:
                label_idx = int(label.split("_", 1)[1])
                label = EXPECTED_LABELS[label_idx]
            except Exception:
                label = str(value)
        normalized[idx] = label
    return normalized


def write_report(
    path: Path,
    model_name: str,
    result_prefix: str,
    input_rows: int,
    output_rows: int,
    batch_size: int,
    max_length: int,
    distribution: pd.DataFrame,
    source_table: pd.DataFrame,
) -> None:
    dist_lines = []
    for row in distribution.itertuples(index=False):
        dist_lines.append(f"| {row.sentiment} | {row.count} | {row.share} | {row.avg_score:.4f} |")
    dist_table = "\n".join(dist_lines)

    source_lines = []
    for row in source_table.itertuples(index=False):
        source_lines.append(
            f"| {row.src} | {row.negative} | {row.neutral} | {row.positive} | {row.total} |"
        )
    source_md = "\n".join(source_lines)

    text = f"""# Отчет по шагу 2: анализ тональности

## Используемая модель

`{model_name}`

Модель применяется к очищенному тексту из поля `clean_text` и возвращает один из трех классов тональности: `neutral`, `positive`, `negative`.

## Параметры запуска

| Параметр | Значение |
|---|---:|
| Количество входных текстов | {input_rows} |
| Количество обработанных текстов | {output_rows} |
| Размер пакета | {batch_size} |
| Максимальная длина токенизации | {max_length} |

## Распределение тональности

| Тональность | Количество текстов | Доля | Средняя уверенность модели |
|---|---:|---:|---:|
{dist_table}

## Распределение тональности по источникам

| Источник | negative | neutral | positive | Всего |
|---|---:|---:|---:|---:|
{source_md}

## Созданные файлы

- `результаты/{result_prefix}sentiment_results.csv` - таблица текстов с результатами анализа тональности.
- `результаты/{result_prefix}step2_sentiment_distribution.csv` - распределение тональности по корпусу.
- `результаты/{result_prefix}step2_sentiment_by_source.csv` - распределение тональности по источникам.

## Методический комментарий

На втором этапе каждый очищенный текст был передан в модель тональной классификации. Для каждого текста сохранены итоговая метка тональности, уверенность модели и вероятности трех классов. Эти признаки используются не как окончательная интерпретация текста, а как формализованный оценочный показатель, который затем может сопоставляться с кластерами и ключевыми словами.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sentiment analysis for cleaned NIRS texts")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick test run")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional prefix for output files, for example full_",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    results_dir = PROJECT_ROOT / "результаты"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    required_columns = {"id", "src", "clean_text"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    texts = df["clean_text"].fillna("").astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    id2label = normalize_id2label(model.config.id2label)
    label_to_idx = {label: idx for idx, label in id2label.items()}

    predictions: list[str] = []
    scores: list[float] = []
    probs_by_label = {label: [] for label in EXPECTED_LABELS}

    with torch.no_grad():
        total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
        for batch in tqdm(batches(texts, args.batch_size), total=total_batches, desc="Sentiment batches"):
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1).cpu()
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
    sentiment_out = results_dir / f"{prefix}sentiment_results.csv"
    out.to_csv(sentiment_out, index=False, encoding="utf-8-sig")

    total = len(out)
    distribution = (
        out.groupby("sentiment", as_index=False)
        .agg(count=("id", "count"), avg_score=("sentiment_score", "mean"))
        .sort_values("sentiment")
    )
    distribution["share"] = distribution["count"].map(lambda x: f"{x / total * 100:.2f}%")
    distribution = distribution[["sentiment", "count", "share", "avg_score"]]
    distribution.to_csv(results_dir / f"{prefix}step2_sentiment_distribution.csv", index=False, encoding="utf-8-sig")

    source_table = pd.crosstab(out["src"], out["sentiment"])
    for label in EXPECTED_LABELS:
        if label not in source_table.columns:
            source_table[label] = 0
    source_table = source_table[["negative", "neutral", "positive"]]
    source_table["total"] = source_table.sum(axis=1)
    source_table = source_table.sort_values("total", ascending=False).reset_index()
    source_table.to_csv(results_dir / f"{prefix}step2_sentiment_by_source.csv", index=False, encoding="utf-8-sig")

    report_out = results_dir / f"{prefix}step2_sentiment_report.md"
    write_report(
        report_out,
        args.model,
        result_prefix=prefix,
        input_rows=len(df),
        output_rows=len(out),
        batch_size=args.batch_size,
        max_length=args.max_length,
        distribution=distribution,
        source_table=source_table,
    )

    print("Step 2 completed")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Rows processed: {len(out)}")
    print(f"Saved: {sentiment_out}")
    print(f"Saved: {report_out}")
    print("Distribution:")
    print(distribution.to_string(index=False))


if __name__ == "__main__":
    main()
