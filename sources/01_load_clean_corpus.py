from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "csv" / "sentiment_dataset.csv"
DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_RANDOM_STATE = 42


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def format_percent(part: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{part / total * 100:.2f}%"


def make_stratified_sample(work: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    sample_size = min(sample_size, len(work))
    sample_parts = []
    used_indices: set[int] = set()

    for _, part in work.groupby("src", sort=False):
        n = max(1, round(len(part) / len(work) * sample_size))
        n = min(n, len(part))
        part_sample = part.sample(n=n, random_state=random_state)
        sample_parts.append(part_sample)
        used_indices.update(part_sample.index.tolist())

    sample = pd.concat(sample_parts, ignore_index=False) if sample_parts else work.head(0)
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=random_state)
    elif len(sample) < sample_size:
        remaining = work.drop(index=list(used_indices), errors="ignore")
        add_n = min(sample_size - len(sample), len(remaining))
        if add_n > 0:
            sample = pd.concat([sample, remaining.sample(n=add_n, random_state=random_state)], ignore_index=False)

    return sample.sort_values(["src", "id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    input_path = args.input.resolve()
    data_dir = PROJECT_ROOT / "данные"
    results_dir = PROJECT_ROOT / "результаты"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    required_columns = {"text", "src"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.loc[:, ["text", "src"]].copy()
    work.insert(0, "id", range(1, len(work) + 1))
    work["src"] = work["src"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    work["clean_text"] = work["text"].map(clean_text)
    work["char_count"] = work["clean_text"].str.len()
    work["word_count"] = work["clean_text"].map(word_count)
    work = work.loc[work["clean_text"].str.len() > 0].copy()
    work = work.drop_duplicates("clean_text", keep="first").reset_index(drop=True)

    source_counts = work["src"].value_counts().rename_axis("src").reset_index(name="count")
    source_counts["share"] = source_counts["count"].map(lambda value: format_percent(int(value), len(work)))

    sample = make_stratified_sample(work, args.sample_size, args.random_state)

    work.to_csv(data_dir / "clean_full_metadata.csv", index=False, encoding="utf-8-sig")
    sample.to_csv(data_dir / "clean_sample.csv", index=False, encoding="utf-8-sig")
    source_counts.to_csv(results_dir / "step1_source_distribution.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
