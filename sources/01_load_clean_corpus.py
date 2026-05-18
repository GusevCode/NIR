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


def write_report(
    report_path: Path,
    input_path: Path,
    initial_rows: int,
    empty_rows: int,
    duplicate_rows: int,
    final_rows: int,
    sample_rows: int,
    source_distribution: pd.DataFrame,
    length_stats: pd.Series,
    word_stats: pd.Series,
) -> None:
    src_rows = []
    for row in source_distribution.itertuples(index=False):
        src_rows.append(f"| {row.src} | {row.count} | {row.share} |")
    src_table = "\n".join(src_rows)

    text = f"""# Отчет по шагу 1: загрузка и очистка корпуса

## Входной файл

`{input_path.as_posix()}`

## Результаты проверки и очистки

| Показатель | Значение |
|---|---:|
| Исходное количество строк | {initial_rows} |
| Пустые тексты после очистки | {empty_rows} |
| Удаленные дубли по очищенному тексту | {duplicate_rows} |
| Количество строк после очистки | {final_rows} |
| Размер рабочей выборки | {sample_rows} |

## Длина очищенных текстов

| Показатель | Символы | Слова |
|---|---:|---:|
| Среднее | {length_stats['mean']:.1f} | {word_stats['mean']:.1f} |
| Медиана | {length_stats['50%']:.0f} | {word_stats['50%']:.0f} |
| Минимум | {length_stats['min']:.0f} | {word_stats['min']:.0f} |
| Максимум | {length_stats['max']:.0f} | {word_stats['max']:.0f} |

## Распределение очищенного корпуса по источникам

| Источник | Количество текстов | Доля |
|---|---:|---:|
{src_table}

## Созданные файлы

- `данные/clean_sample.csv` - рабочая выборка для дальнейшей кластеризации и анализа тональности.
- `данные/clean_full_metadata.csv` - очищенные метаданные полного корпуса без лишних исходных полей.
- `результаты/step1_source_distribution.csv` - распределение текстов по источникам.

## Методический комментарий

На первом этапе корпус был загружен из CSV-файла, после чего тексты были очищены от технических элементов: переносов строк, HTML-разметки, ссылок и лишних пробелов. Затем были удалены пустые строки и полные дубли по очищенному тексту. Для дальнейших экспериментов сформирована рабочая выборка, поскольку полный корпус содержит сотни тысяч записей и требует значительных вычислительных ресурсов при обработке нейросетевыми моделями.
"""
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and clean NIRS text corpus")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    input_path = args.input.resolve()
    data_dir = PROJECT_ROOT / "данные"
    results_dir = PROJECT_ROOT / "результаты"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    initial_rows = len(df)

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

    empty_mask = work["clean_text"].str.len() == 0
    empty_rows = int(empty_mask.sum())
    work = work.loc[~empty_mask].copy()

    duplicate_rows = int(work.duplicated("clean_text").sum())
    work = work.drop_duplicates("clean_text", keep="first").reset_index(drop=True)
    final_rows = len(work)

    source_counts = work["src"].value_counts().rename_axis("src").reset_index(name="count")
    source_counts["share"] = source_counts["count"].map(lambda x: format_percent(int(x), final_rows))

    sample_size = min(args.sample_size, final_rows)
                                                                                             
    sample_parts = []
    used_indices: set[int] = set()
    for _, part in work.groupby("src", sort=False):
        n = max(1, round(len(part) / final_rows * sample_size))
        n = min(n, len(part))
        part_sample = part.sample(n=n, random_state=args.random_state)
        sample_parts.append(part_sample)
        used_indices.update(part_sample.index.tolist())

    sample = pd.concat(sample_parts, ignore_index=False) if sample_parts else work.head(0)
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=args.random_state)
    elif len(sample) < sample_size:
        remaining = work.drop(index=list(used_indices), errors="ignore")
        add_n = min(sample_size - len(sample), len(remaining))
        if add_n > 0:
            sample = pd.concat([
                sample,
                remaining.sample(n=add_n, random_state=args.random_state),
            ], ignore_index=False)
    sample = sample.sort_values(["src", "id"]).reset_index(drop=True)

    full_out = data_dir / "clean_full_metadata.csv"
    sample_out = data_dir / "clean_sample.csv"
    dist_out = results_dir / "step1_source_distribution.csv"
    report_out = results_dir / "step1_cleaning_report.md"

    work.to_csv(full_out, index=False, encoding="utf-8-sig")
    sample.to_csv(sample_out, index=False, encoding="utf-8-sig")
    source_counts.to_csv(dist_out, index=False, encoding="utf-8-sig")

    length_stats = work["char_count"].describe()
    word_stats = work["word_count"].describe()
    write_report(
        report_out,
        input_path,
        initial_rows,
        empty_rows,
        duplicate_rows,
        final_rows,
        len(sample),
        source_counts,
        length_stats,
        word_stats,
    )

    print("Step 1 completed")
    print(f"Input rows: {initial_rows}")
    print(f"Empty rows removed: {empty_rows}")
    print(f"Duplicates removed: {duplicate_rows}")
    print(f"Rows after cleaning: {final_rows}")
    print(f"Sample rows: {len(sample)}")
    print(f"Saved: {sample_out}")
    print(f"Saved: {full_out}")
    print(f"Saved: {report_out}")


if __name__ == "__main__":
    main()

