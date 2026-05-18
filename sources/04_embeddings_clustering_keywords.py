from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "результаты" / "full_sentiment_results.csv"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_BATCH_SIZE = 64
DEFAULT_K_VALUES = "5,7,10,12"
RANDOM_STATE = 42

RUSSIAN_STOPWORDS = {
    "а",
    "без",
    "более",
    "бы",
    "был",
    "была",
    "были",
    "было",
    "быть",
    "в",
    "вам",
    "вас",
    "весь",
    "во",
    "вот",
    "все",
    "всего",
    "всех",
    "вы",
    "где",
    "да",
    "даже",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "есть",
    "еще",
    "же",
    "за",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "ко",
    "когда",
    "кто",
    "ли",
    "либо",
    "мне",
    "может",
    "мы",
    "на",
    "надо",
    "наш",
    "не",
    "него",
    "нее",
    "нет",
    "ни",
    "них",
    "но",
    "ну",
    "о",
    "об",
    "однако",
    "он",
    "она",
    "они",
    "оно",
    "от",
    "очень",
    "по",
    "под",
    "при",
    "с",
    "со",
    "так",
    "также",
    "такой",
    "там",
    "то",
    "тоже",
    "только",
    "у",
    "уже",
    "хотя",
    "чем",
    "что",
    "чтобы",
    "это",
    "этого",
    "этой",
    "этом",
    "этот",
    "я",
}


def stratified_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df.copy()

    frac = sample_size / len(df)
    parts = []
    for _, group in df.groupby("src", sort=False):
        n = max(1, round(len(group) * frac))
        parts.append(group.sample(n=min(n, len(group)), random_state=RANDOM_STATE))

    sampled = pd.concat(parts)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=RANDOM_STATE)
    elif len(sampled) < sample_size:
        remaining = df.drop(index=sampled.index, errors="ignore")
        if len(remaining) > 0:
            extra = remaining.sample(n=min(sample_size - len(sampled), len(remaining)), random_state=RANDOM_STATE)
            sampled = pd.concat([sampled, extra])
    return sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def parse_k_values(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def choose_cluster_count(embeddings: np.ndarray, k_values: list[int]) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=1024,
            n_init=10,
        )
        labels = model.fit_predict(embeddings)
        score = silhouette_score(
            embeddings,
            labels,
            metric="cosine",
            sample_size=min(3000, len(embeddings)),
            random_state=RANDOM_STATE,
        )
        rows.append({"k": k, "silhouette": score, "inertia": model.inertia_})
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)


def extract_cluster_keywords(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    cluster_docs = (
        df.groupby("cluster", as_index=False)
        .agg(cluster_text=("clean_text", lambda values: " ".join(values.astype(str))))
        .sort_values("cluster")
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(RUSSIAN_STOPWORDS),
        token_pattern=r"(?u)\b[а-яА-ЯёЁa-zA-Z][а-яА-ЯёЁa-zA-Z-]{2,}\b",
        ngram_range=(1, 2),
        min_df=1,
        max_features=30000,
    )
    matrix = vectorizer.fit_transform(cluster_docs["cluster_text"])
    terms = np.array(vectorizer.get_feature_names_out())

    rows = []
    for row_idx, cluster_id in enumerate(cluster_docs["cluster"]):
        values = matrix[row_idx].toarray().ravel()
        top_indices = values.argsort()[::-1][:top_n]
        keywords = [terms[i] for i in top_indices if values[i] > 0]
        rows.append({"cluster": cluster_id, "cluster_keywords": ", ".join(keywords)})
    return pd.DataFrame(rows)


def make_cluster_name(keywords: str) -> str:
    first_terms = [item.strip() for item in keywords.split(",")[:3] if item.strip()]
    if not first_terms:
        return "Кластер без выраженных ключевых слов"
    return " / ".join(first_terms)


def write_report(
    path: Path,
    model_name: str,
    input_rows: int,
    analyzed_rows: int,
    selected_k: int,
    k_table: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> None:
    k_lines = "\n".join(
        f"| {row.k} | {row.silhouette:.4f} | {row.inertia:.2f} |"
        for row in k_table.sort_values("k").itertuples(index=False)
    )

    cluster_lines = "\n".join(
        f"| {row.cluster} | {row.count} | {row.share} | {row.main_sentiment} | {row.cluster_name} | {row.cluster_keywords} |"
        for row in cluster_summary.itertuples(index=False)
    )

    text = f"""# Отчет по шагу 3: эмбеддинги, кластеризация и ключевые слова

## Используемая модель эмбеддингов

`{model_name}`

Модель преобразует очищенный текст в числовой вектор. Далее векторы используются для группировки текстов по близости.

## Параметры запуска

| Параметр | Значение |
|---|---:|
| Количество строк во входной таблице | {input_rows} |
| Количество анализируемых текстов | {analyzed_rows} |
| Выбранное количество кластеров | {selected_k} |

## Сравнение числа кластеров

| k | Силуэтный коэффициент | Инерция MiniBatchKMeans |
|---:|---:|---:|
{k_lines}

## Сводка по кластерам

| Кластер | Количество текстов | Доля | Преобладающая тональность | Рабочее название | Ключевые слова |
|---:|---:|---:|---|---|---|
{cluster_lines}

## Методический комментарий

На данном этапе тексты были представлены в виде эмбеддингов, после чего применен алгоритм MiniBatchKMeans. Количество кластеров выбрано по максимальному силуэтному коэффициенту из проверенных значений. Ключевые слова рассчитаны через TF-IDF по объединенным текстам каждого кластера и используются для первичной интерпретации смысловых групп.
"""
    path.write_text(text, encoding="utf-8")


def make_charts(df: pd.DataFrame, cluster_summary: pd.DataFrame, output_prefix: str) -> None:
    figures_dir = PROJECT_ROOT / "рисунки"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cluster_summary["cluster"].astype(str), cluster_summary["count"], color="#34699a")
    ax.set_title("Распределение текстов по смысловым кластерам")
    ax.set_xlabel("Кластер")
    ax.set_ylabel("Количество текстов")
    ax.grid(axis="y", alpha=0.25)
    for bar, share in zip(bars, cluster_summary["share"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), share, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{output_prefix}cluster_distribution.png", dpi=200)
    plt.close(fig)

    sentiment_order = ["negative", "neutral", "positive"]
    crosstab = pd.crosstab(df["cluster"], df["sentiment"])
    for sentiment in sentiment_order:
        if sentiment not in crosstab.columns:
            crosstab[sentiment] = 0
    crosstab = crosstab[sentiment_order]
    shares = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    left = np.zeros(len(shares))
    colors = {"negative": "#c0392b", "neutral": "#7f8c8d", "positive": "#27ae60"}
    labels = {"negative": "Отрицательная", "neutral": "Нейтральная", "positive": "Положительная"}
    for sentiment in sentiment_order:
        ax.barh(shares.index.astype(str), shares[sentiment], left=left, color=colors[sentiment], label=labels[sentiment])
        left += shares[sentiment].values
    ax.set_title("Структура тональности по смысловым кластерам")
    ax.set_xlabel("Доля текстов, %")
    ax.set_ylabel("Кластер")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{output_prefix}cluster_sentiment_structure.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embeddings, clusters and keywords")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--k-values", type=str, default=DEFAULT_K_VALUES)
    parser.add_argument("--output-prefix", type=str, default="sample_")
    parser.add_argument("--top-keywords", type=int, default=12)
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "результаты"
    data_dir = PROJECT_ROOT / "данные"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    usecols = ["id", "src", "clean_text", "sentiment", "sentiment_score"]
    source_df = pd.read_csv(args.input, usecols=usecols, encoding="utf-8-sig")
    source_df = source_df.dropna(subset=["clean_text"]).copy()
    df = stratified_sample(source_df, args.sample_size)
    df["clean_text"] = df["clean_text"].astype(str)

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        df["clean_text"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(data_dir / f"{args.output_prefix}embeddings.npy", embeddings)

    k_values = parse_k_values(args.k_values)
    k_table = choose_cluster_count(embeddings, k_values)
    selected_k = int(k_table.iloc[0]["k"])

    cluster_model = MiniBatchKMeans(
        n_clusters=selected_k,
        random_state=RANDOM_STATE,
        batch_size=1024,
        n_init=20,
    )
    df["cluster"] = cluster_model.fit_predict(embeddings)

    keywords = extract_cluster_keywords(df, args.top_keywords)

    sentiment_by_cluster = pd.crosstab(df["cluster"], df["sentiment"])
    main_sentiment = sentiment_by_cluster.idxmax(axis=1).rename("main_sentiment").reset_index()
    cluster_summary = (
        df.groupby("cluster", as_index=False)
        .agg(count=("id", "count"))
        .merge(main_sentiment, on="cluster", how="left")
        .merge(keywords, on="cluster", how="left")
        .sort_values("cluster")
    )
    cluster_summary["share"] = cluster_summary["count"].map(lambda value: f"{value / len(df) * 100:.2f}%")
    cluster_summary["cluster_name"] = cluster_summary["cluster_keywords"].fillna("").map(make_cluster_name)
    cluster_summary = cluster_summary[
        ["cluster", "count", "share", "main_sentiment", "cluster_name", "cluster_keywords"]
    ]

    df = df.merge(cluster_summary[["cluster", "cluster_name", "cluster_keywords"]], on="cluster", how="left")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(embeddings)
    df["x_pca"] = coords[:, 0]
    df["y_pca"] = coords[:, 1]

    result_path = results_dir / f"{args.output_prefix}clustered_texts.csv"
    summary_path = results_dir / f"{args.output_prefix}cluster_summary.csv"
    k_path = results_dir / f"{args.output_prefix}cluster_k_selection.csv"
    report_path = results_dir / f"{args.output_prefix}step3_clustering_report.md"

    df.to_csv(result_path, index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    k_table.sort_values("k").to_csv(k_path, index=False, encoding="utf-8-sig")

    write_report(
        report_path,
        model_name=args.model,
        input_rows=len(source_df),
        analyzed_rows=len(df),
        selected_k=selected_k,
        k_table=k_table,
        cluster_summary=cluster_summary,
    )
    make_charts(df, cluster_summary, args.output_prefix)

    print("Step 3 completed")
    print(f"Model: {args.model}")
    print(f"Input rows: {len(source_df)}")
    print(f"Analyzed rows: {len(df)}")
    print(f"Selected k: {selected_k}")
    print(f"Saved: {result_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")
    print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()
