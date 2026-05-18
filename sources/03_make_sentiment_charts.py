import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "результаты"
FIGURES_DIR = BASE_DIR / "рисунки"


def make_distribution_chart(prefix: str, title_suffix: str) -> None:
    df = pd.read_csv(RESULTS_DIR / f"{prefix}step2_sentiment_distribution.csv")
    labels = {"negative": "Отрицательная", "neutral": "Нейтральная", "positive": "Положительная"}
    colors = {"negative": "#c0392b", "neutral": "#7f8c8d", "positive": "#27ae60"}

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([labels[item] for item in df["sentiment"]], df["count"], color=[colors[item] for item in df["sentiment"]])
    ax.set_title(f"Распределение тональности{title_suffix}")
    ax.set_xlabel("Тональность")
    ax.set_ylabel("Количество текстов")
    ax.grid(axis="y", alpha=0.25)

    for bar, share in zip(bars, df["share"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), share, ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{prefix}step2_sentiment_distribution.png", dpi=200)
    plt.close(fig)


def make_by_source_chart(prefix: str, title_suffix: str) -> None:
    df = pd.read_csv(RESULTS_DIR / f"{prefix}step2_sentiment_by_source.csv").sort_values("total", ascending=True)
    shares = df[["negative", "neutral", "positive"]].div(df["total"], axis=0) * 100
    colors = {"negative": "#c0392b", "neutral": "#7f8c8d", "positive": "#27ae60"}
    labels = {"negative": "Отрицательная", "neutral": "Нейтральная", "positive": "Положительная"}

    fig, ax = plt.subplots(figsize=(10, 6))
    left = pd.Series([0.0] * len(df), index=df.index)
    for column in ["negative", "neutral", "positive"]:
        ax.barh(df["src"], shares[column], left=left, color=colors[column], label=labels[column])
        left += shares[column]

    ax.set_title(f"Структура тональности по источникам{title_suffix}")
    ax.set_xlabel("Доля текстов, %")
    ax.set_ylabel("Источник")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{prefix}step2_sentiment_by_source.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--title-suffix", type=str, default=" в рабочей выборке")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_distribution_chart(args.prefix, args.title_suffix)
    make_by_source_chart(args.prefix, args.title_suffix)


if __name__ == "__main__":
    main()
