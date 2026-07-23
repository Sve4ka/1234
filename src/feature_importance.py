"""
Экспорт важности признаков финальной модели (models/model.pkl) для
интерпретируемости и финального отчёта проекта.

Работает как с XGBoost (feature_importances_), так и с CatBoost
(get_feature_importance()) — тип модели определяется автоматически.

Запуск:
    python src/feature_importance.py

Вход:  models/model.pkl
Выход: reports/feature_importance.csv   (таблица: признак, важность, ранг, доля %)
       reports/feature_importance.md    (та же таблица в markdown — для отчёта)
       reports/feature_importance.png   (горизонтальная столбчатая диаграмма)
"""

import argparse
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # без графического дисплея (headless-режим)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train import FEATURE_COLUMNS, MODELS_DIR, REPORTS_DIR  # noqa: E402

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
CSV_OUTPUT_PATH = os.path.join(REPORTS_DIR, "feature_importance.csv")
MD_OUTPUT_PATH = os.path.join(REPORTS_DIR, "feature_importance.md")
PNG_OUTPUT_PATH = os.path.join(REPORTS_DIR, "feature_importance.png")


def get_importances(model) -> np.ndarray:
    """Достаёт важность признаков независимо от типа модели
    (XGBoost или CatBoost)."""
    if hasattr(model, "feature_importances_"):
        # XGBoost (и вообще sklearn-совместимые модели)
        return np.asarray(model.feature_importances_)
    elif hasattr(model, "get_feature_importance"):
        # CatBoost
        return np.asarray(model.get_feature_importance())
    else:
        raise TypeError(f"Не знаю, как получить важность признаков "
                         f"для модели типа {type(model)}")


def build_importance_table(importances: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": importances,
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    total = df["importance"].sum()
    df["share_pct"] = (df["importance"] / total * 100).round(2) if total > 0 else 0.0
    return df[["rank", "feature", "importance", "share_pct"]]


def save_markdown(df: pd.DataFrame, model_type_name: str, path: str) -> None:
    lines = [f"# Важность признаков — {model_type_name}\n"]
    lines.append("| # | Признак | Важность | Доля, % |")
    lines.append("|---|---|---|---|")
    for _, row in df.iterrows():
        lines.append(f"| {int(row['rank'])} | {row['feature']} | "
                      f"{row['importance']:.4f} | {row['share_pct']:.1f}% |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_plot(df: pd.DataFrame, model_type_name: str, path: str) -> None:
    df_sorted = df.sort_values("importance", ascending=True)  # для barh снизу вверх
    fig_height = max(4, 0.35 * len(df_sorted))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(df_sorted["feature"], df_sorted["importance"], color="#4C72B0")
    ax.set_xlabel("Важность признака")
    ax.set_title(f"Важность признаков — {model_type_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Экспорт важности признаков финальной модели для отчёта."
    )
    parser.add_argument("--model", "-m", default=MODEL_PATH,
                        help=f"Путь к модели (по умолчанию: {MODEL_PATH})")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    model = joblib.load(args.model)
    model_type_name = type(model).__name__

    importances = get_importances(model)
    df = build_importance_table(importances)

    print(f"Модель: {model_type_name}")
    print(df.to_string(index=False))

    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    save_markdown(df, model_type_name, MD_OUTPUT_PATH)
    save_plot(df, model_type_name, PNG_OUTPUT_PATH)

    print(f"\nСохранено:\n  {CSV_OUTPUT_PATH}\n  {MD_OUTPUT_PATH}\n  {PNG_OUTPUT_PATH}")


if __name__ == "__main__":
    main()