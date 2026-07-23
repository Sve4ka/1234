"""
Обучение Logistic Regression — простой базовой модели (baseline) для
сравнения с XGBoost/CatBoost. Использует то же разбиение train/test по
времени и те же признаки, что и src/train.py, чтобы сравнение было
честным (одинаковый holdout-тест).

В отличие от градиентного бустинга, логистическая регрессия чувствительна
к масштабу признаков, поэтому используется StandardScaler внутри Pipeline.
Балансировка классов — через встроенный class_weight="balanced".

Запуск:
    python src/train_baseline.py
    python src/train_baseline.py --input data/data_dataset_train.csv

Вход:  data/dataset_train.csv (результат src/dataset.py)
Выход: models/model_logreg.pkl
       reports/baseline_metrics.txt
"""

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from train import (  # noqa: E402
    INPUT_PATH, MODELS_DIR, REPORTS_DIR,
    time_based_split, prepare_xy, evaluate_model, print_feature_importance,
    build_metrics_summary_md, CATEGORY_ORDER,
)

MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "model_logreg.pkl")
REPORT_OUTPUT_PATH = os.path.join(REPORTS_DIR, "baseline_metrics.txt")


def train_logreg(X_train, y_train) -> Pipeline:
    """Logistic Regression с масштабированием признаков и балансировкой
    классов. Pipeline гарантирует, что скейлер обучается только на train
    и корректно применяется к любым новым данным при инференсе."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def print_logreg_coefficients(model: Pipeline, report_lines: list) -> None:
    """Для логистической регрессии 'важность признака' — это коэффициенты
    (по одному набору на класс, т.к. задача многоклассовая). Печатаем
    средний абсолютный коэффициент по классам — грубый, но простой прокси
    общей значимости признака."""
    from train import FEATURE_COLUMNS
    import numpy as np

    coefs = model.named_steps["logreg"].coef_  # shape: (n_classes, n_features)
    mean_abs_coef = np.abs(coefs).mean(axis=0)
    order = np.argsort(mean_abs_coef)[::-1]

    print("\n--- Коэффициенты Logistic Regression "
          "(средний |coef| по классам) ---")
    report_lines.append("\n--- Коэффициенты Logistic Regression "
                         "(средний |coef| по классам) ---\n")
    for i in order:
        line = f"{FEATURE_COLUMNS[i]:35s} {mean_abs_coef[i]:.4f}"
        print(line)
        report_lines.append(line + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение Logistic Regression (baseline)."
    )
    parser.add_argument("--input", "-i", default=INPUT_PATH,
                        help=f"Путь к train CSV (по умолчанию: {INPUT_PATH})")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(args.input)
    df_train, df_test = time_based_split(df)
    df_train = df_train.sort_values(["sem_key", "hash"]).reset_index(drop=True)

    X_train, y_train = prepare_xy(df_train)
    X_test, y_test = prepare_xy(df_test)

    report_lines = [f"Вход: {args.input}\n",
                    f"Train: {len(df_train)} строк, Test: {len(df_test)} строк\n"]

    print("\nОбучение Logistic Regression (baseline)...")
    model = train_logreg(X_train, y_train)
    metrics = evaluate_model("Logistic Regression (baseline)", model, X_test, y_test, report_lines)
    print_logreg_coefficients(model, report_lines)

    joblib.dump(model, MODEL_OUTPUT_PATH)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"\n{'='*60}")
    print(f"Logistic Regression macro F1: {metrics['macro_f1']:.4f} "
          f"(критерий приёмки: >= 0.70)")
    print(f"Модель сохранена: {MODEL_OUTPUT_PATH}")
    print(f"Отчёт сохранён: {REPORT_OUTPUT_PATH}")
    print("\nСравните это значение с метриками XGBoost/CatBoost "
          "(reports/metrics_summary.md) — именно ради этого и нужен baseline: "
          "показать, насколько сложные модели лучше простой линейной.")


if __name__ == "__main__":
    main()