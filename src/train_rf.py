"""
Обучение Random Forest — модели, выбранной в задании специально ради
интерпретируемости (в дополнение к XGBoost/CatBoost, нацеленным на
максимальную точность). Random Forest даёт понятную важность признаков
и относительно устойчив к переобучению без сложной настройки.

Использует то же разбиение train/test по времени и те же признаки, что и
src/train.py и src/train_baseline.py — для честного сравнения на одном
и том же holdout-тесте.

Запуск:
    python src/train_rf.py
    python src/train_rf.py --input data/data_dataset_train.csv

Вход:  data/dataset_train.csv (результат src/dataset.py)
Выход: models/model_random_forest.pkl
       reports/random_forest_metrics.txt
"""

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from train import (  # noqa: E402
    INPUT_PATH, MODELS_DIR, REPORTS_DIR,
    time_based_split, prepare_xy, evaluate_model, print_feature_importance,
)

MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "model_random_forest.pkl")
REPORT_OUTPUT_PATH = os.path.join(REPORTS_DIR, "random_forest_metrics.txt")


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Random Forest с балансировкой классов через встроенный
    class_weight="balanced" (аналогично подходу для XGBoost/CatBoost)."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение Random Forest (для интерпретируемости)."
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

    print("\nОбучение Random Forest...")
    model = train_random_forest(X_train, y_train)
    metrics = evaluate_model("Random Forest", model, X_test, y_test, report_lines)
    print_feature_importance("Random Forest", model.feature_importances_, report_lines)

    joblib.dump(model, MODEL_OUTPUT_PATH)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"\n{'='*60}")
    print(f"Random Forest macro F1: {metrics['macro_f1']:.4f} "
          f"(критерий приёмки: >= 0.70)")
    print(f"Модель сохранена: {MODEL_OUTPUT_PATH}")
    print(f"Отчёт сохранён: {REPORT_OUTPUT_PATH}")
    print("\nRandom Forest ценен здесь не ради максимальной точности, а ради "
          "прозрачности: деревья решений и важность признаков легко "
          "объяснить деканату — в отличие от XGBoost/CatBoost, где логика "
          "решений менее наглядна.")


if __name__ == "__main__":
    main()