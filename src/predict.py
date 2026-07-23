"""
Инференс: применение обученной модели (models/model.pkl) к новому
Excel-файлу с оценками — тому самому сценарию из задания "Загрузка нового
файла с данными об успеваемости" -> "ранжированный список студентов с
прогнозами".

КРИТИЧЕСКИ ВАЖНО: признаки сложности (avg_discipline_difficulty,
group_difficulty, specialty_difficulty) строятся с использованием карт,
СОХРАНЁННЫХ ПРИ ОБУЧЕНИИ (models/difficulty_maps.pkl), а не пересчитываются
заново по новому файлу. Если новый файл содержит, например, только одну
группу — пересчёт с нуля дал бы вырожденные, неинформативные значения,
не совместимые с тем, на чём обучалась модель (это подтверждено тестом на
изолированной выгрузке одной группы — см. reports/new_file_test_notes.md).

Запуск:
    python src/predict.py --input data/new_upload.xlsx

Вход:  models/model.pkl
       models/difficulty_maps.pkl
       путь к новому Excel-файлу (сырые данные, формат как data.xlsx)
Выход: data/<имя_файла>_predictions.xlsx — ранжированный список студентов
       с прогнозируемой категорией и вероятностями по каждому классу
"""

import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, load_difficulty_maps, DIFFICULTY_MAPS_PATH  # noqa: E402

INPUT_PATH = "data/dataset_train.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

# Признаки, которые реально идут в модель — см. reports/feature_contract.md
FEATURE_COLUMNS = [
    "n_records", "n_distinct_disciplines",
    "n_otlichno", "n_horosho", "n_udovl", "n_neudovl", "n_neyavka", "n_nezachteno",
    "n_debts", "n_zachet_rows", "n_zachteno",
    "avg_score", "avg_discipline_difficulty", "group_difficulty", "specialty_difficulty",
    "n_retake_rows_in_semester", "has_retake_in_semester", "share_zachteno",
    "semester_number", "prev_avg_score", "score_trend",
]

TARGET_COLUMN = "target_category"
CATEGORY_ORDER = ["0", "1", "2", "3+"]
CATEGORY_TO_INT = {c: i for i, c in enumerate(CATEGORY_ORDER)}
INT_TO_CATEGORY = {i: c for c, i in CATEGORY_TO_INT.items()}

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")


def _default_output_path(input_path: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join("data", f"{base}_predictions.xlsx")


def prepare_x_for_inference(features_df: pd.DataFrame, train_medians: pd.Series = None) -> pd.DataFrame:
    X = features_df[FEATURE_COLUMNS].copy()
    if train_medians is not None:
        X = X.fillna(train_medians)
    else:
        # fallback, если медианы обучения не переданы — использовать
        # медиану внутри самого нового файла (хуже, но лучше, чем ошибка)
        X = X.fillna(X.median(numeric_only=True))
    return X


def run_inference(input_path: str, model_path: str = MODEL_PATH,
                   difficulty_maps_path: str = DIFFICULTY_MAPS_PATH) -> pd.DataFrame:
    model = joblib.load(model_path)
    diff_maps = load_difficulty_maps(difficulty_maps_path)

    # Признаки строятся с картами сложности из ОБУЧЕНИЯ, не из нового файла
    features_df = build_features(input_path, difficulty_maps=diff_maps)

    # Для прогноза берём последний известный семестр каждого студента —
    # именно про него нет данных о будущем, это и есть "предстоящая сессия"
    last_semester_idx = features_df.groupby("hash")["sem_key"].idxmax()
    to_predict = features_df.loc[last_semester_idx].reset_index(drop=True)

    X = prepare_x_for_inference(to_predict)

    pred_class_idx = model.predict(X)
    pred_proba = model.predict_proba(X)

    result = to_predict[["hash", "Номер ЛД", "Учебная группа",
                          "Специальность/направление", "sem_key"]].copy() \
        if "Номер ЛД" in to_predict.columns else \
        to_predict[["hash", "Учебная группа", "Специальность/направление", "sem_key"]].copy()

    result["predicted_category"] = [CATEGORY_ORDER[int(i)] for i in pred_class_idx]
    for i, cls in enumerate(CATEGORY_ORDER):
        result[f"probability_{cls}"] = pred_proba[:, i]

    # Сортировка по убыванию риска: сначала по категории (3+ > 2 > 1 > 0),
    # внутри категории — по вероятности "3+"
    category_rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    result["_rank"] = result["predicted_category"].map(category_rank)
    result = result.sort_values(
        ["_rank", "probability_3+"], ascending=[False, False]
    ).drop(columns="_rank").reset_index(drop=True)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Прогноз задолженностей на предстоящую сессию по "
                    "новому Excel-файлу с оценками."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Путь к новому Excel-файлу с оценками")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"Путь к модели (по умолчанию: {MODEL_PATH})")
    parser.add_argument("--output", "-o", default=None,
                        help="Путь к выходному Excel. Если не указан — "
                             "строится автоматически по имени входного файла.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or _default_output_path(args.input)

    result = run_inference(args.input, model_path=args.model)

    print(f"Вход: {args.input}")
    print(f"Студентов в прогнозной выборке: {len(result)}")
    print("\n--- Распределение прогнозируемых категорий ---")
    print(result["predicted_category"].value_counts())
    print("\n--- Топ-10 студентов по риску ---")
    print(result.head(10).to_string(index=False))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_excel(output_path, index=False)
    print(f"\nСохранено: {output_path}")


if __name__ == "__main__":
    main()