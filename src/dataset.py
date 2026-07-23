"""
Формирование финального датасета для обучения модели прогнозирования
академической задолженности.

Строка = (студент, семестр). Таргет — количество задолженностей в
СЛЕДУЮЩЕМ семестре (уже посчитан в src/features.py со сдвигом shift(-1)
по каждому студенту).

Датасет разбивается на две части:
  - dataset_train.csv   — строки, где таргет известен (есть следующий
                           семестр в истории). Именно эти строки идут
                           в train/test для обучения модели.
  - dataset_predict.csv — последний семестр каждого студента, для
                           которого таргета ещё нет (сессия впереди).
                           Эти строки НЕ участвуют в обучении — это как
                           раз те студенты, для которых система должна
                           выдать реальный прогноз "на предстоящую сессию".

Запуск:
    python src/dataset.py

Вход:  data/data.xlsx (через src/features.py)
Выход: data/dataset_train.csv
       data/dataset_predict.csv
"""

import argparse
import os

from features import build_features

INPUT_PATH = "data/data.xlsx"
TRAIN_OUTPUT_PATH = "data/dataset_train.csv"
PREDICT_OUTPUT_PATH = "data/dataset_predict.csv"


def _default_output_paths(input_path: str):
    """По имени входного файла строит имена выходных train/predict файлов,
    чтобы разные загрузки не перезаписывали друг друга."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    train_path = os.path.join("data", f"{base}_dataset_train.csv")
    predict_path = os.path.join("data", f"{base}_dataset_predict.csv")
    return train_path, predict_path


def build_final_dataset(path: str = INPUT_PATH):
    """
    Возвращает (df_train, df_predict).

    df_train   — строки с известным таргетом (не последний семестр студента).
    df_predict — последний семестр каждого студента (таргет неизвестен,
                 это и есть прогнозная выборка "перед сессией").
    """
    features = build_features(path, save_maps=True)

    has_target = features["target_category"].notna()
    df_train = features[has_target].reset_index(drop=True)
    df_predict = features[~has_target].reset_index(drop=True)

    return df_train, df_predict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Формирует финальный датасет (train/predict) из "
                    "Excel-файла с оценками."
    )
    parser.add_argument(
        "--input", "-i", default=INPUT_PATH,
        help=f"Путь к входному Excel-файлу (по умолчанию: {INPUT_PATH})"
    )
    parser.add_argument(
        "--train-output", default=None,
        help="Путь к выходному train CSV. Если не указан — строится "
             "автоматически по имени входного файла."
    )
    parser.add_argument(
        "--predict-output", default=None,
        help="Путь к выходному predict CSV. Если не указан — строится "
             "автоматически по имени входного файла."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    default_train, default_predict = _default_output_paths(args.input)
    train_output = args.train_output or default_train
    predict_output = args.predict_output or default_predict

    df_train, df_predict = build_final_dataset(args.input)

    print(f"Вход: {args.input}")
    print(f"Финальный датасет для обучения (df_train): "
          f"{df_train.shape[0]} строк, {df_train.shape[1]} столбцов")
    print("\n--- Распределение таргета в обучающей выборке ---")
    print(df_train["target_category"].value_counts())

    print(f"\nПрогнозная выборка (последний семестр каждого студента, "
          f"df_predict): {df_predict.shape[0]} строк")
    print("(таргет здесь неизвестен — это студенты, для которых нужно "
          "предсказать задолженности на предстоящую сессию)")

    # Сверка: сумма должна совпадать с общим числом строк в features
    total = df_train.shape[0] + df_predict.shape[0]
    print(f"\nПроверка: {df_train.shape[0]} + {df_predict.shape[0]} = {total}")

    df_train.to_csv(train_output, index=False, encoding="utf-8-sig")
    df_predict.to_csv(predict_output, index=False, encoding="utf-8-sig")
    print(f"\nСохранено:\n  {train_output}\n  {predict_output}")


if __name__ == "__main__":
    main()