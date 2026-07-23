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

from features import build_features

INPUT_PATH = "data/data.xlsx"
TRAIN_OUTPUT_PATH = "data/dataset_train.csv"
PREDICT_OUTPUT_PATH = "data/dataset_predict.csv"


def build_final_dataset(path: str = INPUT_PATH):
    """
    Возвращает (df_train, df_predict).

    df_train   — строки с известным таргетом (не последний семестр студента).
    df_predict — последний семестр каждого студента (таргет неизвестен,
                 это и есть прогнозная выборка "перед сессией").
    """
    features = build_features(path)

    has_target = features["target_category"].notna()
    df_train = features[has_target].reset_index(drop=True)
    df_predict = features[~has_target].reset_index(drop=True)

    return df_train, df_predict


def main():
    df_train, df_predict = build_final_dataset(INPUT_PATH)

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

    df_train.to_csv(TRAIN_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    df_predict.to_csv(PREDICT_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nСохранено:\n  {TRAIN_OUTPUT_PATH}\n  {PREDICT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()