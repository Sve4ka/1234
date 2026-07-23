"""
Feature Engineering pipeline: сборка датасета "студент x семестр" с признаками
и целевой переменной (количество задолженностей в СЛЕДУЮЩЕМ семестре) для
задачи прогнозирования академической задолженности.

Обучение моделей НЕ входит в этот модуль — только подготовка данных.

Запуск как скрипт:
    python src/features.py
Вход:  data/data.xlsx
Выход: data/student_semester_features.csv

Использование как модуля (для дальнейшего обучения моделей):
    from src.features import build_features
    features_df = build_features("data/data.xlsx")

Допущения (важно для интерпретации признаков):
- "Оценка (успеваемость)" считаем финальной оценкой (после пересдач) — именно
  по ней определяем факт задолженности.
- "Неявка по ув.причине" НЕ считаем задолженностью (уважительная причина).
- Пересдачу внутри семестра детектируем по повторению названия дисциплины
  СТРОГО в пределах одного семестра (студент+год+полугодие+дисциплина
  встречается >1 раза). Повторение дисциплины МЕЖДУ разными семестрами не
  считаем автоматически пересдачей, т.к. ~19% пар (студент, дисциплина)
  законно растянуты на несколько семестров по учебному плану (не так, будто
  сдал/пересдавал), поэтому такой сигнал был бы слишком шумным.
- Полных дублей строк (совпадение по всем 10 столбцам) в данных не найдено
  (df.duplicated().sum() == 0), поэтому строки не удаляются.
"""

import numpy as np
import pandas as pd

INPUT_PATH = "data/data.xlsx"
OUTPUT_PATH = "data/student_semester_features.csv"

# Численный эквивалент экзаменационных оценок
GRADE_TO_SCORE = {
    "Отлично": 5,
    "Хорошо": 4,
    "Удовлетворительно": 3,
    "Неудовлетворительно": 2,
}

# Что считаем задолженностью в финальной оценке (после пересдач)
DEBT_VALUES = {"Неудовлетворительно", "не зачтено", "Неявка", "Не допущен"}


# ------------------------------------------------------------------
# 1. Загрузка и базовая подготовка
# ------------------------------------------------------------------
def load_raw(path: str = INPUT_PATH) -> pd.DataFrame:
    return pd.read_excel(path)


def _semester_key(df: pd.DataFrame) -> pd.Series:
    """Сортируемый числовой ключ семестра из 'Учебный год' + 'Полугодие'."""
    year_start = df["Учебный год"].str.slice(0, 4).astype(int)
    half = df["Полугодие"].map({"I полугодие": 0, "II полугодие": 1})
    return year_start * 2 + half


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sem_key"] = _semester_key(df)
    df["_score"] = df["Оценка (успеваемость)"].map(GRADE_TO_SCORE)
    df["_is_debt"] = df["Оценка (успеваемость)"].isin(DEBT_VALUES)
    df["_is_zachet_row"] = df["Оценка (успеваемость)"].isin({"зачтено", "не зачтено"})
    df["_is_zachteno"] = df["Оценка (успеваемость)"] == "зачтено"
    return df


# ------------------------------------------------------------------
# 2. Признаки сложности (дисциплина / группа / направление)
# ------------------------------------------------------------------
def compute_difficulty_maps(df: pd.DataFrame) -> dict:
    """Доля задолженностей — по дисциплине, группе и направлению подготовки."""
    return {
        "discipline": df.groupby("Дисциплина")["_is_debt"].mean(),
        "group": df.groupby("Учебная группа")["_is_debt"].mean(),
        "specialty": df.groupby("Специальность/направление")["_is_debt"].mean(),
    }


# ------------------------------------------------------------------
# 3. Агрегация по студенту и семестру
# ------------------------------------------------------------------
def build_student_semester_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_derived_columns(df)
    diff_maps = compute_difficulty_maps(df)

    df["_discipline_debt_rate"] = df["Дисциплина"].map(diff_maps["discipline"])
    df["_group_debt_rate"] = df["Учебная группа"].map(diff_maps["group"])
    df["_specialty_debt_rate"] = df["Специальность/направление"].map(diff_maps["specialty"])

    group_cols = ["hash", "sem_key", "Учебный год", "Полугодие",
                  "Уровень подготовки", "Учебная группа", "Специальность/направление"]

    agg = df.groupby(group_cols).agg(
        n_records=("Дисциплина", "count"),
        n_distinct_disciplines=("Дисциплина", "nunique"),
        n_otlichno=("Оценка (успеваемость)", lambda s: (s == "Отлично").sum()),
        n_horosho=("Оценка (успеваемость)", lambda s: (s == "Хорошо").sum()),
        n_udovl=("Оценка (успеваемость)", lambda s: (s == "Удовлетворительно").sum()),
        n_neudovl=("Оценка (успеваемость)", lambda s: (s == "Неудовлетворительно").sum()),
        n_neyavka=("Оценка (успеваемость)", lambda s: (s == "Неявка").sum()),
        n_nezachteno=("Оценка (успеваемость)", lambda s: (s == "не зачтено").sum()),
        n_debts=("_is_debt", "sum"),
        n_zachet_rows=("_is_zachet_row", "sum"),
        n_zachteno=("_is_zachteno", "sum"),
        avg_score=("_score", "mean"),
        avg_discipline_difficulty=("_discipline_debt_rate", "mean"),
        group_difficulty=("_group_debt_rate", "first"),
        specialty_difficulty=("_specialty_debt_rate", "first"),
    ).reset_index()

    # Пересдача СТРОГО внутри семестра: повтор дисциплины в том же периоде
    agg["n_retake_rows_in_semester"] = agg["n_records"] - agg["n_distinct_disciplines"]
    agg["has_retake_in_semester"] = agg["n_retake_rows_in_semester"] > 0

    agg["share_zachteno"] = np.where(
        agg["n_zachet_rows"] > 0, agg["n_zachteno"] / agg["n_zachet_rows"], np.nan
    )

    # --- временные признаки ---
    agg = agg.sort_values(["hash", "sem_key"])
    agg["semester_number"] = agg.groupby("hash").cumcount() + 1
    agg["prev_avg_score"] = agg.groupby("hash")["avg_score"].shift(1)
    agg["score_trend"] = agg["avg_score"] - agg["prev_avg_score"]

    # --- таргет: долги в СЛЕДУЮЩЕМ семестре ---
    agg["target_debts_next"] = agg.groupby("hash")["n_debts"].shift(-1)

    def to_category(x):
        if pd.isna(x):
            return np.nan
        if x == 0:
            return "0"
        if x == 1:
            return "1"
        if x == 2:
            return "2"
        return "3+"

    agg["target_category"] = agg["target_debts_next"].apply(to_category)

    return agg


# ------------------------------------------------------------------
# 4. Публичная точка входа для использования из других скриптов
# ------------------------------------------------------------------
def build_features(path: str = INPUT_PATH) -> pd.DataFrame:
    df = load_raw(path)
    return build_student_semester_features(df)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    features = build_features(INPUT_PATH)

    print(f"Итоговый датасет: {features.shape[0]} строк (студент x семестр), "
          f"{features.shape[1]} столбцов")
    print("\n--- Первые строки ---")
    print(features.head(10))

    print("\n--- Распределение целевой категории (там, где известна) ---")
    print(features["target_category"].value_counts(dropna=False))

    n_no_target = features["target_category"].isna().sum()
    print(f"\nСтрок без таргета (последний семестр студента, "
          f"прогнозная выборка): {n_no_target}")

    print(f"\nСтрок с пересдачей внутри семестра: "
          f"{features['has_retake_in_semester'].sum()} "
          f"({features['has_retake_in_semester'].mean()*100:.1f}%)")

    features.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nСохранено в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()