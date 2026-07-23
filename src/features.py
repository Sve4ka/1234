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

import argparse
import os

import numpy as np
import pandas as pd
import joblib

INPUT_PATH = "data/data.xlsx"
OUTPUT_PATH = "data/student_semester_features.csv"
DIFFICULTY_MAPS_PATH = "models/difficulty_maps.pkl"


def _default_output_path(input_path: str) -> str:
    """По имени входного файла строит имя выходного, чтобы не перезаписывать
    результаты разных загрузок (например, session_2026_spring.xlsx ->
    session_2026_spring_features.csv)."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join("data", f"{base}_features.csv")

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
    """Доля задолженностей — по дисциплине, группе и направлению подготовки.

    ВАЖНО: эти карты должны считаться ОДИН РАЗ на полной обучающей истории
    и затем сохраняться (см. save_difficulty_maps/load_difficulty_maps).
    Если считать их заново на каждом новом небольшом файле при инференсе
    (например, на выгрузке одной группы) — получится, что group_difficulty
    и specialty_difficulty будут отражать статистику ТОЛЬКО этого нового
    файла (часто вырождаясь в одно и то же число на всех строк), а не
    историческую сложность, на которой обучалась модель. Это создаёт
    несоответствие между обучением и инференсом и портит прогноз."""
    return {
        "discipline": df.groupby("Дисциплина")["_is_debt"].mean(),
        "group": df.groupby("Учебная группа")["_is_debt"].mean(),
        "specialty": df.groupby("Специальность/направление")["_is_debt"].mean(),
        # Глобальное среднее — fallback для категорий, которых не было в
        # обучающих данных (новая группа/дисциплина/направление)
        "_global_mean": df["_is_debt"].mean(),
    }


def save_difficulty_maps(diff_maps: dict, path: str = DIFFICULTY_MAPS_PATH) -> None:
    joblib.dump(diff_maps, path)


def load_difficulty_maps(path: str = DIFFICULTY_MAPS_PATH) -> dict:
    return joblib.load(path)


def _map_with_fallback(series: pd.Series, mapping: pd.Series, global_mean: float) -> pd.Series:
    """Как .map(), но неизвестные категории (не встречавшиеся при обучении)
    заполняются глобальным средним, а не NaN."""
    return series.map(mapping).fillna(global_mean)


# ------------------------------------------------------------------
# 3. Агрегация по студенту и семестру
# ------------------------------------------------------------------
def build_student_semester_features(df: pd.DataFrame, difficulty_maps: dict = None) -> pd.DataFrame:
    """
    difficulty_maps=None (по умолчанию) — карты сложности считаются с нуля
        по переданному df. Используется ТОЛЬКО при первичном построении
        обучающего датасета из полной истории (data.xlsx).
    difficulty_maps={...} — использовать заранее сохранённые карты
        (из обучения), с fallback на глобальное среднее для новых
        категорий. Используется при инференсе на новых файлах.
    """
    df = add_derived_columns(df)

    if difficulty_maps is None:
        diff_maps = compute_difficulty_maps(df)
    else:
        diff_maps = difficulty_maps

    global_mean = diff_maps["_global_mean"]
    df["_discipline_debt_rate"] = _map_with_fallback(df["Дисциплина"], diff_maps["discipline"], global_mean)
    df["_group_debt_rate"] = _map_with_fallback(df["Учебная группа"], diff_maps["group"], global_mean)
    df["_specialty_debt_rate"] = _map_with_fallback(df["Специальность/направление"], diff_maps["specialty"], global_mean)

    group_cols = ["hash", "sem_key", "Учебный год", "Полугодие",
                  "Уровень подготовки", "Учебная группа", "Специальность/направление"]

    agg = df.groupby(group_cols).agg(
        n_records=("Дисциплина", "count"),
        n_distinct_disciplines=("Дисциплина", "nunique"),
        n_graded=("Оценка (успеваемость)", lambda s: s.notna().sum()),
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

    # Доля выставленных финальных оценок за семестр. Нужна, чтобы отличать
    # "семестр реально сдан без долгов" от "семестр ещё идёт, оценок просто
    # нет" (иначе текущий незавершённый семестр ошибочно выглядит как
    # идеальный результат — почти все n_debts=0 из-за NaN, а не реальной
    # успеваемости).
    agg["completeness_ratio"] = agg["n_graded"] / agg["n_records"]

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

    # Если следующий семестр студента заполнен оценками меньше, чем
    # наполовину — значит, он ещё идёт (текущая, незавершённая сессия), и
    # его n_debts=0 не означает "нет долгов", а означает "оценок ещё нет".
    # В этом случае таргет считаем неизвестным (NaN), а не "0".
    next_completeness = agg.groupby("hash")["completeness_ratio"].shift(-1)
    INCOMPLETE_THRESHOLD = 0.5
    agg.loc[next_completeness < INCOMPLETE_THRESHOLD, "target_debts_next"] = np.nan

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
def _compute_diff_maps_from_raw(df_raw: pd.DataFrame) -> dict:
    return compute_difficulty_maps(add_derived_columns(df_raw))


def build_features(path: str = INPUT_PATH, difficulty_maps: dict = None,
                    save_maps: bool = False) -> pd.DataFrame:
    """
    difficulty_maps=None (по умолчанию) — карты сложности считаются с нуля
        по файлу path. Подходит для обучения на полной исторической выгрузке.
    difficulty_maps={...} — использовать заранее сохранённые (обученные)
        карты сложности вместо пересчёта по этому файлу. ОБЯЗАТЕЛЬНО для
        инференса на новых файлах — иначе group_difficulty/
        specialty_difficulty/avg_discipline_difficulty будут считаться по
        одному новому файлу (часто по одной группе), а не по истории, на
        которой обучалась модель — это ломает согласованность признаков.
    save_maps=True — сохранить посчитанные карты в DIFFICULTY_MAPS_PATH
        (используется при обучении, чтобы потом переиспользовать их при
        инференсе через src/predict.py).
    """
    df = load_raw(path)

    if difficulty_maps is None:
        diff_maps = _compute_diff_maps_from_raw(df)
        if save_maps:
            save_difficulty_maps(diff_maps)
    else:
        diff_maps = difficulty_maps

    return build_student_semester_features(df, difficulty_maps=diff_maps)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Engineering: строит датасет студент x семестр "
                    "из Excel-файла с оценками."
    )
    parser.add_argument(
        "--input", "-i", default=INPUT_PATH,
        help=f"Путь к входному Excel-файлу (по умолчанию: {INPUT_PATH})"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Путь к выходному CSV. Если не указан — строится автоматически "
             "по имени входного файла (data/<имя>_features.csv)."
    )
    parser.add_argument(
        "--save-maps", action="store_true",
        help="Сохранить карты сложности (дисциплина/группа/направление) в "
             f"{DIFFICULTY_MAPS_PATH} для последующего использования при "
             "инференсе на новых файлах (src/predict.py)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output or _default_output_path(args.input)

    features = build_features(args.input, save_maps=args.save_maps)

    print(f"Вход: {args.input}")
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

    if args.save_maps:
        print(f"\nКарты сложности сохранены в {DIFFICULTY_MAPS_PATH} "
              f"(нужны для src/predict.py на новых файлах)")

    features.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nСохранено в {output_path}")


if __name__ == "__main__":
    main()