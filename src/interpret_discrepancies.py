"""
Анализ расхождений между прогнозом модели и фактическими результатами —
качественный разбор ошибок на holdout-тесте (где факт уже известен), чтобы
понять ПОЧЕМУ модель ошибается, а не только НАСКОЛЬКО (это уже есть в
reports/metrics_summary.md).

Особый фокус — на двух самых важных типах расхождений:
  1. "Пропущенный риск": факт «3+», а модель предсказала меньше — самая
     опасная ошибка (именно её боится минимизировать recall «3+» в задании).
  2. "Ложная тревога": факт «0», а модель предсказала «3+» — лишняя
     нагрузка на кураторов, ложный сигнал.

Запуск:
    python src/interpret_discrepancies.py

Вход:  data/dataset_train.csv (для восстановления holdout-теста)
       models/model.pkl
Выход: reports/discrepancy_analysis.md   (интерпретация с примерами)
       reports/discrepancy_details.csv   (полная таблица факт/прогноз по каждому студенту)
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train import (  # noqa: E402
    INPUT_PATH, MODELS_DIR, REPORTS_DIR, FEATURE_COLUMNS,
    time_based_split, prepare_xy, CATEGORY_ORDER,
)

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
DETAILS_CSV_PATH = os.path.join(REPORTS_DIR, "discrepancy_details.csv")
ANALYSIS_MD_PATH = os.path.join(REPORTS_DIR, "discrepancy_analysis.md")

CATEGORY_TO_INT = {c: i for i, c in enumerate(CATEGORY_ORDER)}


def build_results_table(df_test: pd.DataFrame, model) -> pd.DataFrame:
    X_test, y_test = prepare_xy(df_test)
    y_pred = np.array(model.predict(X_test)).reshape(-1)
    y_proba = model.predict_proba(X_test)

    meta_cols = ["hash", "Учебная группа", "Специальность/направление", "sem_key"]
    meta_cols = [c for c in meta_cols if c in df_test.columns]

    result = df_test[meta_cols].reset_index(drop=True).copy()
    result[FEATURE_COLUMNS] = X_test.reset_index(drop=True)
    result["actual_category"] = [CATEGORY_ORDER[i] for i in y_test]
    result["predicted_category"] = [CATEGORY_ORDER[i] for i in y_pred]
    for i, cls in enumerate(CATEGORY_ORDER):
        result[f"probability_{cls}"] = y_proba[:, i]
    result["match"] = result["actual_category"] == result["predicted_category"]

    return result


def summarize_group(df: pd.DataFrame, label: str, feature_subset: list) -> str:
    """Средние значения ключевых признаков для группы строк — помогает
    понять, чем отличаются ошибочные случаи от правильных."""
    lines = [f"\n**{label}** (n={len(df)})\n"]
    if len(df) == 0:
        lines.append("Случаев не найдено.\n")
        return "".join(lines)
    lines.append("| Признак | Среднее значение |")
    lines.append("|---|---|")
    for feat in feature_subset:
        lines.append(f"| {feat} | {df[feat].mean():.2f} |")
    return "\n".join(lines) + "\n"


def build_analysis_md(result: pd.DataFrame) -> str:
    lines = ["# Анализ расхождений между прогнозом и фактом\n"]
    lines.append(
        "Разбор построен на holdout-тесте (последние по времени семестры, "
        "не участвовавшие в обучении) — там факт уже известен, поэтому "
        "можно честно сравнить прогноз модели с тем, что произошло на "
        "самом деле.\n"
    )

    total = len(result)
    n_match = result["match"].sum()
    lines.append(f"\nВсего строк в тесте: {total}. Совпадений прогноза с "
                  f"фактом: {n_match} ({n_match/total*100:.1f}%).\n")

    # --- Топ-10 студентов с наибольшим прогнозируемым риском (как просит задание) ---
    top10 = result.sort_values("probability_3+", ascending=False).head(10)
    lines.append("\n## Топ-10 студентов с наибольшим прогнозируемым риском\n")
    lines.append("| hash (сокр.) | Группа | Факт | Прогноз | P(3+) | Совпало? |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in top10.iterrows():
        lines.append(
            f"| {row['hash'][:8]}... | {row.get('Учебная группа', '-')} | "
            f"{row['actual_category']} | {row['predicted_category']} | "
            f"{row['probability_3+']:.3f} | {'✅' if row['match'] else '❌'} |"
        )
    n_correct_top10 = top10["match"].sum()
    lines.append(f"\nИз топ-10 самых рискованных по мнению модели — "
                 f"{n_correct_top10}/10 прогнозов совпали с фактической категорией.\n")

    # --- Расхождение №1: пропущенный высокий риск ---
    missed_risk = result[(result["actual_category"] == "3+") &
                          (result["predicted_category"] != "3+")]
    caught_risk = result[(result["actual_category"] == "3+") &
                          (result["predicted_category"] == "3+")]

    lines.append("\n## Расхождение №1: «Пропущенный риск» (самое опасное)\n")
    lines.append(
        f"Факт — «3+» задолженности, но модель предсказала меньше. "
        f"Таких случаев: {len(missed_risk)} из "
        f"{len(missed_risk) + len(caught_risk)} реальных «3+» "
        f"({len(missed_risk)/(len(missed_risk)+len(caught_risk))*100:.1f}% пропущено).\n"
    )

    key_features = ["n_debts", "avg_score", "score_trend", "prev_avg_score",
                     "group_difficulty", "has_retake_in_semester", "semester_number"]
    lines.append(summarize_group(caught_risk, "Правильно пойманные «3+»", key_features))
    lines.append(summarize_group(missed_risk, "Пропущенные «3+»", key_features))

    lines.append(
        "\n**Интерпретация:** если у пропущенных случаев `n_debts` (текущие "
        "долги) в среднем ниже, а `score_trend` (изменение среднего балла) "
        "более отрицательный или сопоставим с пойманными — это означает, "
        "что модель хорошо ловит студентов, которые УЖЕ накопили "
        "задолженности, но хуже предсказывает резкое ухудшение с "
        "благополучного старта (например, неожиданное падение успеваемости "
        "за один семестр без предшествующей истории). Это ожидаемое "
        "ограничение модели, построенной на признаках прошлого поведения — "
        "она хуже видит внезапные, не характерные для студента изменения.\n"
    )

    # --- Расхождение №2: ложная тревога ---
    false_alarm = result[(result["actual_category"] == "0") &
                          (result["predicted_category"] == "3+")]
    true_negative = result[(result["actual_category"] == "0") &
                            (result["predicted_category"] == "0")]

    lines.append("\n## Расхождение №2: «Ложная тревога»\n")
    lines.append(
        f"Факт — «0» задолженностей, но модель предсказала «3+». "
        f"Таких случаев: {len(false_alarm)} из "
        f"{len(false_alarm) + len(true_negative)} реальных «0» "
        f"({len(false_alarm)/(len(false_alarm)+len(true_negative))*100:.1f}% ложных тревог).\n"
    )
    lines.append(summarize_group(false_alarm, "Ложные тревоги", key_features))
    lines.append(summarize_group(true_negative, "Верно распознанные «0»", key_features))
    lines.append(
        "\n**Интерпретация:** если у ложных тревог `group_difficulty` или "
        "`avg_discipline_difficulty` высокие — модель, вероятно, слишком "
        "сильно опирается на \"репутацию\" группы/дисциплин и завышает "
        "риск студентам, которые лично справляются хорошо, даже учась в "
        "\"сложной\" группе. Это стоит учитывать при интерпретации для "
        "кураторов: высокий прогнозируемый риск не всегда означает личные "
        "проблемы студента — иногда это эффект окружения.\n"
    )

    # --- Путаница соседних классов ---
    lines.append("\n## Путаница между соседними классами (\"1\" и \"2\")\n")
    conf_12 = result[result["actual_category"].isin(["1", "2"])]
    n_conf = (conf_12["actual_category"] != conf_12["predicted_category"]).sum()
    lines.append(
        f"Среди {len(conf_12)} случаев с фактом «1» или «2» — "
        f"{n_conf} ({n_conf/len(conf_12)*100:.1f}%) распознаны неверно. "
        f"Это ожидаемо: границы между «1 задолженностью» и «2» размыты "
        f"по имеющимся признакам (небольшая, но реальная разница в объёме "
        f"данных — 297+133 строк против 2693 для класса «0» в тесте), "
        f"и модели не хватает сигнала, чтобы уверенно их разделить.\n"
    )

    # --- Честная оговорка про экспертное мнение кураторов ---
    lines.append("\n## Сравнение с экспертным мнением кураторов\n")
    lines.append(
        "Задание также просит сравнить прогноз с экспертным мнением "
        "кураторов — этих данных в текущем датасете нет, сравнение "
        "невозможно на этом этапе. Рекомендация: собрать хотя бы по "
        "10-20 реальных оценок риска от кураторов вручную (например, по "
        "топ-10 студентам выше) и сверить с прогнозом модели вручную — "
        "это можно сделать как отдельный шаг перед сдачей проекта.\n"
    )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Анализ расхождений между прогнозом и фактом на holdout-тесте."
    )
    parser.add_argument("--input", "-i", default=INPUT_PATH,
                        help=f"Путь к train CSV (по умолчанию: {INPUT_PATH})")
    parser.add_argument("--model", "-m", default=MODEL_PATH,
                        help=f"Путь к модели (по умолчанию: {MODEL_PATH})")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(args.input)
    _, df_test = time_based_split(df)

    model = joblib.load(args.model)
    result = build_results_table(df_test, model)

    result.to_csv(DETAILS_CSV_PATH, index=False, encoding="utf-8-sig")

    analysis_md = build_analysis_md(result)
    with open(ANALYSIS_MD_PATH, "w", encoding="utf-8") as f:
        f.write(analysis_md)

    print(analysis_md)
    print(f"\n\nПолная таблица сохранена: {DETAILS_CSV_PATH}")
    print(f"Анализ сохранён: {ANALYSIS_MD_PATH}")


if __name__ == "__main__":
    main()