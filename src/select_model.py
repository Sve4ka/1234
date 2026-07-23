"""
Выбор лучшей модели среди 6 обученных вариантов (Logistic Regression,
Random Forest, XGBoost/CatBoost x class_weight/SMOTE) и сохранение её в
стандартном виде — models/model.pkl — для использования backend'ом.
Также формирует обоснование выбора.

Запуск:
    python src/select_model.py

Вход:  data/dataset_train.csv (для восстановления holdout-теста)
       models/model_logreg.pkl
       models/model_random_forest.pkl
       models/model_xgboost_classweight.pkl
       models/model_xgboost_smote.pkl
       models/model_catboost_classweight.cbm
       models/model_catboost_smote.cbm
Выход: models/model.pkl               (выбранная модель, единый формат)
       reports/model_selection.md     (обоснование выбора)
"""

import argparse
import os
import sys

import joblib
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(__file__))
from train import (  # noqa: E402
    INPUT_PATH, MODELS_DIR, REPORTS_DIR,
    time_based_split, prepare_xy, evaluate_model, CATEGORY_ORDER,
)

MODEL_PKL_PATH = os.path.join(MODELS_DIR, "model.pkl")
SELECTION_REPORT_PATH = os.path.join(REPORTS_DIR, "model_selection.md")

# Основной критерий приёмки по заданию — macro F1. Recall «3+» отслеживаем
# отдельно и обсуждаем как компромисс, но не он определяет победителя,
# т.к. модель, максимизирующая только recall «3+», может слишком часто
# завышать риск для благополучных студентов (низкий precision по «3+»)
PRIMARY_METRIC = "macro_f1"


def load_all_models():
    return {
        "Logistic Regression (baseline)": joblib.load(
            os.path.join(MODELS_DIR, "model_logreg.pkl")),
        "Random Forest": joblib.load(
            os.path.join(MODELS_DIR, "model_random_forest.pkl")),
        "XGBoost (class_weight)": joblib.load(
            os.path.join(MODELS_DIR, "model_xgboost_classweight.pkl")),
        "XGBoost (SMOTE)": joblib.load(
            os.path.join(MODELS_DIR, "model_xgboost_smote.pkl")),
        "CatBoost (class_weight)": _load_catboost(
            os.path.join(MODELS_DIR, "model_catboost_classweight.cbm")),
        "CatBoost (SMOTE)": _load_catboost(
            os.path.join(MODELS_DIR, "model_catboost_smote.cbm")),
    }


def _load_catboost(path):
    m = CatBoostClassifier()
    m.load_model(path)
    return m


def build_selection_report(results: dict, winner_label: str, runner_up_recall_label: str) -> str:
    w_metrics = results[winner_label]
    r_metrics = results[runner_up_recall_label]

    lines = ["# Обоснование выбора финальной модели\n"]
    lines.append(f"**Выбрана: {winner_label}**\n")
    lines.append(
        f"\n## Критерий выбора\n\n"
        f"Основной критерий — **macro F1** (официальный критерий приёмки "
        f"по заданию, порог >= 0.70). Recall для класса «3+» "
        f"(студенты высокого риска) отслеживается отдельно как важный, "
        f"но вторичный показатель: модель, максимизирующая только "
        f"recall «3+», рискует слишком часто ложно относить благополучных "
        f"студентов к группе риска (низкий precision), что на практике "
        f"означало бы лишнюю нагрузку на кураторов.\n"
    )

    lines.append("\n## Сравнение всех вариантов\n")
    lines.append("| Модель | Accuracy | Macro F1 | Weighted F1 | Recall «3+» |")
    lines.append("|---|---|---|---|---|")
    for label, m in results.items():
        mark = " 🏆" if label == winner_label else ""
        lines.append(f"| {label}{mark} | {m['accuracy']:.3f} | "
                      f"{m['macro_f1']:.3f} | {m['weighted_f1']:.3f} | "
                      f"{m['recall_3plus']:.3f} |")

    lines.append(f"\n## Итог\n")
    lines.append(
        f"**{winner_label}** показала лучший macro F1 ({w_metrics['macro_f1']:.3f}) "
        f"среди всех 6 вариантов, при recall «3+» = {w_metrics['recall_3plus']:.3f}.\n"
    )
    lines.append(
        f"\n**Важная оговорка:** критерий приёмки по заданию — macro F1 >= 0.70 — "
        f"этой моделью пока НЕ достигнут ({w_metrics['macro_f1']:.3f} < 0.70). "
        f"Модель сохранена как рабочий baseline для дальнейшей доработки "
        f"(признаки, объединение классов «1»/«2», больше данных), а не как "
        f"финальный production-результат.\n"
    )

    lines.append(
        f"\n**Альтернатива для обсуждения с заказчиком:** {runner_up_recall_label} "
        f"даёт заметно более высокий recall «3+» ({r_metrics['recall_3plus']:.3f} "
        f"против {w_metrics['recall_3plus']:.3f} у выбранной модели), ценой "
        f"немного более низкого macro F1 ({r_metrics['macro_f1']:.3f}). Если для "
        f"деканата принципиально важнее не пропустить студента высокого риска, "
        f"чем избежать ложных срабатываний — стоит рассмотреть эту модель "
        f"вместо текущего выбора (файл сохранён отдельно в models/).\n"
    )

    lines.append(
        "\n## Как использовать\n\n"
        "```python\n"
        "import joblib\n"
        "model = joblib.load('models/model.pkl')\n"
        "predictions = model.predict(X)          # категория 0/1/2/3\n"
        "probabilities = model.predict_proba(X)   # вероятности по каждому классу\n"
        "```\n\n"
        f"Порядок классов: {CATEGORY_ORDER} (индексы 0-3 соответственно). "
        "Полный список и порядок признаков — см. reports/feature_contract.md.\n"
    )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Выбор лучшей модели по macro F1 на holdout-тесте, "
                    "сохранение как models/model.pkl с обоснованием."
    )
    parser.add_argument("--input", "-i", default=INPUT_PATH,
                        help=f"Путь к train CSV (по умолчанию: {INPUT_PATH})")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(args.input)
    _, df_test = time_based_split(df)
    X_test, y_test = prepare_xy(df_test)

    models = load_all_models()
    results = {}
    report_lines = []  # evaluate_model требует список для побочного логирования

    print("\nПереоценка всех 6 моделей на holdout-тесте...")
    for label, model in models.items():
        results[label] = evaluate_model(label, model, X_test, y_test, report_lines)

    winner_label = max(results, key=lambda k: results[k]["macro_f1"])
    runner_up_recall_label = max(results, key=lambda k: results[k]["recall_3plus"])

    print(f"\n{'='*60}")
    print(f"Выбрана модель: {winner_label} (macro F1 = {results[winner_label]['macro_f1']:.4f})")
    if runner_up_recall_label != winner_label:
        print(f"Альтернатива по recall(3+): {runner_up_recall_label} "
              f"(recall(3+) = {results[runner_up_recall_label]['recall_3plus']:.4f})")

    joblib.dump(models[winner_label], MODEL_PKL_PATH)
    print(f"\nСохранено: {MODEL_PKL_PATH}")

    report_md = build_selection_report(results, winner_label, runner_up_recall_label)
    with open(SELECTION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Обоснование сохранено: {SELECTION_REPORT_PATH}")


if __name__ == "__main__":
    main()