"""
Обучение моделей XGBoost и CatBoost для многоклассовой классификации риска
академической задолженности (категории таргета: "0", "1", "2", "3+").

Разделение train/test — строго по времени (не случайно!): более ранние
семестры идут в обучение, более поздние — в тест, примерно в пропорции
80/20 по количеству строк. Это соответствует реальному сценарию:
модель обучается на прошлом, а проверяется на будущем.

Подбор гиперпараметров — через TimeSeriesSplit-кросс-валидацию ВНУТРИ
обучающей части (той, что осталась после отделения финального теста).
TimeSeriesSplit строит несколько "расширяющихся" фолдов, где обучение
всегда идёт на более раннем периоде, а валидация — на следующем сразу за
ним отрезке времени. Это не даёт "заглянуть в будущее" при подборе
параметров — в отличие от обычной случайной K-Fold-кросс-валидации,
которая для таких данных была бы методологически неверной.

Дисбаланс классов: сравниваются два подхода —
  1) class_weight (без изменения данных) — для XGBoost через sample_weight,
     для CatBoost через class_weights;
  2) SMOTE (синтетическая генерация примеров редких классов) — применяется
     ТОЛЬКО к обучающей выборке (никогда к тесту/валидации), чтобы не
     создавать искусственную "утечку" синтетических примеров в оценку
     качества.
В отчёт попадают метрики обоих подходов, чтобы можно было сравнить и
выбрать. Веса классов (для class_weight) пересчитываются заново на каждом
CV-фолде (а не глобально), чтобы не было утечки информации о полном
распределении классов.

Запуск:
    python src/train.py
    python src/train.py --input data/data_dataset_train.csv
    python src/train.py --n-splits 5      # число фолдов TimeSeriesSplit
    python src/train.py --quick           # без подбора гиперпараметров
                                            (использовать значения по умолчанию,
                                             для быстрой проверки пайплайна)

Вход:  data/dataset_train.csv (результат src/dataset.py)
Выход: models/model_xgboost_classweight.pkl
       models/model_xgboost_smote.pkl
       models/model_catboost_classweight.cbm
       models/model_catboost_smote.cbm
       reports/training_metrics.txt (метрики и сравнение всех 4 вариантов)
"""

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
import joblib

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

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


# ------------------------------------------------------------------
# Разбиение по времени (80/20 по количеству строк, не случайное)
# ------------------------------------------------------------------
def time_based_split(df: pd.DataFrame, train_fraction: float = 0.8):
    # Сортируем по времени; hash — вторичный ключ только для детерминизма
    # (чтобы результат был воспроизводим при равных sem_key).
    df = df.sort_values(["sem_key", "hash"]).reset_index(drop=True)

    cutoff_idx = int(len(df) * train_fraction)
    df_train = df.iloc[:cutoff_idx]
    df_test = df.iloc[cutoff_idx:]

    print(f"Train: {len(df_train)} строк ({len(df_train)/len(df)*100:.1f}%), "
          f"семестры {df_train['sem_key'].min()}..{df_train['sem_key'].max()}")
    print(f"Test:  {len(df_test)} строк ({len(df_test)/len(df)*100:.1f}%), "
          f"семестры {df_test['sem_key'].min()}..{df_test['sem_key'].max()}")
    overlap = set(df_train["sem_key"]) & set(df_test["sem_key"])
    if overlap:
        print(f"(Один семестр {sorted(overlap)} разбит между train и test по "
              f"строкам — так удалось получить пропорцию {train_fraction:.0%}, "
              f"несмотря на неравномерный размер семестров.)")

    return df_train, df_test


def prepare_xy(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    # Пропуски (первый семестр студента, семестры без числовых оценок и т.п.)
    # заполняем медианой обучающей выборки — простое и надёжное решение для MVP.
    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET_COLUMN].map(CATEGORY_TO_INT).values
    return X, y


# ------------------------------------------------------------------
# Сетки гиперпараметров (небольшие, чтобы перебор укладывался в разумное
# время даже на слабом ноутбуке — не десятки часов на GridSearch)
# ------------------------------------------------------------------
XGB_PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [200, 400],
}

CATBOOST_PARAM_GRID = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1],
    "iterations": [200, 400],
}


def _grid_combinations(grid: dict):
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


# ------------------------------------------------------------------
# TimeSeriesSplit кросс-валидация + подбор гиперпараметров
# ------------------------------------------------------------------
def cv_score_xgboost(X, y, params: dict, cv_folds) -> float:
    """Средний macro F1 по фолдам TimeSeriesSplit для набора параметров XGBoost."""
    scores = []
    for train_idx, val_idx in cv_folds:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Веса классов считаем ОТДЕЛЬНО на каждом фолде — иначе утечка
        # информации о полном распределении классов из будущего.
        sw = compute_sample_weight(class_weight="balanced", y=y_tr)

        model = XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=len(CATEGORY_ORDER),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, sample_weight=sw)
        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
    return float(np.mean(scores))


def cv_score_catboost(X, y, params: dict, cv_folds) -> float:
    """Средний macro F1 по фолдам TimeSeriesSplit для набора параметров CatBoost."""
    scores = []
    for train_idx, val_idx in cv_folds:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = CatBoostClassifier(
            **params,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
        )
        model.fit(X_tr, y_tr)
        y_pred = np.array(model.predict(X_val)).reshape(-1)
        scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
    return float(np.mean(scores))


def tune_model(name, grid, score_fn, X, y, cv_folds, report_lines):
    """Полный перебор сетки гиперпараметров, выбор лучшей по среднему
    macro F1 на TimeSeriesSplit-фолдах."""
    print(f"\n--- Подбор гиперпараметров: {name} "
          f"({len(list(_grid_combinations(grid)))} комбинаций x "
          f"{len(cv_folds)} фолдов) ---")
    report_lines.append(f"\n--- Подбор гиперпараметров: {name} ---\n")

    best_score = -1.0
    best_params = None
    t0 = time.time()

    for params in _grid_combinations(grid):
        score = score_fn(X, y, params, cv_folds)
        line = f"{params}  ->  CV macro F1 = {score:.4f}"
        print(line)
        report_lines.append(line + "\n")
        if score > best_score:
            best_score = score
            best_params = params

    elapsed = time.time() - t0
    print(f"Лучшие параметры ({name}): {best_params} "
          f"(CV macro F1 = {best_score:.4f}, перебор занял {elapsed:.1f} сек)")
    report_lines.append(f"Лучшие параметры ({name}): {best_params} "
                         f"(CV macro F1 = {best_score:.4f})\n")

    return best_params, best_score



def apply_smote(X_train, y_train):
    """SMOTE — только на обучающей выборке. Возвращает пересэмплированные
    X, y, где редкие классы дополнены синтетическими примерами до размера
    самого крупного класса."""
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def train_xgboost_smote(X_train, y_train, params: dict = None):
    """XGBoost на данных, сбалансированных через SMOTE — без sample_weight,
    т.к. классы уже уравнены оверсэмплингом."""
    params = params or {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05}
    X_res, y_res = apply_smote(X_train, y_train)
    model = XGBClassifier(
        **params,
        objective="multi:softprob",
        num_class=len(CATEGORY_ORDER),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_res, y_res)
    return model


def train_catboost_smote(X_train, y_train, params: dict = None):
    """CatBoost на данных, сбалансированных через SMOTE — без
    auto_class_weights, т.к. классы уже уравнены оверсэмплингом."""
    params = params or {"iterations": 300, "depth": 6, "learning_rate": 0.05}
    X_res, y_res = apply_smote(X_train, y_train)
    model = CatBoostClassifier(
        **params,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_res, y_res)
    return model



def train_xgboost(X_train, y_train, params: dict = None):
    params = params or {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05}
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = XGBClassifier(
        **params,
        objective="multi:softprob",
        num_class=len(CATEGORY_ORDER),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_catboost(X_train, y_train, params: dict = None):
    params = params or {"iterations": 300, "depth": 6, "learning_rate": 0.05}
    model = CatBoostClassifier(
        **params,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------------
# Оценка
# ------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test, report_lines):
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred).reshape(-1)  # catboost возвращает 2D-массив

    report_dict = classification_report(
        y_test, y_pred, labels=list(range(len(CATEGORY_ORDER))),
        target_names=CATEGORY_ORDER, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_test, y_pred, labels=list(range(len(CATEGORY_ORDER))),
        target_names=CATEGORY_ORDER, zero_division=0
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    accuracy = report_dict["accuracy"]
    recall_3plus = report_dict["3+"]["recall"]

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(report_str)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}  (критерий приёмки: >= 0.70)")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Recall для класса '3+' (высокий риск): {recall_3plus:.4f}")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CATEGORY_ORDER))))
    print(f"\nМатрица ошибок (строки=факт, столбцы=прогноз), порядок {CATEGORY_ORDER}:")
    print(cm)

    report_lines.append(f"\n{'='*60}\n{name}\n{'='*60}\n")
    report_lines.append(report_str)
    report_lines.append(f"Accuracy:    {accuracy:.4f}\n")
    report_lines.append(f"Macro F1:    {macro_f1:.4f}  (критерий приёмки: >= 0.70)\n")
    report_lines.append(f"Weighted F1: {weighted_f1:.4f}\n")
    report_lines.append(f"Recall для класса '3+' (высокий риск): {recall_3plus:.4f}\n")
    report_lines.append(f"Матрица ошибок (строки=факт, столбцы=прогноз), порядок {CATEGORY_ORDER}:\n")
    report_lines.append(str(cm) + "\n")

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "recall_3plus": recall_3plus,
        "per_class": report_dict,  # precision/recall/f1/support по каждому классу
    }
    return metrics


def build_metrics_summary_md(results: dict) -> str:
    """Сводная markdown-таблица метрик по всем вариантам моделей —
    удобно вставить прямо в финальный отчёт проекта."""
    lines = ["# Сводная таблица метрик качества\n"]
    lines.append("Критерий приёмки: macro F1 >= 0.70. "
                  "Отдельно отслеживается recall для класса «3+» "
                  "(важно не пропускать студентов высокого риска).\n")

    lines.append("\n## Общие метрики\n")
    lines.append("| Модель | Accuracy | Macro F1 | Weighted F1 | Recall «3+» |")
    lines.append("|---|---|---|---|---|")
    for label, (_, m) in results.items():
        mark = " ✅" if m["macro_f1"] >= 0.70 else ""
        lines.append(f"| {label} | {m['accuracy']:.3f} | "
                      f"{m['macro_f1']:.3f}{mark} | {m['weighted_f1']:.3f} | "
                      f"{m['recall_3plus']:.3f} |")

    lines.append("\n## Precision / Recall / F1 по каждому классу\n")
    for label, (_, m) in results.items():
        lines.append(f"\n### {label}\n")
        lines.append("| Класс | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for cls in CATEGORY_ORDER:
            c = m["per_class"][cls]
            lines.append(f"| {cls} | {c['precision']:.3f} | {c['recall']:.3f} | "
                          f"{c['f1-score']:.3f} | {int(c['support'])} |")

    best_label = max(results, key=lambda k: results[k][1]["macro_f1"])
    lines.append(f"\n**Лучший вариант по macro F1: {best_label}** "
                 f"({results[best_label][1]['macro_f1']:.3f})\n")

    return "\n".join(lines)



def print_feature_importance(name, importances, report_lines):
    order = np.argsort(importances)[::-1]
    print(f"\n--- Важность признаков ({name}) ---")
    report_lines.append(f"\n--- Важность признаков ({name}) ---\n")
    for i in order:
        line = f"{FEATURE_COLUMNS[i]:35s} {importances[i]:.4f}"
        print(line)
        report_lines.append(line + "\n")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение XGBoost и CatBoost на датасете студент x семестр, "
                    "с подбором гиперпараметров через TimeSeriesSplit."
    )
    parser.add_argument("--input", "-i", default=INPUT_PATH,
                        help=f"Путь к train CSV (по умолчанию: {INPUT_PATH})")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Число фолдов TimeSeriesSplit (по умолчанию: 5)")
    parser.add_argument("--quick", action="store_true",
                        help="Пропустить подбор гиперпараметров, использовать "
                             "значения по умолчанию (для быстрой проверки)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(args.input)
    df_train, df_test = time_based_split(df)

    # df_train сортируем по времени ещё раз явно — TimeSeriesSplit требует,
    # чтобы порядок строк совпадал с хронологическим порядком.
    df_train = df_train.sort_values(["sem_key", "hash"]).reset_index(drop=True)

    X_train, y_train = prepare_xy(df_train)
    X_test, y_test = prepare_xy(df_test)

    report_lines = []
    report_lines.append(f"Вход: {args.input}\n")
    report_lines.append(f"Train (после отделения holdout-теста): {len(df_train)} строк\n")
    report_lines.append(f"Test (holdout, самые поздние семестры): {len(df_test)} строк\n")

    if args.quick:
        print("\n--quick: подбор гиперпараметров пропущен, используются значения "
              "по умолчанию.")
        xgb_params, cat_params = None, None
    else:
        cv_folds = list(TimeSeriesSplit(n_splits=args.n_splits).split(X_train))
        print(f"\nTimeSeriesSplit: {args.n_splits} фолдов внутри обучающей выборки "
              f"({len(df_train)} строк, семестры "
              f"{df_train['sem_key'].min()}..{df_train['sem_key'].max()}).")
        for i, (tr_idx, val_idx) in enumerate(cv_folds):
            print(f"  Фолд {i+1}: train={len(tr_idx)} строк "
                  f"(sem_key <= {df_train['sem_key'].iloc[tr_idx[-1]]}), "
                  f"val={len(val_idx)} строк "
                  f"(sem_key {df_train['sem_key'].iloc[val_idx[0]]}"
                  f"..{df_train['sem_key'].iloc[val_idx[-1]]})")

        xgb_params, xgb_cv_score = tune_model(
            "XGBoost", XGB_PARAM_GRID, cv_score_xgboost,
            X_train, y_train, cv_folds, report_lines
        )
        cat_params, cat_cv_score = tune_model(
            "CatBoost", CATBOOST_PARAM_GRID, cv_score_catboost,
            X_train, y_train, cv_folds, report_lines
        )

    print("\nОбучение финальных моделей (class_weight) на всей обучающей выборке...")
    xgb_model_cw = train_xgboost(X_train, y_train, xgb_params)
    xgb_cw_metrics = evaluate_model("XGBoost (class_weight)", xgb_model_cw, X_test, y_test, report_lines)
    print_feature_importance("XGBoost (class_weight)", xgb_model_cw.feature_importances_, report_lines)

    cat_model_cw = train_catboost(X_train, y_train, cat_params)
    cat_cw_metrics = evaluate_model("CatBoost (class_weight)", cat_model_cw, X_test, y_test, report_lines)
    print_feature_importance("CatBoost (class_weight)", cat_model_cw.get_feature_importance(), report_lines)

    print("\nОбучение финальных моделей (SMOTE) на всей обучающей выборке...")
    xgb_model_sm = train_xgboost_smote(X_train, y_train, xgb_params)
    xgb_sm_metrics = evaluate_model("XGBoost (SMOTE)", xgb_model_sm, X_test, y_test, report_lines)
    print_feature_importance("XGBoost (SMOTE)", xgb_model_sm.feature_importances_, report_lines)

    cat_model_sm = train_catboost_smote(X_train, y_train, cat_params)
    cat_sm_metrics = evaluate_model("CatBoost (SMOTE)", cat_model_sm, X_test, y_test, report_lines)
    print_feature_importance("CatBoost (SMOTE)", cat_model_sm.get_feature_importance(), report_lines)

    results = {
        "XGBoost (class_weight)": (xgb_model_cw, xgb_cw_metrics),
        "CatBoost (class_weight)": (cat_model_cw, cat_cw_metrics),
        "XGBoost (SMOTE)": (xgb_model_sm, xgb_sm_metrics),
        "CatBoost (SMOTE)": (cat_model_sm, cat_sm_metrics),
    }

    print(f"\n{'='*60}\nСРАВНЕНИЕ БАЛАНСИРОВКИ КЛАССОВ\n{'='*60}")
    report_lines.append(f"\n{'='*60}\nСРАВНЕНИЕ БАЛАНСИРОВКИ КЛАССОВ\n{'='*60}\n")
    for label, (_, m) in results.items():
        line = (f"{label:30s} accuracy={m['accuracy']:.4f}  "
                f"macro F1={m['macro_f1']:.4f}  "
                f"weighted F1={m['weighted_f1']:.4f}  "
                f"recall(3+)={m['recall_3plus']:.4f}")
        print(line)
        report_lines.append(line + "\n")

    winner_label = max(results, key=lambda k: results[k][1]["macro_f1"])
    winner_model, winner_metrics = results[winner_label]
    print(f"\nЛучший вариант: {winner_label} (macro F1 = {winner_metrics['macro_f1']:.4f})")
    report_lines.append(f"\nЛучший вариант: {winner_label} "
                         f"(macro F1 = {winner_metrics['macro_f1']:.4f})\n")

    # Сводная markdown-таблица метрик — для финального отчёта проекта
    metrics_summary_md = build_metrics_summary_md(results)
    with open(os.path.join(REPORTS_DIR, "metrics_summary.md"), "w", encoding="utf-8") as f:
        f.write(metrics_summary_md)

    # Сохраняем все 4 варианта — для прозрачности и воспроизводимости
    joblib.dump(xgb_model_cw, os.path.join(MODELS_DIR, "model_xgboost_classweight.pkl"))
    joblib.dump(xgb_model_sm, os.path.join(MODELS_DIR, "model_xgboost_smote.pkl"))
    cat_model_cw.save_model(os.path.join(MODELS_DIR, "model_catboost_classweight.cbm"))
    cat_model_sm.save_model(os.path.join(MODELS_DIR, "model_catboost_smote.cbm"))

    with open(os.path.join(REPORTS_DIR, "training_metrics.txt"), "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"\nВсе 4 модели сохранены в {MODELS_DIR}/")
    print(f"Отчёт сохранён в {REPORTS_DIR}/training_metrics.txt")
    print(f"Сводная таблица метрик: {REPORTS_DIR}/metrics_summary.md")


if __name__ == "__main__":
    main()