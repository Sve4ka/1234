from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
    UPLOAD_DIR,
)
from backend.db.db import (
    create_upload,
    get_predictions,
    get_upload,
    save_predictions,
    update_upload_status,
)
from backend.db.migration import run_migrations
from src.predict import run_inference


app = FastAPI(
    title="Academic Debt Prediction API",
    version="1.0.0",
)


MOCK_RESPONSE = {
    "status": "completed",
    "predictions": [
        {
            "student_hash": "hash_1001",
            "record_number": "1001",
            "predicted_class": "3+",
            "probability_0": 0.05,
            "probability_1": 0.10,
            "probability_2": 0.20,
            "probability_3_plus": 0.65,
        },
        {
            "student_hash": "hash_1002",
            "record_number": "1002",
            "predicted_class": "2",
            "probability_0": 0.10,
            "probability_1": 0.15,
            "probability_2": 0.65,
            "probability_3_plus": 0.10,
        },
        {
            "student_hash": "hash_1003",
            "record_number": "1003",
            "predicted_class": "0",
            "probability_0": 0.80,
            "probability_1": 0.10,
            "probability_2": 0.07,
            "probability_3_plus": 0.03,
        },
    ],
}


@app.on_event("startup")
def startup() -> None:
    run_migrations()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
    }


@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
) -> dict:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Имя файла отсутствует",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только файлы .xlsx",
        )

    file_content = file.file.read()

    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="Загруженный файл пуст",
        )

    if len(file_content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Размер файла превышает 100 МБ",
        )

    upload_id = uuid4()
    stored_filename = f"{upload_id}.xlsx"
    file_path = UPLOAD_DIR / stored_filename

    try:
        file_path.write_bytes(file_content)

        dataframe = pd.read_excel(
            file_path,
            engine="openpyxl",
        )

        if dataframe.empty:
            raise ValueError(
                "Excel-файл не содержит данных"
            )

        required_columns = {
            "hash",
            "Номер ЛД",
        }

        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "В Excel отсутствуют обязательные колонки: "
                + ", ".join(sorted(missing_columns))
            )

        create_upload(
            upload_id=upload_id,
            original_filename=file.filename,
            stored_filename=stored_filename,
            rows_count=len(dataframe),
        )

    except Exception as error:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=f"Ошибка загрузки файла: {error}",
        ) from error

    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "rows_count": len(dataframe),
        "status": "uploaded",
    }


@app.get("/uploads/{upload_id}")
def upload_info(
    upload_id: UUID,
) -> dict:
    upload = get_upload(upload_id)

    if upload is None:
        raise HTTPException(
            status_code=404,
            detail="Загрузка не найдена",
        )

    return upload


@app.get("/predict/mock")
def predict_mock() -> dict:
    return MOCK_RESPONSE


def validate_predictions(
    predictions: list[dict],
) -> None:
    required_fields = {
        "student_hash",
        "record_number",
        "predicted_class",
        "probability_0",
        "probability_1",
        "probability_2",
        "probability_3_plus",
    }

    allowed_classes = {
        "0",
        "1",
        "2",
        "3+",
    }

    if not predictions:
        raise ValueError(
            "ML-модуль вернул пустой список прогнозов"
        )

    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise TypeError(
                f"Прогноз с индексом {index} должен быть объектом"
            )

        missing_fields = (
            required_fields
            - set(prediction)
        )

        if missing_fields:
            raise ValueError(
                f"В прогнозе {index} отсутствуют поля: "
                + ", ".join(sorted(missing_fields))
            )

        student_hash = str(
            prediction["student_hash"]
        ).strip()

        record_number = str(
            prediction["record_number"]
        ).strip()

        if not student_hash:
            raise ValueError(
                f"В прогнозе {index} отсутствует hash"
            )

        if not record_number:
            raise ValueError(
                f"В прогнозе {index} отсутствует номер ЛД"
            )

        predicted_class = str(
            prediction["predicted_class"]
        )

        if predicted_class not in allowed_classes:
            raise ValueError(
                f"Недопустимый класс: {predicted_class}"
            )

        probability_sum = 0.0

        for field in (
            "probability_0",
            "probability_1",
            "probability_2",
            "probability_3_plus",
        ):
            value = float(prediction[field])

            if value < 0 or value > 1:
                raise ValueError(
                    f"{field} должна находиться от 0 до 1"
                )

            probability_sum += value

        if not 0.99 <= probability_sum <= 1.01:
            raise ValueError(
                f"Сумма вероятностей прогноза {index} "
                "должна быть равна 1"
            )


def prepare_predictions(
    result: pd.DataFrame,
    source_dataframe: pd.DataFrame,
) -> list[dict]:
    required_ml_columns = {
        "hash",
        "predicted_category",
        "probability_0",
        "probability_1",
        "probability_2",
        "probability_3+",
    }

    missing_ml_columns = (
        required_ml_columns
        - set(result.columns)
    )

    if missing_ml_columns:
        raise ValueError(
            "В результате ML отсутствуют колонки: "
            + ", ".join(sorted(missing_ml_columns))
        )

    required_source_columns = {
        "hash",
        "Номер ЛД",
    }

    missing_source_columns = (
        required_source_columns
        - set(source_dataframe.columns)
    )

    if missing_source_columns:
        raise ValueError(
            "В исходном Excel отсутствуют колонки: "
            + ", ".join(sorted(missing_source_columns))
        )

    ml_result = result.copy()
    source_students = source_dataframe[
        [
            "hash",
            "Номер ЛД",
        ]
    ].copy()

    # Приводим hash к строке с обеих сторон,
    # чтобы merge не ломался из-за разных типов.
    ml_result["hash"] = (
        ml_result["hash"]
        .astype(str)
        .str.strip()
    )

    source_students["hash"] = (
        source_students["hash"]
        .astype(str)
        .str.strip()
    )

    source_students["Номер ЛД"] = (
        source_students["Номер ЛД"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # В исходном Excel один студент может встречаться
    # несколько раз по разным дисциплинам.
    source_students = source_students.drop_duplicates(
        subset=["hash"]
    )

    # Если ML уже вернул Номер ЛД, удаляем его,
    # чтобы источником номера ЛД всегда был исходный Excel.
    if "Номер ЛД" in ml_result.columns:
        ml_result = ml_result.drop(
            columns=["Номер ЛД"]
        )

    merged_result = ml_result.merge(
        source_students,
        on="hash",
        how="left",
        validate="many_to_one",
    )

    missing_record_numbers = merged_result[
        "Номер ЛД"
    ].isna() | merged_result["Номер ЛД"].eq("")

    if missing_record_numbers.any():
        missing_hashes = (
            merged_result.loc[
                missing_record_numbers,
                "hash",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Для следующих hash не найден номер ЛД: "
            + ", ".join(missing_hashes)
        )

    merged_result = merged_result.rename(
        columns={
            "hash": "student_hash",
            "Номер ЛД": "record_number",
            "predicted_category": "predicted_class",
            "probability_3+": "probability_3_plus",
        }
    )

    predictions = merged_result[
        [
            "student_hash",
            "record_number",
            "predicted_class",
            "probability_0",
            "probability_1",
            "probability_2",
            "probability_3_plus",
        ]
    ].to_dict(
        orient="records"
    )

    normalized_predictions = [
        {
            "student_hash": str(
                prediction["student_hash"]
            ),
            "record_number": str(
                prediction["record_number"]
            ),
            "predicted_class": str(
                prediction["predicted_class"]
            ),
            "probability_0": float(
                prediction["probability_0"]
            ),
            "probability_1": float(
                prediction["probability_1"]
            ),
            "probability_2": float(
                prediction["probability_2"]
            ),
            "probability_3_plus": float(
                prediction["probability_3_plus"]
            ),
        }
        for prediction in predictions
    ]

    validate_predictions(
        normalized_predictions
    )

    return normalized_predictions


@app.post("/predict/{upload_id}")
def predict(
    upload_id: UUID,
) -> dict:
    upload = get_upload(upload_id)

    if upload is None:
        raise HTTPException(
            status_code=404,
            detail="Загрузка не найдена",
        )

    if upload["status"] == "processing":
        raise HTTPException(
            status_code=409,
            detail="Прогноз уже выполняется",
        )

    file_path = (
        UPLOAD_DIR
        / upload["stored_filename"]
    )

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Загруженный файл не найден",
        )

    update_upload_status(
        upload_id=upload_id,
        status="processing",
        error_message=None,
    )

    try:
        result = run_inference(
            str(file_path)
        )

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                "run_inference должна вернуть pandas.DataFrame"
            )

        source_dataframe = pd.read_excel(
            file_path,
            engine="openpyxl",
        )

        predictions = prepare_predictions(
            result=result,
            source_dataframe=source_dataframe,
        )

        save_predictions(
            upload_id=upload_id,
            predictions=predictions,
        )

        update_upload_status(
            upload_id=upload_id,
            status="completed",
            error_message=None,
        )

    except Exception as error:
        update_upload_status(
            upload_id=upload_id,
            status="failed",
            error_message=str(error),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка прогнозирования: {error}",
        ) from error

    return {
        "upload_id": upload_id,
        "status": "completed",
        "predictions_count": len(predictions),
    }


@app.get("/predictions/{upload_id}")
def predictions(
    upload_id: UUID,
) -> dict:
    upload = get_upload(upload_id)

    if upload is None:
        raise HTTPException(
            status_code=404,
            detail="Загрузка не найдена",
        )

    if upload["status"] == "processing":
        raise HTTPException(
            status_code=409,
            detail="Прогноз ещё выполняется",
        )

    if upload["status"] == "failed":
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Прогноз завершился с ошибкой",
                "error": upload["error_message"],
            },
        )

    result = get_predictions(upload_id)

    return {
        "upload_id": upload_id,
        "status": upload["status"],
        "predictions": result,
    }


@app.get("/export/mock")
def export_mock() -> FileResponse:
    predictions = MOCK_RESPONSE["predictions"]

    dataframe = pd.DataFrame(predictions)

    dataframe = dataframe.rename(
        columns={
            "student_hash": "Hash",
            "record_number": "Номер ЛД",
            "predicted_class": "Прогноз",
            "probability_0": "Вероятность 0",
            "probability_1": "Вероятность 1",
            "probability_2": "Вероятность 2",
            "probability_3_plus": "Вероятность 3+",
        }
    )

    result_filename = "mock_predictions.xlsx"
    result_path = UPLOAD_DIR / result_filename

    try:
        dataframe.to_excel(
            result_path,
            index=False,
            engine="openpyxl",
        )

    except Exception as error:
        result_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка mock-экспорта: {error}",
        ) from error

    return FileResponse(
        path=result_path,
        filename=result_filename,
        media_type=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
    )


@app.get("/export")
def export_predictions(
    upload_id: UUID,
) -> FileResponse:
    upload = get_upload(upload_id)

    if upload is None:
        raise HTTPException(
            status_code=404,
            detail="Загрузка не найдена",
        )

    if upload["status"] == "processing":
        raise HTTPException(
            status_code=409,
            detail="Прогноз ещё выполняется",
        )

    if upload["status"] == "failed":
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Прогноз завершился с ошибкой",
                "error": upload["error_message"],
            },
        )

    if upload["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="Прогноз ещё не выполнен",
        )

    predictions_result = get_predictions(
        upload_id
    )

    if not predictions_result:
        raise HTTPException(
            status_code=404,
            detail="Результаты прогноза не найдены",
        )

    dataframe = pd.DataFrame(
        predictions_result
    )

    dataframe = dataframe.rename(
        columns={
            "student_hash": "Hash",
            "record_number": "Номер ЛД",
            "predicted_class": "Прогноз",
            "probability_0": "Вероятность 0",
            "probability_1": "Вероятность 1",
            "probability_2": "Вероятность 2",
            "probability_3_plus": "Вероятность 3+",
        }
    )

    result_filename = (
        f"{upload_id}_result.xlsx"
    )

    result_path = (
        UPLOAD_DIR
        / result_filename
    )

    try:
        dataframe.to_excel(
            result_path,
            index=False,
            engine="openpyxl",
        )

    except Exception as error:
        result_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка экспорта: {error}",
        ) from error

    return FileResponse(
        path=result_path,
        filename=result_filename,
        media_type=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
    )