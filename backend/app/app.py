from importlib import import_module
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


app = FastAPI(
    title="Academic Debt Prediction API",
    version="1.0.0",
)


MOCK_RESPONSE = {
    "status": "completed",
    "predictions": [
        {
            "student_id": "1001",
            "student_name": "Иванов Иван Иванович",
            "predicted_class": "3+",
            "probability_0": 0.05,
            "probability_1": 0.10,
            "probability_2": 0.20,
            "probability_3_plus": 0.65,
        },
        {
            "student_id": "1002",
            "student_name": "Петров Пётр Петрович",
            "predicted_class": "2",
            "probability_0": 0.10,
            "probability_1": 0.15,
            "probability_2": 0.65,
            "probability_3_plus": 0.10,
        },
        {
            "student_id": "1003",
            "student_name": "Сидорова Анна Сергеевна",
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


def call_ml_prediction(
    file_path: Path,
) -> list[dict]:
    try:
        ml_module = import_module("ml.inference")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ML-модуль ml.inference ещё не подключён"
        ) from error

    predict_file = getattr(
        ml_module,
        "predict_file",
        None,
    )

    if not callable(predict_file):
        raise RuntimeError(
            "В ml.inference отсутствует функция predict_file"
        )

    result = predict_file(str(file_path))

    if not isinstance(result, list):
        raise TypeError(
            "predict_file должна вернуть list[dict]"
        )

    return result


def validate_predictions(
    predictions: list[dict],
) -> None:
    required_fields = {
        "student_id",
        "student_name",
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

        predicted_class = str(
            prediction["predicted_class"]
        )

        if predicted_class not in allowed_classes:
            raise ValueError(
                f"Недопустимый класс: {predicted_class}"
            )

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
    )

    try:
        predictions = call_ml_prediction(file_path) # todo подключить функцию

        validate_predictions(predictions)

        normalized_predictions = [
            {
                "student_id": str(
                    prediction["student_id"]
                ),
                "student_name": str(
                    prediction["student_name"]
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

        save_predictions(
            upload_id=upload_id,
            predictions=normalized_predictions,
        )

        update_upload_status(
            upload_id=upload_id,
            status="completed",
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
        "predictions_count": len(
            normalized_predictions
        ),
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

    result = get_predictions(upload_id)

    return {
        "upload_id": upload_id,
        "status": upload["status"],
        "predictions": result,
    }


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

    if upload["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="Прогноз ещё не выполнен",
        )

    predictions = get_predictions(upload_id)

    if not predictions:
        raise HTTPException(
            status_code=404,
            detail="Результаты прогноза не найдены",
        )

    dataframe = pd.DataFrame(predictions)

    dataframe = dataframe.rename(
        columns={
            "student_id": "ID студента",
            "student_name": "ФИО студента",
            "predicted_class": "Прогноз",
            "probability_0": "Вероятность 0",
            "probability_1": "Вероятность 1",
            "probability_2": "Вероятность 2",
            "probability_3_plus": "Вероятность 3+",
        }
    )

    result_filename = f"{upload_id}_result.xlsx"
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

@app.get("/export/mock")
def export_mock() -> FileResponse:
    predictions = MOCK_RESPONSE["predictions"]

    dataframe = pd.DataFrame(predictions)

    dataframe = dataframe.rename(
        columns={
            "student_id": "ID студента",
            "student_name": "ФИО студента",
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