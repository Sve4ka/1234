from typing import Any
from uuid import UUID

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from backend.config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
)


def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def create_upload(
    upload_id: UUID,
    original_filename: str,
    stored_filename: str,
    rows_count: int,
) -> None:
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO uploads (
                    id,
                    original_filename,
                    stored_filename,
                    rows_count,
                    status
                )
                VALUES (%s, %s, %s, %s, 'uploaded');
                """,
                (
                    str(upload_id),
                    original_filename,
                    stored_filename,
                    rows_count,
                ),
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_upload(
    upload_id: UUID,
) -> dict[str, Any] | None:
    connection = get_connection()

    try:
        with connection.cursor(
            cursor_factory=RealDictCursor,
        ) as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    original_filename,
                    stored_filename,
                    rows_count,
                    status,
                    error_message,
                    created_at,
                    updated_at
                FROM uploads
                WHERE id = %s;
                """,
                (str(upload_id),),
            )

            upload = cursor.fetchone()

    finally:
        connection.close()

    if upload is None:
        return None

    return dict(upload)


def update_upload_status(
    upload_id: UUID,
    status: str,
    error_message: str | None = None,
) -> None:
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE uploads
                SET
                    status = %s,
                    error_message = %s,
                    updated_at = NOW()
                WHERE id = %s;
                """,
                (
                    status,
                    error_message,
                    str(upload_id),
                ),
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def save_predictions(
    upload_id: UUID,
    predictions: list[dict[str, Any]],
) -> None:
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM predictions
                WHERE upload_id = %s;
                """,
                (str(upload_id),),
            )

            values = [
                (
                    str(upload_id),
                    str(prediction["student_hash"]),
                    str(prediction["record_number"]),
                    str(prediction["predicted_class"]),
                    float(prediction["probability_0"]),
                    float(prediction["probability_1"]),
                    float(prediction["probability_2"]),
                    float(
                        prediction["probability_3_plus"]
                    ),
                )
                for prediction in predictions
            ]

            if values:
                execute_values(
                    cursor,
                    """
                    INSERT INTO predictions (
                        upload_id,
                        student_hash,
                        record_number,
                        predicted_class,
                        probability_0,
                        probability_1,
                        probability_2,
                        probability_3_plus
                    )
                    VALUES %s;
                    """,
                    values,
                )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_predictions(
    upload_id: UUID,
) -> list[dict[str, Any]]:
    connection = get_connection()

    try:
        with connection.cursor(
            cursor_factory=RealDictCursor,
        ) as cursor:
            cursor.execute(
                """
                SELECT
                    student_hash,
                    record_number,
                    predicted_class,
                    probability_0,
                    probability_1,
                    probability_2,
                    probability_3_plus
                FROM predictions
                WHERE upload_id = %s
                ORDER BY
                    CASE predicted_class
                        WHEN '3+' THEN 4
                        WHEN '2' THEN 3
                        WHEN '1' THEN 2
                        WHEN '0' THEN 1
                        ELSE 0
                    END DESC,
                    probability_3_plus DESC,
                    probability_2 DESC,
                    probability_1 DESC;
                """,
                (str(upload_id),),
            )

            rows = cursor.fetchall()

    finally:
        connection.close()

    return [
        dict(row)
        for row in rows
    ]


def delete_upload(
    upload_id: UUID,
) -> None:
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM uploads
                WHERE id = %s;
                """,
                (str(upload_id),),
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()