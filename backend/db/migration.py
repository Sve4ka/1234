from backend.db.db import get_connection


def run_migrations() -> None:
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS uploads (
                    id UUID PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    rows_count INTEGER NOT NULL,

                    status TEXT NOT NULL DEFAULT 'uploaded',
                    error_message TEXT,

                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                    CONSTRAINT uploads_status_check
                        CHECK (
                            status IN (
                                'uploaded',
                                'processing',
                                'completed',
                                'failed'
                            )
                        )
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGSERIAL PRIMARY KEY,

                    upload_id UUID NOT NULL
                        REFERENCES uploads(id)
                        ON DELETE CASCADE,

                    student_hash TEXT NOT NULL,
                    record_number TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,

                    probability_0 DOUBLE PRECISION NOT NULL,
                    probability_1 DOUBLE PRECISION NOT NULL,
                    probability_2 DOUBLE PRECISION NOT NULL,
                    probability_3_plus DOUBLE PRECISION NOT NULL,

                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                    CONSTRAINT predictions_class_check
                        CHECK (
                            predicted_class IN (
                                '0',
                                '1',
                                '2',
                                '3+'
                            )
                        ),

                    CONSTRAINT predictions_probability_0_check
                        CHECK (
                            probability_0 >= 0
                            AND probability_0 <= 1
                        ),

                    CONSTRAINT predictions_probability_1_check
                        CHECK (
                            probability_1 >= 0
                            AND probability_1 <= 1
                        ),

                    CONSTRAINT predictions_probability_2_check
                        CHECK (
                            probability_2 >= 0
                            AND probability_2 <= 1
                        ),

                    CONSTRAINT predictions_probability_3_plus_check
                        CHECK (
                            probability_3_plus >= 0
                            AND probability_3_plus <= 1
                        )
                );
                """
            )

            # Переименование старых колонок для уже существующей БД.
            cursor.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'predictions'
                          AND column_name = 'student_id'
                    )
                    AND NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'predictions'
                          AND column_name = 'student_hash'
                    )
                    THEN
                        ALTER TABLE predictions
                            RENAME COLUMN student_id
                            TO student_hash;
                    END IF;
                END
                $$;
                """
            )

            cursor.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'predictions'
                          AND column_name = 'student_name'
                    )
                    AND NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'predictions'
                          AND column_name = 'record_number'
                    )
                    THEN
                        ALTER TABLE predictions
                            RENAME COLUMN student_name
                            TO record_number;
                    END IF;
                END
                $$;
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    predictions_upload_id_index
                ON predictions(upload_id);
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    predictions_student_hash_index
                ON predictions(student_hash);
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    predictions_record_number_index
                ON predictions(record_number);
                """
            )

            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    predictions_upload_student_unique_index
                ON predictions(
                    upload_id,
                    student_hash,
                    record_number
                );
                """
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()