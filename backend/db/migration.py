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

                    student_id TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,

                    probability_0 DOUBLE PRECISION NOT NULL,
                    probability_1 DOUBLE PRECISION NOT NULL,
                    probability_2 DOUBLE PRECISION NOT NULL,
                    probability_3_plus DOUBLE PRECISION NOT NULL,

                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    predictions_upload_id_index
                ON predictions(upload_id);
                """
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()