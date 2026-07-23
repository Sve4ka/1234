from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DB_NAME = "postgres"
DB_PORT = 5432
DB_PASSWORD = "1234"
DB_USER = "root"
DB_HOST = "postgres"

UPLOAD_DIR = BASE_DIR / "data"

MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".xlsx"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)