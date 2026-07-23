FROM python:3.12.3

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY main.py ./main.py

CMD ["python", "main.py"]