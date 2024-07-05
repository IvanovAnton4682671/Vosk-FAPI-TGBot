FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN apt update -y && apt upgrade -y && apt install ffmpeg -y

RUN pip install --no-cache-dir  fastapi uvicorn vosk pydub

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
