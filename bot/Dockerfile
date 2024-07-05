FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir aiogram requests

ENV BOT_TOKEN=${BOT_TOKEN}

ENV BACKEND_URL=${BACKEND_URL}

CMD ["python3", "bot.py"]
