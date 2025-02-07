FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY . .


CMD ["python", "main.py"]