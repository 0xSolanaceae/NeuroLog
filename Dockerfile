FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi --no-root

COPY . .

CMD ["python", "main.py"]