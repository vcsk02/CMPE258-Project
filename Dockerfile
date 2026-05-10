FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY examples ./examples

RUN pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

EXPOSE 7860
CMD ["python", "app.py"]
