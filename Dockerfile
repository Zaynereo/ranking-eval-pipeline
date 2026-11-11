FROM python:3.11-slim-bookworm

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-m", "pytest", "-q"]
