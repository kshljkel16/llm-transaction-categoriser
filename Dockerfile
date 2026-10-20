# Step 8 — container for the API. Read GUIDE.md, Step 8, before changing anything.
FROM python:3.12-slim

WORKDIR /app

# Copy dependency metadata first so this layer is cached when only code changes.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# The trained baseline is baked into the image; the API key is NOT (it comes from env at runtime).
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "txncat.api:app", "--host", "0.0.0.0", "--port", "8000"]
