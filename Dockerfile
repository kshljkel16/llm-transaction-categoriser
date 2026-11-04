# Container for the transaction-categoriser API.
#
# Build:  docker build -t txncat .
#         (from China, add a PyPI mirror:
#          docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t txncat .)
# Run:    docker run --rm -p 8000:8000 --env-file .env txncat
#         then open http://localhost:8000/docs

FROM python:3.12-slim

# No .pyc files in the image; logs are printed immediately instead of buffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Which package index pip uses. Default: the official PyPI. Override with --build-arg.
ARG PIP_INDEX_URL=https://pypi.org/simple

# 1. Dependencies first, in their own layer. Only pyproject.toml is copied, plus an
#    empty placeholder package, so this slow layer is rebuilt ONLY when dependencies
#    change — not every time you edit code.
COPY pyproject.toml ./
RUN mkdir -p src/txncat && touch src/txncat/__init__.py \
    && pip install --no-cache-dir --index-url "$PIP_INDEX_URL" . \
    && rm -rf src build

# 2. Now the real code. --no-deps: dependencies are already installed above.
COPY src ./src
RUN pip install --no-cache-dir --no-deps --index-url "$PIP_INDEX_URL" .

# 3. The trained baseline is baked into the image.
#    The API key is NOT: it is passed at runtime with --env-file.
COPY models ./models

# 4. Run as a normal user, not root: if the app were ever compromised,
#    the attacker would not control the container as root.
RUN useradd --create-home appuser && mkdir -p /app/.cache && chown -R appuser /app
USER appuser

EXPOSE 8000

# Docker checks /health every 30 s and marks the container unhealthy if it stops answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "txncat.api:app", "--host", "0.0.0.0", "--port", "8000"]
