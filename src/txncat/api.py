"""Step 8 — Serve the hybrid classifier as an HTTP API.

Run when done:
    uvicorn txncat.api:app --reload
    curl -X POST localhost:8000/categorise -H "Content-Type: application/json" \
         -d '{"transactions": [{"description": "AMZN MKTP UK*2K4LZ93"}]}'

Test:  pytest tests/test_api.py
"""

from __future__ import annotations

from fastapi import FastAPI

MAX_BATCH = 100


def create_app(classifier) -> FastAPI:
    """Build the FastAPI app around any object with .predict(texts) -> list[Prediction].

    Endpoints (tested):
      GET  /health      -> 200 {"status": "ok"}
      POST /categorise
          request : {"transactions": [{"description": str, "amount": float | null,
                                       "currency": str | null}, ...]}
          response: {"results": [{"description", "category", "merchant",
                                  "confidence", "source"}, ...]}  same order as input
          * 422 if transactions is empty or longer than MAX_BATCH
          * 422 if any description is empty or longer than 200 characters
            (use Pydantic models with Field constraints — FastAPI then returns
             422 automatically; you write no validation `if`s)

    Taking `classifier` as an argument (dependency injection) is what lets the
    tests pass a fake: no model file, no API key, no network.
    """
    raise NotImplementedError("Step 8: implement create_app")


def build_default_app() -> FastAPI:
    """Wire the real HybridClassifier: BaselineClassifier.load(), an LLMClassifier built from
    make_client_from_env() and the price env vars, and float(os.getenv("HYBRID_THRESHOLD", 0.8)).
    Then return create_app(that_hybrid)."""
    raise NotImplementedError("Step 8: implement build_default_app")


# uvicorn looks for this module-level `app`. It is created lazily so that
# importing this module in tests does not require a model file or API key.
def __getattr__(name: str):
    if name == "app":
        return build_default_app()
    raise AttributeError(name)
