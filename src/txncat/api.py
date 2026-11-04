"""Step 8 — Serve the hybrid classifier as an HTTP API.

Run:   uvicorn txncat.api:app --reload      then open http://localhost:8000/docs
Test:  pytest tests/test_api.py
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

MAX_BATCH = 100
MAX_DESCRIPTION_LEN = 200


# ---- Request / response shapes (FastAPI validates against these) ----------
class TransactionIn(BaseModel):
    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_LEN)        # blank 1: the length limit above
    amount: float | None = None
    currency: str | None = None


class CategoriseRequest(BaseModel):
    transactions: list[TransactionIn] = Field(min_length=1, max_length=MAX_BATCH)   # blank 2: the batch limit


class CategoryOut(BaseModel):
    description: str
    category: str
    merchant: str | None
    confidence: float | None
    source: str


class CategoriseResponse(BaseModel):
    results: list[CategoryOut]


# ---- The app ---------------------------------------------------------------
def create_app(classifier) -> FastAPI:
    """Build the API around any object with .predict(texts) -> list[Prediction]."""
    app = FastAPI(title="Transaction Categoriser", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/categorise", response_model=CategoriseResponse)
    def categorise(req: CategoriseRequest) -> CategoriseResponse:
        texts = [t.description for t in req.transactions]
        preds = classifier.predict(texts)                                                # blank 3: run the classifier
        return CategoriseResponse(results=[
            CategoryOut(description=t, category=p.category, merchant=p.merchant,
                        confidence=p.confidence, source=p.source)
            for t, p in zip(texts, preds)
        ])

    return app


def build_default_app() -> FastAPI:
    """The real service: saved baseline + Kimi + the threshold chosen in Step 7."""
    from dotenv import load_dotenv

    from txncat.baseline import BaselineClassifier
    from txncat.llm_classifier import LLMClassifier
    from txncat.llm_client import make_client_from_env
    from txncat.router import HybridClassifier

    load_dotenv()
    llm = LLMClassifier(
        make_client_from_env(),
        batch_size=int(os.getenv("LLM_BATCH_SIZE", "20")),
        price_in_per_mtok=float(os.getenv("LLM_PRICE_INPUT_PER_MTOK") or 0),
        price_out_per_mtok=float(os.getenv("LLM_PRICE_OUTPUT_PER_MTOK") or 0),
        cache_path=Path(os.getenv("LLM_CACHE_PATH", ".cache/llm_cache.json")),
    )
    hybrid = HybridClassifier(BaselineClassifier.load(), llm,
                              threshold=float(os.getenv("HYBRID_THRESHOLD", "0.55")))
    return create_app(hybrid)


# uvicorn asks this module for `app`. Building it lazily means importing the
# module in tests needs no model file or API key; _app makes sure it's built once.
_app: FastAPI | None = None


def __getattr__(name: str):
    global _app
    if name == "app":
        if _app is None:
            _app = build_default_app()
        return _app
    raise AttributeError(name)