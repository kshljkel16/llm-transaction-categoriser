"""Step 3 — Classical ML baseline: TF-IDF on character n-grams + logistic regression.

Why build this before touching an LLM? Because "is the LLM worth its cost?"
can only be answered against a cheap, fast alternative. See GUIDE.md, Step 3.

Run when done:
    python -m txncat.baseline          # trains on data/train.csv, saves models/baseline.joblib,
                                       # prints accuracy on val (all / seen / unseen merchants)

Test:  pytest tests/test_baseline.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from txncat.schema import Prediction

MODEL_PATH = Path("models/baseline.joblib")


class BaselineClassifier:
    """Wraps an sklearn Pipeline(TfidfVectorizer -> LogisticRegression).

    Required behaviour (the tests check all of it):
      * fit(texts, labels) trains and returns self.
      * predict(texts) -> list[str] of category names.
      * predict_proba(texts) -> np.ndarray of shape (n_texts, n_classes);
        each row sums to 1. Column order = self.classes_.
      * classes_ -> list[str], the category names in the column order of predict_proba.
      * predict_with_confidence(texts) -> list[Prediction] where
        category = most likely class (a plain Python str, not numpy.str_),
        confidence = its probability (a plain Python float),
        source = "baseline", merchant = None, cost_usd = 0.0,
        latency_s = total time for the call divided by len(texts).
      * save(path) / BaselineClassifier.load(path) round-trip (use joblib).

    Design choices to make (and justify in your README):
      * analyzer: "word", "char" or "char_wb"? and ngram_range?
      * lowercase? sublinear_tf? min_df?
      * LogisticRegression C and max_iter.
    Target: val accuracy >= 0.75 overall and >= 0.95 on merchants seen in training.
    """

    def __init__(self) -> None:
        self.pipeline = None  # build your sklearn Pipeline here

    @property
    def classes_(self) -> list[str]:
        raise NotImplementedError

    def fit(self, texts: list[str], labels: list[str]) -> "BaselineClassifier":
        raise NotImplementedError("Step 3: implement fit")

    def predict(self, texts: list[str]) -> list[str]:
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_with_confidence(self, texts: list[str]) -> list[Prediction]:
        raise NotImplementedError

    def save(self, path: Path = MODEL_PATH) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "BaselineClassifier":
        raise NotImplementedError


def main() -> None:
    """Train on train.csv, report on val.csv, save the model.

    Print three numbers: accuracy on all val rows, on rows whose merchant was
    seen in training, and on rows whose merchant was NOT seen. The gap between
    the last two is the most important finding of this step.
    """
    raise NotImplementedError("Step 3: implement main")


if __name__ == "__main__":
    main()
