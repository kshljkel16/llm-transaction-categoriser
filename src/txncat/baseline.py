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

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

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
        self.pipeline = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
            LogisticRegression(C=10, max_iter=2000),
        )

    @property
    def classes_(self) -> list[str]:
        return list(self.pipeline.classes_)



    def fit(self, texts: list[str], labels: list[str]) -> "BaselineClassifier":
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict_with_confidence(self, texts: list[str]) -> list[Prediction]:
        t0 = time.perf_counter()
        proba = self.predict_proba(texts)  # shape: (n_texts, n_classes)
        elapsed = time.perf_counter() - t0
        per_item = elapsed / max(len(texts), 1)  # max(..., 1) avoids dividing by 0 on empty input

        classes = self.classes_  # e.g. ["Bills", "Cash", "Entertainment", ...]
        predictions = []
        for row in proba:  # one row per text, e.g. [0.01, 0.02, 0.91, ...]
            best = row.argmax()  # position of the largest probability -> 2
            predictions.append(
                Prediction(
                    category=str(classes[best]),  # "Entertainment"  (str(): plain Python string)
                    confidence=float(row[best]),  # 0.91             (float(): plain Python float)
                    source="baseline",
                    latency_s=per_item,
                )
            )
        return predictions


    def save(self, path: Path = MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "BaselineClassifier":
        obj = cls()
        obj.pipeline = joblib.load(path)
        return obj


def main() -> None:
    """Train on train.csv, report on val.csv, save the model.

    Print three numbers: accuracy on all val rows, on rows whose merchant was
    seen in training, and on rows whose merchant was NOT seen. The gap between
    the last two is the most important finding of this step.
    """
    # 1. Load the data
    train = pd.read_csv("data/train.csv")
    val = pd.read_csv("data/val.csv")                                 # same for val

    # 2. Train and save
    model = BaselineClassifier().fit(train.description.tolist(), train.category.tolist())
    model.save()

    # 3. Predict on val and compare with the truth
    pred = np.array(model.predict(val.description.tolist()))
    correct = pred == val.category.to_numpy()   # array of True/False, one per row

    # 4. Split by seen / unseen merchant
    seen = (val.merchant_seen_in_train == 1).to_numpy()                                 # True where merchant_seen_in_train == 1

    print(f"val accuracy (all):    {correct.mean():.3f}")
    print(f"val accuracy (seen):   {correct[seen].mean():.3f}")
    print(f"val accuracy (unseen): {correct[~seen].mean():.3f}")


if __name__ == "__main__":
    main()
