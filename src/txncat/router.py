"""Step 6 — The hybrid: trust the cheap model when it is confident, ask the LLM otherwise.

This is the product decision of the project. A threshold is a dial between
cost and accuracy; Step 7 measures the whole curve so you can choose it with data.

Test:  pytest tests/test_router.py
"""

from __future__ import annotations

from txncat.schema import Prediction


class HybridClassifier:
    """Route each transaction to the baseline or the LLM.

    Requirements (tested):
      * predict(texts) -> list[Prediction], same length and order as texts.
      * Run baseline.predict_with_confidence on ALL texts first (it is ~free).
      * Texts with confidence >= threshold keep the baseline prediction.
      * Only the remaining texts go to llm.predict — in ONE call, as one list.
      * After each predict(), set self.last_llm_share = (#sent to LLM) / len(texts)
        (0.0 for an empty input).
      * Empty input -> [] and no calls at all.
      * threshold=0.0 -> nothing goes to the LLM; threshold > 1.0 -> everything does.

    `baseline` needs .predict_with_confidence(texts); `llm` needs .predict(texts).
    That is all — duck typing is what lets the tests use simple fakes.
    """

    def __init__(self, baseline, llm, threshold: float = 0.8) -> None:
        raise NotImplementedError("Step 6: implement HybridClassifier")

    def predict(self, texts: list[str]) -> list[Prediction]:
        raise NotImplementedError
