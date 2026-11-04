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
        self.baseline = baseline
        self.llm = llm
        self.threshold = threshold
        self.last_llm_share = 0.0

    def predict(self, texts: list[str]) -> list[Prediction]:
        if not texts:                                         # empty input: no calls at all
            self.last_llm_share = 0.0
            return []

        # 1. Baseline on EVERYTHING (it's ~free)
        results = self.baseline.predict_with_confidence(texts)

        # 2. Which positions is the baseline NOT sure about?
        unsure = [i for i, p in enumerate(results) if p.confidence < self.threshold]       # blank 1

        # 3. Send ALL unsure texts to the LLM in ONE call, then put answers back in place
        if unsure:
            llm_preds = self.llm.predict([texts[i] for i in unsure])               # blank 2
            for i, p in zip(unsure, llm_preds):
                results[i] = p

        # 4. Record what share went to the LLM (the cost driver)
        self.last_llm_share = len(unsure) / len(texts)                            # blank 3
        return results
