"""Step 6."""
from txncat.router import HybridClassifier
from txncat.schema import Prediction


class FakeBaseline:
    def __init__(self, confidences):
        self.confidences = confidences
        self.calls = 0

    def predict_with_confidence(self, texts):
        self.calls += 1
        return [Prediction("Groceries", confidence=self.confidences[t]) for t in texts]


class FakeLLM:
    def __init__(self):
        self.received = []

    def predict(self, texts):
        self.received.append(list(texts))
        return [Prediction("Travel", source="llm") for _ in texts]


CONF = {"a": 0.95, "b": 0.40, "c": 0.85, "d": 0.10}


def test_routes_low_confidence_to_llm_in_one_call_preserving_order():
    llm = FakeLLM()
    router = HybridClassifier(FakeBaseline(CONF), llm, threshold=0.8)
    preds = router.predict(["a", "b", "c", "d"])
    assert [p.category for p in preds] == ["Groceries", "Travel", "Groceries", "Travel"]
    assert llm.received == [["b", "d"]]
    assert router.last_llm_share == 0.5


def test_threshold_extremes():
    llm = FakeLLM()
    assert HybridClassifier(FakeBaseline(CONF), llm, threshold=0.0).predict(list("abcd"))
    assert llm.received == []
    r = HybridClassifier(FakeBaseline(CONF), llm, threshold=1.01)
    r.predict(list("abcd"))
    assert r.last_llm_share == 1.0


def test_empty_input():
    base, llm = FakeBaseline(CONF), FakeLLM()
    r = HybridClassifier(base, llm)
    assert r.predict([]) == [] and llm.received == [] and r.last_llm_share == 0.0
