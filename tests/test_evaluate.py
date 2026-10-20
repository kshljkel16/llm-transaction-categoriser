"""Step 7."""
import pytest

from txncat.categories import UNKNOWN
from txncat.evaluate import compute_metrics, merchant_accuracy, simulate_hybrid, summarise_costs
from txncat.schema import Prediction


def test_metrics_small_example():
    y_true = ["Groceries", "Groceries", "Cash", "Travel"]
    y_pred = ["Groceries", "Cash", "Cash", UNKNOWN]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.5)
    g, c, t = m["per_class"]["Groceries"], m["per_class"]["Cash"], m["per_class"]["Travel"]
    assert (g["precision"], g["recall"], g["support"]) == (1.0, 0.5, 2)
    assert (c["precision"], c["recall"]) == (0.5, 1.0)
    assert (t["precision"], t["recall"], t["f1"]) == (0.0, 0.0, 0.0)
    # macro F1 over the 3 classes with support: Groceries 2/3, Cash 2/3, Travel 0
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert m["per_class"]["Bills"]["support"] == 0


def test_merchant_accuracy_normalises_and_skips():
    true = ["Sainsbury's", "M&S Food", None, "", "Tesco"]
    pred = ["sainsburys", "MS FOOD", "whatever", "x", None]
    assert merchant_accuracy(true, pred) == pytest.approx(2 / 3)


def test_summarise_costs():
    preds = [Prediction("Cash", cost_usd=0.001, latency_s=s) for s in (0.1, 0.2, 0.3, 0.4)]
    s = summarise_costs(preds)
    assert s["total_cost_usd"] == pytest.approx(0.004)
    assert s["cost_per_1k_usd"] == pytest.approx(1.0)
    assert s["latency_p50_s"] == pytest.approx(0.25)
    assert summarise_costs([])["total_cost_usd"] == 0


def test_simulate_hybrid():
    base = [Prediction("A", confidence=c) for c in (0.9, 0.5, 0.7)]
    llm = [Prediction("L", source="llm") for _ in range(3)]
    combined, share = simulate_hybrid(base, llm, threshold=0.7)
    assert [p.category for p in combined] == ["A", "L", "A"]
    assert share == pytest.approx(1 / 3)
