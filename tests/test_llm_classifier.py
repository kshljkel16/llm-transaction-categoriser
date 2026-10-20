"""Step 5. All tests use FakeClient — no API key needed."""
import json
import re

import pytest

from txncat.categories import CATEGORIES, UNKNOWN
from txncat.llm_classifier import LLMClassifier, build_messages, parse_response
from txncat.llm_client import FakeClient


# ---------- build_messages ----------
def test_system_prompt_lists_every_category():
    system, _ = build_messages(["TESCO 1"])
    for name, desc in CATEGORIES.items():
        assert f"- {name}: {desc}" in system


def test_user_prompt_is_numbered():
    _, user = build_messages(["TESCO 1", "NETFLIX.COM"])
    assert "1. TESCO 1" in user and "2. NETFLIX.COM" in user


# ---------- parse_response ----------
def test_parse_clean_json():
    text = json.dumps([{"i": 1, "category": "Groceries", "merchant": "Tesco"},
                       {"i": 2, "category": "Subscriptions", "merchant": "Netflix"}])
    preds = parse_response(text, 2)
    assert [(p.category, p.merchant) for p in preds] == [("Groceries", "Tesco"), ("Subscriptions", "Netflix")]
    assert all(p.source == "llm" and p.confidence is None for p in preds)


def test_parse_fenced_and_prose():
    text = 'Sure! Here you go:\n```json\n[{"i": 1, "category": "Transport", "merchant": "Uber"}]\n```\nHope that helps.'
    assert parse_response(text, 1)[0].category == "Transport"


def test_parse_normalises_case():
    assert parse_response('[{"i": 1, "category": "groceries", "merchant": "Lidl"}]', 1)[0].category == "Groceries"


def test_parse_invalid_category_is_unknown():
    assert parse_response('[{"i": 1, "category": "Food", "merchant": "Lidl"}]', 1)[0].category == UNKNOWN


def test_parse_uses_index_not_position_and_fills_missing():
    text = '[{"i": 3, "category": "Cash", "merchant": null}, {"i": 1, "category": "Travel", "merchant": "Ryanair"}]'
    preds = parse_response(text, 3)
    assert [p.category for p in preds] == ["Travel", UNKNOWN, "Cash"]
    assert preds[2].merchant is None


def test_parse_garbage():
    preds = parse_response("I cannot help with that.", 4)
    assert len(preds) == 4 and all(p.category == UNKNOWN for p in preds)
    assert all(p.source == "fallback" for p in preds)


def test_parse_empty_merchant_is_none():
    assert parse_response('[{"i": 1, "category": "Transfers", "merchant": ""}]', 1)[0].merchant is None


# ---------- LLMClassifier ----------
def echo_responder(system, user):
    """Pretends to be a perfect LLM: categorises everything as Shopping."""
    ids = [int(m) for m in re.findall(r"^(\d+)\. ", user, flags=re.M)]
    return json.dumps([{"i": i, "category": "Shopping", "merchant": "Shop"} for i in ids])


def test_batching():
    client = FakeClient(echo_responder)
    clf = LLMClassifier(client, batch_size=10)
    preds = clf.predict([f"SHOP {i}" for i in range(25)])
    assert len(preds) == 25 and client.calls == 3


def test_cache_prevents_repeat_calls_and_cost():
    client = FakeClient(echo_responder)
    clf = LLMClassifier(client, batch_size=10, price_in_per_mtok=1.0, price_out_per_mtok=1.0)
    first = clf.predict(["SHOP A", "SHOP B"])
    second = clf.predict(["shop a ", "SHOP B"])           # same keys after strip+upper
    assert client.calls == 1
    assert all(p.cost_usd > 0 for p in first)
    assert all(p.cost_usd == 0 for p in second)
    assert [p.category for p in second] == ["Shopping", "Shopping"]


def test_cost_maths():
    client = FakeClient(echo_responder)
    clf = LLMClassifier(client, batch_size=50, price_in_per_mtok=3.0, price_out_per_mtok=15.0)
    texts = [f"SHOP {i}" for i in range(4)]
    preds = clf.predict(texts)
    system, user = build_messages(texts)
    out = echo_responder(system, user)
    expected = ((len(system) + len(user)) // 4) / 1e6 * 3.0 + (len(out) // 4) / 1e6 * 15.0
    assert sum(p.cost_usd for p in preds) == pytest.approx(expected)


def test_unknown_is_not_cached():
    client = FakeClient(lambda s, u: "garbage")
    clf = LLMClassifier(client, batch_size=10)
    clf.predict(["X"])
    clf.predict(["X"])
    assert client.calls == 2


def test_cache_persists_to_disk(tmp_path):
    path = tmp_path / "cache.json"
    LLMClassifier(FakeClient(echo_responder), cache_path=path).predict(["SHOP A"])
    client = FakeClient(echo_responder)
    LLMClassifier(client, cache_path=path).predict(["SHOP A"])
    assert client.calls == 0
