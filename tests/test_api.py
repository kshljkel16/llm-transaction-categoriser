"""Step 8. Uses a fake classifier: no model file, no API key."""
from fastapi.testclient import TestClient

import pytest

from txncat.api import MAX_BATCH, create_app
from txncat.schema import Prediction


class FakeClassifier:
    def predict(self, texts):
        return [Prediction("Groceries", merchant="Tesco", confidence=0.9) for _ in texts]


@pytest.fixture
def client():
    return TestClient(create_app(FakeClassifier()))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_categorise_shape_and_order(client):
    body = {"transactions": [{"description": "TESCO 1"}, {"description": "TESCO 2", "amount": 3.5}]}
    r = client.post("/categorise", json=body)
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["description"] for x in results] == ["TESCO 1", "TESCO 2"]
    assert results[0] == {"description": "TESCO 1", "category": "Groceries", "merchant": "Tesco",
                          "confidence": 0.9, "source": "baseline"}


def test_validation(client):
    assert client.post("/categorise", json={"transactions": []}).status_code == 422
    too_many = {"transactions": [{"description": "X"}] * (MAX_BATCH + 1)}
    assert client.post("/categorise", json=too_many).status_code == 422
    assert client.post("/categorise", json={"transactions": [{"description": ""}]}).status_code == 422
    assert client.post("/categorise", json={"transactions": [{"description": "X" * 201}]}).status_code == 422
