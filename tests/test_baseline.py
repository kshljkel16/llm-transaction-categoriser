"""Step 3."""
import numpy as np
import pandas as pd
import pytest

from txncat.baseline import BaselineClassifier


@pytest.fixture(scope="module")
def trained(data_dir):
    train = pd.read_csv(data_dir / "train.csv")
    return BaselineClassifier().fit(train.description.tolist(), train.category.tolist())


@pytest.fixture(scope="module")
def val(data_dir):
    return pd.read_csv(data_dir / "val.csv")


def test_accuracy_targets(trained, val):
    pred = np.array(trained.predict(val.description.tolist()))
    correct = pred == val.category.to_numpy()
    seen = val.merchant_seen_in_train.to_numpy() == 1
    assert correct.mean() >= 0.75, f"overall val accuracy {correct.mean():.3f}"
    assert correct[seen].mean() >= 0.95, f"seen-merchant accuracy {correct[seen].mean():.3f}"


def test_proba_shape_and_rows_sum_to_one(trained, val):
    p = trained.predict_proba(val.description.tolist()[:20])
    assert p.shape == (20, len(trained.classes_))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_predict_with_confidence(trained, data_dir):
    train = pd.read_csv(data_dir / "train.csv")
    rows = train.groupby("category").head(1)           # one known example per category
    preds = trained.predict_with_confidence(rows.description.tolist())
    assert [p.category for p in preds] == rows.category.tolist()
    for p in preds:
        assert type(p.category) is str, "convert numpy strings with str() — JSON needs plain str"
        assert type(p.confidence) is float
        assert p.source == "baseline" and p.cost_usd == 0.0
        assert 0.0 < p.confidence <= 1.0


def test_confidence_is_lower_on_unseen_merchants(trained, val):
    """The property the whole hybrid design relies on."""
    preds = trained.predict_with_confidence(val.description.tolist())
    conf = np.array([p.confidence for p in preds])
    seen = val.merchant_seen_in_train.to_numpy() == 1
    assert conf[seen].mean() > conf[~seen].mean() + 0.1


def test_save_load_roundtrip(trained, tmp_path, val):
    path = tmp_path / "m.joblib"
    trained.save(path)
    loaded = BaselineClassifier.load(path)
    texts = val.description.tolist()[:50]
    assert loaded.predict(texts) == trained.predict(texts)
