"""Step 1 — these pass from the start. Read them: they define the data contract."""
import pandas as pd

from txncat.categories import CATEGORY_NAMES
from txncat.generate_data import generate


def test_generation_is_deterministic():
    a, _, _ = generate(seed=1, n_train=50, n_val=10, n_test=10)
    b, _, _ = generate(seed=1, n_train=50, n_val=10, n_test=10)
    assert [r.description for r in a] == [r.description for r in b]


def test_no_unseen_merchant_leaks_into_train(data_dir):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    unseen_groups = set(test.loc[test.merchant_seen_in_train == 0, "merchant_group"])
    assert unseen_groups, "test must contain unseen merchants"
    assert unseen_groups.isdisjoint(set(train.merchant_group))


def test_every_category_is_present_and_valid(data_dir):
    train = pd.read_csv(data_dir / "train.csv")
    assert set(train.category) == set(CATEGORY_NAMES)


def test_val_and_test_have_about_30_percent_unseen(data_dir):
    for split in ("val", "test"):
        df = pd.read_csv(data_dir / f"{split}.csv")
        share = (df.merchant_seen_in_train == 0).mean()
        assert 0.25 <= share <= 0.35
