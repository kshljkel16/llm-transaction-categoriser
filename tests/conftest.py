"""Shared fixtures. Data is generated once per test session into a temp folder,
so tests never depend on files you may have edited by hand."""
import pytest

from txncat.generate_data import generate, write_csv


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    train, val, test = generate(seed=42)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        write_csv(rows, d / f"{name}.csv")
    return d
