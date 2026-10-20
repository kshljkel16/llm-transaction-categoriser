"""Step 2."""
import sqlite3

import pandas as pd

from txncat.db import load_csv_to_sqlite


def test_load_counts_and_columns(data_dir, tmp_path):
    db = tmp_path / "t.db"
    n = load_csv_to_sqlite(data_dir / "train.csv", db, "train")
    assert n == len(pd.read_csv(data_dir / "train.csv"))
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute("PRAGMA table_info(transactions)")]
    for c in ["id", "description", "amount", "currency", "category", "merchant",
              "merchant_group", "merchant_seen_in_train", "split"]:
        assert c in cols


def test_reload_does_not_duplicate(data_dir, tmp_path):
    db = tmp_path / "t.db"
    load_csv_to_sqlite(data_dir / "val.csv", db, "val")
    load_csv_to_sqlite(data_dir / "val.csv", db, "val")
    load_csv_to_sqlite(data_dir / "test.csv", db, "test")
    con = sqlite3.connect(db)
    counts = dict(con.execute("SELECT split, COUNT(*) FROM transactions GROUP BY split"))
    assert counts == {"val": 1500, "test": 1500}


def test_sql_matches_pandas(data_dir, tmp_path):
    db = tmp_path / "t.db"
    load_csv_to_sqlite(data_dir / "train.csv", db, "train")
    con = sqlite3.connect(db)
    sql = dict(con.execute("SELECT category, COUNT(*) FROM transactions GROUP BY category"))
    assert sql == pd.read_csv(data_dir / "train.csv").category.value_counts().to_dict()


def test_quotes_in_descriptions_survive(tmp_path):
    csv = tmp_path / "q.csv"
    csv.write_text(
        "id,description,amount,currency,category,merchant,merchant_group,merchant_seen_in_train\n"
        "0,SAINSBURY'S LOCAL 12,3.5,GBP,Groceries,Sainsbury's,Groceries:Sainsbury's,1\n")
    db = tmp_path / "q.db"
    load_csv_to_sqlite(csv, db, "train")
    (desc,) = sqlite3.connect(db).execute("SELECT description FROM transactions").fetchone()
    assert desc == "SAINSBURY'S LOCAL 12"
