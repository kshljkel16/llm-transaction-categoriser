"""Step 2 — Load the CSVs into SQLite so you can explore them with SQL.

Run when done:
    python -m txncat.db                 # builds data/transactions.db
    sqlite3 data/transactions.db < sql/analysis.sql

Test:  pytest tests/test_db.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("data/transactions.db")


def load_csv_to_sqlite(csv_path: Path, db_path: Path, split: str) -> int:
    """Append the rows of `csv_path` into table `transactions` in `db_path`.

    Requirements:
      * Table name: `transactions`.
      * Columns: id, description, amount, currency, category, merchant,
        merchant_group, merchant_seen_in_train, split
        (`split` is the extra column filled with the `split` argument: "train"/"val"/"test").
      * Create the table if it does not exist. Use sensible SQL types
        (INTEGER, REAL, TEXT).
      * Calling it twice for the same split must NOT duplicate rows:
        delete that split's rows first.
      * Use parameterised inserts (`executemany` with `?` placeholders),
        never string formatting — see GUIDE.md, Step 2, for why.
      * Return the number of rows inserted.

    Hint: the `csv` module's DictReader, or pandas.read_csv, both work.
    """
    raise NotImplementedError("Step 2: implement load_csv_to_sqlite")


def main() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        n = load_csv_to_sqlite(Path(f"data/{split}.csv"), DB_PATH, split)
        print(f"loaded {n} rows for {split}")


if __name__ == "__main__":
    main()
