"""Step 2 — Load the CSVs into SQLite so you can explore them with SQL.

Run when done:
    python -m txncat.db                 # builds data/transactions.db
    sqlite3 data/transactions.db < sql/analysis.sql

Test:  pytest tests/test_db.py
"""

from __future__ import annotations

import sqlite3
import csv
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
    # 1. Read the CSV into a list of tuples (one tuple = one row, in column order).
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [
            (
                int(r["id"]),
                r["description"],
                float(r["amount"]),
                r["currency"],
                r["category"],
                r["merchant"] or None,  # empty string -> NULL (transfers, cash)
                r["merchant_group"],
                int(r["merchant_seen_in_train"]),
                split,
            )
            for r in csv.DictReader(f)
        ]

    # 2. Write them in one transaction.
    con = sqlite3.connect(db_path)
    try:
        with con:  # commits if the block succeeds, rolls back if it raises
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id                     INTEGER NOT NULL,
                    description            TEXT    NOT NULL,
                    amount                 REAL    NOT NULL,
                    currency               TEXT    NOT NULL,
                    category               TEXT    NOT NULL,
                    merchant               TEXT,
                    merchant_group         TEXT    NOT NULL,
                    merchant_seen_in_train INTEGER NOT NULL,
                    split                  TEXT    NOT NULL,
                    PRIMARY KEY (split, id)
                )
                """
            )
            con.execute("DELETE FROM transactions WHERE split = ?", (split,))
            con.executemany(
                "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
    finally:
        con.close()  # `with con` does NOT close the connection — do it explicitly

    return len(rows)


def main() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        n = load_csv_to_sqlite(Path(f"data/{split}.csv"), DB_PATH, split)
        print(f"loaded {n} rows for {split}")


if __name__ == "__main__":
    main()
