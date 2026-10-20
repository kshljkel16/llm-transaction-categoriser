"""Run every query in a .sql file against the project database and print the results.

Usage (from the project root):
    python sql/run_sql.py                       # runs sql/analysis.sql
    python sql/run_sql.py sql/analysis.sql 2    # runs only the 2nd query

Works on Windows without installing the sqlite3 command-line tool.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/transactions.db")


def split_queries(sql_text: str) -> list[str]:
    """Split a file into complete SQL statements.

    Uses sqlite3.complete_statement, i.e. SQLite's own tokenizer, so a ';' inside
    a -- comment or a 'string' does NOT end a statement. Comment-only leftovers
    (e.g. a final -- Answer: line) are dropped.
    """
    queries, buffer = [], ""
    for line in sql_text.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            queries.append(buffer.strip())
            buffer = ""
    return queries


def _display(query: str) -> str:
    """Show the '-- Qn' title (if any) and the SQL code, hiding other comment lines."""
    keep = []
    for ln in query.splitlines():
        s = ln.strip()
        if not s.startswith("--") or s.startswith("-- Q"):
            keep.append(ln)
    return "\n".join(keep).strip()


def main() -> None:
    sql_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sql/analysis.sql")
    only = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not DB_PATH.exists():
        sys.exit(f"{DB_PATH} not found — run `python -m txncat.db` first.")

    queries = split_queries(sql_file.read_text(encoding="utf-8"))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    con = sqlite3.connect(DB_PATH)
    try:
        for i, q in enumerate(queries, 1):
            if only is not None and i != only:
                continue
            print(f"\n=== Query {i} " + "=" * 60)
            print(_display(q))
            print("-" * 72)
            try:
                print(pd.read_sql_query(q, con).to_string(index=False))
            except Exception as e:  # show the SQL error and keep going
                print(f"ERROR: {e}")
    finally:
        con.close()


if __name__ == "__main__":
    main()