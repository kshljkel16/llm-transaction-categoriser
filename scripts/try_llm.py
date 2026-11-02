"""Step 5, Part D — run the LLM classifier on a small validation sample.

Usage (from the project root):
    python scripts/try_llm.py                               # 200 val rows, batch size 20
    python scripts/try_llm.py --n 100 --batch-sizes 1 10 20 50   # batch-size experiment

Every run uses a FRESH in-memory cache, so each batch size really calls the LLM
(otherwise the second run would be answered from the cache and cost nothing).
Predictions are saved to reports/llm_sample_b<batch>.csv so you can inspect them.

Uses VALIDATION data only — never test (test is touched once, in Step 7).
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from txncat.categories import UNKNOWN
from txncat.llm_classifier import LLMClassifier
from txncat.llm_client import LLMResponse, make_client_from_env


class CountingClient:
    """Wraps any LLM client and records how many calls were made and their latencies."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.latencies: list[float] = []
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        r = self.inner.complete(system, user)
        self.latencies.append(r.latency_s)
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        return r


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower()) if isinstance(name, str) else ""


def run_once(sample: pd.DataFrame, batch_size: int, client, price_in: float, price_out: float) -> dict:
    counting = CountingClient(client)
    clf = LLMClassifier(counting, batch_size=batch_size,
                        price_in_per_mtok=price_in, price_out_per_mtok=price_out,
                        cache_path=None)                      # fresh cache: every row hits the LLM

    t0 = time.perf_counter()
    preds = clf.predict(sample.description.tolist())
    wall_s = time.perf_counter() - t0

    out = sample.copy()
    out["pred"] = [p.category for p in preds]
    out["pred_merchant"] = [p.merchant for p in preds]
    out["correct"] = out.pred == out.category
    seen = out.merchant_seen_in_train == 1

    has_merchant = out.merchant.notna()
    merchant_ok = [_norm(t) == _norm(p) and _norm(p) != ""
                   for t, p in zip(out.merchant[has_merchant], out.pred_merchant[has_merchant])]

    total_cost = sum(p.cost_usd for p in preds)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv(f"reports/llm_sample_b{batch_size}.csv", index=False)

    return {
        "batch": batch_size,
        "rows": len(out),
        "calls": len(counting.latencies),
        "acc_all": out.correct.mean(),
        "acc_seen": out.correct[seen].mean(),
        "acc_unseen": out.correct[~seen].mean(),
        "merchant_acc": sum(merchant_ok) / max(len(merchant_ok), 1),
        "unknown_rate": (out.pred == UNKNOWN).mean(),
        "in_tokens": counting.input_tokens,
        "out_tokens": counting.output_tokens,
        "cost_usd": total_cost,
        "cost_per_1k_usd": total_cost / len(out) * 1000,
        "sec_per_call": sum(counting.latencies) / max(len(counting.latencies), 1),
        "wall_s": wall_s,
        "_errors": out[~out.correct],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of val rows to sample")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[20])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    load_dotenv()
    price_in = float(os.getenv("LLM_PRICE_INPUT_PER_MTOK") or 0)
    price_out = float(os.getenv("LLM_PRICE_OUTPUT_PER_MTOK") or 0)
    if price_in == 0 or price_out == 0:
        print("WARNING: LLM prices are not set in .env — cost will show as 0.\n")

    val = pd.read_csv("data/val.csv")
    sample = val.sample(n=min(args.n, len(val)), random_state=args.seed)
    print(f"Sample: {len(sample)} val rows "
          f"({(sample.merchant_seen_in_train == 0).sum()} from unseen merchants)\n")

    client = make_client_from_env()
    results = []
    for b in args.batch_sizes:
        print(f"Running batch size {b} ...", flush=True)
        r = run_once(sample, b, client, price_in, price_out)
        results.append(r)
        print(f"  done: {r['calls']} calls in {r['wall_s']:.0f} s, cost ${r['cost_usd']:.4f}\n", flush=True)

    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(table.round(4).to_string(index=False))

    errors = results[-1]["_errors"]
    print(f"\nFirst 15 errors (batch size {results[-1]['batch']}):")
    print(errors[["description", "category", "pred", "merchant", "pred_merchant"]]
          .head(15).to_string(index=False))


if __name__ == "__main__":
    main()