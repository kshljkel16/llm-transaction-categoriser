"""Step 7 — The evaluation harness. This is the part Revolut cares about most.

It answers one product question with numbers:
    "For each system, what accuracy do we get, at what cost and latency —
     and where should the hybrid threshold be?"

Run when done:
    python -m txncat.evaluate          # writes reports/results.md, reports/*.png,
                                       # and a `predictions` table in data/transactions.db

Test:  pytest tests/test_evaluate.py
"""

from __future__ import annotations

import numpy as np

from txncat.categories import CATEGORY_NAMES, UNKNOWN
from txncat.schema import Prediction


# ---------------------------------------------------------------------------
# Pure functions — unit tested. No I/O in here.
# ---------------------------------------------------------------------------
def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Return {"accuracy": float, "macro_f1": float, "per_class": {cat: {...}}}.

    * per_class has an entry for every category in CATEGORY_NAMES, each with
      "precision", "recall", "f1", "support" (support = count in y_true).
    * macro_f1 = unweighted mean of per-class F1 over the categories that have
      support > 0. (Classes absent from y_true are excluded from the mean.)
    * UNKNOWN predictions are simply wrong — they lower recall of the true class
      and never count as a correct positive for anything.
    * If precision or recall has a zero denominator, define it as 0.0.
    You may use sklearn.metrics, but implement it once by hand first —
    you will be asked to explain macro vs micro averaging in interviews.
    """
    raise NotImplementedError("Step 7: implement compute_metrics")


def merchant_accuracy(true_merchants: list[str | None], pred_merchants: list[str | None]) -> float:
    """Exact-match merchant accuracy after normalisation, over rows that HAVE a true merchant.

    * Rows whose true merchant is None or "" are skipped (transfers, cash).
    * Normalise both sides: lower-case, keep only letters and digits
      ("Sainsbury's" == "sainsburys", "M&S Food" == "ms food").
    * A None prediction is wrong.
    * If no row has a true merchant, return 0.0.
    """
    raise NotImplementedError("Step 7: implement merchant_accuracy")


def summarise_costs(preds: list[Prediction]) -> dict:
    """Return {"total_cost_usd", "cost_per_1k_usd", "latency_p50_s", "latency_p95_s"}.

    * cost_per_1k_usd = total / len(preds) * 1000.
    * Percentiles over per-prediction latency_s (np.percentile).
    * Empty list -> all zeros.
    """
    raise NotImplementedError("Step 7: implement summarise_costs")


def simulate_hybrid(baseline_preds: list[Prediction], llm_preds: list[Prediction],
                    threshold: float) -> tuple[list[Prediction], float]:
    """Combine precomputed predictions as the router WOULD at `threshold`.

    Returns (combined_predictions, llm_share).
    Row i takes baseline_preds[i] if its confidence >= threshold, else llm_preds[i].

    Why this exists: to sweep 20 thresholds you do NOT call the LLM 20 times.
    Call it once on every row, cache, then simulate every threshold for free.
    """
    raise NotImplementedError("Step 7: implement simulate_hybrid")


# ---------------------------------------------------------------------------
# The experiment — not unit tested; you verify it by reading its output.
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full experiment. Required steps, in order:

    1. Load val.csv and test.csv, the saved baseline, and an LLMClassifier
       (with the disk cache — you will rerun this many times).
    2. Get baseline and LLM predictions for EVERY row of val and of test.
    3. THRESHOLD SELECTION ON VAL ONLY: sweep thresholds 0.0..1.0 (step 0.05)
       with simulate_hybrid. Pick the threshold by a rule you write down BEFORE
       looking at test results, e.g. "lowest LLM share whose accuracy is within
       1 point of LLM-only". Save the sweep plot: x = share sent to LLM,
       y = accuracy, one point per threshold -> reports/threshold_sweep_val.png
    4. On TEST, report baseline, LLM-only and hybrid(chosen threshold):
       accuracy, macro F1, accuracy on seen / unseen merchants, merchant accuracy,
       cost per 1k, p50/p95 latency, LLM share.
       -> reports/results.md as a markdown table.
    5. Confusion matrix of the hybrid on test -> reports/confusion_hybrid.png
    6. Write every test prediction into table `predictions`
       (id, split, system, predicted_category) in data/transactions.db,
       then answer Q7 in sql/analysis.sql.
    """
    raise NotImplementedError("Step 7: implement the experiment")


if __name__ == "__main__":
    main()
