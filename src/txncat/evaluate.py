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
import re

from txncat.categories import CATEGORY_NAMES, UNKNOWN
from txncat.schema import Prediction


# ---------------------------------------------------------------------------
# Pure functions — unit tested. No I/O in here.
# ---------------------------------------------------------------------------
def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    per_class = {}
    for c in CATEGORY_NAMES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)               # blank 1: false alarm
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)               # blank 2: a miss
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn ) if (tp + fn) > 0 else 0.0                                                     # blank 3: like precision, but for misses
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    f1s = [m["f1"] for m in per_class.values() if m["support"] > 0]                    # blank 4: only classes that appear
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s), "per_class": per_class}

def _norm(name: str) -> str:
    """Lower-case and keep only letters and digits: "M&S Food" -> "msfood"."""
    return re.sub(r"[^a-z0-9]", "", name.lower())

def merchant_accuracy(true_merchants: list[str | None], pred_merchants: list[str | None]) -> float:
    pairs = [(t, p) for t, p in zip(true_merchants, pred_merchants) if t]   # skip transfers/cash
    if not pairs:
        return 0.0
    correct = sum(1 for t, p in pairs if p is not None and _norm(t) == _norm(p))
    return correct / len(pairs)


def summarise_costs(preds: list[Prediction]) -> dict:
    if not preds:
        return {"total_cost_usd": 0.0, "cost_per_1k_usd": 0.0, "latency_p50_s": 0.0, "latency_p95_s": 0.0}
    total = sum(p.cost_usd for p in preds)
    latencies = [p.latency_s for p in preds]
    return {
        "total_cost_usd": total,
        "cost_per_1k_usd": total / len(preds) * 1000,                                 # blank 1: average per txn, times 1000
        "latency_p50_s": float(np.percentile(latencies, 50)),
        "latency_p95_s": float(np.percentile(latencies,95)),  # blank 2
    }


def simulate_hybrid(baseline_preds: list[Prediction], llm_preds: list[Prediction],
                    threshold: float) -> tuple[list[Prediction], float]:
    combined = [b if b.confidence >= threshold else l                     # blank 3
                for b, l in zip(baseline_preds, llm_preds)]
    n_llm = sum(1 for b in baseline_preds if b.confidence < threshold)
    return combined, n_llm / max(len(baseline_preds), 1)          # blank 4: avoid dividing by zero


# ---------------------------------------------------------------------------
# The experiment — not unit tested; you verify it by reading its output.
# ---------------------------------------------------------------------------
# Imports used only by the experiment live here, so the pure functions above
# (and their tests) do not need pandas, matplotlib or an API key.
import json
import os
import sqlite3
from pathlib import Path

REPORTS = Path("reports")
DB_PATH = Path("data/transactions.db")
THRESHOLDS = [round(i * 0.05, 2) for i in range(21)]        # 0.00, 0.05, ..., 1.00

# The threshold-selection rule, written down BEFORE looking at any test result:
#   choose the threshold with the LOWEST LLM share whose VALIDATION accuracy is
#   within TOLERANCE of LLM-only validation accuracy.
TOLERANCE = 0.01                                              # 1 accuracy point
BATCH_SIZE = 20                                               # chosen in Step 5


def merchant_accuracy_lenient(true_merchants: list[str | None],
                              pred_merchants: list[str | None]) -> float:
    """Like merchant_accuracy, but a match counts if either normalised name contains the other.

    "Tesco" vs "Tesco Express" and "Pret" vs "Pret A Manger" count as correct (Step 5 notes).
    """
    pairs = [(t, p) for t, p in zip(true_merchants, pred_merchants) if t]
    if not pairs:
        return 0.0
    correct = 0
    for t, p in pairs:
        if p:
            a, b = _norm(t), _norm(p)
            correct += bool(a and b and (a in b or b in a))
    return correct / len(pairs)


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for accuracy: resample rows with replacement, take percentiles."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _llm_predictions(df, split: str, clf) -> list[Prediction]:
    """LLM predictions for every row of `df`, saved to reports/llm_preds_<split>.json.

    Why a file and not just the LLM cache? The cache returns cost 0 and latency 0 for
    repeated descriptions, which would make reruns report different (too low) costs.
    Saving the predictions themselves — with the cost and latency measured on the
    first run — makes every rerun give identical numbers, at zero extra cost.

    Works in chunks and saves after each one, so a network failure loses at most one
    chunk: rerun the script and it continues where it stopped.
    """
    path = REPORTS / f"llm_preds_{split}.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    texts = df.description.tolist()
    chunk = BATCH_SIZE * 5                                    # 100 rows = 5 API calls
    if len(done) < len(texts):
        print(f"  LLM on {split}: {len(done)}/{len(texts)} rows already done, continuing...")
    for start in range(len(done), len(texts), chunk):
        preds = clf.predict(texts[start:start + chunk])
        done.extend(p.to_dict() for p in preds)
        path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        cost = sum(d["cost_usd"] for d in done)
        print(f"  LLM on {split}: {len(done)}/{len(texts)} rows, cost so far ${cost:.4f}", flush=True)
    return [Prediction(**d) for d in done]


def _evaluate_system(name: str, df, preds: list[Prediction], llm_share: float) -> dict:
    y_true = df.category.tolist()
    y_pred = [p.category for p in preds]
    m = compute_metrics(y_true, y_pred)
    correct = np.array([t == p for t, p in zip(y_true, y_pred)])
    seen = (df.merchant_seen_in_train == 1).to_numpy()
    true_merchants = [x if isinstance(x, str) else None for x in df.merchant]   # NaN -> None
    pred_merchants = [p.merchant for p in preds]
    lo, hi = bootstrap_ci(correct)
    costs = summarise_costs(preds)
    extracts_merchants = any(m is not None for m in pred_merchants)   # the baseline never does
    return {
        "system": name,
        "accuracy": m["accuracy"],
        "acc_ci95": f"{lo:.3f}–{hi:.3f}",
        "macro_f1": m["macro_f1"],
        "acc_seen": correct[seen].mean(),
        "acc_unseen": correct[~seen].mean(),
        "merchant_acc_strict": (merchant_accuracy(true_merchants, pred_merchants)
                                if extracts_merchants else "n/a"),
        "merchant_acc_lenient": (merchant_accuracy_lenient(true_merchants, pred_merchants)
                                 if extracts_merchants else "n/a"),
        "llm_share": llm_share,
        "cost_per_1k_usd": costs["cost_per_1k_usd"],
        "latency_p50_ms": costs["latency_p50_s"] * 1000,
        "latency_p95_ms": costs["latency_p95_s"] * 1000,
        "_per_class": m["per_class"],
    }


def _sweep_plot(sweep: list[dict], baseline_acc: float, llm_acc: float, chosen: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")                                    # draw to file, no window
    import matplotlib.pyplot as plt

    xs = [r["llm_share"] for r in sweep]
    ys = [r["accuracy"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker="o", color="#2f6fde", label="Hybrid (one point per threshold)")
    ax.axhline(llm_acc, color="#555555", linestyle="--", label=f"LLM only ({llm_acc:.3f})")
    ax.axhline(baseline_acc, color="#999999", linestyle=":", label=f"Baseline only ({baseline_acc:.3f})")
    ax.scatter([chosen["llm_share"]], [chosen["accuracy"]], s=160, facecolors="none",
               edgecolors="#d62728", linewidths=2, zorder=5,
               label=f"Chosen: threshold {chosen['threshold']:.2f}")
    for r in [r for r in sweep if r["threshold"] in (0.0, 0.2, 0.4, 1.0)]:
        ax.annotate(f"{r['threshold']:.2f}", (r["llm_share"], r["accuracy"]),
                    textcoords="offset points", xytext=(4, -12), fontsize=8, color="#444444")
    ax.set_xlabel("Share of transactions sent to the LLM (cost driver)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Hybrid threshold sweep (validation set)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "threshold_sweep_val.png", dpi=150)
    plt.close(fig)


def _confusion_plot(df, preds: list[Prediction]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = CATEGORY_NAMES + [UNKNOWN]
    idx = {c: i for i, c in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(df.category, preds):
        mat[idx[t], idx.get(p.category, idx[UNKNOWN])] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(np.log1p(mat), cmap="Blues")                   # log colour so small errors are visible
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if mat[i, j]:
                ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=7,
                        color="white" if i == j else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Hybrid — confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_hybrid.png", dpi=150)
    plt.close(fig)


def _save_predictions_sql(df, split: str, system_preds: dict[str, list[Prediction]]) -> None:
    con = sqlite3.connect(DB_PATH)
    try:
        with con:
            con.execute("""CREATE TABLE IF NOT EXISTS predictions (
                               id INTEGER NOT NULL, split TEXT NOT NULL, system TEXT NOT NULL,
                               predicted_category TEXT NOT NULL,
                               PRIMARY KEY (split, system, id))""")
            con.execute("DELETE FROM predictions WHERE split = ?", (split,))
            rows = [(int(i), split, system, p.category)
                    for system, preds in system_preds.items()
                    for i, p in zip(df.id, preds)]
            con.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?)", rows)
    finally:
        con.close()


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _write_results_md(results: list[dict], sweep: list[dict], chosen: dict,
                      llm_val_acc: float, n_test: int) -> None:
    cols = ["system", "accuracy", "acc_ci95", "macro_f1", "acc_seen", "acc_unseen",
            "merchant_acc_strict", "merchant_acc_lenient", "llm_share",
            "cost_per_1k_usd", "latency_p50_ms", "latency_p95_ms"]
    lines = [
        "# Results",
        "",
        f"## Test set ({n_test} rows, evaluated once)",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in results:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    lines += [
        "",
        "Cost uses the prices in `.env`. Latency is per transaction: an LLM call's time is split",
        "evenly across its batch of 20, so a real-time user waiting for one call waits ~20x longer.",
        "acc_ci95 = 95% bootstrap confidence interval for accuracy. Merchant accuracy counts rows",
        "with no extracted merchant as wrong; the baseline never extracts merchants (n/a), so the",
        "hybrid only has merchant names for the rows it sent to the LLM.",
        "",
        "## Threshold selection (validation set only)",
        "",
        f"Rule (fixed before looking at test): lowest LLM share whose validation accuracy is within "
        f"{TOLERANCE:.2f} of LLM-only validation accuracy ({llm_val_acc:.3f}).",
        "",
        f"**Chosen threshold: {chosen['threshold']:.2f}** "
        f"(val accuracy {chosen['accuracy']:.3f}, LLM share {chosen['llm_share']:.3f}).",
        "",
        "| threshold | llm_share | val_accuracy | val_macro_f1 |",
        "|---|---|---|---|",
    ]
    for r in sweep:
        lines.append(f"| {r['threshold']:.2f} | {r['llm_share']:.3f} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |")
    lines += ["", "![threshold sweep](threshold_sweep_val.png)", "",
              "![confusion matrix](confusion_hybrid.png)", ""]
    hybrid = next(r for r in results if r["system"].startswith("hybrid"))
    lines += ["## Hybrid per-class metrics (test)", "",
              "| category | precision | recall | f1 | support |", "|---|---|---|---|---|"]
    for c, m in hybrid["_per_class"].items():
        lines.append(f"| {c} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    (REPORTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full experiment (see the steps below). Safe to rerun: LLM calls happen once."""
    import pandas as pd
    from dotenv import load_dotenv

    from txncat.baseline import BaselineClassifier
    from txncat.llm_classifier import LLMClassifier
    from txncat.llm_client import make_client_from_env

    load_dotenv()
    REPORTS.mkdir(exist_ok=True)

    # 1. Data and models ----------------------------------------------------
    val = pd.read_csv("data/val.csv")
    test = pd.read_csv("data/test.csv")
    baseline = BaselineClassifier.load()
    llm = LLMClassifier(
        make_client_from_env(),
        batch_size=BATCH_SIZE,
        price_in_per_mtok=float(os.getenv("LLM_PRICE_INPUT_PER_MTOK") or 0),
        price_out_per_mtok=float(os.getenv("LLM_PRICE_OUTPUT_PER_MTOK") or 0),
        cache_path=None,           # honest costs: every row really calls the LLM once
    )

    # 2. Predictions for every row (LLM results saved to reports/llm_preds_*.json)
    print("Baseline predictions...")
    base_val = baseline.predict_with_confidence(val.description.tolist())
    base_test = baseline.predict_with_confidence(test.description.tolist())
    print("LLM predictions (first run ~30 min; reruns are instant)...")
    llm_val = _llm_predictions(val, "val", llm)
    llm_test = _llm_predictions(test, "test", llm)

    # 3. Threshold selection on VALIDATION only --------------------------------
    y_val = val.category.tolist()
    sweep = []
    for t in THRESHOLDS:
        combined, share = simulate_hybrid(base_val, llm_val, t)
        m = compute_metrics(y_val, [p.category for p in combined])
        sweep.append({"threshold": t, "llm_share": share,
                      "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    llm_val_acc = compute_metrics(y_val, [p.category for p in llm_val])["accuracy"]
    base_val_acc = compute_metrics(y_val, [p.category for p in base_val])["accuracy"]
    ok = [r for r in sweep if r["accuracy"] >= llm_val_acc - TOLERANCE]
    chosen = min(ok, key=lambda r: (r["llm_share"], r["threshold"]))
    _sweep_plot(sweep, base_val_acc, llm_val_acc, chosen)
    print(f"\nChosen threshold on VAL: {chosen['threshold']:.2f} "
          f"(val acc {chosen['accuracy']:.3f}, LLM share {chosen['llm_share']:.3f}; "
          f"LLM-only val acc {llm_val_acc:.3f})")

    # 4. Final evaluation on TEST — once, with the threshold fixed above -------
    hybrid_test, hybrid_share = simulate_hybrid(base_test, llm_test, chosen["threshold"])
    results = [
        _evaluate_system("baseline", test, base_test, 0.0),
        _evaluate_system("llm_only", test, llm_test, 1.0),
        _evaluate_system(f"hybrid@{chosen['threshold']:.2f}", test, hybrid_test, hybrid_share),
    ]
    _write_results_md(results, sweep, chosen, llm_val_acc, len(test))

    # 5. Confusion matrix --------------------------------------------------------
    _confusion_plot(test, hybrid_test)

    # 6. Predictions into SQL for error analysis (Q7) ---------------------------
    _save_predictions_sql(test, "test", {"baseline": base_test, "llm_only": llm_test,
                                         "hybrid": hybrid_test})

    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("\nTEST RESULTS")
    print(table.apply(lambda col: col.map(_fmt)).to_string(index=False))
    print("\nWrote reports/results.md, reports/threshold_sweep_val.png, reports/confusion_hybrid.png")
    print("and table `predictions` in data/transactions.db")


if __name__ == "__main__":
    main()