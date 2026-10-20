"""Step 5 — Categorise transactions with an LLM: prompt -> structured output -> validated Prediction.

The three hard parts, and where they live:
  * build_messages   : what you ask (prompt design, few-shot examples)
  * parse_response   : never trust the model's output format — validate everything
  * LLMClassifier    : batching (fewer calls), caching (no repeat cost), cost accounting

Run when done:
    python -m txncat.llm_classifier "AMZN MKTP UK*2K4LZ93" "SUMUP *KEBAB KING"

Test:  pytest tests/test_llm_classifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from txncat.categories import CATEGORIES, UNKNOWN
from txncat.llm_client import LLMClient
from txncat.schema import Prediction


SYSTEM_PROMPT_TEMPLATE = """You categorise card transactions for a personal-finance app.

For each numbered transaction description, return:
- "category": exactly one of the categories below
- "merchant": the clean, human-readable merchant name (e.g. "AMZN MKTP UK*2K4" -> "Amazon"),
  or null for person-to-person transfers and cash.

Categories:
{category_list}

{few_shot}
Respond with ONLY a JSON array, one object per transaction, in the same order:
[{{"i": 1, "category": "...", "merchant": "..."}}, ...]"""


def build_messages(descriptions: list[str]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a batch of descriptions.

    Requirements (tested):
      * The system prompt lists EVERY category from CATEGORIES with its description,
        one per line, formatted "- Name: description".
      * The user prompt numbers the descriptions from 1: "1. TESCO STORES 3412".
    Your design work (not tested, but it drives accuracy):
      * Write 3-6 few-shot examples into `few_shot` — pick the tricky cases you
        found in Step 3's error analysis (processor prefixes, truncation, transfers).
        Do NOT copy rows from val/test into the prompt: that is data leakage.
    """
    raise NotImplementedError("Step 5: implement build_messages")


def parse_response(text: str, n: int) -> list[Prediction]:
    """Turn the raw model reply into exactly `n` Predictions, in input order.

    Must handle (each one is a test):
      * a clean JSON array;
      * the array wrapped in ```json ... ``` fences, or with prose before/after it;
      * category in the wrong case ("groceries") -> normalise to "Groceries";
      * a category not in CATEGORIES -> UNKNOWN;
      * an item missing from the array -> that position gets UNKNOWN;
      * completely unparseable text -> all n positions get UNKNOWN;
      * "merchant": null, missing, or "" -> merchant None.
    Every returned Prediction has source="llm" (or "fallback" when it is UNKNOWN
    because parsing failed), confidence=None. Costs are filled in by LLMClassifier.

    Hint: find the first "[" and the last "]" and json.loads what is between.
    Use the "i" field to place items, not their position in the array.
    """
    raise NotImplementedError("Step 5: implement parse_response")


class LLMClassifier:
    """Batches descriptions, calls the client, parses, caches, and accounts for cost.

    Requirements (tested):
      * predict(texts) -> list[Prediction], same length and order as texts.
      * Calls the client once per batch of `batch_size` UNCACHED descriptions.
      * Cache key: the description stripped and upper-cased. A cached description
        never triggers a client call again. Cached predictions have cost_usd=0.0
        (you did not pay again) — keep the category and merchant.
      * If cache_path is given, load it in __init__ (if the file exists) and save
        it after each predict() call, as JSON {key: {"category":..., "merchant":...}}.
        Never cache UNKNOWN results — a retry might succeed next time.
      * Cost of a batch = input_tokens/1e6 * price_in + output_tokens/1e6 * price_out.
        Split it equally across the items of that batch. Same for latency.
    """

    def __init__(self, client: LLMClient, batch_size: int = 20,
                 price_in_per_mtok: float = 0.0, price_out_per_mtok: float = 0.0,
                 cache_path: Path | None = None) -> None:
        raise NotImplementedError("Step 5: implement LLMClassifier.__init__")

    def predict(self, texts: list[str]) -> list[Prediction]:
        raise NotImplementedError("Step 5: implement LLMClassifier.predict")


def main() -> None:
    import os
    from dotenv import load_dotenv
    from txncat.llm_client import make_client_from_env

    load_dotenv()
    clf = LLMClassifier(
        make_client_from_env(),
        price_in_per_mtok=float(os.getenv("LLM_PRICE_INPUT_PER_MTOK") or 0),
        price_out_per_mtok=float(os.getenv("LLM_PRICE_OUTPUT_PER_MTOK") or 0),
        cache_path=Path(".cache/llm_cache.json"),
    )
    for text, pred in zip(sys.argv[1:], clf.predict(sys.argv[1:])):
        print(f"{text!r:40} -> {pred}")


if __name__ == "__main__":
    main()
