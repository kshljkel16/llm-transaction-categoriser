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

FEW_SHOT = """Examples:
HULU.COM GB -> {"category": "Subscriptions", "merchant": "Hulu"}
ESSO UK OIL 4471 -> {"category": "Transport", "merchant": "Esso"}
KAUFLAND SP. Z O.O. 0912 -> {"category": "Groceries", "merchant": "Kaufland"}
SQ *BLUE DOOR BAKERY -> {"category": "Restaurants", "merchant": "Blue Door Bakery"}
TO A KOWALCZYK -> {"category": "Transfers", "merchant": null}
"""

def build_messages(descriptions: list[str]) -> tuple[str, str]:
    category_list = "\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items())
    system = SYSTEM_PROMPT_TEMPLATE.format(category_list=category_list, few_shot=FEW_SHOT)
    user = "\n".join(f"{i}. {d}" for i, d in enumerate(descriptions, start=1))
    return system, user


def parse_response(text: str, n: int) -> list[Prediction]:
    # 0. Safe default: every position starts as "unknown". Anything we fail to parse stays like this.
    results = [Prediction(UNKNOWN, source="fallback") for _ in range(n)]

    # 1. EXTRACT: the array starts at the first "[" and ends at the last "]".
    start = text.find("[")                # blank 1
    end = text.rfind("]")                 # blank 2  (rfind searches from the right)
    if start == -1 or end < start:        # find() returns -1 when not found
        return results

    # 2. PARSE
    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return results
    if not isinstance(items, list):
        return results

    # lookup table: lowercase name -> official name, e.g. "groceries" -> "Groceries"
    valid = {name.lower(): name for name in CATEGORIES}

    # 3. VALIDATE each item, and place it by its "i"
    for item in items:
        if not isinstance(item, dict):
            continue                                       # skip junk like a bare string
        try:
            pos = int(item["i"]) - 1                     # blank 3: "i" starts at 1, list positions at 0
        except (KeyError, TypeError, ValueError):
            continue                                       # no usable "i" -> skip this item
        if not 0 <= pos < n:
            continue                                       # "i": 99 when we sent 3 -> skip

        raw_category = str(item.get("category", "")).strip().lower()
        category = valid.get(raw_category, UNKNOWN)           # blank 4: what if it's not a real category?
        merchant = item.get("merchant") or None           # blank 5: turn "" into "no merchant"
        results[pos] = Prediction(category, merchant=merchant, source="llm")

    return results


class LLMClassifier:
    """Batches descriptions, calls the LLM, parses, caches, and accounts for cost."""

    def __init__(self, client: LLMClient, batch_size: int = 20,
                 price_in_per_mtok: float = 0.0, price_out_per_mtok: float = 0.0,
                 cache_path: Path | None = None) -> None:
        self.client = client
        self.batch_size = batch_size
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():          # load the cache saved last time
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    @staticmethod
    def _key(text: str) -> str:
        return text.strip().upper()                              # "netflix.com " -> "NETFLIX.COM"

    def predict(self, texts: list[str]) -> list[Prediction]:
        results: list[Prediction | None] = [None] * len(texts)    # one empty slot per input
        todo: list[int] = []                                      # positions NOT in the cache

        # 1. Answer from the cache where possible (free, instant)
        for i, text in enumerate(texts):
            hit = self.cache.get(self._key(text))
            if hit is not None:
                results[i] = Prediction(hit["category"], merchant=hit["merchant"], source="llm")
            else:
                todo.append(i)

        # 2. Send the rest to the LLM in batches
        for b in range(0, len(todo), self.batch_size):                        # blank 1: step = how many per call
            batch_pos = todo[b:b + self.batch_size]                           # blank 2: same number
            batch_texts = [texts[i] for i in batch_pos]
            system, user = build_messages(batch_texts)
            resp = self.client.complete(system, user)
            preds = parse_response(resp.text, len(batch_texts))

            batch_cost = (resp.input_tokens / 1e6 * self.price_in
                          + resp.output_tokens / 1e6 * self.price_out)       # blank 3: output price
            for i, p in zip(batch_pos, preds):
                p.cost_usd = batch_cost / len(batch_pos)          # split cost evenly
                p.latency_s = resp.latency_s / len(batch_pos)     # same for time
                results[i] = p
                if p.category != UNKNOWN:                             # blank 4: never cache failures
                    self.cache[self._key(texts[i])] = {"category": p.category, "merchant": p.merchant}

        # 3. Save the cache so the next run reuses it
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        return results


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
