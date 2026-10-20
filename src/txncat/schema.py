"""Shared data types. Every classifier in this project returns a list of `Prediction`.

Using one return type everywhere is what makes the systems comparable:
the evaluation code does not care whether a prediction came from
logistic regression, an LLM, or the router.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

Source = Literal["baseline", "llm", "fallback"]


@dataclass
class Prediction:
    category: str
    merchant: str | None = None      # clean merchant name, e.g. "Tesco"; None if not extracted
    confidence: float | None = None  # baseline: max class probability; LLM: None
    source: Source = "baseline"
    cost_usd: float = 0.0            # money spent to produce THIS prediction
    latency_s: float = 0.0           # wall-clock time attributable to THIS prediction

    def to_dict(self) -> dict:
        return asdict(self)
