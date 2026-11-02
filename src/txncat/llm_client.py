"""Step 4 — A thin, provider-agnostic LLM client.

Everything else in the project talks to `LLMClient`, never to a vendor SDK
directly. That gives you three things:
  1. You can swap Anthropic <-> Moonshot with one environment variable.
  2. Tests run with `FakeClient` — no network, no cost, deterministic.
  3. Retries and timing live in one place.

Test:  pytest tests/test_llm_client.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol
from dotenv import load_dotenv


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Provided: a fake client for tests and offline development.
# ---------------------------------------------------------------------------
class FakeClient:
    """Returns whatever `responder(system, user)` returns. Counts calls.

    Token counts are approximated as len(text) // 4 so cost maths can be tested.
    """

    def __init__(self, responder: Callable[[str, str], str]) -> None:
        self.responder = responder
        self.calls = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls += 1
        text = self.responder(system, user)
        return LLMResponse(text=text,
                           input_tokens=(len(system) + len(user)) // 4,
                           output_tokens=len(text) // 4,
                           latency_s=0.0)


# ---------------------------------------------------------------------------
# Provided: retry helper. Read it — GUIDE.md, Step 4, explains each line.
# ---------------------------------------------------------------------------
def with_retries(fn: Callable[[], LLMResponse], max_attempts: int = 4,
                 base_delay_s: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep) -> LLMResponse:
    """Call fn(); on exception, wait base_delay * 2**attempt and try again."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception:
            if attempt == max_attempts - 1:
                raise
            sleep(base_delay_s * 2 ** attempt)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# TODO: the two real clients.
# ---------------------------------------------------------------------------
class AnthropicClient:
    """Calls the Anthropic Messages API.

    Requirements:
      * Read ANTHROPIC_API_KEY and ANTHROPIC_MODEL from the environment (constructor args override).
      * temperature=0 (we want the same answer for the same input).
      * max_tokens large enough for a batch of answers (e.g. 2000).
      * Measure latency with time.perf_counter() around the call.
      * Fill LLMResponse from the response: text of the first content block,
        usage.input_tokens, usage.output_tokens.
      * Wrap the API call in with_retries.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        raise NotImplementedError("Step 4: implement AnthropicClient")

    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError


class MoonshotClient:
    """Calls Moonshot (Kimi) through its OpenAI-compatible Chat Completions API."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or os.getenv("MOONSHOT_MODEL", "kimi-k2.6")
        self.thinking = os.getenv("MOONSHOT_THINKING", "enabled")
        key = api_key or os.getenv("MOONSHOT_API_KEY")                                  # blank 1
        if not key:
            raise ValueError("MOONSHOT_API_KEY is not set — add it to your .env file")
        self.client = OpenAI(api_key=key, base_url=base_url or os.getenv("MOONSHOT_BASE_URL"))   # blank 2

    def complete(self, system: str, user: str) -> LLMResponse:
        def call() -> LLMResponse:
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=4000,    # NO temperature: Kimi models reject it
                extra_body={"thinking": {"type": self.thinking}},
            )
            return LLMResponse(
                text=resp.choices[0].message.content or "",
                input_tokens=resp.usage.prompt_tokens,                             # blank 3
                output_tokens=resp.usage.completion_tokens,                            # blank 4
                latency_s=time.perf_counter() - t0,
            )

        return with_retries(call)

def make_client_from_env() -> LLMClient:
    """Return AnthropicClient or MoonshotClient depending on LLM_PROVIDER.

    Raise ValueError with a helpful message for any other value.
    Call `dotenv.load_dotenv()` first so a local .env file is picked up.
    """
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER")
    if provider == "moonshot":
        return MoonshotClient()
    raise ValueError(
        f"Unknown LLM_PROVIDER {provider!r}; only 'moonshot' is implemented"
    )

def _smoke_test() -> None:
    """python -m txncat.llm_client  — one real call, prints the reply and token usage."""
    client = make_client_from_env()
    r = client.complete("You are terse.", "Reply with the single word: pong")
    print(r)
    print("provider:", os.getenv("LLM_PROVIDER"))


if __name__ == "__main__":
    _smoke_test()
