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
    """Calls Moonshot (Kimi) through its OpenAI-compatible Chat Completions API.

    Requirements: same as AnthropicClient, but
      * use `openai.OpenAI(api_key=..., base_url=MOONSHOT_BASE_URL)`;
      * the system prompt goes in as a {"role": "system"} message;
      * tokens come from response.usage.prompt_tokens / completion_tokens.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None) -> None:
        raise NotImplementedError("Step 4: implement MoonshotClient")

    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError


def make_client_from_env() -> LLMClient:
    """Return AnthropicClient or MoonshotClient depending on LLM_PROVIDER.

    Raise ValueError with a helpful message for any other value.
    Call `dotenv.load_dotenv()` first so a local .env file is picked up.
    """
    raise NotImplementedError("Step 4: implement make_client_from_env")


def _smoke_test() -> None:
    """python -m txncat.llm_client  — one real call, prints the reply and token usage."""
    client = make_client_from_env()
    r = client.complete("You are terse.", "Reply with the single word: pong")
    print(r)
    print("provider:", os.getenv("LLM_PROVIDER"))


if __name__ == "__main__":
    _smoke_test()
