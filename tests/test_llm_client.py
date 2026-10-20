"""Step 4. The retry tests pass from the start — read them to understand with_retries."""
import pytest

from txncat.llm_client import FakeClient, LLMResponse, make_client_from_env, with_retries


def test_fake_client_counts_calls():
    c = FakeClient(lambda s, u: "ok")
    c.complete("sys", "user")
    c.complete("sys", "user")
    assert c.calls == 2


def test_retry_succeeds_after_failures():
    attempts, sleeps = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError("network blip")
        return LLMResponse("ok", 1, 1, 0.0)

    assert with_retries(flaky, sleep=sleeps.append).text == "ok"
    assert sleeps == [1.0, 2.0]          # exponential backoff


def test_retry_gives_up():
    def always_fails():
        raise TimeoutError

    with pytest.raises(TimeoutError):
        with_retries(always_fails, max_attempts=3, sleep=lambda s: None)


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nonsense")
    with pytest.raises(ValueError):
        make_client_from_env()
