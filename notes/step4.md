- Smoke test "pong" (4 runs): 22 input tokens; output 29–36 tokens (varies — hidden "thinking"
  differs per run, temperature fixed at 1); latency 1.4–1.8 s typical, one outlier of 14.3 s
  caused by a slow network.
- Implication: the LLM is ~1,500x slower than the baseline (~1 ms). The 14 s outlier shows why
  latency must be reported as p50/p95, not the mean. Same input -> different token counts (and
  potentially different answers), so caching matters for both cost and consistency.