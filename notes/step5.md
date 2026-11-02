## Thinking on vs off (200 val rows, batch 20, kimi-k2.6)
| | Thinking ON | Thinking OFF |
|---|---|---|
| Accuracy all / seen / unseen | 1.000 / 1.000 / 1.000 | 0.995 / 0.9925 / 1.000 |
| Output tokens | 23,528 | 4,327 |
| Cost per 1k txns | $0.501 | $0.117 |
| Seconds per call | 61.6 | 14.4 |

- ~82% of thinking-on output tokens were hidden reasoning; output tokens were ~94% of cost.
- Accuracy difference = 1 row of 200 (noise). Thinking off: 4.3x cheaper and faster -> chosen.
- Unseen merchants: 66/66 correct (baseline 0.338). Small sample: by the rule of three the true
  accuracy is likely above ~95%. Synthetic data uses well-known brands, so this is optimistic.
- Only error: TRAINLINE.COM -> Transport (truth: Travel). A seen merchant the baseline gets right,
  and a taxonomy ambiguity ("trains between cities" = Travel). Supports the hybrid: baseline for
  known merchants, LLM for new ones.

## Merchant accuracy (0.84 strict) — mostly a metric problem
16 mismatches in 200 rows; nearly all are valid alternative names, not errors:
- store format/sub-brand: Tesco Express, Carrefour Market, Co-op Food, Hilton Hotels
- short vs full name: Pret / Pret A Manger, Costa / Costa Coffee, NCP, Flix / FlixBus
- product vs company: YouTube Premium / Google, OpenAI / ChatGPT (both defensible)
- label unknowable from text: "TFLCYCLE HIRE" labelled Santander Cycles
Plan: keep strict exact-match (normalised) and add a lenient containment match in Step 7.
Real systems use merchant IDs from the card network + a canonical display-name table.

## Batch size (100 val rows, thinking off)
| Batch | Calls | Acc all | Acc unseen | In tokens | Out tokens | Cost/1k | Sec/call | Total s |
|---|---|---|---|---|---|---|---|---|
| 1  | 100 | 0.97 | 0.966 | 43,712 | 2,615 | $0.520 | 1.7  | 166 |
| 10 | 10  | 0.99 | 1.000 | 5,372  | 2,224 | $0.140 | 7.2  | 72  |
| 20 | 5   | 0.99 | 1.000 | 3,242  | 2,175 | $0.118 | 12.4 | 62  |
| 50 | 2   | 1.00 | 1.000 | 1,964  | 2,155 | $0.105 | 30.0 | 60  |

- Batching saves ~5x, almost entirely on input tokens: batch 1 resends the ~430-token system
  prompt on every call. Output tokens (~2,200) are the answers and don't shrink.
- Diminishing returns: 1->10 saves 73% of cost, 20->50 only 11%.
- Accuracy differences are 3 rows of 100 — noise; no measurable effect of batch size.
- Latency per call grows with batch size (1.7 s -> 30 s) while total time falls (166 s -> 60 s).
  Real-time categorisation (one transaction at a time) needs batch 1; bulk backfills should use 20-50.
- No truncation at batch 50 with thinking off (unknown_rate 0).
- Chosen for evaluation: batch 20 (within 12% of cheapest; a bad response loses 20 rows, not 50).
- Errors at batch 1: <fill in>

- Errors at batch 1 (3/100):
  - PLAYSTATION NETWORK -> Subscriptions (truth Entertainment): taxonomy ambiguity
    (PSN sells games and PS Plus memberships).
  - FLIX SE <ref> x2 -> Entertainment (truth Travel): real error — "FLIX SE" is FlixBus's legal
    entity; with no context the model read "Flix" as streaming. Same merchant, same mistake:
    LLM errors are consistent per merchant, and caching makes them permanent -> production
    systems need a user-correction feedback loop.
- Across all runs, LLM errors are rare and cluster in two types: taxonomy ambiguity
  (Trainline, PlayStation) and legal-entity descriptors with no brand context (FLIX SE).