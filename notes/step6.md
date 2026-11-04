# Step 6 — Hybrid router (live demo, 200 val rows, threshold 0.8, batch 20, thinking off)

| | Acc all | Seen | Unseen | Cost/1k | Instant |
|---|---|---|---|---|---|
| Baseline only | ~0.80 | ~1.00 | ~0.34 | ~$0 | 100% |
| LLM only | 0.995 | 0.993 | 1.000 | $0.117 | 0% |
| Hybrid (0.8) | 0.990 | 1.000 | 0.970 | $0.035 | 70% |

- LLM share 0.30 -> cost 0.30 x $0.117 = $0.035 per 1k (3.3x cheaper than LLM-only).
- Hybrid matches LLM-only within 1 row, and beats it on seen merchants (baseline handles
  merchants like Trainline that the LLM miscategorised).
- Both errors are confident baseline mistakes on unseen merchants that never reached the LLM:
  NETFLIX.COM -> Travel (".COM" spurious correlation) and BRITISHGAS VUN5FEKZWG -> Travel
  (glued word + booking-style code resembles airline descriptors).
- The hybrid's ceiling is set by the baseline's overconfident errors. Options: higher threshold
  (more cost), novelty detection (route unfamiliar-looking descriptors to the LLM), calibration.
- Small sample (200 rows, 66 unseen): Step 7 measures this properly on the full sets.