# Step 7 — Final evaluation (test set, 1,500 rows, evaluated once)

| | Accuracy (95% CI) | Macro F1 | Unseen | Seen | Cost/1k | p50 | p95 |
|---|---|---|---|---|---|---|---|
| Baseline | 0.786 (0.765–0.807) | 0.786 | 0.287 | 1.000 | ~$0 | 0.03 ms | 0.03 ms |
| LLM only | 0.987 (0.981–0.992) | 0.981 | 0.982 | 0.989 | $0.094 | 561 ms | 1,280 ms |
| Hybrid @0.55 | 0.967 (0.957–0.976) | 0.963 | 0.889 | 1.000 | $0.025 | 0.03 ms | 825 ms |

## Threshold selection (val only)
- Rule fixed before seeing test: lowest LLM share with val accuracy within 1 point of LLM-only.
  Chosen 0.55 (val acc 0.976 vs LLM-only 0.986, LLM share 0.238).
- Sweep has a sharp knee: ~22% LLM share buys almost all the accuracy gain.
- The chosen point sits on the steep edge -> fragile. On test the gap grew from 1 to 2 points
  because test has different unseen merchants. Lesson: pick a point with margin.
- In hindsight, ~35% LLM share reached 0.988 on val (above LLM-only: the baseline fixes the LLM's
  seen-merchant mistakes). Not re-chosen after seeing test — that would bias the result.

## Findings
- Hybrid: 3.8x cheaper than LLM-only, 75% of transactions answered instantly (p50 0.03 ms),
  perfect on seen merchants — but 2 points less accurate overall; CIs don't overlap, so it is real.
- Per 1M transactions: LLM-only ~$94 / ~13k errors; hybrid ~$25 / ~33k errors. At Kimi's prices
  the cost saving alone is small; the stronger case for the hybrid is latency and resilience
  (75% of traffic keeps working if the LLM API is down).
- 6 unseen merchants cause ~90% of hybrid errors (Q8): misleading shared fragments
  (EXPRESS, POLSKA, TV, .CO.UK). Errors are concentrated per merchant.
- LLM-only's weakest category: Travel (taxonomy ambiguity). No LLM answer failed to parse in
  3,000 rows (Unknown never predicted).
- Merchant names: LLM 0.812 strict / 0.948 lenient (mostly naming variants). Hybrid only 0.227,
  because the baseline never extracts merchants -> product gap: 75% of rows would show no merchant.

## Next steps
- Merchant override table fed by user corrections; novelty detection to route unfamiliar
  descriptors to the LLM; monitor error rate per merchant.
- Merchant lookup built from LLM answers for descriptors the baseline already handles.
- Clarify Travel vs Transport in the taxonomy (trains/coaches between cities).
- Validate on a fresh test set (or real labelled data) before changing the threshold rule.

## Interview answers
1. Precision vs recall: for Bills, low recall means bills hidden in other categories, so budgets
   look too low; low precision inflates them. Both hurt; which matters more depends on the feature.
2. Macro F1 averages per-class F1 equally, so rare classes (Cash 3%) count; micro F1 pools all
   predictions and equals accuracy for single-label tasks. Report both accuracy and macro F1.
3. Threshold on val, not test: choosing on test selects for luck, so the reported number is
   optimistic. Here val said a 1-point gap; test showed 2.
4. Percentiles, not mean: the hybrid's latency is bimodal (0.03 ms vs ~1 s); a mean describes nobody.
5. Is 1 point meaningful on 1,500 rows? Check the CI: ±~1 point near 98%. Hybrid vs LLM-only (2 pts)
   is real — the CIs don't overlap.
6. For a PM: "The hybrid handles 3 in 4 transactions instantly at no cost and sends the rest to
   the LLM, cutting LLM spend ~4x for a 2-point accuracy drop. A higher threshold buys accuracy
   back at more cost."