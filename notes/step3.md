# Step 3 — Baseline experiments (validation set)

| Setting | All | Seen | Unseen | Features |
|---|---|---|---|---|
| word, (1,1) | 0.765 | 0.979 | 0.267 | 455 |
| **char_wb, (2,5), C=10** | 0.801 | 1.000 | 0.338 | 8,772 |
| char, (2,5) | 0.807 | 1.000 | 0.358 | 13,505 |
| char_wb, C=0.1 | 0.777 | 0.983 | 0.298 | 8,772 |
| char_wb, C=1 | 0.802 | 1.000 | 0.340 | 8,772 |
| char_wb, C=100 | 0.804 | 1.000 | 0.347 | 8,772 |

Val: 1,500 rows (1,050 seen, 450 unseen merchants).

## Findings

1. **Analyzer:** character n-grams beat word tokens on unseen merchants: char 0.358, char_wb 0.338,
   word 0.267 (+9 points char vs word). Word tokens fail on truncation, glued words and
   abbreviations (AMZN vs AMAZON); character n-grams still share pieces.
2. **char vs char_wb:** char is +2.0 points on unseen but has 54% more features (13,505 vs 8,772).
   char also builds n-grams across word boundaries (e.g. "A MA"), which may help with glued words.
   I keep **char_wb**: smaller model, and the +2 points is within noise (see 4).
3. **C:** C=0.1 underfits — strong regularisation shrinks weights so much that even seen merchants
   drop to 0.983. C=1, 10, 100 are practically identical; I keep C=10.
4. **Significance:** unseen accuracy uses 450 rows, so the 95% interval is about ±4–5 points
   (SE = sqrt(0.35×0.65/450) ≈ 0.022). Differences of 1–2 points between settings are noise.
   The real finding: every setting stays at 27–36% on unseen merchants vs 98–100% on seen.
   Tuning cannot close a 66-point gap — the model lacks world knowledge about new merchants,
   which motivates the LLM (Step 5) and the hybrid (Step 6).
5. **Why word has only 455 features:** min_df=2 drops tokens that appear once, which removes almost
   all reference codes; what remains is mostly merchant names, cities and country codes.
   
## Error analysis (val, char_wb, C=10)

- 298 mistakes out of 1,500 (accuracy 0.801). **All 298 are on unseen merchants** — zero errors on
  merchants seen in training. The model memorises merchants rather than learning what a category is.
- **Prior bias:** when unsure, it guesses the largest classes — Restaurants (102 wrong guesses) and
  Groceries (65), which are the two biggest classes in train (19.7% and 17.6%).
- **Confident mistakes (spurious correlation):** NETFLIX.COM → Travel (conf 0.96), ZARA.COM → Travel (0.91).
  In train, ".COM" appears mostly in Travel merchants (BOOKING.COM, TRAINLINE.COM, BA.COM, WIZZAIR.COM),
  so the model learned ".COM ⇒ Travel". About 2% of rows above conf 0.8 are wrong (~24 rows); the hybrid
  will NOT send these to the LLM, so they stay wrong. A risk to report in Step 7.
- **Mistakes no human would make:** SHELL UK OIL → Subscriptions, JMP S.A. BIEDRONKA → Shopping,
  NETFLIX.COM → Travel. They need world knowledge about brands, which an LLM has.
- **Confidence separates right from wrong:** mean conf 0.98 on seen vs 0.39 on unseen merchants;
  0.92 on correct vs 0.34 on wrong predictions.
- **Threshold 0.8 preview:** 73% of rows have conf ≥ 0.8, with 97.8% accuracy; the other 27% are only
  32.3% accurate. Routing just that 27% to an LLM is the idea behind the hybrid (Step 6).

### Patterns for few-shot examples (Step 5)
Write NEW examples with these patterns; don't copy val rows (that would be leakage):
- Streaming/subscription brand ending in .COM → Subscriptions (fights ".COM ⇒ Travel")
- Fuel brand with no fuel word, e.g. "<BRAND> UK OIL" → Transport
- Legal-entity prefix before the brand, e.g. "JMP S.A. <BRAND>" → category of the brand