# Step 1 — Data exploration notes

Data generated with `python -m txncat.generate_data` (seed 42).

## Dataset at a glance

| Split | Rows | From merchants unseen in train |
|---|---|---|
| train | 6,000 | 0 |
| val | 1,500 | 450 (30%) |
| test | 1,500 | 450 (30%) |

**Class balance (train):** Restaurants 19.7%, Groceries 17.6%, Transport 13.9%, Shopping 11.6%,
Subscriptions 7.2%, Entertainment 6.6%, Transfers 6.4%, Bills 5.3%, Health 4.2%, Travel 4.2%, Cash 3.2%.
The data is imbalanced: Restaurants is about 6× more frequent than Cash. So accuracy alone can hide poor
performance on rare classes, and I'll also report macro F1 (Step 7).

**Leakage check:** the set of unseen test merchants ∩ train merchants is empty, so the merchant-level split holds.

## Noise types — one real example each

| Noise type | Example | True merchant / category |
|---|---|---|
| Payment-processor prefix | `ZETTLE_*ITSU LEEDS` | Itsu / Restaurants |
| Truncation (card networks cap at ~22–25 chars) | `VISION EXPRESS PORT` (city "PORTO" cut off) | Vision Express / Health |
| Glued words | `PRETA MANGER PORTO` | Pret A Manger / Restaurants |
| Country-code suffix | `LLOYDS PHARMACY 6926 GB` | Lloyds Pharmacy / Health |
| Mixed case | `Steamgames.Com 8Pajssuu` | Steam / Entertainment |

## Hard cases — even I can't categorise these from the text alone

1. **`EB *6DAHGHDA`** → Eventbrite / Entertainment. The descriptor is just an abbreviation plus a random
   reference code. Without knowing that "EB" means Eventbrite, there is no signal at all.
2. **`APPLE STORE R3506`** (Shopping) vs **`APPLE.COM/BILL`** (Subscriptions). Same brand, two categories.
   The brand name alone isn't enough; the descriptor format decides it.

These are good candidates for few-shot examples in Step 5, and they're a limitation of any text-only approach.

## Answers

### 1. Why is a random row split wrong here? What would accuracy look like with one?

A random row split puts the same merchants in both train and test. For example, `TESCO STORES 3412` lands
in train and `TESCO STORES 9981` in test. The model then only needs to recognise "Tesco" again, which it
does almost perfectly, so test accuracy would come out close to 100%. That number is misleading: in
production, new merchants appear every day, and on merchants the model has never seen, accuracy is much
lower (I'll measure it in Step 3; I expect roughly 30%).

So the split is done by merchant, not by row: about 30% of val/test rows come from merchants that never
appear in training. This is the same idea as `GroupKFold` in scikit-learn — split by group whenever rows
inside a group are correlated.

### 2. What is data leakage? Two examples from this project.

Data leakage is when information that would not be available at prediction time gets into training or
model selection. The model then looks better offline than it will be in production, because it was
evaluated on things it had effectively already seen.

Examples in this project:
1. **Random row split.** The same merchant ends up in train and test, so test accuracy measures memorisation
   instead of generalisation to new merchants.
2. **Tuning on the test set.** For example, copying test descriptions into the LLM prompt as few-shot
   examples (Step 5), or choosing the hybrid threshold by looking at test results (Step 7). Both make the
   final test number optimistically biased. All tuning must use the validation set; test is used once, at
   the end.

### 3. What are the limitations of synthetic data here, and how would you validate on real data?

**Why synthetic:** real card transactions are private, and public datasets are small, unlabelled or US-only.
Generating data gives exact ground truth for category and merchant, and lets me control the difficulty.

**Limitations:**
- **The mess is only as realistic as the generator's rules.** It has five noise types I wrote by hand;
  real descriptors have patterns I didn't think of.
- **No label noise.** Every label is perfectly correct. Real labels come from people or rules and contain errors.
- **No genuinely ambiguous merchants.** In reality a supermarket also sells electronics, and a pharmacy
  sells cosmetics. Here every merchant has exactly one category.
- **A small, repetitive merchant catalogue** (~130 merchants, 2–3 templates each). That's why accuracy on
  seen merchants will be near 100%, which is unrealistically high.
- **Small generator inconsistencies.** Country codes are added at random, so I found `CASHDEPOSIT LEEDS PT`
  (a UK city with a Portuguese country code).

**How I would validate on real data:**
- Take a sample of a few hundred real transactions (e.g. my own statements, anonymised), label them by hand,
  and run the trained systems on them unchanged.
- Compare accuracy on real vs synthetic data. A large gap shows where the generator is unrealistic.
- In production, use customer re-categorisations as labels to keep measuring accuracy over time.
