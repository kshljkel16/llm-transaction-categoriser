# Step 2 — SQL notes

1. **SQL injection** — user input executed as SQL code. Building queries with f-strings lets input
   like `x'); DROP TABLE transactions; --` escape the string and run commands. Parameterised queries
   (`?` placeholders) send the SQL and the data separately, so data is never parsed as code.
   (executemany is just for speed; execute with `?` is equally safe.)
2. **Idempotency** — running the load once or many times gives the same result. Achieved by deleting
   the split's rows before inserting, with PRIMARY KEY (split, id) as a safety net. Matters because
   pipelines are re-run after crashes or code changes; otherwise data silently duplicates.
3. **GROUP BY vs window function** — GROUP BY collapses rows into one per group. A window function
   (`OVER`) computes across rows without collapsing them, e.g. each category's count next to the
   grand total via SUM(COUNT(*)) OVER ().
4. **Money as float** — floats are binary and can't represent most decimals exactly
   (0.1 + 0.2 = 0.30000000000000004); errors accumulate over millions of transactions.
   Store integer minor units (pence) or use Decimal.

## Findings
- Classes imbalanced: Restaurants 19.7% vs Cash 3.3% → report macro F1.
- Amounts right-skewed (Travel max £1,386 vs avg £176) → median better than mean; amount is a useful feature.
- Merchant frequency is flat (121–137 for the top 10); real data follows a power law.
- Currency is derived from the description text in the generator → adds little information as a feature.
- Short bare-brand descriptors ("DROPBOX", "ASOS LU") are hardest for unseen merchants.