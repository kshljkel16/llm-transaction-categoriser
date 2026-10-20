-- Step 2 — Exploratory SQL. Write one query under each question.
-- Run:  sqlite3 -header -column data/transactions.db < sql/analysis.sql
--
-- Write your answers as comments under each query (e.g. "-- Restaurants is the
-- largest class at ~20%"). These notes go into your README later.

-- Q1. Rows per split, and how many come from merchants unseen in train
SELECT split,
       COUNT(*)                         AS n_rows,
       SUM(merchant_seen_in_train = 0)  AS n_unseen
FROM transactions
GROUP BY split
ORDER BY n_rows DESC;
-- Answer: train 6000 / 0 unseen; val and test 1500 / 450 unseen (30%).

-- Q2. Class balance in train
SELECT category,
       COUNT(*)                                            AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)  AS pct
FROM transactions
WHERE split = 'train'
GROUP BY category
ORDER BY n DESC;
-- Answer: imbalanced — Restaurants 19.7% (1183) is ~6x Cash 3.3% (195);
--         so report macro F1, not just accuracy.

-- Q3. Average and max amount per category, GBP only
SELECT category,
       COUNT(*)              AS n,
       ROUND(AVG(amount), 2) AS avg_amount,
       ROUND(MAX(amount), 2) AS max_amount
FROM transactions
WHERE currency = 'GBP'
GROUP BY category
ORDER BY avg_amount DESC;
-- Answer: Travel highest (avg £176), Subscriptions lowest (avg £11.55, max £41).
--         Max is far above avg (Travel £1,386 vs £176) -> amounts are right-skewed.
--         Cash max is exactly £200: ATM withdrawals are round numbers -> amount is a useful feature.


-- Q4. 10 most frequent merchants in train
SELECT merchant,
       category,
       COUNT(*) AS n
FROM transactions
WHERE split = 'train' AND merchant IS NOT NULL
GROUP BY merchant, category
ORDER BY n DESC
LIMIT 10;
-- Answer: top merchants are Glovo 137, Orlen 130, Waitrose 130 ... Starbucks 121.
--         Counts are almost flat (121-137) across 71 train merchants. Real spending
--         follows a power law (a few merchants like Tesco/Amazon dominate) -> another
--         synthetic-data limitation for the README.


-- Q5. Which currencies appear, and in which categories
SELECT currency,
       category,
       COUNT(*) AS n_rows
FROM transactions
GROUP BY currency, category
ORDER BY currency, n_rows DESC;
-- Answer: 4 currencies — GBP 80%, EUR 11%, PLN 7%, AED 2%.
--         Subscriptions and Transfers appear ONLY in GBP.


-- Q6. Description length per category
SELECT category,
       COUNT(*)                            AS n,
       ROUND(AVG(LENGTH(description)), 1)  AS avg_len,
       MIN(LENGTH(description))            AS min_len,
       MAX(LENGTH(description))            AS max_len
FROM transactions
GROUP BY category
ORDER BY avg_len;
-- Answer: average length is similar everywhere (15.7 Shopping - 19.2 Transfers/Cash).
--         Shortest descriptors are bare brand names: "DROPBOX" (5-7 chars), "ASOS LU", "NOW TV".
--         Short = harder for unseen merchants: no context word like PHARMACY or CINEMAS,
--         and few character n-grams for the model to use.

-- Q7 (do this AFTER Step 6, when predictions exist in table `predictions`).
--     Error rate per category for each system, worst first.
--     Table `predictions` has: id, split, system, predicted_category.
--     Join it to `transactions` on (id, split).
