-- Step 2 — Exploratory SQL. Write one query under each question.
-- Run:  sqlite3 -header -column data/transactions.db < sql/analysis.sql
--
-- Write your answers as comments under each query (e.g. "-- Restaurants is the
-- largest class at ~20%"). These notes go into your README later.

-- Q1. How many rows per split, and how many of them come from merchants unseen in train?


-- Q2. Class balance: count and percentage of each category in the TRAIN split,
--     largest first. (Hint: a window function, SUM(COUNT(*)) OVER (), gives the total.)


-- Q3. Median-ish view of money: average and max amount per category in GBP only.


-- Q4. The 10 most frequent merchants in train, with their category.


-- Q5. Which currencies appear, and in which categories?


-- Q6. Description length: average and max number of characters per category.
--     (Why care? Truncated descriptors carry less signal — see GUIDE.md.)


-- Q7 (do this AFTER Step 6, when predictions exist in table `predictions`).
--     Error rate per category for each system, worst first.
--     Table `predictions` has: id, split, system, predicted_category.
--     Join it to `transactions` on (id, split).
