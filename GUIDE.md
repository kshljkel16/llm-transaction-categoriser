# Build Guide — Transaction Categoriser

This guide is the "requirements + teacher" for the project. Each step has the same shape:

1. **Why** — what problem this step solves, and why a team like Revolut's would care.
2. **Concepts** — the theory you need, explained from the ground up.
3. **Your task** — exactly what to build. The file you edit has the precise requirements in its docstrings.
4. **Done when** — the test that must pass, plus what you should see when you run it.
5. **Be able to answer** — questions an interviewer could ask about this step. If you can't answer one, reread the Concepts.

Work in order. Each step builds on the one before. Commit to git after every step: a clean history is part of the portfolio.

**Time budget:** about 2 weeks at 2–3 hours a day. Steps 3, 5 and 7 are the heavy ones.

**The golden rule:** never look at test-set results while you are still making design decisions. You'll understand why in Step 7.

---

## Step 0 — What you are building, and the story it tells

### The product problem

A card statement line looks like `SQ *KEBAB KING GLASGOW` or `AMZN MKTP UK*2K4LZ93`. A banking app wants to show **"Restaurants · Kebab King"** and **"Shopping · Amazon"**. Categories power spending analytics, budgets, and insights, so accuracy is directly visible to customers.

There are three ways to do it:

| Approach | Strength | Weakness |
|---|---|---|
| Classical ML (TF-IDF + logistic regression) | Almost free, ~1 ms, very accurate on merchants it has seen | Fails on new merchants: it has never seen "Kebab King" |
| LLM | World knowledge: knows a kebab shop is a restaurant | Costs money per call, slow (hundreds of ms to seconds), output must be validated |
| **Hybrid** | Baseline handles easy cases, LLM handles hard ones | Needs a threshold. Choosing it IS the product decision |

### The story you will tell in an interview

> "I compared a classical baseline, an LLM, and a confidence-based hybrid on transaction categorisation. I built an evaluation harness measuring accuracy, cost per thousand transactions and latency, and showed that the hybrid keeps X% of the LLM's accuracy while sending only Y% of traffic to it."

That sentence hits almost every line of the Revolut job description: LLMs, evaluation, data pipelines, prompt workflows, metrics, and measurable business outcomes. Everything below exists to make that sentence true, with your own numbers.

### Setup

```bash
cd txn-categoriser
git init && git add . && git commit -m "Initial scaffold"
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                          # 7 pass, the rest fail — that is your to-do list
```

**Why a virtual environment?** Each project gets its own isolated set of package versions. Without one, upgrading scikit-learn for project A can silently break project B.

**Why `pip install -e .` (editable)?** It installs the `txncat` package from `src/`. Your edits apply immediately, and `import txncat` works from anywhere, including tests.

**Why the `src/` layout?** With the code in `src/txncat/`, Python can't accidentally import your package from the current directory instead of the installed one. That is a classic "works on my machine" bug. (You already met its cousin: a file called `types.py` would shadow Python's standard `types` module. That's why the shared types live in `schema.py`.)

Create an empty GitHub repo and push to it. Make it public only when Step 9 is done.

---

## Step 1 — The data

**File to read:** `src/txncat/generate_data.py`. You don't edit it, but you must understand it.
**Run:** `python -m txncat.generate_data`
**Tests (already pass):** `pytest tests/test_data.py`

### Why synthetic data

Real transaction data is private. Public datasets are small, unlabelled, or US-only. Generating data gives **exact ground truth** for both category and merchant name, and lets you control difficulty.

The price: the "mess" is only as realistic as the generator's rules. Real data has things this doesn't: label noise from human annotators, merchants that genuinely sit in two categories (a supermarket that also sells TVs), and a long tail of tiny local businesses. **Say this in your README.** Naming your data's limitations reads as maturity, not weakness.

### What makes it realistic

Open a CSV and look at 30 rows. You'll see:

- **Payment-processor prefixes:** `SQ *` (Square), `SUMUP *`, `PAYPAL *`. Small businesses take cards through these, so the actual merchant is hidden after the prefix.
- **Truncation:** card networks cap descriptors at around 22–25 characters, so `PRET A MANGER LONDON` can become `PRET A MANGER LON`.
- **Noise:** store numbers, reference codes, country codes, glued words.

### The most important design decision: the merchant-level split

The naive way to split data is to shuffle rows randomly into train/val/test. Here that would be **data leakage**. "TESCO STORES 3412" would sit in train and "TESCO STORES 9981" in test. The test would only check whether the model can spot Tesco again, and accuracy would look near-perfect.

In production, new merchants appear every day. So the generator holds out entire merchants: about 30% of val and test rows come from merchants that **never** appear in training (`merchant_seen_in_train = 0`). This is the same idea as `GroupKFold` in scikit-learn. You split by group, not by row, whenever rows within a group are correlated.

**Leakage, defined:** information that would not be available at prediction time leaks into training or model selection, so your offline numbers are better than reality.

### Train / validation / test — the three roles

- **Train:** the model learns parameters from it.
- **Validation:** *you* learn from it. Choose hyperparameters, prompts, and thresholds here.
- **Test:** touched once, at the end, to report the final number. If you tune on test, it becomes a second validation set and your reported number is optimistically biased.

### Be able to answer
1. Why is a random row split wrong for this problem? What would the accuracy look like with one?
2. What is data leakage? Give two examples from this project.
3. What are the limitations of synthetic data here, and how would you validate on real data?

---

## Step 2 — SQL exploration

**File:** `src/txncat/db.py`, then `sql/analysis.sql`
**Done when:** `pytest tests/test_db.py` passes and every question in `analysis.sql` has a query plus a one-line comment with the answer.

### Why SQL, when pandas exists?

Revolut lists SQL as a core requirement, and in industry the data almost always lives in a warehouse you query with SQL. Here you practise the habit: load once, then ask questions in SQL.

### Concepts

**SQLite** is a whole database in one file. There's no server, and `sqlite3` is built into Python. It's perfect for local work.

**Types:** use `INTEGER` for ids and flags, `REAL` for amounts, `TEXT` for strings. (Real money systems store integer minor units, i.e. pence, to avoid floating-point rounding. Mention that if asked about money and floats.)

**Parameterised queries — never build SQL with f-strings.**

```python
# WRONG: breaks on "SAINSBURY'S" (the apostrophe ends the string) and is an injection hole
con.execute(f"INSERT INTO t VALUES ('{desc}')")

# RIGHT: the driver handles quoting
con.executemany("INSERT INTO t VALUES (?, ?)", rows)
```

The test `test_quotes_in_descriptions_survive` exists precisely to catch the wrong version.

**Idempotency:** running the load twice must give the same result as running it once. That's why you delete the split's rows before inserting. Pipelines get re-run all the time, after a crash or a code change, and non-idempotent steps silently duplicate data.

**Window functions (for Q2):** `SUM(COUNT(*)) OVER ()` computes the grand total while you're still grouped by category. Percentage share is then `100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()`. Window functions come up in nearly every SQL interview, so learn them properly here.

### Be able to answer
1. What is SQL injection, and how do parameterised queries prevent it?
2. Why make a data-loading step idempotent?
3. Explain `GROUP BY` vs a window function with `OVER ()`.
4. Why shouldn't money be stored as a float?

---

## Step 3 — The baseline

**File:** `src/txncat/baseline.py`
**Done when:** `pytest tests/test_baseline.py` passes, and `python -m txncat.baseline` prints val accuracy split into seen and unseen merchants.

### Why build a "worse" model first?

Without a baseline you can't say whether the LLM is *worth* its cost. Teams that skip this step spend money on LLM calls for problems a 1-millisecond model already solves. "What's your baseline?" is one of the most common questions in an ML interview.

### Concepts

**Turning text into numbers.** A model needs vectors. **TF-IDF** gives each "term" a weight:

- **TF** (term frequency): how often the term appears in this description.
- **IDF** (inverse document frequency): `log(N / number of documents containing the term)`. Rare terms get high weight; terms in every row (like `GB`) get low weight.
- `sublinear_tf=True` uses `1 + log(tf)` instead of the raw count, so one term repeated five times doesn't dominate.

**Words or characters?** This is the key design choice. Word tokens break here: `AMZN`, `AMAZON.CO.UK*X7` and `AMZNMKTP` share no whole word. **Character n-grams** — every substring of length 2 to 5 — capture the shared pieces `AMZ`, `MZN`, `AMA`. They're robust to truncation, glued words and reference codes. `analyzer="char_wb"` builds n-grams only inside word boundaries, which reduces noise from the spaces between words.

**Run the experiment:** train with `analyzer="word"` and with `"char_wb"`, compare val accuracy, and put both numbers in your README. A design choice backed by a measurement is exactly what reviewers look for.

**Logistic regression** learns one weight per feature per class. It turns the weighted sum into class probabilities with softmax. Why it's the classic text baseline:

- Fast to train and predict, even with 100k+ sparse features.
- Interpretable: you can list the n-grams with the highest weight for each class.
- `predict_proba` gives a probability you can use as a **confidence score**. The whole hybrid in Step 6 depends on this.
- `C` is inverse regularisation strength. Small C means a simpler model; large C fits training data harder. Try 0.1, 1, 10 on validation.

**Calibration caveat:** "probability 0.9" doesn't guarantee being right 90% of the time. Logistic regression is usually reasonably calibrated, but you don't need perfect calibration here. You only need confidence to **rank** easy cases above hard ones. The test `test_confidence_is_lower_on_unseen_merchants` checks exactly that property.

### What you should find

Overall val accuracy around 0.8 — but split it. Expect roughly 100% on seen merchants and roughly 30% on unseen ones. **That gap is the most important finding of the project.** It motivates everything after this step. Write it down.

**Error analysis.** Print 20 misclassified unseen-merchant rows. What do they have in common? Which ones would a human get right instantly, and why? Keep the list: the tricky patterns become your few-shot examples in Step 5.

### Be able to answer
1. What do TF and IDF each capture? Why does IDF down-weight `GB`?
2. Why do character n-grams beat word tokens on this data? Show it with your measured numbers.
3. What does `C` control in logistic regression? How did you pick it?
4. Is `predict_proba` a calibrated probability? Does it matter for this project?
5. Why is accuracy near 100% on seen merchants? Is that realistic? (Hint: synthetic templates. Say so.)

---

## Step 4 — The LLM client

**File:** `src/txncat/llm_client.py`
**Done when:** `pytest tests/test_llm_client.py` passes, and `python -m txncat.llm_client` makes one real call and prints the reply and token counts.

### Why an abstraction layer

The rest of the code depends on one tiny interface, `complete(system, user) -> LLMResponse`, not on a vendor SDK. Three wins:

1. **Switch providers with one env var.** Use Kimi (reachable from Beijing) or Claude. Running the eval on both makes a strong extra row in your results table.
2. **Tests run on `FakeClient`.** No network, no cost, deterministic.
3. **Retries and timing live in one place.**

`LLMClient` is a `typing.Protocol`, which is "duck typing with type hints". Any class with a matching `complete` method qualifies, without inheriting from anything.

### Concepts

**Tokens and pricing.** LLMs read and write *tokens*, roughly ¾ of an English word. Descriptors full of codes like `2K4LZ93` take more tokens than you'd guess. Providers charge per million input tokens and per million output tokens, and output is usually several times more expensive. Copy the current prices for your chosen model from the provider's pricing page into `.env`. Your cost numbers are only as good as those prices, so note the date you copied them.

**System vs user prompt.** The system prompt holds standing instructions (role, categories, output format). The user prompt holds this request's data. Keeping them separate also makes the instruction part cacheable on providers that support prompt caching.

**Temperature = 0.** Temperature controls randomness when the model picks tokens. For classification you want the same answer for the same input, so use 0. (Even at 0, outputs aren't *guaranteed* identical across calls. That's one more reason to cache.)

**Retries with exponential backoff** (read `with_retries`, which is provided). Network calls fail transiently: timeouts, `429 Too Many Requests` (rate limit), `5xx` server errors. Retrying immediately makes a rate limit worse, so you wait 1 s, 2 s, 4 s and so on. Production systems also add **jitter** (a random extra delay) so a thousand clients that failed together don't retry at the same instant. They also retry only on retryable errors: a `401 Unauthorized` will never succeed on retry. Improving `with_retries` this way is a good extension.

**Secrets.** API keys live in `.env`, which `.gitignore` excludes. Run `git status` before every commit until checking becomes a habit. A key pushed to a public GitHub repo gets scraped within minutes. If it ever happens, revoke the key immediately; deleting the commit doesn't help.

**Moonshot speaks the OpenAI API format**, so you use the `openai` Python package with a different `base_url`. Many providers do this. It's worth knowing as a pattern.

### Be able to answer
1. What is a token? Why do descriptors with reference codes cost more?
2. Why temperature 0 for classification?
3. Explain exponential backoff and jitter. Which HTTP errors should and shouldn't be retried?
4. Why put an interface between your code and the vendor SDK?

---

## Step 5 — The LLM classifier

**File:** `src/txncat/llm_classifier.py`
**Done when:** `pytest tests/test_llm_classifier.py` passes, and `python -m txncat.llm_classifier "SUMUP *KEBAB KING" "AMZN MKTP UK*2K4LZ93"` gives sensible answers.

This is the core of the "prompt workflows" requirement in the job description.

### Concepts

**Anatomy of the prompt.** It's already sketched in `SYSTEM_PROMPT_TEMPLATE`:

1. **Role and task:** one sentence.
2. **The taxonomy with descriptions.** Category *names* alone are ambiguous. Is a gym "Entertainment" or "Health"? The descriptions in `categories.py` resolve that, so the model and your evaluation agree on definitions. Taxonomy design is a real product decision; mention it.
3. **Few-shot examples:** 3–6 input→output pairs. These are the most effective lever you have. Choose them from Step 3's error analysis: a processor prefix, a truncated name, a transfer, a merchant where the brand doesn't reveal the category.
4. **Output format:** exact JSON shape, and "ONLY JSON".

**Leakage again:** don't copy val or test rows into the few-shot examples. Write your own or take them from train. If you tune the prompt on val, report on test.

**Batching.** Sending 20 descriptions per call instead of 1 shares the long system prompt across 20 items, so cost per item drops sharply. The trade-offs:
- One bad response now ruins 20 predictions (your parser must degrade per item, not per batch).
- Very long batches can lower accuracy and risk exceeding output limits.

**Run the experiment:** take 200 val rows and compare batch sizes 1, 10, 20, 50 on accuracy and cost per 1k. That's a real finding for the README.

**Structured output — never trust the format.** Models sometimes wrap JSON in ```` ```json ```` fences, add "Sure! Here you go", return `"groceries"` in lowercase, invent a category like `"Food"`, or skip an item. Every one of those is a test case. The pattern is:

1. **Extract:** find the outermost `[...]`.
2. **Parse:** `json.loads` inside try/except.
3. **Validate:** category in the allowed set (case-insensitive), index in range.
4. **Degrade gracefully:** anything invalid becomes `UNKNOWN` for that item only. It never crashes and never makes up an answer.

**Why the `"i"` field?** If the model drops item 2, a positional parse shifts every later answer onto the wrong transaction, which is silent and catastrophic. Matching by index makes the damage local and visible.

**Native structured outputs.** Many APIs can enforce a JSON schema or a "JSON mode" for you. Build the manual version first so you understand what can break, then check your provider's docs and mention native support as an improvement. Even with native support you still validate. Defence in depth.

**Caching.** The same merchant shows up thousands of times (every Netflix bill). The cache key is the normalised description (stripped, upper-cased). Only valid results are cached: if a call failed, retrying later might succeed. Cached predictions cost $0. In production this is the single biggest cost saver.

**Cost accounting.** Each batch's cost is split evenly across its items, so every `Prediction` carries its own `cost_usd`. Summing any subset — "how much did the unseen merchants cost us?" — becomes trivial in Step 7.

### Be able to answer
1. Walk me through your prompt. Why each part? What did few-shot examples change, in numbers?
2. What failure modes did you see in LLM output, and how does your parser handle each?
3. Batch size: what's the trade-off, and what did you measure?
4. How does caching change cost in production, and what should never be cached?
5. Why match results by index instead of by position?

---

## Step 6 — The hybrid router

**File:** `src/txncat/router.py`
**Done when:** `pytest tests/test_router.py` passes.

### Concepts

This is **selective prediction** (also called "learning to defer"). A cheap model answers when it's confident and defers to an expensive expert otherwise. The same pattern shows up all over fintech: auto-approve confident cases, send the rest to a human reviewer or a bigger model.

The threshold is a dial:
- **Threshold 0:** never defer. Baseline cost, baseline accuracy.
- **Threshold above 1:** always defer. LLM cost, LLM accuracy.
- **In between:** a curve. Where you sit on it is a **business decision**, a trade of cents per thousand transactions against wrongly categorised spend that customers see.

Implementation notes:
- Run the baseline on everything first. It's nearly free.
- Send all low-confidence texts to the LLM in **one** `predict` call, so the LLM classifier can batch and cache them.
- Write back into the original positions. Order must be preserved.
- Track `last_llm_share`: this is the cost driver you'll report.

### Be able to answer
1. Why does the router work at all? (Answer with Step 3's evidence: confidence is lower on unseen merchants.)
2. What happens to cost and accuracy at the two extremes of the threshold?
3. Where else in a bank does "confident model plus fallback" apply?

---

## Step 7 — The evaluation harness

**File:** `src/txncat/evaluate.py`
**Done when:** `pytest tests/test_evaluate.py` passes, and `python -m txncat.evaluate` writes `reports/results.md`, `reports/threshold_sweep_val.png` and `reports/confusion_hybrid.png`.

This step is what separates "I called an API" from an applied AI engineer. The job description literally says "manage the models, data, architecture, and evaluation needed to deliver measurable business outcomes."

### Concepts

**Accuracy** is the share of predictions that are correct. It's simple, but it hides class imbalance. If 20% of rows are Restaurants, a model that never predicts Cash loses only 3% accuracy while being useless for Cash.

**Precision, recall, F1 — per class.** For one class C:
- **Precision:** of everything predicted C, how much really was C? Low precision means false alarms.
- **Recall:** of everything that really was C, how much did we find? Low recall means misses.
- **F1:** their harmonic mean, `2PR / (P + R)`. It's high only when both are high.

**Macro vs micro averaging.**
- **Macro F1:** compute F1 per class, then take the plain average. Every class counts equally, so rare classes like Cash or Health matter.
- **Micro F1:** pool all predictions, then compute. For single-label classification this equals accuracy.
- **Weighted F1:** per-class F1 weighted by class size.

Report accuracy *and* macro F1. Implement `compute_metrics` by hand once (the docstring says how), then check it against `sklearn.metrics.classification_report`. Being able to derive these by hand is a classic interview test.

**Confusion matrix:** rows are the true class, columns the predicted class. Off-diagonal cells show *which* confusions happen — Subscriptions vs Shopping for Apple? Travel vs Transport? It tells you what to fix next.

**Slice metrics.** Always report accuracy on seen *and* unseen merchants separately. An average over both hides the very effect this project is about. "Evaluate on slices" is a habit worth naming in interviews.

**Cost per 1,000 transactions** is the unit a product manager can reason about. Multiply by monthly volume and you have a budget line.

**Latency percentiles, not the mean.** A mean of 200 ms can hide 5% of requests taking 4 seconds. p50 is the typical user; p95 is your unlucky users. Note that the hybrid has *bimodal* latency: baseline rows are near-instant, LLM rows are slow. Percentiles show that, a mean doesn't.

**Choosing the threshold on validation — the golden rule, explained.** If you sweep thresholds on test and pick the best one, your reported test number contains a small amount of luck you selected for. It's optimistically biased. The rule:

1. Write your selection rule down **before** looking at results, for example "lowest LLM share whose val accuracy is within 1 point of LLM-only."
2. Apply it on val.
3. Report test exactly once.

**The sweep trick** (`simulate_hybrid`). Don't call the LLM 21 times to test 21 thresholds. Call it once per row (cached), keep the baseline predictions, then *simulate* every threshold by combining the two lists. That costs the same as one run and gives you the full cost-accuracy curve. This is the chart that goes at the top of your README.

**How sure are you?** With 1,500 test rows, an accuracy around 85% carries roughly ±2 percentage points of uncertainty (standard error `sqrt(p(1-p)/n)` ≈ 0.009, times about 2 for 95%). So a 1-point difference between two systems may be noise. Mention this, and as an extension bootstrap a confidence interval: resample the test set with replacement 1,000 times and take the 2.5th and 97.5th percentiles of accuracy.

**Store predictions in SQL**, then answer Q7 in `analysis.sql` with a join. You now have a queryable record of errors, the way analytics works in a real team.

### Be able to answer
1. Precision vs recall: in this product, which error is worse for which category? (Think about a customer's budget for Bills.)
2. Macro vs micro F1. Why report both accuracy and macro F1?
3. Why choose the threshold on val, not test? What exactly goes wrong otherwise?
4. Why percentiles instead of mean latency?
5. Is a 1-point accuracy difference on 1,500 rows meaningful?
6. What's your recommended threshold, and how would you explain it to a product manager in two sentences?

---

## Step 8 — Serve it: API and Docker

**File:** `src/txncat/api.py`, `Dockerfile`
**Done when:** `pytest tests/test_api.py` passes; `make serve` and `http://localhost:8000/docs` let you try the endpoint in the browser; `make docker` runs the same thing in a container.

### Concepts

**FastAPI + Pydantic.** You declare request and response shapes as Pydantic models with constraints (`Field(min_length=1, max_length=200)`). FastAPI validates every request against them and returns **422 Unprocessable Entity** with a precise message, with no `if` statements in your code. You also get interactive docs at `/docs` for free.

**Input limits are a production concern:** an unbounded batch size or description length is a cost and denial-of-service risk, since every character goes to a paid LLM.

**Dependency injection.** `create_app(classifier)` takes the classifier as an argument instead of building it inside. Tests pass a fake, production passes the real hybrid. It's the same idea as `LLMClient` in Step 4, and the most reusable design habit in this project.

**Lazy `app`.** The module-level `__getattr__` builds the real app only when uvicorn asks for `app`, so importing the module in tests needs no model file or API key.

**Docker.**
- Copying `pyproject.toml` before the code lets Docker cache the dependency layer.
- The trained model goes *into* the image. The API key does **not** — it arrives at runtime via `--env-file .env`. Images get shared and pushed to registries, so secrets baked in leak with them.
- `.dockerignore` keeps `.env`, data and venvs out of the build context.

**Optional:** deploy the container to a free hosting tier and put the live `/docs` link in your README. Put a hard cap on spending in your provider console first.

### Be able to answer
1. What does a 422 mean, and who produces it in your API?
2. Why inject the classifier instead of creating it inside the app?
3. Why must the API key not be in the Docker image?
4. What limits did you put on inputs, and why do they matter for an LLM-backed endpoint?

---

## Step 9 — The write-up (this is what gets you the interview)

Recruiters read the README, not the code. Spend real time here.

### README

Fill in `README.md` with your **real** numbers:

1. **The results table** — all three systems.
2. **One key-finding sentence** at the top, with numbers.
3. **The threshold-sweep chart.**
4. **Design decisions with evidence:** char vs word n-grams (measured), batch size (measured), few-shot vs zero-shot (measured), the threshold selection rule.
5. **Limitations, honestly:** synthetic data, near-perfect accuracy on known merchants is an artefact of templates, single run without confidence intervals (unless you did the bootstrap).
6. **What I'd do next:** see Extensions below.

### CV bullet — template

Use this only after it's built, with your measured numbers in the brackets:

> **Transaction Categoriser — LLM vs ML vs hybrid** · [GitHub link]
> - Built an evaluation harness comparing a TF-IDF/logistic-regression baseline, an LLM ([model]) and a confidence-routed hybrid on [N]k card transactions with a merchant-level held-out split.
> - Hybrid reached [X]% accuracy (LLM-only [Y]%, baseline [Z]%) while sending only [S]% of transactions to the LLM — [K]× lower cost per 1k transactions; threshold chosen on validation, reported once on test.
> - Shipped as a FastAPI + Docker service with batched, cached, schema-validated LLM calls.
> *Tech: Python, scikit-learn, LLM APIs, SQL, FastAPI, Docker, pytest*

### Your 60-second interview pitch

Problem (messy descriptors, new merchants) → baseline and its blind spot (seen ~100%, unseen ~30%) → why an LLM helps → why the LLM alone is too expensive → hybrid plus threshold sweep → result in one number → what you'd do next. Practise it out loud until it fits in 60 seconds.

---

## Extensions — pick one or two if you have time

- **Use the amount and currency.** Cash withdrawals are round numbers; travel is expensive. Add them as features to the baseline or as context in the prompt, and measure the change.
- **MCC codes.** Real card transactions carry a Merchant Category Code from the card network. Explain how that would change the design: it's a strong feature, but known to be noisy.
- **Learn from user corrections.** When a customer recategorises a transaction, that's a free label. Design how those would flow back into training.
- **Monitoring.** Track the LLM share and the UNKNOWN rate per day. A sudden rise means drift (new merchants, a changed descriptor format) or a broken prompt.
- **Distillation.** Use the LLM's answers on unlabelled data as training labels for the cheap model. The baseline gets better at unseen merchants and LLM share drops over time.
- **Second provider.** Run the evaluation with both Kimi and Claude and add a row per model.
- **Bootstrap confidence intervals** for every number in the results table.
