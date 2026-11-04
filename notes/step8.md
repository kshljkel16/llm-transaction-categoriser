# Step 8 — API

- FastAPI service: GET /health, POST /categorise (max 100 transactions, descriptions 1–200 chars;
  Pydantic validation returns 422 automatically). Interactive docs at /docs.
- create_app(classifier) uses dependency injection: tests pass a fake, production the real hybrid
  (saved baseline + Kimi, threshold 0.55 from Step 7, batch 20, disk cache).

## Live check
- "TESCO STORES 3412" -> Groceries, baseline, conf 0.987 (instant, no merchant name).
- "SQ *KEBAB KING GLASGOW" -> Restaurants via LLM, but merchant "Kebab King Glasgow": the LLM
  kept the city. Fix: prompt instruction + few-shot example (merchant without city/store/country).
- "VISION EXPRESS LEEDS" -> Health via LLM (correct). Baseline confidence depends on the exact
  descriptor, so the same merchant can be routed differently -> merchant-level overrides are
  more reliable than threshold tuning.
- Repeating the request was served from the LLM cache (no call, no cost).

## Limitations
- The cache is a JSON file: concurrent requests can overwrite each other's updates.
  Production: a shared store (e.g. Redis).
- Real-time requests wait for a whole LLM call (~1–2 s for one transaction; more in a batch).

## Interview answers
1. 422 = Unprocessable Entity: the request was valid JSON but broke the schema; FastAPI produces it
   from the Pydantic models.
2. Dependency injection: the app receives the classifier, so tests use a fake (no model, key or
   network) and production swaps in the real one without code changes.
3. The API key is not in the Docker image because images are shared and pushed to registries;
   it is passed at runtime with --env-file.
4. Input limits (100 transactions, 200 chars) cap cost and abuse: every description may go to a
   paid LLM.

## Docker
- Image builds and serves the API (same answers as local uvicorn); `docker ps` shows (healthy).
- Dependencies installed in a separate layer -> code-only rebuilds are fast.
- Model baked in; API key passed at runtime (--env-file), never in the image. Runs as non-root.
- scikit-learn pinned to the version that saved models/baseline.joblib: a pickled model is tied
  to the library version that created it.

   - Built and ran in Docker: `docker ps` -> Up (healthy). Dependency layer 147 s on first build,
     code layer 11.5 s; code-only rebuilds reuse the cached dependency layer.
   - First build failed: Docker Desktop couldn't reach Docker Hub from China without a proxy;
     fixed by configuring the proxy. Python packages via the Tsinghua PyPI mirror (--build-arg).