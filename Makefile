.PHONY: setup data db baseline test eval serve docker

setup:      ## create venv and install
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

data:       ## Step 1
	python -m txncat.generate_data

db:         ## Step 2
	python -m txncat.db && sqlite3 -header -column data/transactions.db < sql/analysis.sql

baseline:   ## Step 3
	python -m txncat.baseline

test:
	pytest -q

eval:       ## Step 7
	python -m txncat.evaluate

serve:      ## Step 8
	uvicorn txncat.api:app --reload

docker:     ## Step 8
	docker build -t txncat . && docker run --rm -p 8000:8000 --env-file .env txncat
