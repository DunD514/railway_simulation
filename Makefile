PY ?= python
PORT ?= 8000

.PHONY: all install test train train-quick eval web serve dev clean

all: install test train eval web   ## full pipeline

install:
	$(PY) -m pip install -r requirements.txt
	cd web && npm ci

test:
	$(PY) -m pytest -q

train:          ## main runs, ablations, hyperparameter sweep, heuristic reference curves
	$(PY) -m experiments.train

train-quick:    ## smoke run: few episodes and seeds, no sweep (OVERWRITES results/)
	$(PY) -m experiments.train --episodes 40 --seeds 2 --no-sweep

eval:           ## held-out evaluation, statistics, behaviour analysis, replays
	$(PY) -m experiments.evaluate

web:
	cd web && npm run build

serve:          ## API + built UI on http://127.0.0.1:$(PORT)
	$(PY) -m uvicorn api.main:app --port $(PORT)

dev:            ## Vite dev server on :5173 (run `make serve` alongside for /api)
	cd web && npm run dev

clean:
	rm -rf results web/dist .pytest_cache
