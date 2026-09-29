.PHONY: install test lint format typecheck eval demo-traces api gitleaks manifest all

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest --cov=src --cov-report=term-missing --cov-fail-under=70

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m black .

typecheck:
	$(PY) -m mypy --strict src/

eval:
	$(PY) -m eval.run

demo-traces:
	$(PY) scripts/record_demo_traces.py

manifest:
	$(PY) scripts/make_manifest.py

api:
	$(PY) -m uvicorn src.api.main:app --reload --port 8000

gitleaks:
	gitleaks detect --no-banner --redact --source .

all: lint typecheck test
