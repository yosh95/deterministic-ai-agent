.PHONY: install install-dev format check type-check test all run-sample

# Install the package normally
install:
	pip install .

# Install with development tools (Ruff, Mypy, Pytest)
install-dev:
	pip install -e ".[dev]"

# Formatting using Ruff
format:
	ruff format .
	ruff check --select I --fix .

# Linter check using Ruff
check:
	ruff check .

# Type check using Mypy
type-check:
	mypy deterministic_ai_agent

# Run tests (includes Ruff format/check and Mypy because of pytest.ini_options)
test:
	PYTHONPATH=. pytest

# Build production-ready models (Requires [train] dependencies)
build-models:
	python tools/build_production_models.py

# Run all checks and tests
all: format check type-check test

# Run the sample simulation
run-sample:
	python -m deterministic_ai_agent.executor.engine
