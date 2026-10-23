.DEFAULT_GOAL := help
.PHONY: help check-uv venv lock upgrade install sync-frozen add add-dev lint format test test-rust clean

UV ?= uv
PY := .venv/bin/python
VENV_PROMPT := rag-agent

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

venv: check-uv ## Create .venv (prompt: rag-agent) with the pinned Python (.python-version)
	$(UV) venv --allow-existing --prompt $(VENV_PROMPT)

lock: check-uv ## Resolve dependencies and write uv.lock
	$(UV) lock

upgrade: check-uv ## Re-resolve uv.lock to the newest allowed versions
	$(UV) lock --upgrade

install: venv ## Sync .venv to uv.lock (runtime + dev)
	$(UV) sync

sync-frozen: venv ## Install exactly what uv.lock says; fail if it is stale (CI)
	$(UV) sync --locked

add: check-uv ## Add a runtime dependency: make add PKG=httpx
	$(UV) add $(PKG)

add-dev: check-uv ## Add a dev dependency: make add-dev PKG=mypy
	$(UV) add --dev $(PKG)

lint: ## Lint with ruff
	$(UV) run ruff check src

format: ## Format with ruff
	$(UV) run ruff format src
	$(UV) run ruff check --fix src

test: ## Run pytest
	$(UV) run pytest

test-rust: ## Run Rust unit tests (rust/dedup)
	cd rust/dedup && PYO3_PYTHON=$(CURDIR)/$(PY) cargo test

clean: ## Remove .venv and caches (keeps uv.lock)
	rm -rf .venv .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
