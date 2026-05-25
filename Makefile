# Makefile — MAPS project convenience commands
#
# All targets wrap `uv` invocations. Run `make help` (default) to
# discover. Sprint 11.6 — pattern inspired by bilevel-fishery.

.DEFAULT_GOAL := help

# ============================================================================
# Setup
# ============================================================================

.PHONY: install
install:  ## Install all extras (blindsight, agl, sarl, dev) + pre-commit hooks
	uv sync --extra blindsight --extra agl --extra sarl --extra dev
	uv run pre-commit install

.PHONY: sync
sync:  ## uv sync (no extras change)
	uv sync

# ============================================================================
# Tests
# ============================================================================

.PHONY: test
test:  ## Run full pytest suite (verbose)
	uv run pytest tests/ -v

.PHONY: test-fast
test-fast:  ## Run pytest quiet (CI-style)
	uv run pytest tests/ -q

.PHONY: test-core
test-core:  ## Sprint 11 core/ tests only (cascade + losses + second_order + parity + seeding)
	uv run pytest tests/unit/core/ tests/parity/core/ tests/unit/utils/test_seeding.py -v

.PHONY: test-slow
test-slow:  ## Reproduction tests (paper z-scores, multi-seed) — long
	uv run pytest tests/reproduction -m slow

# ============================================================================
# Code quality
# ============================================================================

.PHONY: lint
lint:  ## Ruff check (read-only)
	uv run ruff check .

.PHONY: format
format:  ## Ruff format (modifies files)
	uv run ruff format .

.PHONY: format-check
format-check:  ## Ruff format check (read-only, CI-friendly)
	uv run ruff format --check .

.PHONY: check
check: lint format-check test  ## All checks: lint + format-check + test

# ============================================================================
# Maintenance
# ============================================================================

.PHONY: clean
clean:  ## Purge .pyc, __pycache__, .DS_Store, .coverage, ruff/pytest cache
	@find . -name "__pycache__" -type d \
		-not -path "./.venv/*" -not -path "./.git/*" -not -path "./external/*" \
		-exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" \
		-not -path "./.venv/*" -not -path "./.git/*" -not -path "./external/*" \
		-delete 2>/dev/null || true
	@find . -name ".DS_Store" \
		-not -path "./.venv/*" -not -path "./.git/*" -not -path "./external/*" \
		-delete 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache .coverage
	@echo "[clean] done"

# ============================================================================
# Help
# ============================================================================

.PHONY: help
help:  ## Show this help (default target)
	@echo "MAPS — Makefile targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make install      # first-time setup"
	@echo "  make test-core    # Sprint 11 core tests"
	@echo "  make check        # full lint + format + test sweep"
	@echo "  make clean        # purge caches & cruft"
