.PHONY: install run debug clean lint lint-strict

# Pass a command and its options through ARGS, for example:
#   make run ARGS="search 'how to serve a LoRA adapter' --k 5"
ARGS ?= index

MYPY_FLAGS = --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" \
		-exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
