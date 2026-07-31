.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb src

clean:
	rm -rf .mypy_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ingores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

list-strict:
	uv run flake8 .
	uv run mypy . --strict
