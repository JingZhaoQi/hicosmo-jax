# Repository Guidelines

## Project Structure & Module Organization
Core JAX code, samplers, and visual tools live in `hicosmo/`, with corresponding configs under `hicosmo/configs/` (for example `pantheon_plus.yaml`). Legacy SciPy comparisons stay in `qcosmc/` to preserve baseline parity. Tests live under `tests/` with domain folders such as `tests/analysis/` and `tests/chains/`, plus top-level `test_*.py` entry points. Examples, docs, and published figures reside in `examples/`, `docs/`, and `results/`; update notebooks when APIs shift.

## Build, Test, and Development Commands
- `pip install -e ".[dev]"`: installs runtime plus linting, typing, and test extras for local work.
- `pytest` or `pytest -m "not slow"`: runs the default suite; reserve `-m slow` or `-m gpu` for hardware-specific checks.
- `pytest --cov=hicosmo --cov-report=term-missing`: verify coverage and spot missing lines before publishing.
- `black . && isort .`: enforce formatting; follow with `mypy hicosmo` and `flake8 hicosmo tests` to keep typing and lint clean.

## Coding Style & Naming Conventions
Follow Black’s 88-character layout and isort’s Black profile; avoid hand formatting. Prefer pure, type-annotated functions and only gate `@jax.jit` or `@jax.vmap` after measuring wins. Use `snake_case` for modules and functions, `CamelCase` for classes, and uppercase for cosmological constants or dataset tags.

## Testing Guidelines
Stick to pytest markers (`slow`, `gpu`, `integration`) so CI stays deterministic. Add deterministic fixtures for stochastic samplers and refresh plots in `tests/chains/` when distributions change. Keep new tests alongside related modules and align their names with existing `test_*.py` patterns.

## Commit & Pull Request Guidelines
Write imperative commit subjects (emoji optional) and include bodies for context, metrics, or bilingual notes. Reference issues using `Refs #123` or `Fixes #123`, and flag breaking API changes explicitly. PRs should list validation commands, attach refreshed figures for visualization changes, and note CPU versus GPU behavior when relevant.

## Environment & Configuration Tips
Support Python 3.9–3.11 and set `JAX_PLATFORM_NAME=cpu` for deterministic CI runs. Install GPU extras via `pip install "hicosmo-jax[gpu]"` when CUDA hardware is available, and document any required `XLA_FLAGS` or dataset paths in PR notes.
