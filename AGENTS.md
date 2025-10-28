# Repository Guidelines

## Project Structure & Module Organization
- Core JAX models, samplers, and visualization live in `hicosmo/`; mirror new functionality there and colocate configs under `hicosmo/configs/` (e.g., `pantheon_plus.yaml`).
- Legacy scipy comparisons remain in `qcosmc/`; touch only when keeping baseline parity.
- Tests reside in `tests/` with domain folders (`tests/analysis/`, `tests/chains/`) plus top-level `test_*.py` entry points. Align new tests with existing markers.
- Examples, documentation, and published figures stay in `examples/`, `docs/`, and `results/`; update notebooks whenever APIs change.

## Build, Test, and Development Commands
- `pip install -e ".[dev]"` — installs the project with linting, typing, and test extras for iterative work.
- `pytest` or `pytest -m "not slow"` — runs the default unit and regression suite; reserve `-m slow`/`-m gpu` for targeted hardware runs.
- `pytest --cov=hicosmo --cov-report=term-missing` — verifies coverage before publishing changes.
- `black . && isort .` followed by `mypy hicosmo` and `flake8 hicosmo tests` — enforces formatting, typing, and linting standards.

## Coding Style & Naming Conventions
- Adhere to Black’s 88-character layout and isort’s Black profile; avoid manual formatting tweaks.
- Prefer pure, type-annotated functions; gate `@jax.jit` or `@jax.vmap` behind demonstrated performance wins.
- Use `snake_case` for modules and functions, `CamelCase` for classes, and uppercase for cosmological constants or dataset tags.

## Testing Guidelines
- Honour pytest markers (`slow`, `gpu`, `integration`) so CI remains deterministic; add deterministic fixtures for stochastic samplers.
- Update plots in `tests/chains/` when sampler distributions change and document intentional coverage deltas in follow-up issues.

## Commit & Pull Request Guidelines
- Write imperative commit subjects (emoji optional) and add bodies for context, metrics, or bilingual notes.
- Reference issues with `Refs #123` or `Fixes #123`; call out breaking API changes explicitly.
- PRs should list validation commands, attach refreshed figures for visualization tweaks, and note CPU vs GPU behaviour when relevant.

## Environment & Configuration Tips
- Support Python 3.9–3.11; set `JAX_PLATFORM_NAME=cpu` for deterministic CI and document any required `XLA_FLAGS` or dataset locations in PR notes.
- Install GPU extras via `pip install "hicosmo-jax[gpu]"` when CUDA hardware is available; mention hardware specs in performance reports.
