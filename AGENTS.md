# Repository Guidelines

Use this guide to orient new contributions and keep the HIcosmo toolchain predictable across CPU and GPU environments.

## Project Structure & Module Organization
- `hicosmo/` holds production JAX code (models, samplers, likelihoods, visualization). Mirror new components in this tree.
- `qcosmc/` is the legacy scipy baseline; touch only when syncing behaviour for regression comparisons.
- `tests/` and top-level `test_*.py` contain regression, benchmarking, and diagnostics; group additions by domain (e.g., `tests/analysis/`).
- `examples/`, `docs/`, and published `results/` artefacts document and showcase functionality; keep notebooks and figures consistent with API changes.

## Build, Test, and Development Commands
- `pip install -e ".[dev]"` — install editable sources with linting, typing, and test extras.
- `pytest` or `pytest -m "not slow"` — run the default suite; reserve `-m slow` and `-m gpu` for dedicated hardware checks.
- `pytest --cov=hicosmo --cov-report=term-missing` — confirm coverage before merging feature branches.
- `black . && isort .`, then `mypy hicosmo` and `flake8 hicosmo tests` — match repo formatting and static checks; use `pre-commit run --all-files` for a single pass.

## Coding Style & Naming Conventions
- Enforce Black’s 88-character layout and isort’s Black profile for imports.
- Prefer pure, type-annotated functions; decorate with `@jax.jit` or `@jax.vmap` only after profiling benefits.
- Use `snake_case` for modules/functions, `CamelCase` for classes, and upper-case constants for cosmological parameters.
- Place reusable YAML configs under `hicosmo/configs/` with dataset-based filenames (e.g., `pantheon_plus.yaml`).

## Testing Guidelines
- Honour pytest markers: gate heavy runs behind `slow`, GPU-specific checks behind `gpu`, and scenario tests behind `integration`.
- Provide deterministic fixtures for new samplers or likelihoods; update comparison plots in `tests/chains/` when distributions change.
- Preserve coverage trends; document any intentional drop-off in the PR and create a follow-up issue if needed.

## Commit & Pull Request Guidelines
- Follow existing history: short imperative subjects (emoji optional), with bodies for context, metrics, or bilingual notes.
- Reference issues using `Refs #123` or `Fixes #123`; highlight breaking API changes explicitly.
- PRs must list validation commands, attach figures for visualization tweaks, and mention GPU vs CPU behaviour when relevant.
- Keep PR scope tight; split sweeping refactors into preparatory commits linked through the description.

## Environment & Configuration Tips
- Support Python 3.9–3.11; install GPU extras via `pip install "hicosmo-jax[gpu]"` when CUDA is available.
- Set `JAX_PLATFORM_NAME=cpu` for deterministic CI runs and describe any required `XLA_FLAGS` or dataset locations in the PR notes.
