"""Run HIcosmo from a Cobaya-style configuration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from .components import build_likelihoods, resolve_sampler, resolve_theory
from .config import (
    build_parameter_registry,
    load_config,
    normalize_config,
    validate_config,
)
from .datasets import ensure_dataset, resolve_data_root
from ..parameters.setup import apply_requested_free_params, register_model_parameters
from ..utils.logging import get_logger
from ..utils.manifest import write_run_manifest

logger = get_logger(__name__)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "+"} else "_" for ch in value)


def run_from_config(
    config: Union[str, Path, Dict[str, Any]],
    *,
    run: bool = True,
    save_results: bool = True,
    output_format: str = "hdf5",
) -> Dict[str, Any]:
    """Build and optionally execute an inference run from config."""
    raw_cfg = load_config(config)
    cfg = normalize_config(raw_cfg)
    validate_config(cfg)

    theory_class = resolve_theory(cfg.get("theory"))
    registry = build_parameter_registry(cfg, apply_selection=False)
    model_param_names = register_model_parameters(registry, theory_class)

    data_cfg = cfg.get("data") or {}
    data_root = resolve_data_root(data_cfg.get("root"))
    datasets = data_cfg.get("datasets") or []
    for dataset in datasets:
        ensure_dataset(str(dataset), data_root=str(data_root))

    likelihoods = build_likelihoods(
        cfg.get("likelihood", []),
        theory_class=theory_class,
        data_root=data_root,
    )

    for likelihood in likelihoods:
        registry.add_from_likelihood(likelihood)

    free_list = cfg.get("free") or []
    fixed_list = cfg.get("fixed") or []
    if free_list:
        apply_requested_free_params(registry, list(free_list), model_param_names)
    if fixed_list:
        registry.set_fixed(list(fixed_list))

    sampler_name, sampler_options = resolve_sampler(cfg.get("sampler"))

    output_cfg = cfg.get("output") or {}
    root = Path(output_cfg.get("root", "results"))
    chain_name = output_cfg.get("chain_name") or cfg.get("name") or "chain"
    lik_name = _safe_name(
        "+".join(getattr(lik, "name", lik.__class__.__name__) for lik in likelihoods)
    )
    theory_name = _safe_name(getattr(theory_class, "__name__", "theory"))
    sampler_tag = _safe_name(sampler_name)
    run_id = output_cfg.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / theory_name / lik_name / sampler_tag / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    from ..hicosmo import InferenceRunner

    mcmc_config = {"sampler": sampler_name, "chain_name": chain_name}
    mcmc_config.update(sampler_options)

    runner = InferenceRunner(
        cosmology_class=theory_class,
        likelihoods=likelihoods,
        registry=registry,
        mcmc_config=mcmc_config,
        setup_mcmc=sampler_name not in {"dynesty", "nested"},
    )

    result: Dict[str, Any] = {
        "runner": runner,
        "output_dir": output_dir,
        "config": cfg,
    }

    if run:
        if sampler_name in {"dynesty", "nested"}:
            from ..samplers.nested import run_nested, save_nested_results

            nested_results = run_nested(
                runner.likelihood_func, registry.get_free(), sampler_options
            )
            result["samples"] = nested_results.samples

            if save_results:
                filename = output_dir / f"{chain_name}.npz"
                save_nested_results(nested_results, str(filename), format="npz")
        else:
            samples = runner.run()
            result["samples"] = samples

            if save_results:
                filename = output_dir / chain_name
                runner.save_results(str(filename), format=output_format)

        write_run_manifest(output_dir, cfg, chain_name=chain_name)

    return result
