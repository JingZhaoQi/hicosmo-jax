"""Sampler option normalization shared by high-level entry points."""

from __future__ import annotations

from typing import Any, Dict

MCMC_INIT_OPTION_KEYS = frozenset(
    {
        "strict_mode",
        "chain_method",
        "optimize_init",
        "max_opt_iterations",
        "opt_learning_rate",
        "enable_checkpoints",
        "checkpoint_interval",
        "checkpoint_interval_seconds",
        "checkpoint_dir",
        "backup_versions",
        "save_warmup",
        "compression",
        "auto_resume",
    }
)


def split_mcmc_options(config: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split backend sampler options from :class:`MCMC` constructor options."""
    sampler_options = {}
    init_options = {}
    for key, value in config.items():
        if key in MCMC_INIT_OPTION_KEYS:
            init_options[key] = value
        else:
            sampler_options[key] = value
    return sampler_options, init_options
