#!/usr/bin/env python3
"""
HIcosmo high-level API and convenience runner.
"""

from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Type, Union

from .api_registry import (
    combine_likelihoods,
    list_cosmology_names,
    list_likelihood_names,
    resolve_cosmology_class,
    resolve_likelihoods,
)
from .models.base import CosmologyBase
from .parameters import ParameterRegistry
from .parameters.setup import apply_requested_free_params, register_model_parameters
from .samplers import MCMC
from .samplers.options import split_mcmc_options


class InferenceRunner:
    """High-level inference wrapper around :class:`hicosmo.samplers.MCMC`."""

    def __init__(
        self,
        cosmology_class: Union[Type[CosmologyBase], str],
        likelihoods: Union[Any, List[Any], str],
        free_params: Optional[List[str]] = None,
        registry: Optional[ParameterRegistry] = None,
        mcmc_config: Optional[Dict[str, Any]] = None,
        preset: str = "planck2018",
        setup_mcmc: bool = True,
    ) -> None:
        self.cosmology_class = resolve_cosmology_class(cosmology_class)
        self.likelihoods = resolve_likelihoods(self.cosmology_class, likelihoods)
        self.likelihood_func = combine_likelihoods(self.likelihoods)

        self.registry = registry or ParameterRegistry.from_defaults(preset)

        model_param_names = register_model_parameters(
            self.registry, self.cosmology_class
        )

        for likelihood in self.likelihoods:
            self.registry.add_from_likelihood(likelihood)

        apply_requested_free_params(self.registry, free_params, model_param_names)

        raw_mcmc_config = dict(mcmc_config or {})
        self.mcmc_options, self.mcmc_init_options = split_mcmc_options(
            {
                k: v
                for k, v in raw_mcmc_config.items()
                if k not in {"sampler", "chain_name", "derived"}
            }
        )
        self.mcmc_config = raw_mcmc_config
        self.sampler_name = str(self.mcmc_config.get("sampler", "numpyro"))
        self.chain_name = self.mcmc_config.get("chain_name")
        self.mcmc: Optional[MCMC] = None

        if setup_mcmc:
            self._setup_mcmc()

    def _setup_mcmc(self) -> None:
        """Instantiate the MCMC backend from current config."""
        config_dict = self.registry.to_dict()
        config_dict["mcmc"] = dict(self.mcmc_options)
        self.mcmc = MCMC(
            config_dict,
            self.likelihood_func,
            chain_name=self.chain_name,
            sampler=self.sampler_name,
            likelihoods=self.likelihoods,
            **self.mcmc_init_options,
        )

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run sampling and return posterior samples."""
        if self.mcmc is None:
            self._setup_mcmc()
        assert self.mcmc is not None
        return self.mcmc.run(**kwargs)

    def get_samples(self, param: Optional[str] = None) -> Union[Dict[str, Any], List]:
        """Get all samples or a single parameter chain."""
        if self.mcmc is None:
            raise RuntimeError("MCMC not initialized.")
        return self.mcmc.get_samples(param=param)

    def save_results(
        self, filename: Optional[str] = None, format: str = "hdf5"
    ) -> None:
        """Persist current samples to disk."""
        if self.mcmc is None:
            raise RuntimeError("MCMC not initialized.")
        self.mcmc.save_results(filename=filename, format=format)

    def summary(self, prob: float = 0.9, burnin_frac: float = 0.1) -> None:
        """Print summary statistics."""
        if self.mcmc is None:
            raise RuntimeError("MCMC not initialized.")
        self.mcmc.print_summary(prob=prob, burnin_frac=burnin_frac)

    def corner_plot(
        self,
        filename: Union[str, Path],
        params: Optional[List[str]] = None,
        **kwargs,
    ):
        """Generate a corner plot from current samples."""
        if self.mcmc is None:
            raise RuntimeError("MCMC not initialized.")

        from .visualization import Plotter

        plotter = Plotter(
            self.get_samples(),
            labels=self.mcmc.labels,
            ranges=self.mcmc.ranges,
        )
        return plotter.corner(params=params, filename=filename, **kwargs)


# ============================================================================
# Main API
# ============================================================================


def hicosmo(
    cosmology: Union[Type[CosmologyBase], str],
    likelihood: Union[Any, List[Any], str],
    free_params: List[str],
    preset: str = "planck2018",
    num_samples: int = 8000,
    num_chains: int = 4,
    chain_name: Optional[str] = None,
    sampler: str = "numpyro",
    **mcmc_kwargs,
) -> MCMC:
    """
    Create cosmological parameter inference with zero configuration.
    """
    cosmo_cls = resolve_cosmology_class(cosmology)
    resolved = resolve_likelihoods(cosmo_cls, likelihood)
    combined = combine_likelihoods(resolved)

    registry = ParameterRegistry.from_defaults(preset)

    model_param_names = register_model_parameters(registry, cosmo_cls)

    registry.add_from_likelihood(combined)
    apply_requested_free_params(registry, free_params, model_param_names)

    sampler_options, mcmc_init_options = split_mcmc_options(mcmc_kwargs)

    mcmc_config = registry.to_dict()
    mcmc_config["mcmc"] = {
        "num_samples": num_samples,
        "num_chains": num_chains,
        **sampler_options,
    }

    return MCMC(
        mcmc_config,
        combined,
        chain_name=chain_name,
        sampler=sampler,
        likelihoods=resolved,
        **mcmc_init_options,
    )


# ============================================================================
# Helper functions
# ============================================================================


def list_likelihoods() -> List[str]:
    """List available likelihood string identifiers."""
    return list_likelihood_names()


def list_cosmologies() -> List[str]:
    """List available cosmology model names."""
    return list_cosmology_names()


__all__ = ["hicosmo", "InferenceRunner", "list_likelihoods", "list_cosmologies"]


class _CallableHicosmoModule(types.ModuleType):
    """Make import-order-dependent ``from hicosmo import hicosmo`` callable."""

    def __call__(self, *args, **kwargs):
        return hicosmo(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableHicosmoModule
