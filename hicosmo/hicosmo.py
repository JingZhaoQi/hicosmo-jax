#!/usr/bin/env python3
"""
HIcosmo high-level API and convenience runner.
"""

from functools import reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .models.base import CosmologyBase
from .parameters import ParameterRegistry
from .samplers import MCMC
from .utils.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Lazy-loaded registries (avoid circular imports)
# ============================================================================

_COSMOLOGY_MAP: Dict[str, Type[CosmologyBase]] = {}
_LIKELIHOOD_MAP: Dict[str, Callable[[Type[CosmologyBase]], Any]] = {}


def _init_cosmology_map() -> None:
    """Initialize cosmology string-to-class mapping."""
    global _COSMOLOGY_MAP
    if _COSMOLOGY_MAP:
        return

    from .models import LCDM

    _COSMOLOGY_MAP = {"LCDM": LCDM, "lcdm": LCDM}
    try:
        from .models import CPL, ILCDM, wCDM

        _COSMOLOGY_MAP.update(
            {
                "wCDM": wCDM,
                "wcdm": wCDM,
                "CPL": CPL,
                "cpl": CPL,
                "ILCDM": ILCDM,
                "ilcdm": ILCDM,
            }
        )
    except ImportError:
        pass


def _init_likelihood_map() -> None:
    """Initialize likelihood string-to-factory mapping."""
    global _LIKELIHOOD_MAP
    if _LIKELIHOOD_MAP:
        return

    from .likelihoods import (
        BAO_likelihood,
        H0LiCOWLikelihood,
        Planck2018DistancePriorsLikelihood,
        SH0ESLikelihood,
        SN_likelihood,
        TDCOSMOLikelihood,
    )

    _LIKELIHOOD_MAP = {
        # Supernovae
        "sn": lambda c: SN_likelihood(c, "pantheon+", M_B="marginalize"),
        "sn_shoes": lambda c: SN_likelihood(c, "pantheon+shoes", M_B="marginalize"),
        # BAO
        "bao": lambda c: BAO_likelihood(c, "desi2024"),
        "bao_sdss": lambda c: BAO_likelihood(c, "sdss_dr16"),
        # Strong lensing
        "h0licow": lambda c: H0LiCOWLikelihood(cosmology_class=c),
        "tdcosmo": lambda c: TDCOSMOLikelihood(cosmology_class=c),
        # CMB
        "planck": lambda c: Planck2018DistancePriorsLikelihood(cosmology_class=c),
        "cmb": lambda c: Planck2018DistancePriorsLikelihood(cosmology_class=c),
        # H0
        "shoes": lambda c: SH0ESLikelihood(),
        "sh0es": lambda c: SH0ESLikelihood(),
    }


def _resolve_cosmology_class(
    cosmology: Union[Type[CosmologyBase], str],
) -> Type[CosmologyBase]:
    """Resolve a cosmology specification to a model class."""
    _init_cosmology_map()
    cosmo_cls = (
        _COSMOLOGY_MAP.get(cosmology, cosmology)
        if isinstance(cosmology, str)
        else cosmology
    )
    if isinstance(cosmo_cls, str):
        raise ValueError(
            f"Unknown cosmology '{cosmology}'. Available: {list(_COSMOLOGY_MAP.keys())}"
        )
    return cosmo_cls


def _resolve_likelihoods(
    cosmology_class: Type[CosmologyBase],
    likelihood: Union[Any, List[Any], str],
) -> List[Any]:
    """Resolve likelihood spec(s) into instantiated likelihood objects."""
    _init_likelihood_map()
    like_list = [likelihood] if not isinstance(likelihood, list) else likelihood
    resolved = []

    for like in like_list:
        if isinstance(like, str):
            if like not in _LIKELIHOOD_MAP:
                raise ValueError(
                    f"Unknown likelihood '{like}'. Available: {list(_LIKELIHOOD_MAP.keys())}"
                )
            resolved.append(_LIKELIHOOD_MAP[like](cosmology_class))
        else:
            if hasattr(like, "cosmology_class") and like.cosmology_class is None:
                like.cosmology_class = cosmology_class
            resolved.append(like)

    if not resolved:
        raise ValueError("At least one likelihood must be provided.")
    return resolved


def _combine_likelihoods(likelihoods: List[Any]) -> Any:
    """Combine likelihood objects with the + operator."""
    if len(likelihoods) == 1:
        return likelihoods[0]
    return reduce(lambda a, b: a + b, likelihoods)


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
        self.cosmology_class = _resolve_cosmology_class(cosmology_class)
        self.likelihoods = _resolve_likelihoods(self.cosmology_class, likelihoods)
        self.likelihood_func = _combine_likelihoods(self.likelihoods)

        self.registry = registry or ParameterRegistry.from_defaults(preset)

        if hasattr(self.cosmology_class, "get_parameters"):
            for param in self.cosmology_class.get_parameters():
                if param.name not in self.registry:
                    self.registry.add(
                        param.name,
                        value=param.value,
                        free=param.free,
                        prior=param.prior,
                        latex_label=param.latex_label,
                    )

        for likelihood in self.likelihoods:
            self.registry.add_from_likelihood(likelihood)

        if free_params:
            self.registry.set_free(free_params)

        self.mcmc_config = dict(mcmc_config or {})
        self.sampler_name = str(self.mcmc_config.get("sampler", "numpyro"))
        self.chain_name = self.mcmc_config.get("chain_name")
        self.mcmc_options = {
            k: v
            for k, v in self.mcmc_config.items()
            if k not in {"sampler", "chain_name", "derived"}
        }
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
    cosmo_cls = _resolve_cosmology_class(cosmology)
    resolved = _resolve_likelihoods(cosmo_cls, likelihood)
    combined = _combine_likelihoods(resolved)

    registry = ParameterRegistry.from_defaults(preset)

    if hasattr(cosmo_cls, "get_parameters"):
        for param in cosmo_cls.get_parameters():
            if param.name not in registry:
                registry.add(
                    param.name,
                    value=param.value,
                    free=param.free,
                    prior=param.prior,
                    latex_label=param.latex_label,
                )

    registry.add_from_likelihood(combined)
    registry.set_free(free_params)

    mcmc_config = registry.to_dict()
    mcmc_config["mcmc"] = {
        "num_samples": num_samples,
        "num_chains": num_chains,
        **mcmc_kwargs,
    }

    return MCMC(
        mcmc_config,
        combined,
        chain_name=chain_name,
        sampler=sampler,
        likelihoods=resolved,
    )


# ============================================================================
# Helper functions
# ============================================================================


def list_likelihoods() -> List[str]:
    """List available likelihood string identifiers."""
    _init_likelihood_map()
    return sorted(_LIKELIHOOD_MAP.keys())


def list_cosmologies() -> List[str]:
    """List available cosmology model names."""
    _init_cosmology_map()
    return sorted({cls.__name__ for cls in _COSMOLOGY_MAP.values()})


__all__ = ["hicosmo", "InferenceRunner", "list_likelihoods", "list_cosmologies"]
