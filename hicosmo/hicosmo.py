#!/usr/bin/env python3
"""
HIcosmo High-Level API - Minimal Interface for Cosmological Parameter Estimation.

This module provides the `hicosmo()` factory function for zero-configuration
parameter inference with automatic parameter matching, multi-likelihood support,
and intelligent defaults.

Examples
--------
>>> from hicosmo import hicosmo
>>>
>>> # Single likelihood
>>> inference = hicosmo('LCDM', 'sn', ['H0', 'Omega_m'])
>>> samples = inference.run(num_samples=8000)
>>>
>>> # Multiple likelihoods (joint analysis)
>>> inference = hicosmo('LCDM', ['sn', 'bao', 'cmb'], ['H0', 'Omega_m'])
>>> samples = inference.run(num_samples=16000)
"""

from typing import Type, Union, List, Dict, Any, Optional
from functools import reduce

from .models.base import CosmologyBase
from .parameters import ParameterRegistry
from .samplers import MCMC
from .likelihoods.combined import CombinedLikelihood
from .utils.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Lazy-loaded registries (avoid circular imports)
# ============================================================================

_COSMOLOGY_MAP: Dict[str, Type[CosmologyBase]] = {}
_LIKELIHOOD_MAP: Dict[str, callable] = {}


def _init_cosmology_map():
    """Initialize cosmology string-to-class mapping."""
    global _COSMOLOGY_MAP
    if _COSMOLOGY_MAP:
        return
    from .models import LCDM

    _COSMOLOGY_MAP = {"LCDM": LCDM, "lcdm": LCDM}
    try:
        from .models import wCDM, CPL

        _COSMOLOGY_MAP.update({"wCDM": wCDM, "wcdm": wCDM, "CPL": CPL, "cpl": CPL})
    except ImportError:
        pass


def _init_likelihood_map():
    """Initialize likelihood string-to-factory mapping."""
    global _LIKELIHOOD_MAP
    if _LIKELIHOOD_MAP:
        return
    from .likelihoods import (
        SN_likelihood,
        BAO_likelihood,
        H0LiCOWLikelihood,
        Planck2018DistancePriorsLikelihood,
        SH0ESLikelihood,
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
        "h0licow": lambda c: H0LiCOWLikelihood(),
        "tdcosmo": lambda c: TDCOSMOLikelihood(),
        # CMB
        "planck": lambda c: Planck2018DistancePriorsLikelihood(),
        "cmb": lambda c: Planck2018DistancePriorsLikelihood(),
        # H0
        "shoes": lambda c: SH0ESLikelihood(),
        "sh0es": lambda c: SH0ESLikelihood(),
    }


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

    Parameters
    ----------
    cosmology : Type[CosmologyBase] or str
        Cosmology model class or string ('LCDM', 'wCDM', 'CPL').
    likelihood : Likelihood, List[Likelihood], or str
        Likelihood(s): 'sn', 'bao', 'cmb', 'h0licow', 'tdcosmo', 'shoes', etc.
    free_params : List[str]
        Parameters to vary: ['H0', 'Omega_m']
    preset : str
        Parameter preset: 'planck2018', 'wmap9'
    num_samples : int
        Total MCMC samples (across all chains).
    num_chains : int
        Number of parallel chains.
    chain_name : str, optional
        Name for the MCMC chain file.
    sampler : str
        Sampler backend: 'numpyro' (default) or 'emcee'.
    **mcmc_kwargs
        Additional MCMC options (num_warmup, target_accept, etc.)

    Returns
    -------
    MCMC
        Configured MCMC sampler. Call `.run()` to start sampling.

    Examples
    --------
    >>> # Simple single-probe
    >>> mcmc = hicosmo('LCDM', 'bao', ['H0', 'Omega_m'])
    >>> mcmc.run()
    >>> mcmc.report()
    >>>
    >>> # Multi-probe joint analysis
    >>> mcmc = hicosmo('LCDM', ['sn', 'bao', 'cmb'], ['H0', 'Omega_m'])
    >>> mcmc.run(num_samples=16000, num_chains=8)
    """
    _init_cosmology_map()
    _init_likelihood_map()

    # 1. Resolve cosmology class
    cosmo_cls = (
        _COSMOLOGY_MAP.get(cosmology, cosmology)
        if isinstance(cosmology, str)
        else cosmology
    )
    if isinstance(cosmo_cls, str):
        raise ValueError(
            f"Unknown cosmology '{cosmology}'. Available: {list(_COSMOLOGY_MAP.keys())}"
        )

    # 2. Resolve and combine likelihoods (use existing CombinedLikelihood!)
    like_list = [likelihood] if not isinstance(likelihood, list) else likelihood
    resolved = []
    for like in like_list:
        if isinstance(like, str):
            if like not in _LIKELIHOOD_MAP:
                raise ValueError(
                    f"Unknown likelihood '{like}'. Available: {list(_LIKELIHOOD_MAP.keys())}"
                )
            resolved.append(_LIKELIHOOD_MAP[like](cosmo_cls))
        else:
            # Set cosmology_class if not set
            if hasattr(like, "cosmology_class") and like.cosmology_class is None:
                like.cosmology_class = cosmo_cls
            resolved.append(like)

    # Combine using + operator (leverages existing CombinedLikelihood)
    combined = reduce(lambda a, b: a + b, resolved)

    # 3. Build parameter registry
    registry = ParameterRegistry.from_defaults(preset)

    # Merge model-specific parameters (e.g., w for wCDM)
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

    # Add nuisance parameters from likelihood
    registry.add_from_likelihood(combined)

    # Set free parameters
    registry.set_free(free_params)

    # 4. Create MCMC (let MCMC handle nuisance collection, parameter mapping, etc.)
    mcmc_config = registry.to_dict()
    mcmc_config["mcmc"] = {
        "num_samples": num_samples,
        "num_chains": num_chains,
        **mcmc_kwargs,
    }

    return MCMC(mcmc_config, combined, chain_name=chain_name, sampler=sampler)


# ============================================================================
# Helper functions
# ============================================================================


def list_likelihoods() -> List[str]:
    """List available likelihood strings."""
    _init_likelihood_map()
    return list(_LIKELIHOOD_MAP.keys())


def list_cosmologies() -> List[str]:
    """List available cosmology strings."""
    _init_cosmology_map()
    return list(set(_COSMOLOGY_MAP.values()))


__all__ = ["hicosmo", "list_likelihoods", "list_cosmologies"]
