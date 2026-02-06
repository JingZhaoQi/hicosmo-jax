"""
HIcosmo Likelihood System
=========================

Clean, minimal likelihood implementations following CLAUDE.md design principles.

Submodules:
- bao/: Baryon Acoustic Oscillations
- sn/: Type Ia Supernovae
- cmb/: Cosmic Microwave Background
- h0/: Direct H0 measurements
- gw/: Gravitational Wave Standard Sirens
- lensing/: Strong Gravitational Lensing

Quick Start:
    >>> from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
    >>> from hicosmo.models import LCDM
    >>> sne = SN_likelihood(LCDM, "pantheon+", M_B="free")
    >>> bao = BAO_likelihood(LCDM, "desi2024")
    >>> log_L = sne(H0=70, Omega_m=0.3, M_B=-19.3) + bao(H0=70, Omega_m=0.3)

Discover Available Datasets:
    >>> from hicosmo.likelihoods import list_all_datasets, show_available_datasets
    >>> list_all_datasets()  # Returns dict of all available datasets
    >>> show_available_datasets()  # Pretty print all datasets
"""

# Base classes (kept at root level)
from .base import Likelihood, NuisanceList, GaussianLikelihood

# Combined likelihood
from .combined import CombinedLikelihood

# BAO submodule
from .bao import (
    BAOCollection,
    BAOLikelihood,
    BAO_likelihood,
    BAOLikelihoodFactory,
    BOSSDR12BAO,
    CustomBAO,
    DESI2024BAO,
    SDSSDR12BAO,
    SDSSDR16BAO,
    SixDFBAO,
    available_bao_datasets,
    create_bao_likelihood,
    get_available_datasets,
    register_bao_dataset,
)

# SN submodule
from .sn import (
    PantheonPlusLikelihood,
    SN_likelihood,
    SNLikelihood,
    available_sn_datasets,
    create_pantheonplus_likelihood,
    register_sn_dataset,
)

# CMB submodule
from .cmb import Planck2018DistancePriorsLikelihood

# H0 submodule
from .h0 import H0LiCOWLikelihood, SH0ESLikelihood

# GW submodule
from .gw import (
    BNSMassPrior,
    BrokenPowerLawMass,
    CosmologyAdapter,
    GWCatalogData,
    GWEventData,
    GWGalaxyCatalog,
    GWInjectionSet,
    GWPopulationRateModel,
    GWRateModel,
    GWStandardSirenLikelihood,
    Injections,
    MadauRate,
    MadauRateEvaluator,
    MassPriorEvaluator,
    PowerLawMass,
    PowerLawPeak,
    create_mock_gw_catalog,
    detector2source,
    detector2source_jacobian,
    detector2source_jacobian_q,
    detector2source_jacobian_single_mass,
    load_gw_event_hdf5,
    load_gw_injections,
    load_gwtc3_catalog,
    load_gwtc3_events,
)

# Lensing submodule
from .lensing import (
    ExternalLensLikelihood,
    HierarchicalGWLikelihood,
    HierarchicalTDCOSMO,
    KappaPrior,
    TDCOSMOLikelihood,
)

__all__ = [
    # Base
    "Likelihood",
    "NuisanceList",
    "GaussianLikelihood",
    "CombinedLikelihood",
    # Supernova
    "PantheonPlusLikelihood",
    "create_pantheonplus_likelihood",
    "SN_likelihood",
    "SNLikelihood",
    "register_sn_dataset",
    "available_sn_datasets",
    # BAO base
    "BAOLikelihood",
    "BAOCollection",
    # Specific BAO datasets
    "SDSSDR12BAO",
    "SDSSDR16BAO",
    "BOSSDR12BAO",
    "DESI2024BAO",
    "SixDFBAO",
    "CustomBAO",
    # BAO utilities
    "get_available_datasets",
    "create_bao_likelihood",
    "BAO_likelihood",
    "BAOLikelihoodFactory",
    "register_bao_dataset",
    "available_bao_datasets",
    # Strong lensing
    "H0LiCOWLikelihood",
    # CMB distance priors
    "Planck2018DistancePriorsLikelihood",
    "SH0ESLikelihood",
    "TDCOSMOLikelihood",
    "KappaPrior",
    "ExternalLensLikelihood",
    "HierarchicalTDCOSMO",
    # Gravitational Waves
    "GWStandardSirenLikelihood",
    "GWEventData",
    "GWCatalogData",
    "GWInjectionSet",
    "GWGalaxyCatalog",
    "PowerLawMass",
    "BrokenPowerLawMass",
    "PowerLawPeak",
    "BNSMassPrior",
    "MadauRate",
    "GWRateModel",
    "create_mock_gw_catalog",
    "load_gw_injections",
    "load_gw_event_hdf5",
    "load_gwtc3_catalog",
    "load_gwtc3_events",
    "Injections",
    "HierarchicalGWLikelihood",
    "CosmologyAdapter",
    "detector2source",
    "detector2source_jacobian",
    "detector2source_jacobian_q",
    "detector2source_jacobian_single_mass",
    "MassPriorEvaluator",
    "MadauRateEvaluator",
    "GWPopulationRateModel",
    # Convenience aliases (short names)
    "H0LiCOW",
    "Planck",
    "TDCOSMO",
    # Dataset discovery
    "list_all_datasets",
    "show_available_datasets",
]


# ============================================================
# Convenience Aliases for Simple API
# ============================================================
def H0LiCOW(cosmology_class=None, **kwargs):
    """
    Convenience alias for H0LiCOWLikelihood.

    Parameters
    ----------
    cosmology_class : type, optional
        Cosmology model class (LCDM, wCDM, CPL, etc.). Default: LCDM.
    **kwargs
        Additional arguments passed to H0LiCOWLikelihood.

    Returns
    -------
    H0LiCOWLikelihood
        Configured likelihood object.

    Examples
    --------
    >>> from hicosmo.likelihoods import H0LiCOW
    >>> from hicosmo.models import LCDM
    >>> lensing = H0LiCOW(LCDM)
    >>> log_L = lensing(H0=70, Omega_m=0.3)
    """
    if cosmology_class is not None:
        kwargs["cosmology_class"] = cosmology_class
    return H0LiCOWLikelihood(**kwargs)


def Planck(cosmology_class=None, **kwargs):
    """
    Convenience alias for Planck2018DistancePriorsLikelihood.

    Parameters
    ----------
    cosmology_class : type, optional
        Cosmology model class (LCDM, wCDM, CPL, etc.).
    **kwargs
        Additional arguments passed to Planck2018DistancePriorsLikelihood.

    Returns
    -------
    Planck2018DistancePriorsLikelihood
        Configured likelihood object.

    Examples
    --------
    >>> from hicosmo.likelihoods import Planck
    >>> from hicosmo.models import LCDM
    >>> cmb = Planck(LCDM)
    >>> log_L = cmb(H0=70, Omega_m=0.3, Omega_b=0.05)
    """
    if cosmology_class is not None:
        kwargs["cosmology_class"] = cosmology_class
    return Planck2018DistancePriorsLikelihood(**kwargs)


def TDCOSMO(cosmology_class=None, **kwargs):
    """
    Convenience alias for TDCOSMOLikelihood (hierarchical analysis).

    Full hierarchical Bayesian analysis of strong-lensing time delays,
    including mass-sheet transformation (MST) and stellar anisotropy parameters.

    Parameters
    ----------
    cosmology_class : type, optional
        Cosmology model class (LCDM, wCDM, CPL, etc.).
    **kwargs
        Additional arguments passed to TDCOSMOLikelihood.

    Returns
    -------
    TDCOSMOLikelihood
        Configured likelihood object with nuisance parameters.

    Nuisance Parameters
    -------------------
    The likelihood has 5 nuisance parameters that can be sampled:
    - lambda_int_mean : Internal MST population mean (default: 1.0, bounds: 0.5-1.5)
    - lambda_int_sigma : Internal MST scatter (default: 0.05, bounds: 0.001-0.2)
    - alpha_lambda : Slope of λ with R_eff/θ_E (default: 0.0, bounds: -1.0-1.0)
    - a_ani_mean : Stellar anisotropy mean (default: 1.0, bounds: 0.1-5.0)
    - a_ani_sigma : Anisotropy scatter (default: 0.1, bounds: 0.001-0.5)

    Examples
    --------
    >>> from hicosmo.likelihoods import TDCOSMO
    >>> from hicosmo.models import LCDM
    >>> tdcosmo = TDCOSMO(LCDM)

    >>> # Get nuisance parameters for MCMC setup
    >>> for name, default, lo, hi in tdcosmo.nuisance_parameters():
    ...     print(f"{name}: {default} ({lo}, {hi})")

    >>> # Evaluate with nuisance parameters
    >>> log_L = tdcosmo(H0=70, Omega_m=0.3, lambda_int_mean=1.0, a_ani_mean=1.0)
    """
    if cosmology_class is not None:
        kwargs["cosmology_class"] = cosmology_class
    # Prefer the bundled TDCOSMO2025 dataset when available so that
    # `TDCOSMO(LCDM)` reproduces the latest published analysis by default.
    if "use_tdcosmo2025" not in kwargs:
        from pathlib import Path

        # Go up 2 levels: likelihoods/ -> hicosmo/ -> project root
        tdcosmo2025_path = (
            Path(__file__).resolve().parents[2]
            / "TDCOSMO2025_public"
            / "TDCOSMO_sample"
        )
        if tdcosmo2025_path.exists():
            kwargs["use_tdcosmo2025"] = True
            kwargs.setdefault("anisotropy_model", "const")
    return TDCOSMOLikelihood(**kwargs)


# ============================================================
# Dataset Discovery Functions
# ============================================================
def list_all_datasets(refresh: bool = False):
    """
    List all available datasets across all likelihood types.

    Auto-discovers datasets by scanning the data directory. No hardcoding
    required - new datasets are automatically detected when added.

    Parameters
    ----------
    refresh : bool, optional
        If True, rescan the data directory (default: False).

    Returns
    -------
    dict
        Dictionary mapping category to list of dataset names.
        Categories: 'bao', 'sn', 'cmb', 'h0', 'gw', 'lensing'

    Examples
    --------
    >>> from hicosmo.likelihoods import list_all_datasets
    >>> datasets = list_all_datasets()
    >>> print(datasets)
    {'bao': ['boss_dr12', 'desi_2024', 'sdss_dr12', 'sdss_dr16', 'sixdf'],
     'sn': ['pantheon+shoes'],
     'cmb': ['planck2018_distance'],
     'h0': ['h0licow'],
     'gw': ['gwtc-3'],
     'lensing': ['tdcosmo', 'tdcosmo2025']}

    >>> # Use discovered datasets
    >>> bao = BAO_likelihood(LCDM, datasets['bao'][0])
    """
    from ..data_registry import list_all_datasets as _list_all

    return _list_all(refresh=refresh)


def show_available_datasets(category=None):
    """
    Pretty print all available datasets.

    Displays a formatted table of available datasets with descriptions,
    making it easy for users to discover what data is available.

    Parameters
    ----------
    category : str, optional
        If specified, show only datasets for this category.
        Options: 'bao', 'sn', 'cmb', 'h0', 'gw', 'lensing'

    Examples
    --------
    >>> from hicosmo.likelihoods import show_available_datasets

    >>> # Show all datasets
    >>> show_available_datasets()
    ======================================================================
                        HIcosmo Available Datasets
    ======================================================================

      BAO (Baryon Acoustic Oscillations)
      --------------------------------------------------
        • desi_2024        : DESI 2024 DR1 BAO measurements (z=0.3-2.3)
        • boss_dr12        : BOSS DR12 Luminous Red Galaxy BAO
        • sdss_dr12        : SDSS DR12 consensus BAO
        • sdss_dr16        : SDSS DR16 LRG and QSO BAO
        • sixdf            : 6dF Galaxy Survey BAO (z=0.1)

      SN (Type Ia Supernovae)
      --------------------------------------------------
        • pantheon+shoes   : Pantheon+ with SH0ES Cepheid calibration
    ...

    >>> # Show only BAO datasets
    >>> show_available_datasets('bao')
    """
    from ..data_registry import show_available_datasets as _show

    _show(category=category)
