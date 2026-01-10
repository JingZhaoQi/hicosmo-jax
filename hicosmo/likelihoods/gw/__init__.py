"""
Gravitational Wave (GW) Likelihoods
===================================

Provides likelihood implementations for gravitational wave standard sirens.

Available Classes:
- GWStandardSirenLikelihood: Main GW standard siren likelihood
- GWEventData, GWCatalogData, GWInjectionSet: Data structures
- PowerLawMass, BrokenPowerLawMass, PowerLawPeak: Mass distributions
- MadauRate, GWRateModel: Rate models

Quick Start:
    >>> from hicosmo.likelihoods.gw import GWStandardSirenLikelihood
    >>> from hicosmo.models import LCDM
    >>> gw = GWStandardSirenLikelihood(LCDM, catalog_path="gwtc3.hdf5")
    >>> log_L = gw(H0=70, Omega_m=0.3)
"""

from .standard_siren import (
    GWCatalogData,
    GWEventData,
    GWGalaxyCatalog,
    GWInjectionSet,
    GWRateModel,
    GWStandardSirenLikelihood,
    MadauRate,
    PowerLawMass,
    PowerLawPeak,
    create_mock_gw_catalog,
    load_gw_injections,
    load_gw_event_hdf5,
    load_gwtc3_catalog,
    load_gwtc3_events,
)
from .population import BNSMassPrior, BrokenPowerLawMass
from .population_rate import (
    CosmologyAdapter,
    GWPopulationRateModel,
    MadauRateEvaluator,
    MassPriorEvaluator,
    detector2source,
    detector2source_jacobian,
    detector2source_jacobian_q,
    detector2source_jacobian_single_mass,
)
from .injections import Injections

__all__ = [
    # Main likelihood
    "GWStandardSirenLikelihood",
    # Data structures
    "GWEventData",
    "GWCatalogData",
    "GWInjectionSet",
    "GWGalaxyCatalog",
    # Mass distributions
    "PowerLawMass",
    "BrokenPowerLawMass",
    "PowerLawPeak",
    "BNSMassPrior",
    # Rate models
    "MadauRate",
    "GWRateModel",
    # Utilities
    "create_mock_gw_catalog",
    "load_gw_injections",
    "load_gw_event_hdf5",
    "load_gwtc3_catalog",
    "load_gwtc3_events",
    "Injections",
    # Population rate
    "CosmologyAdapter",
    "GWPopulationRateModel",
    "MadauRateEvaluator",
    "MassPriorEvaluator",
    "detector2source",
    "detector2source_jacobian",
    "detector2source_jacobian_q",
    "detector2source_jacobian_single_mass",
]
