"""
Supernova (SN Ia) Likelihoods
=============================

Provides likelihood implementations for Type Ia supernova distance measurements.

Available Datasets:
- PantheonPlus: Pantheon+ sample (1701 SNe Ia)
- Union3: Union3/UNITY1.5 compressed Gaussian (2087 SNe Ia, 22 bins)
- DESY5: DES Year 5 supernova sample (1820 SNe Ia)

Quick Start:
    >>> from hicosmo.likelihoods import SN_likelihood
    >>> from hicosmo.models import LCDM
    >>> sne = SN_likelihood(LCDM, "pantheon+", M_B="free")
    >>> log_L = sne(H0=70, Omega_m=0.3, M_B=-19.3)

    >>> # Union3 (no M_B needed)
    >>> union3 = SN_likelihood(LCDM, "union3")
    >>> log_L = union3(H0=70, Omega_m=0.3)

    >>> # DES-SN5YR (M analytically marginalized by default)
    >>> desy5 = SN_likelihood(LCDM, "desy5")
    >>> log_L = desy5(H0=70, Omega_m=0.3)
"""

from .pantheonplus import PantheonPlusLikelihood, create_pantheonplus_likelihood
from .union3 import Union3Likelihood, create_union3_likelihood
from .desy5 import DESY5Likelihood, create_desy5_likelihood
from .factory import (
    SN_likelihood,
    SNLikelihood,
    register_sn_dataset,
    available_sn_datasets,
)

__all__ = [
    # Likelihood classes
    "PantheonPlusLikelihood",
    "Union3Likelihood",
    "DESY5Likelihood",
    "SNLikelihood",
    # Factory functions
    "SN_likelihood",
    "create_pantheonplus_likelihood",
    "create_union3_likelihood",
    "create_desy5_likelihood",
    "register_sn_dataset",
    "available_sn_datasets",
]
