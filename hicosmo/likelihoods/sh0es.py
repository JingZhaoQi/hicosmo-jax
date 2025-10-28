"""SH0ES local H0 prior likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax.numpy as jnp

from .base import Likelihood


@dataclass
class SH0ESData:
    """Gaussian prior on the Hubble constant from SH0ES."""

    H0_mean: float = 73.04  # km/s/Mpc (Riess et al. 2021)
    H0_sigma: float = 1.04


class SH0ESLikelihood(Likelihood):
    """Implements the SH0ES local H0 prior as a Gaussian likelihood."""

    def __init__(self, data: Optional[SH0ESData] = None, name: Optional[str] = None, **kwargs: Any) -> None:
        self.data = data or SH0ESData()
        super().__init__(name=name or "sh0es", data_path=None, **kwargs)
        self.initialize()

    def _default_dataset_name(self) -> str:
        return "sh0es"

    def _load_data(self) -> None:  # type: ignore[override]
        return

    def _setup_covariance(self) -> None:  # type: ignore[override]
        self.inv_cov = None

    def get_requirements(self) -> Dict[str, Any]:  # type: ignore[override]
        return {}

    def theory(self, cosmology, **kwargs):  # type: ignore[override]
        raise NotImplementedError("SH0ES likelihood is evaluated via log_likelihood only.")

    def log_likelihood(self, cosmology, **kwargs) -> float:
        H0 = jnp.asarray(cosmology.params["H0"], dtype=jnp.float32)
        diff = (H0 - self.data.H0_mean) / self.data.H0_sigma
        return -0.5 * diff**2 - jnp.log(self.data.H0_sigma * jnp.sqrt(2.0 * jnp.pi))

    def get_derived_params(self, cosmology) -> Dict[str, float]:
        return {"H0": float(cosmology.params["H0"])}
