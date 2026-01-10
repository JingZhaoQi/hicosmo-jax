"""Planck 2018 distance prior likelihood compatible with HIcosmo and JAX."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import jax
import jax.numpy as jnp
from jax import jit

from ..base import Likelihood, NuisanceList
from ...utils.constants import c_km_s
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PlanckDistancePriorData:
    R: float = 1.750235
    l_a: float = 301.4707
    omega_b_h2: float = 0.02235976
    inv_cov: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [
                [9.43923971e04, -1.3604913e03, 1.6645172916e06],
                [-1.3604913e03, 1.6143490e02, 3.6716180e03],
                [1.6645172916e06, 3.6716180e03, 7.97191825162e07],
            ],
            dtype=jnp.float32,
        )
    )

    def data_vector(self) -> jnp.ndarray:
        return jnp.array([self.R, self.l_a, self.omega_b_h2], dtype=jnp.float32)


class Planck2018DistancePriorsLikelihood(Likelihood):
    """Compressed Planck 2018 TT,TE,EE+lowE distance prior.

    Note on Sound Horizon Calibration
    ----------------------------------
    The EH98 fitting formula combined with the Hu--Sugiyama z_star fit yields
    a sound horizon that is slightly smaller than the value implied by the
    Planck 2018 distance-prior l_a at the fiducial cosmology. We therefore
    apply a constant calibration factor derived by matching the Planck l_a
    value at the Planck 2018 best-fit parameters:

        r_s^{Planck} / r_s^{EH98} = 144.64 / 144.42 = 1.00157

    This is a pragmatic consistency correction for the compressed distance
    priors and ensures the likelihood peaks at the intended cosmology.

    Supports consistent API with other HIcosmo likelihoods:

    Examples
    --------
    >>> from hicosmo.likelihoods import Planck2018DistancePriorsLikelihood
    >>> from hicosmo.models import LCDM
    >>>
    >>> # Create with cosmology class
    >>> cmb = Planck2018DistancePriorsLikelihood(cosmology_class=LCDM)
    >>> log_L = cmb(H0=70, Omega_m=0.3, Omega_b=0.05)
    >>>
    >>> # Or use convenience alias
    >>> from hicosmo.likelihoods import Planck
    >>> cmb = Planck(LCDM)
    >>> log_L = cmb(H0=70, Omega_m=0.3, Omega_b=0.05)
    """

    GRID_POINTS = 4096
    # Calibration to align EH98 r_s(z_*) with Planck distance-prior l_a at best fit.
    RS_ZSTAR_CALIBRATION = 1.00157

    def __init__(
        self,
        cosmology_class: Optional[Type] = None,
        dataset: Optional[PlanckDistancePriorData] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.cosmology_class = cosmology_class
        self.dataset = dataset or PlanckDistancePriorData()
        self.obs_vec = self.dataset.data_vector()
        self.inv_cov = self.dataset.inv_cov
        self._default_params = {}
        if self.cosmology_class is not None:
            try:
                default_model = self.cosmology_class()
                params = default_model.params
                if hasattr(params, "to_dict"):
                    self._default_params = params.to_dict()
                elif isinstance(params, dict):
                    self._default_params = dict(params)
            except Exception:
                self._default_params = {}
        super().__init__(name=name or "planck2018_distance", data_path=None, **kwargs)
        self.initialize()
        self._z_base = jnp.linspace(0.0, 1.0, self.GRID_POINTS, dtype=jnp.float32)
        self._initialize_fast_likelihood()

        if cosmology_class is not None:
            logger.info("Planck 2018 distance priors loaded")
            logger.info("Observables: R (shift), l_a (acoustic scale), omega_b*h^2")
            logger.info("Reference: Planck Collaboration 2018")

    def _default_dataset_name(self) -> str:
        return "planck2018_distance"

    def _load_data(self) -> None:
        return

    def _setup_covariance(self) -> None:
        return

    def get_requirements(self) -> Dict[str, Any]:
        return {}

    def theory(self, cosmology, **kwargs):
        raise NotImplementedError

    @property
    def nuisance_parameters(self) -> List:
        """Return nuisance parameters (none for Planck distance priors)."""
        return NuisanceList()

    def __call__(self, **params) -> float:
        """Compute log-likelihood from cosmological parameters.

        Parameters
        ----------
        **params : dict
            Cosmological parameters (H0, Omega_m, Omega_b required).

        Returns
        -------
        float
            Log-likelihood value.

        Examples
        --------
        >>> cmb = Planck2018DistancePriorsLikelihood(cosmology_class=LCDM)
        >>> log_L = cmb(H0=70, Omega_m=0.3, Omega_b=0.05)
        """
        if self.cosmology_class is None:
            raise ValueError(
                "cosmology_class not set. Create with: "
                "Planck2018DistancePriorsLikelihood(cosmology_class=LCDM)"
            )

        if self._can_use_fast_params(params):
            params_jax = self._prepare_params_dict(params)
            return self._loglike_fast(params_jax)

        cosmology = self.cosmology_class(**params)
        return self.log_likelihood(cosmology)

    # __add__ and __radd__ inherited from Likelihood base class

    def get_info(self) -> Dict[str, Any]:
        """Return likelihood information."""
        return {
            'name': self.name,
            'type': 'Planck2018DistancePriors',
            'observables': ['R', 'l_a', 'omega_b_h2'],
            'reference': 'Planck Collaboration 2018',
            'cosmology_class': self.cosmology_class.__name__ if self.cosmology_class else None,
        }

    def __repr__(self) -> str:
        cosmo_name = self.cosmology_class.__name__ if self.cosmology_class else "None"
        return f"Planck2018DistancePriorsLikelihood(cosmology_class={cosmo_name})"

    def log_likelihood(self, cosmology, **kwargs) -> float:
        H0 = jnp.asarray(cosmology.params["H0"], dtype=jnp.float32)
        Omega_m = jnp.asarray(cosmology.params["Omega_m"], dtype=jnp.float32)
        Omega_b = cosmology.params.get("Omega_b")
        if Omega_b is None:
            raise ValueError("Planck distance priors require Omega_b in the cosmology parameters.")
        Omega_b = jnp.asarray(Omega_b, dtype=jnp.float32)

        z_star = cosmology.recombination_redshift()
        D_A = cosmology.angular_diameter_distance(z_star)
        r_s = cosmology.sound_horizon(z_star) * self.RS_ZSTAR_CALIBRATION
        h = H0 / 100.0
        omega_b_h2 = Omega_b * h**2

        shift_R = jnp.sqrt(Omega_m) * (H0 / c_km_s) * D_A * (1.0 + z_star)
        l_a = jnp.pi * D_A * (1.0 + z_star) / r_s

        model_vec = jnp.stack([shift_R, l_a, omega_b_h2])
        diff = model_vec - self.obs_vec
        chi2 = diff @ (self.inv_cov @ diff)
        return -0.5 * chi2

    def get_derived_params(self, cosmology) -> Dict[str, float]:
        H0 = jnp.asarray(cosmology.params["H0"], dtype=jnp.float32)
        Omega_m = jnp.asarray(cosmology.params["Omega_m"], dtype=jnp.float32)
        Omega_b = jnp.asarray(cosmology.params["Omega_b"], dtype=jnp.float32)
        h = H0 / 100.0
        omega_b_h2 = Omega_b * h**2
        z_star = cosmology.recombination_redshift()
        D_A = cosmology.angular_diameter_distance(z_star)
        r_s = cosmology.sound_horizon(z_star) * self.RS_ZSTAR_CALIBRATION
        shift_R = jnp.sqrt(Omega_m) * (H0 / c_km_s) * D_A * (1.0 + z_star)
        l_a = jnp.pi * D_A * (1.0 + z_star) / r_s

        return {
            "R": float(shift_R),
            "l_a": float(l_a),
            "omega_b_h2": float(omega_b_h2),
            "z_star": float(z_star),
            "r_s_zstar": float(r_s),
        }


    def _prepare_params_dict(self, params: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """Convert params to JAX arrays using cosmology normalization."""
        return self._prepare_params_for_jax(
            params, dtype=jnp.float32, default_params=self._default_params
        )

    def _can_use_fast_params(self, params: Dict[str, Any]) -> bool:
        if self.cosmology_class is None:
            return False
        if not hasattr(self.cosmology_class, "compute_grid_traced"):
            return False
        if not hasattr(self.cosmology_class, "sound_horizon_traced"):
            return False
        return ('H0' in params or 'h' in params)

    def _initialize_fast_likelihood(self) -> None:
        """Build JIT-compiled fast likelihood path for JAX samplers."""
        if self.cosmology_class is None:
            self._loglike_fast = None
            return
        if not hasattr(self.cosmology_class, "compute_grid_traced"):
            self._loglike_fast = None
            return
        if not hasattr(self.cosmology_class, "sound_horizon_traced"):
            self._loglike_fast = None
            return
        if not hasattr(self.cosmology_class, "recombination_redshift_traced"):
            self._loglike_fast = None
            return

        obs_vec = self.obs_vec
        inv_cov = self.inv_cov
        z_base = self._z_base
        cosmology_class = self.cosmology_class

        @jit
        def _loglike_impl(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            H0 = params["H0"]
            Omega_m = params["Omega_m"]
            Omega_b = params["Omega_b"]

            h = H0 / 100.0
            omega_b_h2 = Omega_b * h**2
            z_star = cosmology_class.recombination_redshift_traced(params)

            z_grid = z_star * z_base
            cosmo_grid = cosmology_class.compute_grid_traced(z_grid, params)
            D_M = cosmo_grid["D_M"][-1]
            D_A = D_M / (1.0 + z_star)

            r_s = cosmology_class.sound_horizon_traced(z_star, params) * self.RS_ZSTAR_CALIBRATION

            shift_R = jnp.sqrt(Omega_m) * (H0 / c_km_s) * D_A * (1.0 + z_star)
            l_a = jnp.pi * D_A * (1.0 + z_star) / r_s

            model_vec = jnp.stack([shift_R, l_a, omega_b_h2])
            diff = model_vec - obs_vec
            chi2 = diff @ (inv_cov @ diff)
            return -0.5 * chi2

        self._loglike_fast = _loglike_impl
