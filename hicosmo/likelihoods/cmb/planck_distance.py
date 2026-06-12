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
            dtype=jnp.float64,
        )
    )

    def data_vector(self) -> jnp.ndarray:
        return jnp.array([self.R, self.l_a, self.omega_b_h2], dtype=jnp.float64)


class Planck2018DistancePriorsLikelihood(Likelihood):
    """Compressed Planck 2018 TT,TE,EE+lowE distance prior.

    Note on Sound Horizon Calibration
    ----------------------------------
    The EH98 r_s fitting formula and the Hu--Sugiyama z_star fit deviate
    from the full CAMB Boltzmann computation at the 0.1--0.2% level, with a
    parameter-dependent drift across the sampled (omega_b, omega_m) range.
    Because the Planck distance-prior covariance constrains l_a to ~3e-4
    relative precision, both quantities are rescaled by quadratic
    CAMB-calibration ratios in (omega_b h^2, omega_m h^2); see
    _RS_CAL_COEF/_ZSTAR_CAL_COEF below. After this calibration the
    (r_s, z_*) pipeline matches CAMB to better than 2e-5 across the grid.

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
    # Quadratic CAMB calibration of the EH98 r_s(z_*) and Hu-Sugiyama z_*
    # fitting formulae, fitted on a 7x9 CAMB background grid over
    # omega_b_h2 in [0.018, 0.028] and omega_m_h2 in [0.10, 0.22]
    # (fit residuals < 1.4e-5; the ratios depend on the physical densities
    # only — H0-independence verified numerically). This replaces the old
    # constant factor 1.00157, which matched l_a only at the Planck best
    # fit and silently compensated a radiation-density rounding error that
    # has since been fixed in compute_omega_r.
    # ratio = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2,
    # x = (omega_b_h2 - 0.02237)/0.01, y = (omega_m_h2 - 0.1430)/0.1
    _RS_CAL_COEF = (
        9.98975730e-01,
        -5.91048262e-04,
        3.37320595e-04,
        -6.28955632e-05,
        1.59326538e-04,
        -7.70064498e-05,
    )
    _ZSTAR_CAL_COEF = (
        9.98242260e-01,
        8.44026766e-04,
        1.15633371e-04,
        1.41197434e-04,
        -1.72541508e-04,
        -8.48441118e-05,
    )

    @staticmethod
    def _camb_ratio(coef, omega_b_h2, omega_m_h2):
        """Evaluate the quadratic CAMB-calibration ratio (JAX-traceable)."""
        x = (omega_b_h2 - 0.02237) / 0.01
        y = (omega_m_h2 - 0.1430) / 0.1
        return (
            coef[0]
            + coef[1] * x
            + coef[2] * y
            + coef[3] * x * x
            + coef[4] * x * y
            + coef[5] * y * y
        )

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
        # Square-stretched grid: a UNIFORM grid to z_*~1090 (dz~0.27) leaves a
        # 0.09% trapezoid error in D_M(z_*) from the low-z curvature of 1/E(z)
        # — about 6 sigma of the Planck l_a covariance. u^2 concentrates the
        # points at low z and removes the error at identical cost.
        u = jnp.linspace(0.0, 1.0, self.GRID_POINTS, dtype=jnp.float64)
        self._z_base = u * u
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

        self._warn_unknown_params(params)
        if self._can_use_fast_params(params):
            params_jax = self._prepare_params_dict(params)
            return self._loglike_fast(params_jax)

        cosmology = self.cosmology_class(**params)
        return self.log_likelihood(cosmology)

    # __add__ and __radd__ inherited from Likelihood base class

    def get_info(self) -> Dict[str, Any]:
        """Return likelihood information."""
        return {
            "name": self.name,
            "type": "Planck2018DistancePriors",
            "observables": ["R", "l_a", "omega_b_h2"],
            "reference": "Planck Collaboration 2018",
            "cosmology_class": (
                self.cosmology_class.__name__ if self.cosmology_class else None
            ),
        }

    def __repr__(self) -> str:
        cosmo_name = self.cosmology_class.__name__ if self.cosmology_class else "None"
        return f"Planck2018DistancePriorsLikelihood(cosmology_class={cosmo_name})"

    def log_likelihood(self, cosmology, **kwargs) -> float:
        H0 = jnp.asarray(cosmology.params["H0"], dtype=jnp.float64)
        Omega_m = jnp.asarray(cosmology.params["Omega_m"], dtype=jnp.float64)
        Omega_b = cosmology.params.get("Omega_b")
        if Omega_b is None:
            raise ValueError(
                "Planck distance priors require Omega_b in the cosmology parameters."
            )
        Omega_b = jnp.asarray(Omega_b, dtype=jnp.float64)

        h = H0 / 100.0
        omega_b_h2 = Omega_b * h**2
        omega_m_h2 = Omega_m * h**2

        z_star_raw = cosmology.recombination_redshift()
        z_star = z_star_raw * self._camb_ratio(
            self._ZSTAR_CAL_COEF, omega_b_h2, omega_m_h2
        )
        r_s = cosmology.sound_horizon(z_star_raw) * self._camb_ratio(
            self._RS_CAL_COEF, omega_b_h2, omega_m_h2
        )
        D_A = cosmology.angular_diameter_distance(z_star)

        shift_R = jnp.sqrt(Omega_m) * (H0 / c_km_s) * D_A * (1.0 + z_star)
        l_a = jnp.pi * D_A * (1.0 + z_star) / r_s

        model_vec = jnp.stack([shift_R, l_a, omega_b_h2])
        diff = model_vec - self.obs_vec
        chi2 = diff @ (self.inv_cov @ diff)
        return -0.5 * chi2

    def get_derived_params(self, cosmology) -> Dict[str, float]:
        H0 = jnp.asarray(cosmology.params["H0"], dtype=jnp.float64)
        Omega_m = jnp.asarray(cosmology.params["Omega_m"], dtype=jnp.float64)
        Omega_b = jnp.asarray(cosmology.params["Omega_b"], dtype=jnp.float64)
        h = H0 / 100.0
        omega_b_h2 = Omega_b * h**2
        omega_m_h2 = Omega_m * h**2
        z_star_raw = cosmology.recombination_redshift()
        z_star = z_star_raw * self._camb_ratio(
            self._ZSTAR_CAL_COEF, omega_b_h2, omega_m_h2
        )
        r_s = cosmology.sound_horizon(z_star_raw) * self._camb_ratio(
            self._RS_CAL_COEF, omega_b_h2, omega_m_h2
        )
        D_A = cosmology.angular_diameter_distance(z_star)
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
            params, dtype=jnp.float64, default_params=self._default_params
        )

    def _can_use_fast_params(self, params: Dict[str, Any]) -> bool:
        if self.cosmology_class is None:
            return False
        if not hasattr(self.cosmology_class, "compute_grid_traced"):
            return False
        if not hasattr(self.cosmology_class, "sound_horizon_traced"):
            return False
        return "H0" in params or "h" in params

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

        rs_cal_coef = self._RS_CAL_COEF
        zstar_cal_coef = self._ZSTAR_CAL_COEF
        camb_ratio = self._camb_ratio

        @jit
        def _loglike_impl(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            H0 = params["H0"]
            Omega_m = params["Omega_m"]
            Omega_b = params["Omega_b"]

            h = H0 / 100.0
            omega_b_h2 = Omega_b * h**2
            omega_m_h2 = Omega_m * h**2

            # Hu-Sugiyama z_* and EH98 r_s(z_*), both rescaled by the
            # quadratic CAMB calibration. r_s is evaluated at the RAW
            # Hu-Sugiyama z_* because the calibration ratio was fitted
            # against exactly that pipeline.
            z_star_raw = cosmology_class.recombination_redshift_traced(params)
            z_star = z_star_raw * camb_ratio(zstar_cal_coef, omega_b_h2, omega_m_h2)
            r_s = cosmology_class.sound_horizon_traced(
                z_star_raw, params
            ) * camb_ratio(rs_cal_coef, omega_b_h2, omega_m_h2)

            # Grid from 0 to (calibrated) z_star with GRID_POINTS points
            z_grid = z_star * z_base
            cosmo_grid = cosmology_class.compute_grid_traced(z_grid, params)
            D_M = cosmo_grid["D_M"][-1]
            D_A = D_M / (1.0 + z_star)

            shift_R = jnp.sqrt(Omega_m) * (H0 / c_km_s) * D_A * (1.0 + z_star)
            l_a = jnp.pi * D_A * (1.0 + z_star) / r_s

            model_vec = jnp.stack([shift_R, l_a, omega_b_h2])
            diff = model_vec - obs_vec
            chi2 = diff @ (inv_cov @ diff)
            return -0.5 * chi2

        self._loglike_fast = _loglike_impl
