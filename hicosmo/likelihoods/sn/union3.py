"""
Union3 Supernova Likelihood Function
=====================================

Implementation of Union3 (Rubin et al. 2023) Type Ia supernova likelihood.
Uses the compressed Gaussian approximation from UNITY1.5 Bayesian hierarchical
standardization, providing 22 binned distance moduli with full covariance.

Key differences from Pantheon+:
- Data format: 22 binned distance moduli (not 1701 individual SNe)
- M_B already marginalized in the UNITY1.5 framework (no M_B parameter)
- Inverse covariance matrix provided directly (no need to invert)
- Redshift range: 0.05 < z < 2.26 (2087 SNe Ia compressed into 22 bins)

Data source: Rubin et al. 2023, arXiv:2311.12098
             https://github.com/rubind/union3_release
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path

from ...models.base import CosmologyBase, compute_background_grid_for_model
from ..base import Likelihood, NuisanceList
from ...utils.jax_tools import interp_linspace
from ...utils.logging import get_logger
from ...models import LCDM

logger = get_logger(__name__)


def _get_default_data_path() -> Path:
    """Get default data path for Union3 data."""
    package_dir = Path(__file__).parent
    project_root = package_dir.parent.parent
    return project_root / "data" / "sne" / "union3"


class Union3Likelihood(Likelihood):
    """
    Union3 Type Ia supernova likelihood function.

    Uses the compressed Gaussian approximation from UNITY1.5 hierarchical
    standardization. The data consists of 22 binned distance moduli with
    a full inverse covariance matrix. M_B is already marginalized in the
    UNITY framework, so no M_B parameter is needed.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to Union3 data directory containing union3_data.npz.
        If None, uses the default package data directory.
    cosmology_class : type, optional
        Cosmology model class (LCDM, wCDM, CPL, etc.). Default: LCDM.
    verbose : bool, default False
        Print diagnostic information during initialization.
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        cosmology_class: type = None,
        verbose: bool = False,
    ):
        if data_path is None:
            self.data_path = str(_get_default_data_path())
        else:
            self.data_path = str(Path(data_path))
        self._cosmology_class = cosmology_class if cosmology_class is not None else LCDM
        self.verbose = verbose

        # Load data
        self._load_data()
        self._precompute()
        self._prepare_distance_grid()

        if self.verbose:
            logger.info(f"Union3 loaded: {self.n_bins} redshift bins")
            logger.info(
                f"Redshift range: {float(self.redshifts.min()):.3f} - "
                f"{float(self.redshifts.max()):.3f}"
            )

    def _load_data(self):
        """Load Union3 compressed Gaussian data from npz file."""
        data_path = Path(self.data_path)
        npz_file = data_path / "union3_data.npz"

        if not npz_file.exists():
            raise FileNotFoundError(
                f"Union3 data file not found: {npz_file}. "
                f"Please ensure the Union3 data is installed in {data_path}."
            )

        data = np.load(npz_file)
        self.redshifts = jnp.array(data["redshifts"])
        self.mu_obs = jnp.array(data["mu"])
        self.cov_inv = jnp.array(data["cov_inv"])
        self.n_bins = len(self.redshifts)

        # Compute log determinant of covariance for normalization
        # cov_inv is already provided; compute log|C| = -log|C^{-1}|
        sign, logdet_inv = jnp.linalg.slogdet(self.cov_inv)
        self.log_det_cov = -logdet_inv

        # Validate
        assert self.cov_inv.shape == (self.n_bins, self.n_bins), (
            f"Inverse covariance shape {self.cov_inv.shape} inconsistent "
            f"with {self.n_bins} bins"
        )

    def _setup_covariance(self) -> None:
        """Compatibility with Likelihood base class."""
        pass

    def get_requirements(self) -> Dict[str, Any]:
        """Return cosmology requirements (handled internally)."""
        return {}

    def theory(self, cosmology: CosmologyBase, **kwargs) -> jnp.ndarray:
        """Return theory distance modulus vector at Union3 redshift bins."""
        return self._distance_modulus_from_model(cosmology)

    def _precompute(self):
        """Precompute JIT-compiled likelihood function.

        Applies analytic marginalization over the absolute magnitude offset.
        Although UNITY handles M_B internally, the binned distance moduli
        retain an overall offset degeneracy with H0.  Following the standard
        Pantheon+ prescription (and Cobaya's implementation), we project out
        this offset by modifying the inverse covariance matrix:
            C^{-1} -> C^{-1} - C^{-1} 1 (1^T C^{-1} 1)^{-1} 1^T C^{-1}
        This makes the likelihood insensitive to a constant shift in mu,
        ensuring H0 is not spuriously constrained.
        """
        dtype = self.mu_obs.dtype
        self.mu_obs = jnp.asarray(self.mu_obs, dtype=dtype)
        self.cov_inv = jnp.asarray(self.cov_inv, dtype=dtype)
        self.log_det_cov = jnp.asarray(self.log_det_cov, dtype=dtype)

        # Analytic marginalization over absolute magnitude offset
        ones = jnp.ones(self.n_bins, dtype=dtype)
        Cinv_ones = self.cov_inv @ ones
        fisher_M = ones @ Cinv_ones
        self.cov_inv = self.cov_inv - jnp.outer(Cinv_ones, Cinv_ones) / fisher_M

        # Capture constants in closure for JIT
        mu_obs = self.mu_obs
        cov_inv = self.cov_inv

        @jit
        def _likelihood_impl(mu_theory: jnp.ndarray) -> jnp.ndarray:
            residual = mu_obs - mu_theory
            chi2 = residual @ (cov_inv @ residual)
            return -0.5 * chi2

        self._likelihood_jit = _likelihood_impl

        # Build grid-accepting version (for CombinedLikelihood shared grid)
        redshifts = self.redshifts

        @jit
        def _loglike_from_grid_impl(
            cosmo_grid: Dict, grid_z: jnp.ndarray, params: Dict
        ) -> jnp.ndarray:
            d_L = interp_linspace(redshifts, grid_z, cosmo_grid["d_L"])
            mu = 5.0 * jnp.log10(d_L) + 25.0
            return _likelihood_impl(mu)

        self._loglike_from_grid = _loglike_from_grid_impl

    def _prepare_distance_grid(self) -> None:
        """Precompute redshift grid for JAX-based cosmology interpolation."""
        max_z = float(self.redshifts.max()) + 0.5
        n_grid = int(jnp.clip(max_z / 1e-3, 2048, 8192))
        self._z_grid = jnp.linspace(0.0, max_z, n_grid, dtype=self.mu_obs.dtype)

    def _distance_modulus_from_model(self, model: CosmologyBase) -> jnp.ndarray:
        """Compute distance modulus vector using JAX traced cosmology."""
        cosmo_params = self._extract_cosmo_params(model)
        cosmo_grid = compute_background_grid_for_model(
            self._cosmology_class, self._z_grid, cosmo_params
        )
        d_L_grid = cosmo_grid["d_L"]
        d_L = interp_linspace(self.redshifts, self._z_grid, d_L_grid)
        return 5.0 * jnp.log10(d_L) + 25.0

    def _distance_modulus_from_params(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Compute distance modulus directly from parameter dict using traced grids."""
        dtype = self.mu_obs.dtype
        cosmo_params_raw = self._cosmology_class.normalize_params(params, dtype=dtype)
        cosmo_params = {}
        for k, v in cosmo_params_raw.items():
            if isinstance(v, (int, float)):
                cosmo_params[k] = jnp.asarray(v, dtype=dtype)
            elif isinstance(v, jax.core.Tracer):
                cosmo_params[k] = v
            elif hasattr(v, "shape"):
                cosmo_params[k] = jnp.asarray(v, dtype=dtype)
        cosmo_grid = compute_background_grid_for_model(
            self._cosmology_class, self._z_grid, cosmo_params
        )
        d_L_grid = cosmo_grid["d_L"]
        d_L = interp_linspace(self.redshifts, self._z_grid, d_L_grid)
        return 5.0 * jnp.log10(d_L) + 25.0

    def _extract_cosmo_params(self, model: CosmologyBase) -> Dict[str, jnp.ndarray]:
        """Convert model.params to JAX-compatible dict."""
        params_dict = model.params.to_dict()
        dtype = self.mu_obs.dtype
        return {
            k: jnp.asarray(v, dtype=dtype)
            for k, v in params_dict.items()
            if isinstance(v, (int, float, jnp.ndarray, np.ndarray))
        }

    def _can_use_fast_params(self, params: Dict[str, Any]) -> bool:
        """Check whether we can bypass model construction."""
        if not hasattr(self._cosmology_class, "compute_grid_traced"):
            return False
        if "H0" not in params:
            return False
        if "h" in params and "H0" not in params:
            return False
        return True

    def log_likelihood(self, model: CosmologyBase, **kwargs) -> float:
        """
        Compute log-likelihood for Union3 data.

        Parameters
        ----------
        model : CosmologyBase
            Cosmological model instance.
        **kwargs : dict
            Additional parameters (unused).

        Returns
        -------
        float
            Log-likelihood value (-chi2/2).
        """
        if model is None:
            raise ValueError("model argument must be provided")
        theory_mu = self._distance_modulus_from_model(model)
        return self._likelihood_jit(theory_mu)

    def __call__(self, **params) -> float:
        """
        Callable interface for MCMC sampling.

        Constructs cosmology model from parameters and computes log-likelihood.
        No M_B parameter needed (already marginalized in UNITY framework).

        Parameters
        ----------
        **params : dict
            Cosmological parameters (H0, Omega_m, etc.).

        Returns
        -------
        float
            Log-likelihood value.

        Examples
        --------
        >>> union3 = Union3Likelihood(cosmology_class=LCDM)
        >>> log_L = union3(H0=70, Omega_m=0.3)
        """
        self._warn_unknown_params(params)
        # No nuisance parameters to filter for Union3
        cosmo_params = dict(params)

        if self._can_use_fast_params(cosmo_params):
            theory_mu = self._distance_modulus_from_params(cosmo_params)
        else:
            model = self._cosmology_class(**cosmo_params)
            theory_mu = self._distance_modulus_from_model(model)

        return self._likelihood_jit(theory_mu)

    def chi2(self, model: CosmologyBase) -> float:
        """
        Compute chi-squared for given model.

        Parameters
        ----------
        model : CosmologyBase
            Cosmological model.

        Returns
        -------
        float
            Chi-squared value.
        """
        theory_mu = self._distance_modulus_from_model(model)
        residual = self.mu_obs - jnp.asarray(theory_mu, dtype=self.mu_obs.dtype)
        return float(residual @ (self.cov_inv @ residual))

    def get_normalization_constant(self) -> float:
        """Return log-likelihood normalization constant."""
        return -0.5 * float(self.log_det_cov)

    def get_num_data(self) -> int:
        """Return number of data points (binned redshift bins)."""
        return self.n_bins

    @property
    def nuisance_parameters(self):
        """Union3 has no nuisance parameters (M_B marginalized in UNITY)."""
        return NuisanceList()

    def get_info(self) -> Dict:
        """Get dataset information."""
        return {
            "name": "Union3",
            "n_bins": self.n_bins,
            "n_sne": 2087,
            "redshift_range": (
                float(self.redshifts.min()),
                float(self.redshifts.max()),
            ),
            "standardization": "UNITY1.5",
            "data_type": "compressed_gaussian",
        }

    def __repr__(self):
        info = self.get_info()
        return (
            f"Union3({info['n_sne']} SNe → {info['n_bins']} bins, "
            f"z=[{info['redshift_range'][0]:.3f}, {info['redshift_range'][1]:.3f}])"
        )


def create_union3_likelihood(**kwargs) -> Union3Likelihood:
    """
    Create Union3 likelihood instance.

    Parameters
    ----------
    **kwargs
        Arguments passed to Union3Likelihood.

    Returns
    -------
    Union3Likelihood
        Configured likelihood instance.
    """
    return Union3Likelihood(**kwargs)
