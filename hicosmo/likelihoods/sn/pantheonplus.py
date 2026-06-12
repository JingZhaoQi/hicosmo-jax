"""
PantheonPlus Supernova Likelihood Function
==========================================

Clean implementation of Pantheon+ (2022) 1701 Type Ia supernova likelihood.
Following CLAUDE.md design principles: minimal, efficient, and JAX-optimized.

Data source: Brout et al. 2022, ApJ, 938, 110
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np
from typing import Optional, Dict, Tuple, Union, Any
from pathlib import Path

from ...models.base import CosmologyBase, compute_background_grid_for_model
from ..base import Likelihood, NuisanceList
from ...utils.jax_tools import interp_linspace
from ...utils.logging import get_logger
from ...models import LCDM

logger = get_logger(__name__)


def _get_default_data_path() -> Path:
    """
    Get default data path relative to package location.

    This ensures data can be found regardless of user's working directory.

    Returns
    -------
    Path
        Absolute path to default data directory (hicosmo_package/../data/sne)
    """
    # Get package directory (hicosmo/likelihoods/)
    package_dir = Path(__file__).parent

    # Navigate to project root: hicosmo/likelihoods/ -> hicosmo/ -> project_root/
    project_root = package_dir.parent.parent

    # Default data location: project_root/data/sne/
    return project_root / "data" / "sne"


class PantheonPlusLikelihood(Likelihood):
    """
    PantheonPlus Type Ia supernova likelihood function.

    Clean, minimal implementation following CLAUDE.md principles:
    - Single responsibility: only SNe Ia likelihood
    - Performance optimized: JAX JIT compilation
    - Simple interface: just cosmological models

    Parameters
    ----------
    data_path : str or Path, optional
        Path to PantheonPlus data directory containing official data release files.
        If None, automatically uses '{hicosmo_package}/../data/sne/' where
        Pantheon+SH0ES data is installed. Can be absolute or relative path.
    include_shoes : bool, default False
        If True, use Pantheon+SH0ES combined dataset (SNe + Cepheids).
        If False, use Pantheon+ supernova-only dataset.
    include_systematics : bool, default True
        Include systematic uncertainties in covariance matrix.
    z_min : float, default 0.01
        Minimum redshift cut. SNe with z < z_min are excluded.
    z_max : float, optional
        Maximum redshift cut. If None, no upper limit applied.
    apply_z_cut : bool, default True
        Whether to apply redshift cuts. Set False to use all data.
    marginalize_M_B : bool, default True
        If ``True`` analytically marginalize absolute magnitude ``M_B`` (matching the
        official CosmoSIS likelihood). If ``False`` treat ``M_B`` as an explicit
        parameter to be sampled alongside cosmology.
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        include_shoes: bool = False,
        include_systematics: bool = True,
        z_min: float = 0.01,
        z_max: Optional[float] = None,
        apply_z_cut: bool = True,
        marginalize_M_B: bool = True,
        cosmology_class: type = None,
        verbose: bool = False,
    ):

        # Resolve data path: use default if None, otherwise convert to Path
        if data_path is None:
            self.data_path = str(_get_default_data_path())
        else:
            self.data_path = str(Path(data_path))
        self.include_shoes = include_shoes
        self.include_systematics = include_systematics
        self.z_min = z_min
        self.z_max = z_max
        self.apply_z_cut = apply_z_cut
        self.marginalize_M_B = marginalize_M_B
        self._cosmology_class = cosmology_class if cosmology_class is not None else LCDM
        self.verbose = verbose

        # Load data
        self._load_data()

        # Precompute covariance matrix inverse for efficiency
        self._precompute_covariance()
        self._prepare_distance_grid()

        # Simplified output (user-requested format)
        dataset_name = "PantheonPlus+SH0ES" if self.include_shoes else "PantheonPlus"
        if self.verbose:
            logger.info(f"{dataset_name} loaded: {self.n_sne} objects")
        # Always show z_min if z_cut is applied (default is 0.01)
        if self.apply_z_cut:
            if self.verbose:
                logger.info(f"Cut redshift: z_min = {self.z_min}")
        if self.verbose:
            logger.info(
                f"Redshift range: {self.redshifts.min():.3f} - {self.redshifts.max():.3f}"
            )
            logger.info(
                f"Systematics: {'included' if self.include_systematics else 'excluded'}"
            )
            logger.info(
                f"M_B treatment: {'marginalized' if self.marginalize_M_B else 'free parameter'}"
            )

    def _load_data(self):
        """Load real PantheonPlus data from configured data_path."""
        self._load_real_data()

    def _setup_covariance(self) -> None:
        """Compatibility with Likelihood base class."""
        self._precompute_covariance()

    def get_requirements(self) -> Dict[str, Any]:
        """Return cosmology requirements (handled internally)."""
        return {}

    def theory(self, cosmology: CosmologyBase, **kwargs) -> jnp.ndarray:
        """Return theory distance modulus vector for the current dataset."""
        if self.include_shoes:
            return self._compute_theory_shoes(cosmology)
        return self._compute_theory_pantheon(cosmology)

    def _load_real_data(self):
        """Load real PantheonPlus data following official cosmosis implementation."""
        import pandas as pd

        # Data file paths - support both official release structure and simplified structure
        data_path = Path(self.data_path)

        # Try simplified structure first (data/sne/)
        data_file = data_path / "Pantheon+SH0ES.dat"
        if self.include_systematics:
            cov_file = data_path / "Pantheon+SH0ES_STAT+SYS.cov"
        else:
            cov_file = data_path / "Pantheon+SH0ES_STATONLY.cov"

        # Fall back to official release structure if simplified not found
        if not data_file.exists():
            data_file = (
                data_path
                / "Pantheon+_Data"
                / "4_DISTANCES_AND_COVAR"
                / "Pantheon+SH0ES.dat"
            )
            if self.include_systematics:
                cov_file = (
                    data_path
                    / "Pantheon+_Data"
                    / "4_DISTANCES_AND_COVAR"
                    / "Pantheon+SH0ES_STAT+SYS.cov"
                )
            else:
                cov_file = (
                    data_path
                    / "Pantheon+_Data"
                    / "4_DISTANCES_AND_COVAR"
                    / "Pantheon+SH0ES_STATONLY.cov"
                )

        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}. "
                f"Please download the official PantheonPlus data release."
            )

        if not cov_file.exists():
            raise FileNotFoundError(
                f"Covariance file not found: {cov_file}. "
                f"Please download the official PantheonPlus data release."
            )

        # Load supernova data - following official cosmosis implementation
        data = pd.read_csv(data_file, comment="#", sep=r"\s+")

        # Use official column names exactly as in cosmosis code
        z_hd_raw = data["zHD"].values  # Hubble flow redshift (pec-velocity corrected)
        z_cmb_raw = data["zCMB"].values  # Provided for completeness
        z_hel_raw = data["zHEL"].values  # Heliocentric redshift
        m_b_corr_raw = data["m_b_corr"].values  # Corrected apparent magnitude
        is_calibrator_raw = data["IS_CALIBRATOR"].values.astype(bool)
        ceph_dist_raw = data["CEPH_DIST"].values

        # Official data selection mask
        if self.include_shoes:
            # SH0ES mode: z_HD > 0.01 OR is_calibrator
            self.ww = (z_hd_raw > 0.01) | is_calibrator_raw
        else:
            # Pantheon+ mode: z_HD > 0.01 AND NOT is_calibrator
            self.ww = (z_hd_raw > 0.01) & (~is_calibrator_raw)

        # Apply additional redshift cuts if requested
        if self.apply_z_cut and (self.z_min != 0.01 or self.z_max is not None):
            z_mask = z_hd_raw > self.z_min
            if self.z_max is not None:
                z_mask = z_mask & (z_hd_raw <= self.z_max)
            self.ww = self.ww & z_mask

        # Extract final datasets
        self.z_cmb = jnp.array(z_hd_raw[self.ww])
        self.z_hel = jnp.array(z_hel_raw[self.ww])
        self.z_hd = jnp.array(z_hd_raw[self.ww])
        self.m_obs = jnp.array(m_b_corr_raw[self.ww])
        self.is_calibrator = jnp.array(is_calibrator_raw[self.ww])
        self.ceph_dist = jnp.array(ceph_dist_raw[self.ww])

        # Store redshifts for backward compatibility
        self.redshifts = self.z_hd
        self._original_indices = np.where(self.ww)[0]

        # Load covariance matrix
        try:
            ff = np.loadtxt(cov_file)
            n_dim = len(data)
            cov_full = ff[1:].reshape(n_dim, n_dim)
            self.covariance = jnp.array(cov_full[self.ww, :][:, self.ww])
        except Exception as e:
            if self.verbose:
                logger.warning(f"Could not load covariance matrix: {e}")
            n_data = len(self.m_obs)
            typical_error = 0.1
            self.covariance = jnp.eye(n_data) * typical_error**2

        self.n_sne = len(self.redshifts)

        # Validate data consistency
        assert len(self.redshifts) == len(
            self.m_obs
        ), "Redshift and magnitude arrays must have same length"
        assert self.covariance.shape == (
            self.n_sne,
            self.n_sne,
        ), "Covariance matrix dimensions inconsistent"

    def _precompute_covariance(self):
        """Precompute covariance matrix inverse and determinant."""
        # Numerical stability
        regularization = 1e-8 * jnp.eye(self.n_sne)
        stable_cov = self.covariance + regularization

        self.cov_inv = jnp.linalg.inv(stable_cov)
        self.log_det_cov = jnp.linalg.slogdet(stable_cov)[1]

        if self.verbose:
            condition_number = jnp.linalg.cond(stable_cov)
            logger.info(
                f"Covariance: {self.n_sne}×{self.n_sne}, condition number: {condition_number:.1e}"
            )

        self._initialize_jit_likelihood()

    def log_likelihood(
        self, model: CosmologyBase, M_B: Optional[float] = None, **kwargs
    ) -> float:
        """
        Compute log-likelihood following official cosmosis implementation.

        Parameters
        ----------
        model : CosmologyBase
            Cosmological model instance.
        M_B : float, optional
            Absolute magnitude parameter. Required if marginalize_M_B=False.
            Ignored if marginalize_M_B=True (default).
        **kwargs : dict
            Additional parameters (for future extensions).

        Returns
        -------
        float
            Log-likelihood value

        Raises
        ------
        ValueError
            If model is not provided.
        """
        if model is None:
            raise ValueError("model argument must be provided")

        # Compute theory vector
        if self.include_shoes:
            theory_mu = self._compute_theory_shoes(model)
        else:
            theory_mu = self._compute_theory_pantheon(model)

        # Choose computation path based on marginalization setting
        if self.marginalize_M_B:
            # Analytically marginalize over M_B
            return self._marginalized_likelihood(theory_mu)
        else:
            # Treat M_B as explicit parameter
            if M_B is None:
                raise ValueError("M_B must be provided when marginalize_M_B=False")
            return self._fixed_likelihood(theory_mu, M_B)

    def get_normalization_constant(self) -> float:
        """
        Return log-likelihood normalization constant for information criteria.

        The Gaussian log-likelihood has the form:
            log(L) = -χ²/2 + normalization_constant

        For AIC/BIC we need χ² = -2 * (log_likelihood - normalization_constant)

        Returns
        -------
        float
            Normalization constant including:
            - -0.5 * log|C| (covariance determinant)
            - +0.5 * log(1ᵀC⁻¹1) if M_B is marginalized
        """
        norm = -0.5 * float(self.log_det_cov)
        if self.marginalize_M_B and hasattr(self, "_mb_aux"):
            _, _, _, mb_log_den_const = self._mb_aux
            norm += 0.5 * float(mb_log_den_const)
        return norm

    def get_num_data(self) -> int:
        """Return number of SNe data points."""
        return self.n_sne

    def __call__(self, **params) -> float:
        """
        Callable interface for MCMC sampling.

        Automatically constructs cosmology model from parameters and computes
        log-likelihood. This enables the simplified API:

            sne = PantheonPlusLikelihood(cosmology_class=LCDM)
            MCMC(params, sne, chain_name='lcdm_sn').run(...)

        Parameters
        ----------
        **params : dict
            Cosmological parameters (H0, Omega_m, etc.) passed to cosmology_class.
            When marginalize_M_B=False, M_B should be included in params.

        Returns
        -------
        float
            Log-likelihood value.

        Examples
        --------
        >>> # With M_B marginalized (default)
        >>> sne = PantheonPlusLikelihood(cosmology_class=LCDM)
        >>> log_L = sne(H0=70, Omega_m=0.3)

        >>> # With M_B as free parameter
        >>> sne = PantheonPlusLikelihood(cosmology_class=LCDM, marginalize_M_B=False)
        >>> log_L = sne(H0=70, Omega_m=0.3, M_B=-19.3)
        """
        self._warn_unknown_params(params)
        # Extract M_B if present and marginalize_M_B=False
        # Use .get() instead of .pop() to avoid modifying original dict
        M_B = None
        if not self.marginalize_M_B:
            M_B = params.get("M_B", None)
            if M_B is None:
                raise ValueError(
                    "M_B must be provided when marginalize_M_B=False. "
                    "Either include M_B in params or set marginalize_M_B=True."
                )

        # Filter out M_B from cosmology params (don't pass to cosmology model)
        cosmo_params = {k: v for k, v in params.items() if k != "M_B"}

        # Fast path: avoid constructing model when direct parameters are available
        if self._can_use_fast_params(cosmo_params):
            mu_hubble = self._distance_modulus_from_params(cosmo_params)
            if self.include_shoes:
                theory_mu = jnp.where(self.is_calibrator, self.ceph_dist, mu_hubble)
            else:
                theory_mu = mu_hubble
        else:
            # Fallback: construct cosmology model (handles derived parameters)
            model = self._cosmology_class(**cosmo_params)
            if self.include_shoes:
                theory_mu = self._compute_theory_shoes(model)
            else:
                theory_mu = self._compute_theory_pantheon(model)

        # Choose computation path based on marginalization setting
        if self.marginalize_M_B:
            return self._marginalized_likelihood(theory_mu)
        else:
            return self._fixed_likelihood(theory_mu, M_B)

    def _can_use_fast_params(self, params: Dict[str, Any]) -> bool:
        """Check whether we can bypass model construction and use traced params directly."""
        if not hasattr(self._cosmology_class, "compute_grid_traced"):
            return False
        # Require explicit H0 to avoid missing derived conversions (h -> H0)
        if "H0" not in params:
            return False
        # If alternative bases are provided without H0, fall back to model
        if "h" in params and "H0" not in params:
            return False
        if (
            any(k in params for k in ("omega_b_h2", "omega_c_h2", "omega_m_h2"))
            and "H0" not in params
        ):
            return False
        return True

    def _distance_modulus_from_params(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Compute distance modulus directly from parameter dict using traced grids.

        Note: Must accept JAX Tracer objects during MCMC sampling, not just
        concrete types. Use duck typing to detect numeric/array-like values.
        """
        dtype = self.m_obs.dtype
        cosmo_params_raw = self._cosmology_class.normalize_params(params, dtype=dtype)
        cosmo_params = {}
        for k, v in cosmo_params_raw.items():
            # Accept: int, float, jnp.ndarray, np.ndarray, and JAX Tracer objects
            if isinstance(v, (int, float)):
                cosmo_params[k] = jnp.asarray(v, dtype=dtype)
            elif isinstance(v, jax.core.Tracer):
                # JAX Tracer during MCMC - keep as-is, already traced
                cosmo_params[k] = v
            elif hasattr(v, "shape"):
                # Array-like (jnp.ndarray, np.ndarray)
                cosmo_params[k] = jnp.asarray(v, dtype=dtype)
        cosmo_grid = compute_background_grid_for_model(
            self._cosmology_class, self._z_grid, cosmo_params
        )
        d_L_grid = cosmo_grid["d_L"]
        d_L = interp_linspace(self.z_cmb, self._z_grid, d_L_grid)
        d_A = d_L / (1.0 + self.z_cmb) ** 2
        return 5.0 * jnp.log10((1.0 + self.z_cmb) * (1.0 + self.z_hel) * d_A) + 25.0

    def _initialize_jit_likelihood(self) -> None:
        """Precompile both fixed and marginalized likelihood functions."""
        dtype = self.covariance.dtype
        self.m_obs = jnp.asarray(self.m_obs, dtype=dtype)
        self.cov_inv = jnp.asarray(self.cov_inv, dtype=dtype)
        self.log_det_cov = jnp.asarray(self.log_det_cov, dtype=dtype)
        self.ceph_dist = jnp.asarray(self.ceph_dist, dtype=dtype)
        self.is_calibrator = jnp.asarray(self.is_calibrator, dtype=bool)
        ones_vec = jnp.ones((self.n_sne,), dtype=dtype)
        cov_inv_ones = self.cov_inv @ ones_vec
        mb_denominator = ones_vec @ cov_inv_ones
        self._mb_aux = (ones_vec, cov_inv_ones, mb_denominator, jnp.log(mb_denominator))

        # Capture constants in closure for JIT compilation
        m_obs = self.m_obs
        cov_inv = self.cov_inv
        log_det_cov = self.log_det_cov
        n_sne = self.n_sne
        ones_vec_const, cov_inv_ones_const, mb_denominator_const, mb_log_den_const = (
            self._mb_aux
        )

        # JIT compile fixed M_B likelihood
        # Returns -χ²/2 only (no normalization) - normalization via get_normalization_constant()
        def _fixed_likelihood_impl(
            mu_theory: jnp.ndarray, M_B: jnp.ndarray
        ) -> jnp.ndarray:
            residual = m_obs - (mu_theory + M_B)
            chi2 = residual @ (cov_inv @ residual)
            return -0.5 * chi2

        self._fixed_likelihood_jit = jit(_fixed_likelihood_impl)

        # JIT compile marginalized M_B likelihood
        # Returns -χ²/2 only (no normalization) - normalization via get_normalization_constant()
        def _marginalized_likelihood_impl(mu_theory: jnp.ndarray) -> jnp.ndarray:
            residual = m_obs - mu_theory
            numerator = residual @ cov_inv_ones_const
            M_B_best = numerator / mb_denominator_const
            marginalized_residual = residual - M_B_best * ones_vec_const
            chi2 = marginalized_residual @ (cov_inv @ marginalized_residual)
            return -0.5 * chi2

        self._marginalized_likelihood_jit = jit(_marginalized_likelihood_impl)

        # Build grid-accepting version (for CombinedLikelihood shared grid)
        z_cmb = self.z_cmb
        z_hel = self.z_hel
        include_shoes = self.include_shoes
        is_calibrator = self.is_calibrator
        ceph_dist = self.ceph_dist
        marginalize = self.marginalize_M_B

        @jit
        def _loglike_from_grid_impl(
            cosmo_grid: Dict, grid_z: jnp.ndarray, params: Dict
        ) -> jnp.ndarray:
            d_L = interp_linspace(z_cmb, grid_z, cosmo_grid["d_L"])
            d_A = d_L / (1.0 + z_cmb) ** 2
            mu = 5.0 * jnp.log10((1.0 + z_cmb) * (1.0 + z_hel) * d_A) + 25.0
            if include_shoes:
                mu = jnp.where(is_calibrator, ceph_dist, mu)
            if marginalize:
                return _marginalized_likelihood_impl(mu)
            else:
                M_B = params.get("M_B", jnp.asarray(-19.3, dtype=mu.dtype))
                return _fixed_likelihood_impl(mu, M_B)

        self._loglike_from_grid = _loglike_from_grid_impl

    def _prepare_distance_grid(self) -> None:
        """Precompute redshift grid for JAX-based cosmology interpolation."""
        max_z = float(self.z_cmb.max()) + 0.5
        # Target ~1e-3 spacing, clamp between 2048 and 8192 points
        n_grid = int(jnp.clip(max_z / 1e-3, 2048, 8192))
        self._z_grid = jnp.linspace(0.0, max_z, n_grid, dtype=self.m_obs.dtype)

    def _extract_cosmo_params(self, model: CosmologyBase) -> Dict[str, jnp.ndarray]:
        """Convert model.params to JAX-compatible dict (no filtering, just type conversion)."""
        params_dict = model.params.to_dict()
        dtype = self.m_obs.dtype
        return {
            k: jnp.asarray(v, dtype=dtype)
            for k, v in params_dict.items()
            if isinstance(v, (int, float, jnp.ndarray, np.ndarray))
        }

    def _distance_modulus_from_model(self, model: CosmologyBase) -> jnp.ndarray:
        """Compute distance modulus vector using pure JAX traced cosmology."""
        cosmo_params = self._extract_cosmo_params(model)
        cosmo_grid = compute_background_grid_for_model(
            self._cosmology_class, self._z_grid, cosmo_params
        )
        d_L_grid = cosmo_grid["d_L"]
        d_L = interp_linspace(self.z_cmb, self._z_grid, d_L_grid)
        d_A = d_L / (1.0 + self.z_cmb) ** 2
        return 5.0 * jnp.log10((1.0 + self.z_cmb) * (1.0 + self.z_hel) * d_A) + 25.0

    def _fixed_likelihood(self, mu_theory: jnp.ndarray, M_B: float) -> float:
        """Wrapper for JIT-compiled fixed M_B likelihood."""
        mu_theory = jnp.asarray(mu_theory, dtype=self.m_obs.dtype)
        M_B_array = jnp.asarray(M_B, dtype=self.m_obs.dtype)
        return self._fixed_likelihood_jit(mu_theory, M_B_array)

    def _marginalized_likelihood(self, mu_theory: jnp.ndarray) -> float:
        """Wrapper for JIT-compiled marginalized likelihood."""
        mu_theory = jnp.asarray(mu_theory, dtype=self.m_obs.dtype)
        return self._marginalized_likelihood_jit(mu_theory)

    def chi2(self, model: CosmologyBase, M_B: float) -> float:
        """
        Compute χ² for given model and M_B.

        Parameters
        ----------
        model : CosmologyBase
            Cosmological model
        M_B : float
            Absolute magnitude parameter

        Returns
        -------
        float
            χ² value
        """
        # Get theoretical distance moduli
        if self.include_shoes:
            theory_mu = self._compute_theory_shoes(model)
        else:
            theory_mu = self._compute_theory_pantheon(model)

        theory_mu = jnp.asarray(theory_mu, dtype=self.m_obs.dtype)
        residual = self.m_obs - (theory_mu + jnp.asarray(M_B, dtype=self.m_obs.dtype))
        chi2 = residual @ (self.cov_inv @ residual)
        return float(chi2)

    def best_fit_M_B(self, model: CosmologyBase) -> float:
        """
        Find optimal absolute magnitude for given model.

        Parameters
        ----------
        model : CosmologyBase
            Cosmological model

        Returns
        -------
        float
            Optimal absolute magnitude M_B
        """
        # Get theoretical distance moduli
        if self.include_shoes:
            theory_mu = self._compute_theory_shoes(model)
        else:
            theory_mu = self._compute_theory_pantheon(model)

        residual = self.m_obs - jnp.asarray(theory_mu, dtype=self.m_obs.dtype)
        ones_vec, cov_inv_ones, denominator, _ = self._mb_aux
        numerator = residual @ cov_inv_ones
        return float(numerator / denominator)

    def get_info(self) -> Dict:
        """Get dataset information."""
        return {
            "name": "PantheonPlus",
            "n_supernovae": self.n_sne,
            "redshift_range": (
                float(self.redshifts.min()),
                float(self.redshifts.max()),
            ),
            "include_systematics": self.include_systematics,
            "data_type": "PantheonPlusSH0ES" if self.include_shoes else "PantheonPlus",
        }

    def _compute_theory_pantheon(self, model: CosmologyBase) -> jnp.ndarray:
        """Theoretical distance moduli for Pantheon+ (CMB-frame SNe only)."""
        return self._distance_modulus_from_model(model)

    def _compute_theory_shoes(self, model: CosmologyBase) -> jnp.ndarray:
        """
        Compute theoretical distance moduli for SH0ES mode.
        Following official cosmosis implementation lines 102-109.

        Calibrators: use fixed Cepheid distances
        Hubble flow SNe: use cosmology with redshift corrections
        """
        mu_hubble = self._distance_modulus_from_model(model)
        calibrator_mask = self.is_calibrator
        return jnp.where(calibrator_mask, self.ceph_dist, mu_hubble)

    @property
    def nuisance_parameters(self):
        """
        Return nuisance parameters for automatic registry integration.

        Returns
        -------
        List[Parameter]
            List of nuisance parameters. Returns M_B if marginalize_M_B=False,
            empty list if marginalize_M_B=True (analytically marginalized).

        Examples
        --------
        >>> # Case 1: Sample M_B explicitly
        >>> pantheon = PantheonPlusLikelihood(data_path='...', marginalize_M_B=False)
        >>> registry.add_from_likelihood(pantheon)  # Auto-adds M_B

        >>> # Case 2: Marginalize M_B (default, more efficient)
        >>> pantheon = PantheonPlusLikelihood(data_path='...', marginalize_M_B=True)
        >>> registry.add_from_likelihood(pantheon)  # No parameters added
        """
        if self.marginalize_M_B:
            # M_B analytically marginalized - no nuisance parameter needed
            return NuisanceList()
        else:
            # M_B should be sampled - return as Parameter
            from ...parameters import Parameter

            return NuisanceList(
                [
                    Parameter(
                        name="M_B",
                        value=-19.23,  # Typical value for SNe Ia
                        free=True,
                        prior={"dist": "uniform", "min": -20.0, "max": -18.0},
                        latex_label=r"$\mathcal{M}_B$",
                        description="SNe Ia absolute magnitude in B-band",
                    )
                ]
            )

    # __add__ and __radd__ inherited from Likelihood base class

    def __repr__(self):
        """String representation."""
        info = self.get_info()
        return (
            f"PantheonPlus({info['n_supernovae']} SNe, "
            f"z=[{info['redshift_range'][0]:.3f}, {info['redshift_range'][1]:.3f}], "
            f"systematics={info['include_systematics']})"
        )


# Convenience function
def create_pantheonplus_likelihood(**kwargs) -> PantheonPlusLikelihood:
    """
    Create PantheonPlus likelihood instance.

    Parameters
    ----------
    **kwargs
        Arguments passed to PantheonPlusLikelihood

    Returns
    -------
    PantheonPlusLikelihood
        Configured likelihood instance
    """
    return PantheonPlusLikelihood(**kwargs)
