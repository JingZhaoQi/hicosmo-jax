"""
DES Year 5 Supernova Likelihood Function
=========================================

Implementation of the DES-SN5YR (2024) 1820 Type Ia supernova likelihood.
Following the official Dovekie cosmosis likelihood with analytic M marginalization.

Data source: DES Collaboration, 2024, ApJL, 973, L14 (arXiv: 2401.02929)
GitHub: https://github.com/des-science/DES-SN5YR
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
    """
    Get default data path relative to package location.

    Returns
    -------
    Path
        Absolute path to default DESY5 data directory (hicosmo/data/sne/desy5)
    """
    package_dir = Path(__file__).parent
    project_root = package_dir.parent.parent
    return project_root / "data" / "sne" / "desy5"


class DESY5Likelihood(Likelihood):
    """
    DES Year 5 Type Ia supernova likelihood function.

    Implements the DES-SN5YR Dovekie likelihood with analytic M marginalization,
    following the official cosmosis implementation (arXiv: 2401.02929).

    The DES-SN5YR sample contains 1829 photometrically classified SNe Ia
    (1635 DES + 194 external low-z), with 1820 passing quality cuts.
    Distance moduli are provided directly from the BBC+SALT3 pipeline.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to DESY5 data directory containing DES-Dovekie_HD.csv and
        covariance files. If None, uses the bundled data.
    include_systematics : bool, default True
        If True, use STAT+SYS covariance. If False, use STATONLY.
    z_min : float, default 0.025
        Minimum redshift cut. SNe with z < z_min are excluded.
    z_max : float, optional
        Maximum redshift cut. If None, no upper limit applied.
    apply_z_cut : bool, default True
        Whether to apply redshift cuts.
    marginalize_M : bool, default True
        If True, analytically marginalize the absolute magnitude offset M
        following Goliath et al. (2001), arXiv:astro-ph/0104009, Eq. A9-12.
        If False, treat M as a free parameter to be sampled.
    cosmology_class : type, optional
        Cosmological model class. Defaults to LCDM.
    verbose : bool, default False
        If True, print diagnostic information during initialization.

    Notes
    -----
    Key differences from Pantheon+:
    - Data provides distance moduli (MU) directly, not apparent magnitudes.
    - Covariance is stored as inverse covariance in upper-triangular NPZ format.
    - No SH0ES calibration mode.
    - Official constraint: flat LCDM Omega_m = 0.352 +/- 0.017.
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        include_systematics: bool = True,
        z_min: float = 0.025,
        z_max: Optional[float] = None,
        apply_z_cut: bool = True,
        marginalize_M: bool = True,
        cosmology_class: type = None,
        verbose: bool = False,
    ):
        if data_path is None:
            self.data_path = str(_get_default_data_path())
        else:
            self.data_path = str(Path(data_path))
        self.include_systematics = include_systematics
        self.z_min = z_min
        self.z_max = z_max
        self.apply_z_cut = apply_z_cut
        self.marginalize_M = marginalize_M
        self._cosmology_class = cosmology_class if cosmology_class is not None else LCDM
        self.verbose = verbose

        self._load_data()
        self._precompute_covariance()
        self._prepare_distance_grid()

        if self.verbose:
            logger.info(f"DES-SN5YR loaded: {self.n_sne} supernovae")
            if self.apply_z_cut:
                logger.info(f"Cut redshift: z_min = {self.z_min}")
            logger.info(
                f"Redshift range: {self.redshifts.min():.4f} - {self.redshifts.max():.4f}"
            )
            logger.info(
                f"Systematics: {'included' if self.include_systematics else 'excluded'}"
            )
            logger.info(
                f"M treatment: {'marginalized' if self.marginalize_M else 'free parameter'}"
            )

    def _load_data(self):
        """Load DES-SN5YR data in SNANA format."""
        data_path = Path(self.data_path)
        data_file = data_path / "DES-Dovekie_HD.csv"

        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}. "
                f"Please download from https://github.com/des-science/DES-SN5YR"
            )

        # Parse SNANA-style format: header lines start with #, column names
        # follow VARNAMES:, data lines start with SN:
        cid_list = []
        idsurvey_list = []
        z_hd_list = []
        z_hel_list = []
        mu_list = []
        mu_err_list = []

        with open(data_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("VARNAMES:"):
                    continue
                if line.startswith("SN:"):
                    parts = line.split()
                    # SN: CID IDSURVEY zHD zHEL MU MUERR ...
                    cid_list.append(parts[1])
                    idsurvey_list.append(int(float(parts[2])))
                    z_hd_list.append(float(parts[3]))
                    z_hel_list.append(float(parts[4]))
                    mu_list.append(float(parts[5]))
                    mu_err_list.append(float(parts[6]))

        z_hd_raw = np.array(z_hd_list)
        z_hel_raw = np.array(z_hel_list)
        mu_raw = np.array(mu_list)
        mu_err_raw = np.array(mu_err_list)
        n_total = len(z_hd_raw)

        # Official cut: zHD > 0 (all pass in current release)
        self.ww = z_hd_raw > 0.0

        # Additional redshift cuts
        if self.apply_z_cut:
            if self.z_min is not None:
                self.ww = self.ww & (z_hd_raw >= self.z_min)
            if self.z_max is not None:
                self.ww = self.ww & (z_hd_raw <= self.z_max)

        # Extract selected data as JAX arrays
        self.z_cmb = jnp.array(z_hd_raw[self.ww])
        self.z_hel = jnp.array(z_hel_raw[self.ww])
        self.z_hd = self.z_cmb
        self.mu_obs = jnp.array(mu_raw[self.ww])
        self.mu_err = jnp.array(mu_err_raw[self.ww])
        self.redshifts = self.z_hd
        self._original_indices = np.where(self.ww)[0]
        self.n_sne = len(self.redshifts)

        # Load inverse covariance from NPZ (upper triangular format)
        self._load_inverse_covariance(data_path, n_total)

        # Validate
        assert len(self.redshifts) == len(
            self.mu_obs
        ), "Redshift and distance modulus arrays must have same length"
        assert self.cov_inv.shape == (
            self.n_sne,
            self.n_sne,
        ), "Inverse covariance matrix dimensions inconsistent"

    def _load_inverse_covariance(self, data_path: Path, n_total: int):
        """
        Load inverse covariance from NPZ file.

        The DES-SN5YR release stores the inverse covariance matrix in compressed
        upper-triangular format. This avoids the costly double-inversion
        (inv -> cov -> inv) that the cosmosis implementation performs.
        """
        if self.include_systematics:
            cov_file = data_path / "STAT+SYS.npz"
        else:
            cov_file = data_path / "STATONLY.npz"

        if not cov_file.exists():
            raise FileNotFoundError(
                f"Covariance file not found: {cov_file}. "
                f"Please download from https://github.com/des-science/DES-SN5YR"
            )

        d = np.load(cov_file)
        n = d[d.files[0]][0]
        if n != n_total:
            raise ValueError(
                f"Covariance dimension ({n}) does not match data size ({n_total})"
            )

        # Reconstruct full symmetric inverse covariance from upper triangle
        inv_cov_full = np.zeros((n, n), dtype=np.float64)
        inv_cov_full[np.triu_indices(n)] = d[d.files[1]].astype(np.float64)
        i_lower = np.tril_indices(n, -1)
        inv_cov_full[i_lower] = inv_cov_full.T[i_lower]

        # Apply selection mask
        inv_cov_selected = inv_cov_full[self.ww][:, self.ww]

        # Store as JAX array
        self.cov_inv = jnp.array(inv_cov_selected)

        # Compute regular covariance for normalization and diagnostics
        regularization = 1e-8 * jnp.eye(self.n_sne)
        stable_inv = self.cov_inv + regularization
        self.covariance = jnp.linalg.inv(stable_inv)

    def _setup_covariance(self) -> None:
        """Compatibility with Likelihood base class."""
        pass

    def _precompute_covariance(self):
        """Precompute auxiliary quantities for likelihood evaluation."""
        self.log_det_cov = -jnp.linalg.slogdet(self.cov_inv)[1]

        if self.verbose:
            condition_number = jnp.linalg.cond(self.cov_inv)
            logger.info(
                f"Inv. covariance: {self.n_sne}x{self.n_sne}, "
                f"condition number: {condition_number:.1e}"
            )

        self._initialize_jit_likelihood()

    def _initialize_jit_likelihood(self) -> None:
        """Precompile JIT-compiled likelihood functions."""
        dtype = self.cov_inv.dtype
        self.mu_obs = jnp.asarray(self.mu_obs, dtype=dtype)
        self.cov_inv = jnp.asarray(self.cov_inv, dtype=dtype)
        self.log_det_cov = jnp.asarray(self.log_det_cov, dtype=dtype)

        # Precompute M-marginalization auxiliaries
        ones_vec = jnp.ones((self.n_sne,), dtype=dtype)
        cov_inv_ones = self.cov_inv @ ones_vec
        m_denominator = ones_vec @ cov_inv_ones
        self._m_aux = (ones_vec, cov_inv_ones, m_denominator, jnp.log(m_denominator))

        # Capture constants in closures for JIT
        mu_obs = self.mu_obs
        cov_inv = self.cov_inv
        log_det_cov = self.log_det_cov
        ones_vec_const, cov_inv_ones_const, m_denominator_const, m_log_den_const = (
            self._m_aux
        )

        # Fixed M likelihood: -chi2/2
        def _fixed_likelihood_impl(
            mu_theory: jnp.ndarray, M: jnp.ndarray
        ) -> jnp.ndarray:
            residual = mu_obs - (mu_theory + M)
            chi2 = residual @ (cov_inv @ residual)
            return -0.5 * chi2

        self._fixed_likelihood_jit = jit(_fixed_likelihood_impl)

        # Marginalized M likelihood following Goliath et al. (2001)
        # arXiv:astro-ph/0104009, Eq. A9-12
        # chi2 = delta @ C^-1 @ delta - (delta @ C^-1 @ 1)^2 / (1 @ C^-1 @ 1)
        #         + log(1 @ C^-1 @ 1 / (2*pi))
        def _marginalized_likelihood_impl(mu_theory: jnp.ndarray) -> jnp.ndarray:
            delta = mu_theory - mu_obs
            chit2 = delta @ (cov_inv @ delta)
            B = delta @ cov_inv_ones_const
            C = m_denominator_const
            chi2 = chit2 - (B**2 / C) + jnp.log(C / (2.0 * jnp.pi))
            return -0.5 * chi2

        self._marginalized_likelihood_jit = jit(_marginalized_likelihood_impl)

        # Grid-accepting version for CombinedLikelihood shared grid
        z_cmb = self.z_cmb
        z_hel = self.z_hel
        marginalize = self.marginalize_M

        @jit
        def _loglike_from_grid_impl(
            cosmo_grid: Dict, grid_z: jnp.ndarray, params: Dict
        ) -> jnp.ndarray:
            d_L = interp_linspace(z_cmb, grid_z, cosmo_grid["d_L"])
            d_A = d_L / (1.0 + z_cmb) ** 2
            mu = 5.0 * jnp.log10((1.0 + z_cmb) * (1.0 + z_hel) * d_A) + 25.0
            if marginalize:
                return _marginalized_likelihood_impl(mu)
            else:
                M = params.get("M", jnp.asarray(-19.3, dtype=mu.dtype))
                return _fixed_likelihood_impl(mu, M)

        self._loglike_from_grid = _loglike_from_grid_impl

    def _prepare_distance_grid(self) -> None:
        """Precompute redshift grid for JAX-based cosmology interpolation."""
        max_z = float(self.z_cmb.max()) + 0.5
        n_grid = int(jnp.clip(max_z / 1e-3, 2048, 8192))
        self._z_grid = jnp.linspace(0.0, max_z, n_grid, dtype=self.mu_obs.dtype)

    def _extract_cosmo_params(self, model: CosmologyBase) -> Dict[str, jnp.ndarray]:
        """Convert model.params to JAX-compatible dict."""
        params_dict = model.params.to_dict()
        dtype = self.mu_obs.dtype
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
        d_L = interp_linspace(self.z_cmb, self._z_grid, d_L_grid)
        d_A = d_L / (1.0 + self.z_cmb) ** 2
        return 5.0 * jnp.log10((1.0 + self.z_cmb) * (1.0 + self.z_hel) * d_A) + 25.0

    def _can_use_fast_params(self, params: Dict[str, Any]) -> bool:
        """Check whether we can bypass model construction and use traced params directly."""
        if not hasattr(self._cosmology_class, "compute_grid_traced"):
            return False
        if "H0" not in params:
            return False
        if "h" in params and "H0" not in params:
            return False
        if (
            any(k in params for k in ("omega_b_h2", "omega_c_h2", "omega_m_h2"))
            and "H0" not in params
        ):
            return False
        return True

    def get_requirements(self) -> Dict[str, Any]:
        """Return cosmology requirements."""
        return {}

    def theory(self, cosmology: CosmologyBase, **kwargs) -> jnp.ndarray:
        """Return theory distance modulus vector for the current dataset."""
        return self._distance_modulus_from_model(cosmology)

    def log_likelihood(
        self, model: CosmologyBase, M: Optional[float] = None, **kwargs
    ) -> float:
        """
        Compute log-likelihood following official DES-SN5YR implementation.

        Parameters
        ----------
        model : CosmologyBase
            Cosmological model instance.
        M : float, optional
            Absolute magnitude offset. Required if marginalize_M=False.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        float
            Log-likelihood value.
        """
        if model is None:
            raise ValueError("model argument must be provided")

        theory_mu = self._distance_modulus_from_model(model)

        if self.marginalize_M:
            return self._marginalized_likelihood(theory_mu)
        else:
            if M is None:
                raise ValueError("M must be provided when marginalize_M=False")
            return self._fixed_likelihood(theory_mu, M)

    def __call__(self, **params) -> float:
        """
        Callable interface for MCMC sampling.

        Parameters
        ----------
        **params : dict
            Cosmological parameters (H0, Omega_m, etc.).
            When marginalize_M=False, M should be included in params.

        Returns
        -------
        float
            Log-likelihood value.
        """
        self._warn_unknown_params(params)
        M = None
        if not self.marginalize_M:
            M = params.get("M", None)
            if M is None:
                raise ValueError(
                    "M must be provided when marginalize_M=False. "
                    "Either include M in params or set marginalize_M=True."
                )

        # Filter out M from cosmology params
        cosmo_params = {k: v for k, v in params.items() if k != "M"}

        if self._can_use_fast_params(cosmo_params):
            theory_mu = self._distance_modulus_from_params(cosmo_params)
        else:
            model = self._cosmology_class(**cosmo_params)
            theory_mu = self._distance_modulus_from_model(model)

        if self.marginalize_M:
            return self._marginalized_likelihood(theory_mu)
        else:
            return self._fixed_likelihood(theory_mu, M)

    def _fixed_likelihood(self, mu_theory: jnp.ndarray, M: float) -> float:
        """Wrapper for JIT-compiled fixed M likelihood."""
        mu_theory = jnp.asarray(mu_theory, dtype=self.mu_obs.dtype)
        M_array = jnp.asarray(M, dtype=self.mu_obs.dtype)
        return self._fixed_likelihood_jit(mu_theory, M_array)

    def _marginalized_likelihood(self, mu_theory: jnp.ndarray) -> float:
        """Wrapper for JIT-compiled marginalized likelihood."""
        mu_theory = jnp.asarray(mu_theory, dtype=self.mu_obs.dtype)
        return self._marginalized_likelihood_jit(mu_theory)

    def chi2(self, model: CosmologyBase, M: float) -> float:
        """
        Compute chi-squared for given model and M.

        Parameters
        ----------
        model : CosmologyBase
            Cosmological model
        M : float
            Absolute magnitude offset

        Returns
        -------
        float
            chi-squared value
        """
        theory_mu = self._distance_modulus_from_model(model)
        theory_mu = jnp.asarray(theory_mu, dtype=self.mu_obs.dtype)
        residual = self.mu_obs - (theory_mu + jnp.asarray(M, dtype=self.mu_obs.dtype))
        chi2 = residual @ (self.cov_inv @ residual)
        return float(chi2)

    def best_fit_M(self, model: CosmologyBase) -> float:
        """
        Find optimal absolute magnitude offset for given model.

        Parameters
        ----------
        model : CosmologyBase
            Cosmological model

        Returns
        -------
        float
            Optimal magnitude offset M
        """
        theory_mu = self._distance_modulus_from_model(model)
        # M_best minimizes chi2: residual = mu_obs - mu_theory - M
        # d(chi2)/dM = 0 gives M = (1^T C^-1 residual) / (1^T C^-1 1)
        # where residual = mu_obs - mu_theory
        residual = self.mu_obs - jnp.asarray(theory_mu, dtype=self.mu_obs.dtype)
        ones_vec, cov_inv_ones, denominator, _ = self._m_aux
        numerator = residual @ cov_inv_ones
        return float(numerator / denominator)

    def get_normalization_constant(self) -> float:
        """
        Return log-likelihood normalization constant.

        Returns
        -------
        float
            Normalization constant: -0.5 * log|C| + 0.5 * log(1^T C^-1 1)
            for the marginalized case.
        """
        norm = -0.5 * float(self.log_det_cov)
        if self.marginalize_M and hasattr(self, "_m_aux"):
            _, _, _, m_log_den_const = self._m_aux
            norm += 0.5 * float(m_log_den_const)
        return norm

    def get_num_data(self) -> int:
        """Return number of SNe data points."""
        return self.n_sne

    def get_info(self) -> Dict:
        """Get dataset information."""
        return {
            "name": "DES-SN5YR",
            "n_supernovae": self.n_sne,
            "redshift_range": (
                float(self.redshifts.min()),
                float(self.redshifts.max()),
            ),
            "include_systematics": self.include_systematics,
            "data_type": "DES-SN5YR",
        }

    @property
    def nuisance_parameters(self):
        """
        Return nuisance parameters for automatic registry integration.

        When marginalize_M=True (default), M is analytically marginalized
        and no nuisance parameter is needed. When marginalize_M=False,
        M is returned as a free parameter to be sampled.
        """
        if self.marginalize_M:
            return NuisanceList()
        else:
            from ...parameters import Parameter

            return NuisanceList(
                [
                    Parameter(
                        name="M",
                        value=-19.23,
                        free=True,
                        prior={"dist": "uniform", "min": -20.0, "max": -18.0},
                        latex_label=r"$\mathcal{M}$",
                        description="SNe Ia absolute magnitude offset (DES-SN5YR)",
                    )
                ]
            )

    def __repr__(self):
        """String representation."""
        info = self.get_info()
        return (
            f"DESY5({info['n_supernovae']} SNe, "
            f"z=[{info['redshift_range'][0]:.4f}, {info['redshift_range'][1]:.4f}], "
            f"systematics={info['include_systematics']})"
        )


def create_desy5_likelihood(**kwargs) -> DESY5Likelihood:
    """
    Create DES-SN5YR likelihood instance.

    Parameters
    ----------
    **kwargs
        Arguments passed to DESY5Likelihood

    Returns
    -------
    DESY5Likelihood
        Configured likelihood instance
    """
    return DESY5Likelihood(**kwargs)
