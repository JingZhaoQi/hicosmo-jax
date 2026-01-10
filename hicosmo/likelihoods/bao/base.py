"""
BAO (Baryon Acoustic Oscillations) likelihood base class for HIcosmo.

This module provides a high-performance, JAX-optimized implementation of BAO likelihoods,
compatible with various BAO datasets including SDSS, BOSS, DESI, etc.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import os
from pathlib import Path
import yaml
import warnings

from ..base import Likelihood
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BAODataPoint:
    """Single BAO measurement data point."""
    z: float  # Effective redshift
    value: float  # Measured value
    error: float  # Error on measurement
    observable: str  # Type: 'DM_over_rd', 'DH_over_rd', 'DV_over_rd', 'rs_over_DV', 'fsigma8', etc.


@dataclass
class BAODataset:
    """Container for a complete BAO dataset."""
    name: str
    data_points: List[BAODataPoint]
    covariance: Optional[np.ndarray] = None
    correlation: Optional[np.ndarray] = None
    reference: str = ""
    year: int = 0


class BAOLikelihood(Likelihood):
    """
    Base class for all BAO likelihoods.

    Supports various BAO observables:
    - DM/rd: Comoving angular diameter distance over sound horizon
    - DH/rd: Hubble distance over sound horizon
    - DV/rd: Volume-averaged distance over sound horizon
    - rs/DV: Sound horizon over volume-averaged distance
    - f*sigma8: Growth rate times sigma8

    Attributes:
        dataset_name: Name of the BAO dataset to use
        data_path: Path to BAO data files
        use_rd_fid: Whether to use fiducial rd value from data
        marginalize_rd: Whether to marginalize over rd
    """

    def __init__(self,
                 dataset_name: Optional[str] = None,
                 data_path: Optional[str] = None,
                 use_rd_fid: bool = False,
                 marginalize_rd: bool = False,
                 cosmology_class: Optional[type] = None,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize BAO likelihood.

        Args:
            dataset_name: Specific BAO dataset name (e.g., 'sdss_dr12_consensus')
            data_path: Override default data path
            use_rd_fid: Use fiducial rd value from dataset
            marginalize_rd: Marginalize over sound horizon at drag epoch
            verbose: Print loading information
        """
        # Default data path (go up 3 levels: bao/ -> likelihoods/ -> hicosmo/ -> data/)
        if data_path is None:
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'data', 'bao_data'
            )

        super().__init__(name=dataset_name, data_path=data_path, **kwargs)

        self.dataset_name = dataset_name or self._default_dataset_name()
        self.use_rd_fid = use_rd_fid
        self.marginalize_rd = marginalize_rd
        self.verbose = verbose
        if cosmology_class is None:
            from ...models import LCDM
            self._cosmology_class = LCDM
        else:
            self._cosmology_class = cosmology_class

        # Cache default cosmology parameters for missing fixed values (e.g., Omega_b)
        self._default_params = {}
        try:
            default_model = self._cosmology_class()
            params = default_model.params
            if hasattr(params, "to_dict"):
                self._default_params = params.to_dict()
            elif isinstance(params, dict):
                self._default_params = dict(params)
        except Exception:
            self._default_params = {}

        # Data containers
        self.dataset: Optional[BAODataset] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.rd_fid: Optional[float] = None

        # Initialize on creation
        self.initialize()

        # Cache JAX-ready arrays for faster likelihood evaluation
        self._prepare_jax_cache()

    @abstractmethod
    def _default_dataset_name(self) -> str:
        """Return default dataset name for this likelihood."""
        pass

    @abstractmethod
    def _load_dataset(self) -> BAODataset:
        """Load the specific BAO dataset."""
        pass

    def _load_data(self) -> None:
        """Load BAO data from files."""
        self.dataset = self._load_dataset()

        if self.verbose:
            logger.debug(f"{self.dataset.name} loaded: {len(self.dataset.data_points)} measurements")
            logger.debug(
                f"Redshift range: {min(p.z for p in self.dataset.data_points):.3f} - "
                f"{max(p.z for p in self.dataset.data_points):.3f}"
            )
            unique_obs = set(p.observable for p in self.dataset.data_points)
            logger.debug(f"Observables: {', '.join(sorted(unique_obs))}")
            if self.dataset.reference:
                logger.debug(f"Reference: {self.dataset.reference}")

    def _prepare_jax_cache(self) -> None:
        """Prepare JAX-friendly cached arrays for fast evaluation."""
        if self.dataset is None:
            return

        self._z = jnp.array([p.z for p in self.dataset.data_points], dtype=jnp.float64)
        self._data_vec = jnp.array([p.value for p in self.dataset.data_points], dtype=jnp.float64)
        self._obs_codes = jnp.array(
            [self._observable_code(p.observable) for p in self.dataset.data_points],
            dtype=jnp.int32
        )
        self._inv_cov_jax = jnp.array(self.inv_cov)
        self._has_fsigma8 = any(p.observable == 'fsigma8' for p in self.dataset.data_points)
        self._dtype = self._data_vec.dtype
        self._prepare_distance_grid()
        self._initialize_fast_likelihood()

    def _prepare_distance_grid(self) -> None:
        """Precompute redshift grid for fast JAX distance interpolation."""
        z_max = float(self._z.max()) + 0.5
        z_min = 0.0
        n_grid = int(jnp.clip(z_max / 5e-4, 2048, 8192))
        self._z_grid = jnp.linspace(z_min, z_max, n_grid, dtype=self._dtype)

    @staticmethod
    def _observable_code(observable: str) -> int:
        """Map observable string to integer code for fast vectorized selection."""
        codes = {
            'DM_over_rd': 0,
            'DH_over_rd': 1,
            'DV_over_rd': 2,
            'rs_over_DV': 3,
            'Hz_rs': 4,
            'fsigma8': 5,
            'DM_over_DH': 6,
        }
        if observable not in codes:
            raise ValueError(f"Unknown BAO observable: {observable}")
        return codes[observable]

    def _setup_covariance(self) -> None:
        """Setup the covariance matrix and its inverse."""
        if self.dataset.covariance is not None:
            # Use provided covariance matrix
            self.inv_cov = np.linalg.inv(self.dataset.covariance)
            if self.verbose:
                cond = np.linalg.cond(self.dataset.covariance)
                logger.debug(
                    f"Covariance: {self.dataset.covariance.shape[0]}×{self.dataset.covariance.shape[0]}, "
                    f"condition number: {cond:.1e}"
                )
        else:
            # Use diagonal errors
            errors = np.array([p.error for p in self.dataset.data_points])
            self.inv_cov = np.diag(1.0 / errors**2)
            if self.verbose:
                logger.debug("Using diagonal errors")

    def get_requirements(self) -> Dict[str, Any]:
        """
        Return theory requirements for this likelihood.

        Returns:
            Dictionary of required theory calculations
        """
        reqs = {
            'DM_z': {'z': [p.z for p in self.dataset.data_points]},
            'DH_z': {'z': [p.z for p in self.dataset.data_points]},
            'rd': None,  # Sound horizon at drag epoch
        }

        # Add fsigma8 requirement if needed
        if any(p.observable == 'fsigma8' for p in self.dataset.data_points):
            reqs['fsigma8_z'] = {'z': [p.z for p in self.dataset.data_points
                                       if p.observable == 'fsigma8']}

        return reqs

    def _prepare_params_dict(self, params: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """Convert parameter dict to JAX arrays via model normalization."""
        return self._prepare_params_for_jax(
            params, dtype=self._dtype, default_params=self._default_params
        )

    def _can_use_fast_params(self, params: Dict[str, Any]) -> bool:
        if not hasattr(self, "_cosmology_class"):
            return False
        if not hasattr(self._cosmology_class, "compute_grid_traced"):
            return False
        if self._has_fsigma8:
            return False
        if not hasattr(self._cosmology_class, "sound_horizon_drag_traced"):
            return False
        return ('H0' in params) or ('h' in params)

    def _compute_rd_from_params(self, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        if hasattr(self._cosmology_class, "sound_horizon_drag_traced"):
            return self._cosmology_class.sound_horizon_drag_traced(params)
        model = self._cosmology_class(
            **{k: float(v) if hasattr(v, "__float__") else v for k, v in params.items()}
        )
        return jnp.asarray(model.sound_horizon_drag(), dtype=self._dtype)

    def _initialize_fast_likelihood(self) -> None:
        """Build JIT-compiled fast likelihood path (params -> logL)."""
        if not hasattr(self._cosmology_class, "compute_grid_traced"):
            self._loglike_fast = None
            return
        if not hasattr(self._cosmology_class, "sound_horizon_drag_traced"):
            self._loglike_fast = None
            return

        z_grid = self._z_grid
        z_obs = self._z
        obs_codes = self._obs_codes
        data_vec = self._data_vec
        inv_cov = self._inv_cov_jax
        use_rd_fid = bool(self.use_rd_fid)
        rd_fid = jnp.asarray(self.rd_fid if self.rd_fid is not None else 0.0, dtype=self._dtype)
        cosmology_class = self._cosmology_class
        rd_func = getattr(cosmology_class, "sound_horizon_drag_traced", None)

        @jit
        def _loglike_impl(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            cosmo_grid = cosmology_class.compute_grid_traced(z_grid, params)
            DM_grid = cosmo_grid['D_M']
            DH_grid = cosmo_grid['D_H']
            E_z_grid = cosmo_grid['E_z']

            volume_factor = z_grid * DM_grid**2 * DH_grid
            DV_grid = jnp.exp((1.0 / 3.0) * jnp.log(volume_factor + 1e-30))
            H_grid = params['H0'] * E_z_grid

            DM_obs = jnp.interp(z_obs, z_grid, DM_grid)
            DH_obs = jnp.interp(z_obs, z_grid, DH_grid)
            DV_obs = jnp.interp(z_obs, z_grid, DV_grid)
            H_obs = jnp.interp(z_obs, z_grid, H_grid)

            if use_rd_fid:
                rd_val = rd_fid
            else:
                rd_val = rd_func(params)

            theory = jnp.zeros_like(z_obs)
            theory = jnp.where(obs_codes == 0, DM_obs / rd_val, theory)
            theory = jnp.where(obs_codes == 1, DH_obs / rd_val, theory)
            theory = jnp.where(obs_codes == 2, DV_obs / rd_val, theory)
            theory = jnp.where(obs_codes == 3, rd_val / DV_obs, theory)
            theory = jnp.where(obs_codes == 4, H_obs * rd_val, theory)
            theory = jnp.where(obs_codes == 6, DM_obs / DH_obs, theory)

            diff = theory - data_vec
            chi2 = diff @ (inv_cov @ diff)
            return -0.5 * chi2

        self._loglike_fast = _loglike_impl

    def log_likelihood_from_params(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Fast log-likelihood path using parameter dict (JAX-friendly)."""
        if not self._can_use_fast_params(params) or self._loglike_fast is None:
            model = self._cosmology_class(**params)
            return self.log_likelihood(model)
        params_jax = self._prepare_params_dict(params)
        return self._loglike_fast(params_jax)

    def theory(self, cosmology, **kwargs) -> jnp.ndarray:
        """
        Compute theoretical predictions for BAO observables.

        Args:
            cosmology: Cosmology object
            **kwargs: Additional parameters

        Returns:
            Theory vector as JAX array
        """
        rd = kwargs.get('rd', None)
        theory_vec = self._compute_theory_vector(cosmology, rd=rd)
        return jnp.array(theory_vec)

    def _compute_theory_vector(self, cosmology, rd: Optional[float] = None) -> jnp.ndarray:
        """
        Compute theoretical predictions for all data points.

        Args:
            cosmology: Cosmology object with distance/growth calculations
            rd: Sound horizon at drag epoch (if None, computed from cosmology)

        Returns:
            Array of theoretical predictions matching data points
        """
        if rd is None:
            rd = cosmology.sound_horizon_drag()  # Sound horizon at drag epoch

        from ...utils.constants import c_km_s

        z = self._z
        obs_codes = self._obs_codes

        # Core distances
        H_z = cosmology.H_z(z)
        DH = c_km_s / H_z
        DM = (1.0 + z) * cosmology.angular_diameter_distance(z)
        DV = (z * DM**2 * DH) ** (1.0 / 3.0)

        # Optional growth term
        if self._has_fsigma8:
            if hasattr(cosmology, 'fsigma8'):
                fsigma8_vals = cosmology.fsigma8(z)
            else:
                fsigma8_vals = cosmology.f_sigma8(z)
        else:
            fsigma8_vals = jnp.zeros_like(z)

        # Assemble theory vector with vectorized selection
        theory = jnp.zeros_like(z)
        theory = jnp.where(obs_codes == 0, DM / rd, theory)          # DM_over_rd
        theory = jnp.where(obs_codes == 1, DH / rd, theory)          # DH_over_rd
        theory = jnp.where(obs_codes == 2, DV / rd, theory)          # DV_over_rd
        theory = jnp.where(obs_codes == 3, rd / DV, theory)          # rs_over_DV
        theory = jnp.where(obs_codes == 4, H_z * rd, theory)         # Hz_rs
        theory = jnp.where(obs_codes == 5, fsigma8_vals, theory)     # fsigma8
        theory = jnp.where(obs_codes == 6, DM / DH, theory)          # DM_over_DH

        return theory

    def _chi_squared(self, theory: jnp.ndarray, data: jnp.ndarray,
                     inv_cov: jnp.ndarray) -> float:
        """
        Compute chi-squared for BAO data.

        Args:
            theory: Theoretical predictions
            data: Observed values
            inv_cov: Inverse covariance matrix

        Returns:
            Chi-squared value
        """
        diff = theory - data
        return jnp.dot(diff, jnp.dot(inv_cov, diff))

    def log_likelihood(self, cosmology, **kwargs) -> float:
        """
        Compute log likelihood for BAO data.

        Args:
            cosmology: Cosmology object
            **kwargs: Additional parameters (e.g., nuisance parameters)

        Returns:
            Log likelihood value
        """
        # Get sound horizon if using fiducial value
        rd = self.rd_fid if self.use_rd_fid else None

        # Compute theoretical predictions using the theory method
        theory_jax = self.theory(cosmology, rd=rd)

        # Compute chi-squared
        chi2 = self._chi_squared(
            theory_jax,
            self._data_vec,
            self._inv_cov_jax
        )

        # Return log likelihood
        return -0.5 * chi2

    def __call__(self, **params) -> float:
        """Callable interface for samplers."""
        return self.log_likelihood_from_params(params)

    def get_derived_params(self, cosmology) -> Dict[str, float]:
        """
        Get derived parameters from this likelihood.

        Returns:
            Dictionary of derived parameters
        """
        derived = {}

        # Add rd if computed
        if not self.use_rd_fid:
            derived['rd'] = cosmology.sound_horizon_drag()

        # Add chi2
        theory = self._compute_theory_vector(cosmology)
        chi2 = self._chi_squared(
            jnp.array(theory),
            self._data_vec,
            self._inv_cov_jax
        )
        derived[f'chi2_{self.dataset_name}'] = float(chi2)

        return derived

    def get_num_data(self) -> int:
        """Return number of BAO data points for BIC calculation."""
        return len(self.dataset.data_points)

    def get_normalization_constant(self) -> float:
        """
        Return log-likelihood normalization constant.

        For BAO Gaussian likelihood: log(L) = -χ²/2 + normalization_constant
        The normalization includes -0.5 * log|C| (covariance determinant).
        """
        if self.dataset.covariance is not None:
            sign, logdet = np.linalg.slogdet(self.dataset.covariance)
            return -0.5 * logdet
        else:
            # Diagonal case: sum of log(sigma^2)
            errors = np.array([p.error for p in self.dataset.data_points])
            return -0.5 * np.sum(np.log(errors**2))


class BAOCollection:
    """
    Collection of multiple BAO datasets for joint analysis.

    This class allows combining multiple BAO measurements with proper
    handling of correlations between datasets when applicable.

    Note: This does not inherit from Likelihood base class as it's a
    collection/wrapper rather than a single likelihood.
    """

    def __init__(self,
                 datasets: List[Union[str, BAOLikelihood]],
                 data_path: Optional[str] = None,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize collection of BAO datasets.

        Args:
            datasets: List of dataset names or BAOLikelihood objects
            data_path: Path to BAO data files
            verbose: Print information
        """
        self.name = 'BAO_collection'
        self.data_path = data_path
        self.verbose = verbose
        self.likelihoods: List[BAOLikelihood] = []

        # Create likelihood objects
        for dataset in datasets:
            if isinstance(dataset, str):
                # Create likelihood from name
                like = self._create_likelihood(dataset, data_path)
            else:
                # Already a likelihood object
                like = dataset
            self.likelihoods.append(like)

        if self.verbose:
            logger.debug(f"BAO Collection initialized with {len(self.likelihoods)} datasets")

    def _create_likelihood(self, name: str, data_path: Optional[str]) -> BAOLikelihood:
        """Create a BAO likelihood from dataset name."""
        # Import specific implementations
        from . import bao_datasets

        # Map dataset names to classes
        dataset_map = {
            'sdss_dr12': bao_datasets.SDSSDR12BAO,
            'sdss_dr16': bao_datasets.SDSSDR16BAO,
            'boss_dr12': bao_datasets.BOSSDR12BAO,
            'desi_2024': bao_datasets.DESI2024BAO,
            'sixdf': bao_datasets.SixDFBAO,
        }

        if name not in dataset_map:
            raise ValueError(f"Unknown BAO dataset: {name}. "
                           f"Available: {list(dataset_map.keys())}")

        return dataset_map[name](data_path=data_path, verbose=self.verbose)

    def log_likelihood(self, cosmology, **kwargs) -> float:
        """
        Compute combined log likelihood.

        Args:
            cosmology: Cosmology object
            **kwargs: Additional parameters

        Returns:
            Total log likelihood
        """
        total = 0.0
        for like in self.likelihoods:
            total += like.log_likelihood(cosmology, **kwargs)
        return total

    def get_requirements(self) -> Dict[str, Any]:
        """Combine requirements from all datasets."""
        reqs = {}
        for like in self.likelihoods:
            like_reqs = like.get_requirements()
            # Merge requirements
            for key, value in like_reqs.items():
                if key not in reqs:
                    reqs[key] = value
                elif isinstance(value, dict) and 'z' in value:
                    # Merge redshift arrays
                    existing_z = reqs[key].get('z', [])
                    new_z = value['z']
                    # Combine and remove duplicates
                    all_z = sorted(set(existing_z + new_z))
                    reqs[key] = {'z': all_z}
        return reqs
