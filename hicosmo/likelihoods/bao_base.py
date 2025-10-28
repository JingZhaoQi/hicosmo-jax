"""
BAO (Baryon Acoustic Oscillations) likelihood base class for HIcosmo.

This module provides a high-performance, JAX-optimized implementation of BAO likelihoods,
compatible with various BAO datasets including SDSS, BOSS, DESI, etc.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import os
from pathlib import Path
import yaml
import warnings

from .base import Likelihood


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
        # Default data path
        if data_path is None:
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data', 'bao_data'
            )

        super().__init__(name=dataset_name, data_path=data_path, **kwargs)

        self.dataset_name = dataset_name or self._default_dataset_name()
        self.use_rd_fid = use_rd_fid
        self.marginalize_rd = marginalize_rd
        self.verbose = verbose

        # Data containers
        self.dataset: Optional[BAODataset] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.rd_fid: Optional[float] = None

        # Initialize on creation
        self.initialize()

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
            print(f"✅ {self.dataset.name} loaded: {len(self.dataset.data_points)} measurements")
            print(f"   Redshift range: {min(p.z for p in self.dataset.data_points):.3f} - "
                  f"{max(p.z for p in self.dataset.data_points):.3f}")
            unique_obs = set(p.observable for p in self.dataset.data_points)
            print(f"   Observables: {', '.join(sorted(unique_obs))}")
            if self.dataset.reference:
                print(f"   Reference: {self.dataset.reference}")

    def _setup_covariance(self) -> None:
        """Setup the covariance matrix and its inverse."""
        if self.dataset.covariance is not None:
            # Use provided covariance matrix
            self.inv_cov = np.linalg.inv(self.dataset.covariance)
            if self.verbose:
                cond = np.linalg.cond(self.dataset.covariance)
                print(f"   Covariance: {self.dataset.covariance.shape[0]}×{self.dataset.covariance.shape[0]}, "
                      f"condition number: {cond:.1e}")
        else:
            # Use diagonal errors
            errors = np.array([p.error for p in self.dataset.data_points])
            self.inv_cov = np.diag(1.0 / errors**2)
            if self.verbose:
                print(f"   Using diagonal errors")

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

    def _compute_theory_vector(self, cosmology, rd: Optional[float] = None) -> np.ndarray:
        """
        Compute theoretical predictions for all data points.

        Args:
            cosmology: Cosmology object with distance/growth calculations
            rd: Sound horizon at drag epoch (if None, computed from cosmology)

        Returns:
            Array of theoretical predictions matching data points
        """
        if rd is None:
            rd = cosmology.rs_drag()  # Sound horizon at drag epoch

        from ..utils.constants import c_km_s

        theory = []
        for point in self.dataset.data_points:
            z = point.z

            if point.observable == 'DM_over_rd':
                # Transverse comoving distance (DM) over rd
                DM = (1.0 + z) * cosmology.angular_diameter_distance(z)
                value = DM / rd

            elif point.observable == 'DH_over_rd':
                # Hubble distance / rd = c/(H(z)*rd)
                value = c_km_s / cosmology.H_z(z) / rd

            elif point.observable == 'DV_over_rd':
                # Volume-averaged distance / rd (uses DM = (1+z) * D_A)
                DM = (1.0 + z) * cosmology.angular_diameter_distance(z)
                DH = c_km_s / cosmology.H_z(z)
                DV = (z * DM**2 * DH)**(1/3)
                value = DV / rd

            elif point.observable == 'rs_over_DV':
                # rd / Volume-averaged distance (inverse of DV/rd)
                DM = (1.0 + z) * cosmology.angular_diameter_distance(z)
                DH = c_km_s / cosmology.H_z(z)
                DV = (z * DM**2 * DH)**(1/3)
                value = rd / DV

            elif point.observable == 'Hz_rs':
                # H(z) * rd (commonly noted as H*z*s)
                value = cosmology.H_z(z) * rd

            elif point.observable == 'fsigma8':
                # Growth rate * sigma8(z)
                value = cosmology.fsigma8(z)

            elif point.observable == 'DM_over_DH':
                # DM/DH ratio (rd-independent)
                DM = (1.0 + z) * cosmology.angular_diameter_distance(z)
                DH = c_km_s / cosmology.H_z(z)
                value = DM / DH

            else:
                raise ValueError(f"Unknown BAO observable: {point.observable}")

            theory.append(value)

        return jnp.array(theory)

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

        # Get observed values
        data = jnp.array([p.value for p in self.dataset.data_points])

        # Compute chi-squared
        chi2 = self._chi_squared(
            theory_jax,
            data,
            jnp.array(self.inv_cov)
        )

        # Return log likelihood
        return -0.5 * chi2

    def get_derived_params(self, cosmology) -> Dict[str, float]:
        """
        Get derived parameters from this likelihood.

        Returns:
            Dictionary of derived parameters
        """
        derived = {}

        # Add rd if computed
        if not self.use_rd_fid:
            derived['rd'] = cosmology.rs_drag()

        # Add chi2
        theory = self._compute_theory_vector(cosmology)
        data = np.array([p.value for p in self.dataset.data_points])
        chi2 = self._chi_squared(
            jnp.array(theory),
            jnp.array(data),
            jnp.array(self.inv_cov)
        )
        derived[f'chi2_{self.dataset_name}'] = float(chi2)

        return derived


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
            print(f"✅ BAO Collection initialized with {len(self.likelihoods)} datasets")

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
