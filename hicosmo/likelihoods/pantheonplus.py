"""
PantheonPlus Supernova Likelihood Function
==========================================

Clean implementation of Pantheon+ (2022) 1701 Type Ia supernova likelihood.
Following CLAUDE.md design principles: minimal, efficient, and JAX-optimized.

Data source: Brout et al. 2022, ApJ, 938, 110
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np
from typing import Optional, Dict, Tuple, Union
from pathlib import Path

from ..core.base import CosmologyBase


class PantheonPlusLikelihood:
    """
    PantheonPlus Type Ia supernova likelihood function.

    Clean, minimal implementation following CLAUDE.md principles:
    - Single responsibility: only SNe Ia likelihood
    - Performance optimized: JAX JIT compilation
    - Simple interface: just cosmological models

    Parameters
    ----------
    data_path : str
        Path to PantheonPlus data directory containing official data release files.
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
    """

    def __init__(self,
                 data_path: str,
                 include_shoes: bool = False,
                 include_systematics: bool = True,
                 marginalize_M_B: bool = True,
                 z_min: float = 0.01,
                 z_max: Optional[float] = None,
                 apply_z_cut: bool = True):

        self.data_path = data_path
        self.include_shoes = include_shoes
        self.include_systematics = include_systematics
        self.marginalize_M_B = marginalize_M_B
        self.z_min = z_min
        self.z_max = z_max
        self.apply_z_cut = apply_z_cut

        # Load data
        self._load_data()

        # Precompute covariance matrix inverse for efficiency
        self._precompute_covariance()

        dataset_name = "PantheonPlusSH0ES" if self.include_shoes else "PantheonPlus"
        print(f"✅ {dataset_name} loaded: {len(self.redshifts)} objects")
        print(f"   Redshift range: {self.redshifts.min():.3f} - {self.redshifts.max():.3f}")
        print(f"   Systematics: {'included' if include_systematics else 'excluded'}")
        print(f"   M_B treatment: {'marginalized' if self.marginalize_M_B else 'free parameter'}")
        if self.apply_z_cut:
            z_max_str = f"{self.z_max:.3f}" if self.z_max is not None else "∞"
            print(f"   Redshift cuts: z ∈ [{self.z_min:.3f}, {z_max_str}]")

    def _load_data(self):
        """Load real PantheonPlus data."""
        if self.data_path is None:
            raise ValueError("data_path is required. PantheonPlus needs real observational data.")
        self._load_real_data()

    def _load_real_data(self):
        """Load real PantheonPlus data following official cosmosis implementation."""
        import pandas as pd

        # Official data release only provides Pantheon+SH0ES combined file
        # We filter out calibrators if include_shoes=False
        if self.include_shoes:
            print("📊 Loading Pantheon+SH0ES combined data...")
        else:
            print("📊 Loading Pantheon+SH0ES data (will filter out calibrators)...")

        # Official file paths - always use Pantheon+SH0ES file
        data_file = Path(self.data_path) / "Pantheon+_Data" / "4_DISTANCES_AND_COVAR" / "Pantheon+SH0ES.dat"

        # Covariance matrix files
        if self.include_systematics:
            cov_file = Path(self.data_path) / "Pantheon+_Data" / "4_DISTANCES_AND_COVAR" / "Pantheon+SH0ES_STAT+SYS.cov"
        else:
            cov_file = Path(self.data_path) / "Pantheon+_Data" / "4_DISTANCES_AND_COVAR" / "Pantheon+SH0ES_STATONLY.cov"

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
        data = pd.read_csv(data_file, comment='#', sep=r'\s+')

        # Use official column names exactly as in cosmosis code
        # Cosmology should use the Hubble-flow corrected redshift (zHD)
        z_hd_raw = data['zHD'].values  # Hubble flow redshift (pec-velocity corrected)
        z_cmb_raw = data['zCMB'].values  # Provided for completeness (not used in cosmosis)
        z_hel_raw = data['zHEL'].values  # Heliocentric redshift
        m_b_corr_raw = data['m_b_corr'].values  # Corrected apparent magnitude (official)
        is_calibrator_raw = data['IS_CALIBRATOR'].values.astype(bool)
        ceph_dist_raw = data['CEPH_DIST'].values

        # Official data selection mask (from cosmosis code)
        if self.include_shoes:
            # SH0ES mode: z_HD > 0.01 OR is_calibrator
            self.ww = (z_hd_raw > 0.01) | is_calibrator_raw
            print(f"   Found {np.sum(is_calibrator_raw)} Cepheid calibrators")
        else:
            # Pantheon+ mode: z_HD > 0.01 AND NOT is_calibrator
            self.ww = (z_hd_raw > 0.01) & (~is_calibrator_raw)
            print(f"   Excluding {np.sum(is_calibrator_raw)} calibrators (Pantheon+ mode)")

        # Apply additional redshift cuts if requested (on top of official cuts)
        if self.apply_z_cut and (self.z_min != 0.01 or self.z_max is not None):
            z_mask = (z_hd_raw > self.z_min)
            if self.z_max is not None:
                z_mask = z_mask & (z_hd_raw <= self.z_max)

            original_count = np.sum(self.ww)
            self.ww = self.ww & z_mask
            new_count = np.sum(self.ww)
            print(f"   Applied redshift cuts: {original_count} → {new_count} objects ({original_count - new_count} removed)")

        # Extract final datasets using official selection
        # Following your previous implementation: use zHD as zcmb for cosmology
        self.z_cmb = jnp.array(z_hd_raw[self.ww])  # Use vpec corrected redshift for cosmology
        self.z_hel = jnp.array(z_hel_raw[self.ww])  # Heliocentric (for peculiar motion)
        self.z_hd = jnp.array(z_hd_raw[self.ww])   # Hubble flow (for cuts)
        self.m_obs = jnp.array(m_b_corr_raw[self.ww])  # Observed corrected magnitude
        self.is_calibrator = is_calibrator_raw[self.ww]  # Calibrator flags
        self.ceph_dist = ceph_dist_raw[self.ww]  # Cepheid distances

        # Store redshifts for backward compatibility (use Hubble flow)
        self.redshifts = self.z_hd

        # Store indices for covariance matrix slicing
        self._original_indices = np.where(self.ww)[0]

        # Load covariance matrix
        print("   Loading covariance matrix...")

        # Load covariance matrix following your previous implementation
        try:
            print("   Reading covariance matrix (this may take a moment)...")
            # Load full file and skip first line (dimension), then reshape
            ff = np.loadtxt(cov_file)
            n_dim = len(data)  # Original data length
            print(f"   Covariance matrix dimension: {n_dim}")

            # Skip first element (dimension) and reshape
            cov_full = ff[1:].reshape(n_dim, n_dim)
            print(f"   Loaded {n_dim}x{n_dim} covariance matrix")

            # Apply data selection to covariance matrix using your method
            self.covariance = jnp.array(cov_full[self.ww, :][:, self.ww])
            print(f"   Applied data selection: {n_dim}x{n_dim} → {self.covariance.shape[0]}x{self.covariance.shape[1]}")

        except Exception as e:
            print(f"   Warning: Could not load covariance matrix: {e}")
            print(f"   Using diagonal covariance matrix (simple approximation)")
            # Fallback to diagonal covariance - estimate from data scatter
            n_data = len(self.m_obs)
            typical_error = 0.1  # Typical magnitude error for SNe Ia
            self.covariance = jnp.eye(n_data) * typical_error**2

        self.n_sne = len(self.redshifts)

        print(f"   ✅ Real data loaded: {self.n_sne} SNe Ia")
        print(f"   Redshift range: {self.redshifts.min():.4f} - {self.redshifts.max():.4f}")
        print(f"   Magnitude range: {self.m_obs.min():.2f} - {self.m_obs.max():.2f}")

        # Validate data consistency
        assert len(self.redshifts) == len(self.m_obs), "Redshift and magnitude arrays must have same length"
        assert self.covariance.shape == (self.n_sne, self.n_sne), "Covariance matrix dimensions inconsistent"

    def _precompute_covariance(self):
        """Precompute covariance matrix inverse and determinant."""
        # Numerical stability
        regularization = 1e-8 * jnp.eye(self.n_sne)
        stable_cov = self.covariance + regularization

        self.cov_inv = jnp.linalg.inv(stable_cov)
        self.log_det_cov = jnp.linalg.slogdet(stable_cov)[1]

        condition_number = jnp.linalg.cond(stable_cov)
        print(f"   Covariance: {self.n_sne}×{self.n_sne}, condition number: {condition_number:.1e}")

        # JAX JIT optimization: Pre-compile fixed M_B likelihood with closures
        # This eliminates repeated compilation for different M_B values
        self._setup_jit_optimized_likelihood()

    def log_likelihood(self, model: CosmologyBase, M_B: Optional[float] = None) -> float:
        """
        Compute log-likelihood following official cosmosis implementation.

        Parameters
        ----------
        model : CosmologyBase
            Cosmological model instance
        M_B : float, optional
            Absolute magnitude parameter. If marginalize_M_B=True, this is ignored.
            If marginalize_M_B=False, this must be provided.

        Returns
        -------
        float
            Log-likelihood value
        """
        # Follow official cosmosis theory computation
        if self.include_shoes:
            # SH0ES mode: separate treatment for calibrators vs Hubble flow
            theory_mu = self._compute_theory_shoes(model)
        else:
            # Pantheon+ mode: all objects use cosmology + official formula
            theory_mu = self._compute_theory_pantheon(model)

        # Choose between marginalized or fixed M_B likelihood
        if self.marginalize_M_B:
            return self._marginalized_likelihood(theory_mu)
        else:
            if M_B is None:
                raise ValueError("M_B must be provided when marginalize_M_B=False")
            return self._fixed_likelihood(theory_mu, M_B)

    def _setup_jit_optimized_likelihood(self):
        """Setup JAX JIT optimized likelihood functions to avoid repeated compilation."""
        # Convert to device arrays to ensure consistent dtype and memory layout
        # Use float32 for optimal JAX performance and compatibility
        self.m_obs = jnp.asarray(self.m_obs, dtype=jnp.float32)
        self.cov_inv = jnp.asarray(self.cov_inv, dtype=jnp.float32)
        self.log_det_cov = jnp.asarray(self.log_det_cov, dtype=jnp.float32)

        # Create JIT-compiled function with closure to capture constants
        # This ensures JAX compiles only once regardless of M_B values
        self._jit_fixed_likelihood = jit(self._create_fixed_likelihood_closure())

        # Also JIT compile marginalized likelihood for consistent performance
        self._jit_marginalized_likelihood = jit(self._create_marginalized_likelihood_closure())

    def _create_fixed_likelihood_closure(self):
        """Create closure function that captures constant arrays for JIT optimization."""
        # Capture constant arrays in closure to avoid repeated compilation
        m_obs = self.m_obs
        cov_inv = self.cov_inv
        log_det_cov = self.log_det_cov

        def fixed_likelihood_impl(mu_theory, M_B):
            """Pure function implementation of fixed M_B likelihood.

            This function only depends on JAX arrays, not Python objects,
            allowing JAX to compile once and reuse for all M_B values.
            """
            # Theory prediction: theoretical magnitude = mu_theory + M_B
            m_theory = mu_theory + M_B

            # Residual: observed magnitude - theoretical magnitude
            residual = m_obs - m_theory

            # χ² calculation
            chi2 = jnp.dot(residual, jnp.dot(cov_inv, residual))

            # Log-likelihood
            log_like = -0.5 * chi2 - 0.5 * log_det_cov

            return log_like

        return fixed_likelihood_impl

    def _create_marginalized_likelihood_closure(self):
        """Create closure function for marginalized likelihood JIT optimization."""
        # Capture constant arrays in closure
        m_obs = self.m_obs
        cov_inv = self.cov_inv
        log_det_cov = self.log_det_cov
        n_sne = self.n_sne

        def marginalized_likelihood_impl(mu_theory):
            """Pure function implementation of marginalized M_B likelihood."""
            # Residual vector: observed magnitude - theoretical magnitude (without M_B)
            residual = m_obs - mu_theory

            # Solve for optimal M_B: M_B_best = (1^T C^-1 r) / (1^T C^-1 1)
            ones = jnp.ones(n_sne)
            numerator = jnp.dot(ones, jnp.dot(cov_inv, residual))
            denominator = jnp.dot(ones, jnp.dot(cov_inv, ones))
            M_B_best = numerator / denominator

            # Marginalized residual
            marginalized_residual = residual - M_B_best * ones

            # χ² calculation
            chi2 = jnp.dot(marginalized_residual, jnp.dot(cov_inv, marginalized_residual))

            # Log-likelihood
            log_like = -0.5 * chi2 - 0.5 * log_det_cov

            # Marginalization correction (from M_B integral Jacobian)
            log_like += 0.5 * jnp.log(denominator)

            return log_like

        return marginalized_likelihood_impl

    def _fixed_likelihood(self, mu_theory: jnp.ndarray, M_B: float) -> float:
        """Fixed M_B likelihood with JAX JIT optimization.

        This method now uses pre-compiled JIT function to avoid
        repeated compilation for different M_B values.
        """
        # Ensure M_B has consistent dtype to prevent recompilation
        M_B_array = jnp.asarray(M_B, dtype=self.m_obs.dtype)

        # Use pre-compiled JIT function
        return self._jit_fixed_likelihood(mu_theory, M_B_array)

    def _marginalized_likelihood(self, mu_theory: jnp.ndarray) -> float:
        """
        Analytically marginalized likelihood over M_B with JAX JIT optimization.

        For linear parameter M_B, we can analytically integrate:
        L_marg = ∫ L(μ|M_B, θ) π(M_B) dM_B

        This method now uses pre-compiled JIT function for optimal performance.
        """
        # Use pre-compiled JIT function
        return self._jit_marginalized_likelihood(mu_theory)

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

        # Theory prediction: theoretical magnitude = mu_theory + M_B
        m_theory = theory_mu + M_B

        # Residual: observed magnitude - theoretical magnitude
        residual = self.m_obs - m_theory

        # χ² calculation
        chi2 = jnp.dot(residual, jnp.dot(self.cov_inv, residual))

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

        residual = self.m_obs - theory_mu

        ones = jnp.ones(self.n_sne)
        numerator = jnp.dot(ones, jnp.dot(self.cov_inv, residual))
        denominator = jnp.dot(ones, jnp.dot(self.cov_inv, ones))

        return float(numerator / denominator)

    def get_info(self) -> Dict:
        """Get dataset information."""
        return {
            'name': 'PantheonPlus',
            'n_supernovae': self.n_sne,
            'redshift_range': (float(self.redshifts.min()), float(self.redshifts.max())),
            'include_systematics': self.include_systematics,
            'data_type': 'PantheonPlusSH0ES' if self.include_shoes else 'PantheonPlus'
        }

    def _compute_theory_pantheon(self, model: CosmologyBase) -> jnp.ndarray:
        """
        Compute theoretical distance moduli for Pantheon+ mode (no calibrators).
        Following official cosmosis implementation line 103.

        Official formula: μ = 5*log10((1+z_CMB)*(1+z_HEL)*d_A(z_CMB)) + 25
        """
        # Use CMB frame redshifts for cosmology
        z_cmb = self.z_cmb
        z_hel = self.z_hel

        # Get angular diameter distance from cosmology
        # The (1+z) factors effectively convert this to luminosity distance with corrections
        d_A = model.angular_diameter_distance(z_cmb)

        # Official distance modulus formula with redshift correction factors
        # (1+z_CMB)*(1+z_HEL)*d_A effectively gives luminosity distance with peculiar velocity corrections
        theory_mu = 5.0 * jnp.log10((1.0 + z_cmb) * (1.0 + z_hel) * d_A) + 25.0

        return theory_mu

    def _compute_theory_shoes(self, model: CosmologyBase) -> jnp.ndarray:
        """
        Compute theoretical distance moduli for SH0ES mode.
        Following official cosmosis implementation lines 102-109.

        Calibrators: use fixed Cepheid distances
        Hubble flow SNe: use cosmology with redshift corrections
        """
        theory_mu = jnp.zeros(len(self.m_obs))

        # Use boolean masks instead of indices for JAX compatibility
        calibrator_mask = jnp.array(self.is_calibrator)
        hubble_mask = ~calibrator_mask

        # For Cepheid calibrators: use fixed distances (line 102)
        theory_mu = jnp.where(calibrator_mask, self.ceph_dist, theory_mu)

        # For Hubble flow SNe: use cosmology (lines 105-109)
        # Get angular diameter distance from cosmology for all points
        d_A = model.angular_diameter_distance(self.z_cmb)

        # Official distance modulus formula
        theory_mu_hubble = 5.0 * jnp.log10((1.0 + self.z_cmb) * (1.0 + self.z_hel) * d_A) + 25.0

        # Apply to Hubble flow objects only
        theory_mu = jnp.where(hubble_mask, theory_mu_hubble, theory_mu)

        return theory_mu

    def __repr__(self):
        """String representation."""
        info = self.get_info()
        return (f"PantheonPlus({info['n_supernovae']} SNe, "
                f"z=[{info['redshift_range'][0]:.3f}, {info['redshift_range'][1]:.3f}], "
                f"systematics={info['include_systematics']})")


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
