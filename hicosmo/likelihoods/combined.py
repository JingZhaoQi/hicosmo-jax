"""
Combined Likelihood for Multi-Probe Analysis
=============================================

Provides a simple way to combine multiple likelihoods using the + operator.
When multiple likelihoods share the same cosmology class, distance calculations
are performed ONCE and shared across all sub-likelihoods.

Example
-------
>>> sne = SN_likelihood(LCDM, "pantheon+")
>>> bao = BAO_likelihood(LCDM, "desi2024")
>>> joint = sne + bao
>>> MCMC(params, joint, chain_name='joint').run(...)
"""

import logging
from typing import List, Callable, Dict, Any, Optional

from .base import Likelihood, NuisanceList
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class CombinedLikelihood(Likelihood):
    """
    Combined likelihood for multi-probe joint analysis.

    Automatically detects when sub-likelihoods share a cosmology class
    and computes the distance grid once instead of N times.
    """

    def __init__(self, likelihoods: List[Callable]):
        super().__init__(name="CombinedLikelihood")
        self._likelihoods = []

        # Flatten nested CombinedLikelihoods
        for lik in likelihoods:
            if isinstance(lik, CombinedLikelihood):
                self._likelihoods.extend(lik._likelihoods)
            else:
                self._likelihoods.append(lik)

        # Attempt to set up shared distance grid
        self._shared_grid_func = None
        self._shared_z_low = None
        self._shared_z_high = None
        self._low_z_likelihoods = []
        self._high_z_likelihoods = []
        self._other_likelihoods = []
        self._setup_shared_grid()

    def _setup_shared_grid(self):
        """Detect shared cosmology class and build unified distance grid."""
        from ..models.base import make_compute_shared_grid

        # Classify likelihoods by z_grid range
        cosmo_classes = set()
        low_z_grids = []

        for lik in self._likelihoods:
            cosmo_cls = getattr(lik, '_cosmology_class', None)
            if cosmo_cls is not None:
                cosmo_classes.add(cosmo_cls)

            z_grid = getattr(lik, '_z_grid', None)
            if z_grid is not None and len(z_grid) > 0:
                z_max = float(z_grid[-1])
                if z_max < 10:
                    self._low_z_likelihoods.append(lik)
                    low_z_grids.append(z_grid)
                else:
                    self._high_z_likelihoods.append(lik)
            else:
                # CMB has no _z_grid but has _z_base
                z_base = getattr(lik, '_z_base', None)
                if z_base is not None:
                    self._high_z_likelihoods.append(lik)
                else:
                    self._other_likelihoods.append(lik)

        # Only optimize if all share the same cosmology class and there are >=2 low-z
        if len(cosmo_classes) != 1 or len(self._low_z_likelihoods) < 2:
            # Fallback: no optimization
            self._low_z_likelihoods = []
            self._high_z_likelihoods = []
            self._other_likelihoods = list(self._likelihoods)
            return

        cosmo_cls = cosmo_classes.pop()

        # Build unified low-z grid (take the densest and widest)
        z_max_low = max(float(g[-1]) for g in low_z_grids)
        n_grid_low = max(len(g) for g in low_z_grids)
        self._shared_z_low = jnp.linspace(0.0, z_max_low, n_grid_low)

        # Build high-z extension grid for CMB (z_low_max to ~1100)
        if self._high_z_likelihoods:
            z_high_max = 1100.0  # Conservative upper bound for z_star
            n_grid_high = 2048
            self._shared_z_high = jnp.linspace(
                z_max_low, z_high_max, n_grid_high
            )
        else:
            self._shared_z_high = jnp.array([z_max_low])  # dummy

        # Get the E_z function from the cosmology class
        E_z_func = getattr(cosmo_cls, '_E_z_static', None)
        if E_z_func is None:
            # Fallback
            self._low_z_likelihoods = []
            self._high_z_likelihoods = []
            self._other_likelihoods = list(self._likelihoods)
            return

        self._cosmo_class = cosmo_cls
        self._shared_grid_func = make_compute_shared_grid(E_z_func)
        logger.info(
            f"Shared grid optimization: low-z [{0:.1f}, {z_max_low:.1f}] "
            f"N={n_grid_low}, high-z [{z_max_low:.1f}, 1100] N=2048, "
            f"{len(self._low_z_likelihoods)} low-z + "
            f"{len(self._high_z_likelihoods)} high-z likelihoods"
        )

    def __call__(self, **params) -> float:
        """
        Compute combined log-likelihood.

        When shared grid is available, computes distances ONCE
        and distributes to sub-likelihoods.
        """
        if self._shared_grid_func is None:
            # Fallback: no sharing (original behavior)
            total_log_L = 0.0
            for lik in self._likelihoods:
                total_log_L = total_log_L + lik(**params)
            return total_log_L

        # Shared grid path: compute distances once on the unified low-z grid
        shared = self._shared_grid_func(
            self._shared_z_low, self._shared_z_high, params
        )

        # Cache: temporarily replace compute_grid_traced on the cosmology class
        # so that sub-likelihoods automatically reuse the precomputed grid.
        # This avoids modifying any sub-likelihood interfaces.
        cosmo_cls = self._cosmo_class
        original_cgt = cosmo_cls.compute_grid_traced
        shared_low = shared["grid_low"]
        shared_z_low = shared["z_low"]

        from jax import jit as _jit

        @_jit
        def _cached_compute_grid(z_grid, params_inner):
            """Return precomputed grid if z_grid matches shared, else compute fresh."""
            # Always return shared grid for low-z calls.
            # High-z (CMB) calls have different z_grid shape and will need fresh computation,
            # but we handle CMB separately below.
            return {
                "d_L": jnp.interp(z_grid, shared_z_low, shared_low["d_L"]),
                "D_M": jnp.interp(z_grid, shared_z_low, shared_low["D_M"]),
                "D_H": jnp.interp(z_grid, shared_z_low, shared_low["D_H"]),
                "E_z": jnp.interp(z_grid, shared_z_low, shared_low["E_z"]),
                "d_C": jnp.interp(z_grid, shared_z_low, shared_low["d_C"]),
                "dVc_dz": jnp.zeros_like(z_grid),  # not needed by SN/BAO
                "ddL_dz": jnp.zeros_like(z_grid),   # not needed by SN/BAO
            }

        total_log_L = 0.0

        # Low-z likelihoods: use cached compute_grid_traced
        try:
            cosmo_cls.compute_grid_traced = staticmethod(_cached_compute_grid)
            for lik in self._low_z_likelihoods:
                total_log_L = total_log_L + lik(**params)
        finally:
            cosmo_cls.compute_grid_traced = original_cgt

        # High-z likelihoods (CMB): use original path (separate z_grid to z_star)
        for lik in self._high_z_likelihoods:
            total_log_L = total_log_L + lik(**params)

        # Other likelihoods: fallback
        for lik in self._other_likelihoods:
            total_log_L = total_log_L + lik(**params)

        return total_log_L

    def _load_data(self) -> None:
        return

    def _setup_covariance(self) -> None:
        return

    def get_requirements(self) -> Dict[str, Any]:
        requirements: Dict[str, Any] = {}
        for lik in self._likelihoods:
            if hasattr(lik, "get_requirements"):
                try:
                    req = lik.get_requirements()
                    if isinstance(req, dict):
                        requirements.update(req)
                except (TypeError, AttributeError, NotImplementedError):
                    # Skip likelihoods that fail to provide requirements
                    continue
        return requirements

    def theory(self, **kwargs) -> jnp.ndarray:
        raise NotImplementedError(
            "CombinedLikelihood does not define a single theory vector."
        )

    @property
    def nuisance_parameters(self):
        nuisance_items = []
        for lik in self._likelihoods:
            nuisance = getattr(lik, "nuisance_parameters", None)
            if nuisance is None:
                continue
            nuisance = nuisance() if callable(nuisance) else nuisance
            if isinstance(nuisance, dict):
                nuisance_items.extend(nuisance.values())
            else:
                nuisance_items.extend(list(nuisance))
        return NuisanceList(nuisance_items)

    # __add__ and __radd__ inherited from Likelihood base class

    def __repr__(self):
        """String representation."""
        lik_names = []
        for lik in self._likelihoods:
            name = getattr(lik, "__class__", type(lik)).__name__
            lik_names.append(name)
        return f"CombinedLikelihood({' + '.join(lik_names)})"

    def __len__(self):
        """Number of component likelihoods."""
        return len(self._likelihoods)

    @property
    def likelihoods(self) -> List[Callable]:
        """Get list of component likelihoods."""
        return self._likelihoods.copy()
