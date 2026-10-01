"""
Specific BAO dataset implementations for HIcosmo.

This module contains implementations for various BAO surveys:
- SDSS DR12 and DR16
- BOSS DR12
- DESI 2024
- 6dFGS
"""

import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict, Any
import yaml

from .base import BAOLikelihood, BAODataset, BAODataPoint
from ..base import NuisanceList
from ...utils.jax_tools import interp_linspace
from ...utils.logging import get_logger

logger = get_logger(__name__)


# Mapping between different dataset naming conventions and HIcosmo observables
_OBSERVABLE_MAP = {
    "DM_over_rs": "DM_over_rd",
    "DM_over_rd": "DM_over_rd",
    "bao_DM_over_rs": "DM_over_rd",
    "DH_over_rs": "DH_over_rd",
    "DH_over_rd": "DH_over_rd",
    "bao_DH_over_rs": "DH_over_rd",
    "DV_over_rs": "DV_over_rd",
    "DV_over_rd": "DV_over_rd",
    "bao_DV_over_rs": "DV_over_rd",
    "rs_over_DV": "rs_over_DV",
    "bao_rs_over_DV": "rs_over_DV",
    "Hz_rs": "Hz_rs",
    "bao_Hz_rs": "Hz_rs",
    "f_sigma8": "fsigma8",
}


def _normalize_observable(raw: str) -> str:
    """Map observable labels from external datasets into HIcosmo conventions."""
    observable = raw.strip()
    if observable.startswith("bao_") and observable not in _OBSERVABLE_MAP:
        observable = observable[4:]
    if observable not in _OBSERVABLE_MAP:
        raise ValueError(f"Unsupported BAO observable label '{raw}'.")
    return _OBSERVABLE_MAP[observable]


def _read_data_file(file_path: Path) -> List[Tuple[float, float, Optional[float], str]]:
    """Parse a BAO data file containing rows of measurements."""
    entries: List[Tuple[float, float, Optional[float], str]] = []
    with open(file_path, "r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue

            z = float(parts[0])
            value = float(parts[1])

            if len(parts) >= 4:
                error = float(parts[2])
                observable_raw = parts[3]
            else:
                error = None
                observable_raw = parts[2]

            observable = _normalize_observable(observable_raw)
            entries.append((z, value, error, observable))

    if not entries:
        raise ValueError(f"No BAO data found in '{file_path}'.")
    return entries


def _build_dataset(
    name: str,
    entries: Sequence[Tuple[float, float, Optional[float], str]],
    covariance: Optional[np.ndarray],
    reference: str,
    year: int,
) -> BAODataset:
    """Create a BAODataset from parsed entries and covariance information."""
    num_points = len(entries)
    if covariance is not None:
        covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
        if covariance.shape != (num_points, num_points):
            raise ValueError(
                f"Covariance shape {covariance.shape} does not match number of data points {num_points}."
            )
        errors = np.sqrt(np.diag(covariance))
    else:
        errors = np.array(
            [entry[2] if entry[2] is not None else None for entry in entries],
            dtype=float,
        )
        if np.any(np.isnan(errors)):
            raise ValueError(
                f"Covariance not provided and some entries missing errors for dataset '{name}'."
            )
        covariance = np.diag(errors**2)

    data_points: List[BAODataPoint] = []
    for idx, (z, value, error_override, observable) in enumerate(entries):
        error = float(errors[idx])
        data_points.append(
            BAODataPoint(
                z=float(z),
                value=float(value),
                error=error,
                observable=observable,
            )
        )

    return BAODataset(
        name=name,
        data_points=data_points,
        covariance=covariance,
        reference=reference,
        year=year,
    )


import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from ...models import LCDM
from ...models.base import compute_background_grid_for_model


def _rescale_dataset(
    entries: Sequence[Tuple[float, float, Optional[float], str]],
    covariance: np.ndarray,
    scale: Dict[str, float],
) -> Tuple[List[Tuple[float, float, Optional[float], str]], np.ndarray]:
    """Multiply each measurement by scale[observable] and propagate to the covariance."""
    s = np.array([scale.get(obs, 1.0) for _, _, _, obs in entries])
    new_entries = [
        (z, value * f, None if err is None else err * f, obs)
        for (z, value, err, obs), f in zip(entries, s)
    ]
    return new_entries, covariance * np.outer(s, s)


def _load_grid_term(
    path: Path, z: float, observables: Tuple[str, ...], upsample: int = 4
) -> Dict[str, Any]:
    """Tabulated (non-Gaussian) BAO likelihood, as released by eBOSS DR16.

    The file holds the probability on a regular grid of one or two BAO
    observables. As in Cobaya, log(prob) is interpolated with a cubic spline;
    the spline is evaluated here on an ``upsample``-times finer grid that is
    then interpolated linearly inside JAX.
    """
    from scipy.interpolate import RectBivariateSpline, UnivariateSpline

    data = np.loadtxt(path)
    if data.shape[1] == 2:
        x = data[:, 0]
        spline = UnivariateSpline(x, np.log(data[:, 1]), s=0)
        xf = np.linspace(x[0], x[-1], upsample * (len(x) - 1) + 1)
        axes = (xf,)
        logp = spline(xf)
    else:
        x, y = np.unique(data[:, 0]), np.unique(data[:, 1])
        table = np.log(data[:, 2]).reshape(len(x), len(y))
        spline = RectBivariateSpline(x, y, table, kx=3, ky=3)
        xf = np.linspace(x[0], x[-1], upsample * (len(x) - 1) + 1)
        yf = np.linspace(y[0], y[-1], upsample * (len(y) - 1) + 1)
        axes = (xf, yf)
        logp = spline(xf, yf)
    return {
        "z": float(z),
        "observables": observables,
        "origin": [float(a[0]) for a in axes],
        "step": [float(a[1] - a[0]) for a in axes],
        "logp": jnp.asarray(logp, dtype=jnp.float64),
    }


def _grid_terms_loglike(grid_terms, DM_grid, DH_grid, DV_grid, grid_z, rd_val):
    """Sum of the tabulated log-likelihoods (grid_terms is a static list)."""
    total = 0.0
    distances = {"DM_over_rd": DM_grid, "DH_over_rd": DH_grid, "DV_over_rd": DV_grid}
    for term in grid_terms:
        z = jnp.asarray([term["z"]], dtype=grid_z.dtype)
        coords = [
            (interp_linspace(z, grid_z, distances[obs]) / rd_val - o) / d
            for obs, o, d in zip(term["observables"], term["origin"], term["step"])
        ]
        total = total + map_coordinates(term["logp"], coords, order=1, mode="nearest")[0]
    return total


class DESI2024BAO(BAOLikelihood):
    """
    DESI 2024 BAO measurements (Data Release 1).

    Reference: DESI Collaboration 2024 (arXiv:2404.03002)

    Parameters
    ----------
    cosmology_class : type, optional
        Cosmology model class (LCDM, wCDM, etc.). Default is LCDM.
    omega_b_mode : str, default 'h0rd'
        How to handle the r_d calculation (sound horizon at drag epoch):

        * ``'h0rd'`` (default): sample ``H0_rd = H0*r_d/(100 km/s)`` directly
          as a nuisance parameter. This samples the BAO-constrained
          combination, avoiding the need to compute r_d from Omega_b;
          r_d is derived as ``r_d = H0_rd*100/H0``.
        * ``'bbn_prior'``: sample Omega_b with the BBN prior
          (Omega_b*h^2 = 0.02218 ± 0.00055).
        * ``'free'``: sample Omega_b freely (uniform prior 0.03-0.07).
        * ``'fixed'``: use the Planck 2018 value (Omega_b = 0.0493). This
          implicitly breaks the H0-r_d degeneracy (not recommended for
          BAO-only analysis).

    Notes
    -----
    BAO measurements constrain D_M/r_d and D_H/r_d, where r_d depends on Omega_b.
    DESI BAO alone can only constrain H0*r_d (Eq. 4.1 in arXiv:2404.03002).
    The default omega_b_mode='h0rd' directly samples this combination.

    Examples
    --------
    >>> # Default: sample H0_rd directly (recommended for BAO-only)
    >>> bao = BAO_likelihood(LCDM, "desi2024")
    >>>
    >>> # With BBN prior on Omega_b
    >>> bao = BAO_likelihood(LCDM, "desi2024", omega_b_mode='bbn_prior')
    >>>
    >>> # With fixed Omega_b (Planck value)
    >>> bao = BAO_likelihood(LCDM, "desi2024", omega_b_mode='fixed')
    """

    # BBN constraint: Omega_b * h^2 = 0.02218 ± 0.00055 (Schoneberg 2024, as used by DESI 2024)
    BBN_OMEGA_B_H2 = 0.02218
    BBN_OMEGA_B_H2_ERR = 0.00055

    def _default_dataset_name(self) -> str:
        return "desi_2024_dr1"

    def _load_dataset(self) -> BAODataset:
        """Load DESI 2024 DR1 BAO data."""
        base_path = Path(self.data_path) / "desi_2024"
        data_file = base_path / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
        cov_file = base_path / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"

        entries = _read_data_file(data_file)
        covariance = np.loadtxt(cov_file)

        dataset = _build_dataset(
            name="DESI 2024 DR1 BAO",
            entries=entries,
            covariance=covariance,
            reference="DESI Collaboration 2024",
            year=2024,
        )
        return dataset

    def __init__(
        self, cosmology_class: type = None, omega_b_mode: str = "h0rd", **kwargs
    ):
        self._cosmology_class = cosmology_class if cosmology_class is not None else LCDM
        self.omega_b_mode = omega_b_mode.lower()

        if self.omega_b_mode not in ("fixed", "bbn_prior", "free", "h0rd"):
            raise ValueError(
                f"omega_b_mode must be 'fixed', 'bbn_prior', 'free', or 'h0rd', got '{omega_b_mode}'"
            )

        super().__init__(cosmology_class=self._cosmology_class, **kwargs)
        self._setup_jax_metadata()

        # Print mode info
        mode_desc = {
            "fixed": "fixed (Planck 2018)",
            "bbn_prior": "sampled with BBN prior (Omega_b)",
            "free": "free Omega_b (BAO constrains H0*r_d only)",
            "h0rd": "H0_rd as nuisance (prior: 95-110)",
        }
        logger.debug(f"Mode: {mode_desc[self.omega_b_mode]}")

    def _setup_jax_metadata(self):
        values = self.dataset.data_points
        dtype = jnp.asarray(values[0].value, dtype=jnp.float64).dtype
        self._dtype = dtype
        self._z_obs = jnp.array([p.z for p in values], dtype=dtype)
        self._observables = [p.observable for p in values]
        self._obs_masks = {
            obs: jnp.array([o == obs for o in self._observables], dtype=bool)
            for obs in set(self._observables)
        }
        self._grid_terms = self._load_grid_terms()
        z_grid_terms = [t["z"] for t in self._grid_terms]
        self._z_grid_max = max([float(self._z_obs.max()), *z_grid_terms]) + 0.5
        self._z_grid_min = 0.0
        grid_points = int(jnp.clip(self._z_grid_max / 5e-4, 4096, 16384))
        self._n_grid = grid_points

        # Initialize fast likelihood with h0rd mode support
        self._initialize_desi_fast_likelihood()

    def _initialize_desi_fast_likelihood(self) -> None:
        """Build JIT-compiled fast likelihood path with h0rd mode support."""
        from jax import jit

        if not hasattr(self._cosmology_class, "compute_grid_traced"):
            self._loglike_fast = None
            return

        z_grid = jnp.linspace(
            self._z_grid_min, self._z_grid_max, self._n_grid, dtype=self._dtype
        )
        z_obs = self._z_obs
        obs_codes = jnp.array(
            [self._observable_code(p.observable) for p in self.dataset.data_points],
            dtype=jnp.int32,
        )
        data_vec = self._data_vec
        inv_cov = self._inv_cov_jax
        cosmology_class = self._cosmology_class
        omega_b_mode = self.omega_b_mode
        grid_terms = self._grid_terms
        # Planck 2018 best-fit ombh2, used when omega_b_mode='fixed'
        _OMBH2_FIXED = jnp.asarray(0.02237, dtype=self._dtype)
        # BBN prior constants (Schoneberg 2024, as used by DESI 2024), captured for the closures
        bbn_mean = jnp.asarray(self.BBN_OMEGA_B_H2, dtype=self._dtype)
        bbn_err = jnp.asarray(self.BBN_OMEGA_B_H2_ERR, dtype=self._dtype)
        bbn_log_norm = jnp.log(bbn_err * jnp.sqrt(2.0 * jnp.pi))

        def _maybe_add_bbn_prior(loglike, params):
            """Apply the BBN prior exactly once, inside the JIT closures.

            Both the standalone fast path and the shared-grid path go through
            this helper, so the prior can never be double-counted by an outer
            wrapper or silently dropped by the CombinedLikelihood dispatch.
            The branch is static: omega_b_mode and dict-key presence are
            trace-time constants.
            """
            if omega_b_mode == "bbn_prior" and "Omega_b" in params:
                h = params["H0"] / 100.0
                omega_b_h2 = params["Omega_b"] * h * h
                diff = (omega_b_h2 - bbn_mean) / bbn_err
                loglike = loglike - 0.5 * diff * diff - bbn_log_norm
            return loglike

        @jit
        def _loglike_impl(params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            cosmo_grid = compute_background_grid_for_model(
                cosmology_class, z_grid, params
            )
            DM_grid = cosmo_grid["D_M"]
            DH_grid = cosmo_grid["D_H"]
            E_z_grid = cosmo_grid["E_z"]

            volume_factor = z_grid * DM_grid**2 * DH_grid
            DV_grid = jnp.exp((1.0 / 3.0) * jnp.log(volume_factor + 1e-30))
            H_grid = params["H0"] * E_z_grid

            DM_obs = interp_linspace(z_obs, z_grid, DM_grid)
            DH_obs = interp_linspace(z_obs, z_grid, DH_grid)
            DV_obs = interp_linspace(z_obs, z_grid, DV_grid)
            H_obs = interp_linspace(z_obs, z_grid, H_grid)

            # Compute rd based on mode
            if omega_b_mode == "h0rd":
                # rd = H0_rd * 100 / H0
                H0_rd = params.get("H0_rd", jnp.asarray(101.0, dtype=DM_obs.dtype))
                rd_val = H0_rd * 100.0 / params["H0"]
            elif omega_b_mode == "fixed":
                # Fix ombh2 (not Omega_b) to match Cobaya/CAMB convention
                h = params["H0"] / 100.0
                rd_params = {**params, "Omega_b": _OMBH2_FIXED / (h * h)}
                rd_val = cosmology_class.sound_horizon_drag_traced(rd_params)
            else:
                # bbn_prior or free: Omega_b is sampled, use as-is
                rd_val = cosmology_class.sound_horizon_drag_traced(params)

            theory = jnp.zeros_like(z_obs)
            theory = jnp.where(obs_codes == 0, DM_obs / rd_val, theory)  # DM_over_rd
            theory = jnp.where(obs_codes == 1, DH_obs / rd_val, theory)  # DH_over_rd
            theory = jnp.where(obs_codes == 2, DV_obs / rd_val, theory)  # DV_over_rd
            theory = jnp.where(obs_codes == 3, rd_val / DV_obs, theory)  # rs_over_DV
            theory = jnp.where(obs_codes == 4, H_obs * rd_val, theory)  # Hz_rs
            theory = jnp.where(obs_codes == 6, DM_obs / DH_obs, theory)  # DM_over_DH

            diff = theory - data_vec
            chi2 = diff @ (inv_cov @ diff)
            loglike = -0.5 * chi2 + _grid_terms_loglike(
                grid_terms, DM_grid, DH_grid, DV_grid, z_grid, rd_val
            )
            return _maybe_add_bbn_prior(loglike, params)

        self._loglike_fast = _loglike_impl

        # Also build grid-accepting version (for CombinedLikelihood shared grid)
        @jit
        def _loglike_from_grid_impl(
            cosmo_grid: Dict[str, jnp.ndarray],
            grid_z: jnp.ndarray,
            params: Dict[str, jnp.ndarray],
        ) -> jnp.ndarray:
            DM_grid = cosmo_grid["D_M"]
            DH_grid = cosmo_grid["D_H"]
            E_z_g = cosmo_grid["E_z"]

            volume_factor = grid_z * DM_grid**2 * DH_grid
            DV_grid = jnp.exp((1.0 / 3.0) * jnp.log(volume_factor + 1e-30))
            H_grid = params["H0"] * E_z_g

            DM_obs = interp_linspace(z_obs, grid_z, DM_grid)
            DH_obs = interp_linspace(z_obs, grid_z, DH_grid)
            DV_obs = interp_linspace(z_obs, grid_z, DV_grid)
            H_obs = interp_linspace(z_obs, grid_z, H_grid)

            # Same rd logic as _loglike_impl
            if omega_b_mode == "h0rd":
                H0_rd = params.get("H0_rd", jnp.asarray(101.0, dtype=DM_obs.dtype))
                rd_val = H0_rd * 100.0 / params["H0"]
            elif omega_b_mode == "fixed":
                h = params["H0"] / 100.0
                rd_params = {**params, "Omega_b": _OMBH2_FIXED / (h * h)}
                rd_val = cosmology_class.sound_horizon_drag_traced(rd_params)
            else:
                rd_val = cosmology_class.sound_horizon_drag_traced(params)

            theory = jnp.zeros_like(z_obs)
            theory = jnp.where(obs_codes == 0, DM_obs / rd_val, theory)
            theory = jnp.where(obs_codes == 1, DH_obs / rd_val, theory)
            theory = jnp.where(obs_codes == 2, DV_obs / rd_val, theory)
            theory = jnp.where(obs_codes == 3, rd_val / DV_obs, theory)
            theory = jnp.where(obs_codes == 4, H_obs * rd_val, theory)
            theory = jnp.where(obs_codes == 6, DM_obs / DH_obs, theory)

            diff = theory - data_vec
            chi2 = diff @ (inv_cov @ diff)
            loglike = -0.5 * chi2 + _grid_terms_loglike(
                grid_terms, DM_grid, DH_grid, DV_grid, grid_z, rd_val
            )
            return _maybe_add_bbn_prior(loglike, params)

        self._loglike_from_grid = _loglike_from_grid_impl

    def _load_grid_terms(self) -> List[Dict[str, Any]]:
        """Tabulated non-Gaussian components; none for the DESI releases."""
        return []

    @staticmethod
    def _observable_code(observable: str) -> int:
        """Map observable string to integer code for fast vectorized selection."""
        codes = {
            "DM_over_rd": 0,
            "DH_over_rd": 1,
            "DV_over_rd": 2,
            "rs_over_DV": 3,
            "Hz_rs": 4,
            "fsigma8": 5,
            "DM_over_DH": 6,
        }
        return codes.get(observable, -1)

    def _extract_cosmo_params(self, cosmology) -> Dict[str, jnp.ndarray]:
        """Convert model.params to JAX-compatible dict (no filtering, just type conversion)."""
        params_dict = cosmology.params.to_dict()
        dtype = self._dtype
        return {
            k: jnp.asarray(v, dtype=dtype)
            for k, v in params_dict.items()
            if isinstance(v, (int, float, jnp.ndarray, np.ndarray))
        }

    def _compute_rd(self, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Compute sound horizon.

        In 'h0rd' mode: rd = H0_rd * 100 / H0
        Otherwise: use CAMB for high precision (~0.1% accuracy).
        """
        if self.omega_b_mode == "h0rd":
            # H0_rd = H0 * r_d / 100 => r_d = H0_rd * 100 / H0
            H0_rd = params.get("H0_rd", jnp.asarray(101.0, dtype=self._dtype))
            H0 = params["H0"]
            return H0_rd * 100.0 / H0

        # Create model instance to use CAMB-based rd calculation
        # This is more accurate than the traced EH98 version
        model = self._cosmology_class(
            **{k: float(v) if hasattr(v, "__float__") else v for k, v in params.items()}
        )
        return jnp.asarray(model.sound_horizon_drag(use_camb=True), dtype=self._dtype)

    def theory(self, cosmology, rd: Optional[float] = None, **kwargs) -> jnp.ndarray:
        """
        Compute BAO theory predictions using cosmology model's compute_grid_traced.

        All distance calculations (D_M, D_H, etc.) are delegated to the cosmology
        model layer, ensuring proper separation of concerns.
        """
        cosmo_params = self._extract_cosmo_params(cosmology)

        # In h0rd mode, get H0_rd from kwargs (passed from MCMC sampler)
        if self.omega_b_mode == "h0rd" and "H0_rd" in kwargs:
            cosmo_params["H0_rd"] = jnp.asarray(kwargs["H0_rd"], dtype=self._dtype)

        z_grid = jnp.linspace(
            self._z_grid_min, self._z_grid_max, self._n_grid, dtype=self._dtype
        )

        # Get background distances from cosmology model (curvature handled there!)
        grid = compute_background_grid_for_model(
            self._cosmology_class, z_grid, cosmo_params
        )

        # Use D_M and D_H directly from model (no duplicate calculation!)
        DM_grid = grid["D_M"]  # Transverse comoving distance
        DH_grid = grid["D_H"]  # Hubble distance c/H(z)
        E_z_grid = grid["E_z"]

        # Compute volume-averaged distance DV = (z * DM^2 * DH)^(1/3)
        volume_factor = z_grid * DM_grid**2 * DH_grid
        DV_grid = jnp.exp((1.0 / 3.0) * jnp.log(volume_factor + 1e-30))
        H_grid = cosmo_params["H0"] * E_z_grid

        # Interpolate to observed redshifts
        DM_obs = interp_linspace(self._z_obs, z_grid, DM_grid)
        DH_obs = interp_linspace(self._z_obs, z_grid, DH_grid)
        DV_obs = interp_linspace(self._z_obs, z_grid, DV_grid)
        H_obs = interp_linspace(self._z_obs, z_grid, H_grid)

        # Compute sound horizon
        rd_val = (
            self._compute_rd(cosmo_params)
            if rd is None
            else jnp.asarray(rd, dtype=self._dtype)
        )

        # Build theory vector
        theory = jnp.zeros_like(self._z_obs, dtype=self._dtype)

        def apply_obs(theory_vec, mask, values):
            return jnp.where(mask, values, theory_vec)

        for obs_name, mask in self._obs_masks.items():
            if obs_name == "DM_over_rd":
                theory = apply_obs(theory, mask, DM_obs / rd_val)
            elif obs_name == "DH_over_rd":
                theory = apply_obs(theory, mask, DH_obs / rd_val)
            elif obs_name == "DV_over_rd":
                theory = apply_obs(theory, mask, DV_obs / rd_val)
            elif obs_name == "rs_over_DV":
                theory = apply_obs(theory, mask, rd_val / DV_obs)
            elif obs_name == "Hz_rs":
                theory = apply_obs(theory, mask, H_obs * rd_val)
            elif obs_name == "DM_over_DH":
                theory = apply_obs(theory, mask, DM_obs / DH_obs)
            else:
                # Fallback to base implementation for unsupported observables
                return super().theory(cosmology, rd=float(rd_val))

        return theory

    @property
    def nuisance_parameters(self):
        """
        Return nuisance parameters based on omega_b_mode.

        Returns
        -------
        List[Parameter]
            Nuisance parameters depend on omega_b_mode:
            - 'h0rd': H0_rd (BAO-constrained combination)
            - 'bbn_prior': Omega_b with BBN prior in likelihood
            - 'free': Omega_b with uniform prior
            - 'fixed': No nuisance (Omega_b fixed to Planck value)
        """
        from ...parameters import Parameter

        if self.omega_b_mode == "h0rd":
            # H0_rd = H0 * r_d / 100 in units of [100 km/s]
            # Typical: H0=70, r_d=147 Mpc => H0_rd = 102.9
            return NuisanceList(
                [
                    Parameter(
                        name="H0_rd",
                        value=102.5,
                        free=True,
                        prior={"dist": "uniform", "min": 95.0, "max": 110.0},
                        latex_label=r"$H_0 r_{\rm d} / (100\,{\rm km\,s^{-1}})$",
                        description="H0*r_d in units of 100 km/s",
                    )
                ]
            )

        elif self.omega_b_mode == "bbn_prior":
            # Sample Omega_b, BBN prior applied in likelihood
            # BBN: Omega_b * h^2 = 0.02218 ± 0.00055
            # For h=0.67: Omega_b ≈ 0.0494
            return NuisanceList(
                [
                    Parameter(
                        name="Omega_b",
                        value=0.0493,
                        free=True,
                        prior={"dist": "uniform", "min": 0.03, "max": 0.07},
                        latex_label=r"$\Omega_{\rm b}$",
                        description="Baryon density (BBN prior in likelihood)",
                    )
                ]
            )

        elif self.omega_b_mode == "free":
            # Sample Omega_b freely without BBN prior
            return NuisanceList(
                [
                    Parameter(
                        name="Omega_b",
                        value=0.0493,
                        free=True,
                        prior={"dist": "uniform", "min": 0.03, "max": 0.07},
                        latex_label=r"$\Omega_{\rm b}$",
                        description="Baryon density (free)",
                    )
                ]
            )

        else:  # 'fixed'
            # Omega_b fixed to Planck 2018 value, no nuisance needed
            return NuisanceList([])

    # NOTE: the BBN prior for the fast paths lives INSIDE the JIT closures
    # (see _maybe_add_bbn_prior in _initialize_desi_fast_likelihood), so the
    # params-dict path, __call__ and the shared-grid path all count it exactly
    # once. Only the cosmology-object path below adds it separately, because
    # that path bypasses the closures entirely.

    def log_likelihood(self, cosmology, **kwargs) -> float:
        """Log-likelihood with optional BBN prior on Omega_b h^2."""
        if self._grid_terms:
            # Tabulated terms are evaluated only in the JIT path (which already
            # includes the BBN prior)
            params = self._prepare_params_dict(self._extract_cosmo_params(cosmology))
            return self._loglike_fast(params)
        loglike = super().log_likelihood(cosmology, **kwargs)
        if self.omega_b_mode == "bbn_prior":
            H0 = jnp.asarray(cosmology.params["H0"], dtype=self._dtype)
            Omega_b = jnp.asarray(cosmology.params["Omega_b"], dtype=self._dtype)
            h = H0 / jnp.asarray(100.0, dtype=self._dtype)
            omega_b_h2 = Omega_b * h * h
            diff = (omega_b_h2 - self.BBN_OMEGA_B_H2) / self.BBN_OMEGA_B_H2_ERR
            norm = self.BBN_OMEGA_B_H2_ERR * jnp.sqrt(2.0 * jnp.pi)
            loglike = loglike + (-0.5 * diff * diff - jnp.log(norm))
        return loglike

    def derived_parameters(self, **params) -> dict:
        """
        Compute derived parameters from MCMC samples.

        Parameters
        ----------
        **params : dict
            Sampled cosmological parameters including H0_rd.

        Returns
        -------
        dict
            Derived parameters including:
            - rd: Sound horizon at drag epoch [Mpc]
            - rd_h: r_d * h = r_d * H0/100 [Mpc]
            - H0_rd: H0 * r_d [km/s] (the BAO-constrained combination)
        """
        H0 = params.get("H0", 70.0)
        h = H0 / 100.0

        # rd = H0_rd * 100 / H0
        H0_rd = params.get("H0_rd", 101.0)
        rd = H0_rd * 100.0 / H0

        return {
            "rd": rd,
            "rd_h": rd * h,
            "H0_rd": H0 * rd,
        }

    def derived_parameters_vectorized(self, samples: dict) -> dict:
        """Compute BAO derived parameters for a full sample array at once."""
        import numpy as np

        H0 = np.asarray(samples.get("H0", 70.0), dtype=float)
        H0_rd = np.asarray(samples.get("H0_rd", 101.0), dtype=float)
        rd = H0_rd * 100.0 / H0
        return {
            "rd": rd,
            "rd_h": rd * H0 / 100.0,
            "H0_rd": H0 * rd,
        }

    def __call__(self, **params) -> float:
        """
        Callable interface for MCMC sampling.

        Automatically constructs cosmology model from parameters and computes
        log-likelihood. This enables the simplified API:

            bao = BAO_likelihood(LCDM, "desi2024")
            MCMC(params, bao, chain_name='lcdm_bao').run(...)

        Parameters
        ----------
        **params : dict
            Cosmological parameters (H0, Omega_m, etc.) passed to cosmology_class.
            All modes use H0_rd as nuisance parameter.

        Returns
        -------
        float
            Log-likelihood value.

        Notes
        -----
        - When omega_b_mode='bbn_prior', the BBN prior on Omega_b*h^2 is applied
          as an additional log-likelihood term.
        - When omega_b_mode='h0rd', r_d is derived from H0_rd: r_d = H0_rd * 100 / H0.
        """
        # Fast path already includes the BBN prior inside its JIT closure;
        # adding it here again would double-count it (previous bug).
        return self.log_likelihood_from_params(params)


class DESIDR2BAO(DESI2024BAO):
    """
    DESI DR2 BAO measurements (Data Release 2 / Year 3).

    Reference: DESI Collaboration 2025 (arXiv:2503.14738)

    This class inherits all functionality from DESI2024BAO (DR1), including:
    - All omega_b_mode options ('h0rd', 'bbn_prior', 'free', 'fixed')
    - JAX JIT-compiled fast likelihood paths
    - Shared grid support for CombinedLikelihood
    - BBN prior on Omega_b h^2

    The only difference is the data: DR2 has 13 measurements across 7 redshift
    bins (vs DR1's 12 measurements across 7 bins). The QSO bin (z_eff=1.484)
    now provides separate DM/rd and DH/rd instead of a single DV/rd.

    Parameters
    ----------
    cosmology_class : type, optional
        Cosmology model class (LCDM, wCDM, etc.). Default is LCDM.
    omega_b_mode : str, default 'h0rd'
        How to handle the r_d calculation. See DESI2024BAO for details.

    Examples
    --------
    >>> from hicosmo.likelihoods import BAO_likelihood
    >>> from hicosmo.models import LCDM
    >>> bao = BAO_likelihood(LCDM, "desi_dr2")
    >>>
    >>> # With BBN prior
    >>> bao = BAO_likelihood(LCDM, "desi_dr2", omega_b_mode='bbn_prior')
    """

    def _default_dataset_name(self) -> str:
        return "desi_dr2"

    def _load_dataset(self) -> BAODataset:
        """Load DESI DR2 BAO data."""
        base_path = Path(self.data_path) / "desi_dr2"
        data_file = base_path / "desi_dr2_gaussian_bao_ALL_GCcomb_mean.txt"
        cov_file = base_path / "desi_dr2_gaussian_bao_ALL_GCcomb_cov.txt"

        entries = _read_data_file(data_file)
        covariance = np.loadtxt(cov_file)

        dataset = _build_dataset(
            name="DESI DR2 BAO",
            entries=entries,
            covariance=covariance,
            reference="DESI Collaboration 2025",
            year=2025,
        )
        return dataset


class SDSSDR12BAO(DESI2024BAO):
    """
    BOSS DR12 consensus BAO (Alam et al. 2017), three bins z = 0.38, 0.51, 0.61.

    The release quotes D_M r_d,fid / r_d [Mpc] and H r_d / r_d,fid [km/s/Mpc]
    with r_d,fid = 147.78 Mpc; they are converted on load to D_M/r_d and
    H r_d [km/s], with the covariance transformed accordingly. Shares the
    r_d / omega_b treatment of DESI2024BAO. Overlaps with ``boss_dr12`` and
    ``sdss_dr16`` (same galaxies), see CombinedLikelihood.
    """

    RD_FID = 147.78  # Mpc

    def _default_dataset_name(self) -> str:
        return "sdss_dr12_consensus"

    def _load_dataset(self) -> BAODataset:
        base_path = Path(self.data_path) / "sdss_dr12"
        entries = _read_data_file(base_path / "sdss_DR12Consensus_bao.dat")
        covariance = np.loadtxt(base_path / "BAO_consensus_covtot_dM_Hz.txt")
        entries, covariance = _rescale_dataset(
            entries, covariance, {"DM_over_rd": 1.0 / self.RD_FID, "Hz_rs": self.RD_FID}
        )
        return _build_dataset(
            name="BOSS DR12 consensus BAO",
            entries=entries,
            covariance=covariance,
            reference="Alam et al. 2017, MNRAS 470, 2617",
            year=2017,
        )


class BOSSDR12BAO(DESI2024BAO):
    """
    BOSS DR12 BAO at z = 0.38, 0.51 (D_M/r_d, D_H/r_d), as re-released with
    eBOSS DR16 (Alam et al. 2021) for combination with the DR16 samples.
    """

    def _default_dataset_name(self) -> str:
        return "boss_dr12"

    def _load_dataset(self) -> BAODataset:
        base_path = Path(self.data_path) / "boss_dr12"
        entries = _read_data_file(base_path / "sdss_DR12_LRG_BAO_DMDH.dat")
        covariance = np.loadtxt(base_path / "sdss_DR12_LRG_BAO_DMDH_covtot.txt")
        return _build_dataset(
            name="BOSS DR12 BAO (z=0.38, 0.51)",
            entries=entries,
            covariance=covariance,
            reference="Alam et al. 2017; Alam et al. 2021",
            year=2021,
        )


class SDSSDR16BAO(DESI2024BAO):
    """
    eBOSS DR16 BAO-only measurements (Alam et al. 2021):

    * LRG, z = 0.698: D_M/r_d, D_H/r_d (Gaussian, full covariance)
    * QSO, z = 1.48:  D_M/r_d, D_H/r_d (Gaussian, full covariance)
    * ELG, z = 0.845: D_V/r_d (tabulated likelihood)
    * Lya auto and Lya x QSO, z = 2.334: D_M/r_d, D_H/r_d (tabulated likelihoods)

    Designed to be combined with ``boss_dr12`` (z = 0.38, 0.51), as in the
    eBOSS DR16 cosmology analysis.
    """

    def _default_dataset_name(self) -> str:
        return "sdss_dr16_bao"

    def _load_dataset(self) -> BAODataset:
        base_path = Path(self.data_path) / "sdss_dr16"
        lrg = _read_data_file(base_path / "sdss_DR16_LRG_BAO_DMDH.dat")
        qso = _read_data_file(base_path / "sdss_DR16_QSO_BAO_DMDH.txt")
        cov_lrg = np.loadtxt(base_path / "sdss_DR16_LRG_BAO_DMDH_covtot.txt")
        cov_qso = np.loadtxt(base_path / "sdss_DR16_QSO_BAO_DMDH_covtot.txt")
        covariance = np.zeros((4, 4))
        covariance[:2, :2] = cov_lrg
        covariance[2:, 2:] = cov_qso
        return _build_dataset(
            name="eBOSS DR16 BAO",
            entries=[*lrg, *qso],
            covariance=covariance,
            reference="Alam et al. 2021, PRD 103, 083533",
            year=2021,
        )

    def _load_grid_terms(self) -> List[Dict[str, Any]]:
        base_path = Path(self.data_path) / "sdss_dr16"
        return [
            _load_grid_term(
                base_path / "sdss_DR16_ELG_BAO_DVtable.txt", 0.845, ("DV_over_rd",)
            ),
            _load_grid_term(
                base_path / "sdss_DR16_LYAUTO_BAO_DMDHgrid.txt",
                2.334,
                ("DM_over_rd", "DH_over_rd"),
            ),
            _load_grid_term(
                base_path / "sdss_DR16_LYxQSO_BAO_DMDHgrid.txt",
                2.334,
                ("DM_over_rd", "DH_over_rd"),
            ),
        ]


class SixDFBAO(DESI2024BAO):
    """
    6dF Galaxy Survey BAO, r_s/D_V(z=0.106) = 0.336 +/- 0.015 (Beutler et al. 2011).

    The measurement uses the Eisenstein-Hu r_s convention; as in Cobaya it is
    mapped to the CAMB-like r_d with r_s/r_d = 153.9/149.8, applied here to
    the data so that the theory is r_d/D_V.
    """

    RS_RESCALE = 153.9 / 149.8

    def _default_dataset_name(self) -> str:
        return "sixdf_2011"

    def _load_dataset(self) -> BAODataset:
        entries = _read_data_file(Path(self.data_path) / "sixdf" / "sixdf_bao_data.txt")
        errors = np.array([e[2] for e in entries])
        entries, covariance = _rescale_dataset(
            entries, np.diag(errors**2), {"rs_over_DV": 1.0 / self.RS_RESCALE}
        )
        return _build_dataset(
            name="6dF Galaxy Survey",
            entries=entries,
            covariance=covariance,
            reference="Beutler et al. 2011, MNRAS 416, 3017",
            year=2011,
        )


class CustomBAO(BAOLikelihood):
    """
    Custom BAO dataset loaded from user-provided files.

    Expects YAML configuration file with data points and covariance.
    """

    def __init__(self, config_file: str, **kwargs):
        """
        Initialize custom BAO dataset.

        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file
        super().__init__(**kwargs)

    def _default_dataset_name(self) -> str:
        return "custom_bao"

    def _load_dataset(self) -> BAODataset:
        """Load custom BAO data from configuration file."""
        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)

        # Parse data points
        data_points = []
        for point in config["data_points"]:
            data_points.append(
                BAODataPoint(
                    z=point["z"],
                    value=point["value"],
                    error=point.get("error", 0.0),
                    observable=point["observable"],
                )
            )

        # Load covariance if provided
        covariance = None
        if "covariance_file" in config:
            cov_path = os.path.join(
                os.path.dirname(self.config_file), config["covariance_file"]
            )
            covariance = np.loadtxt(cov_path)
        elif "covariance" in config:
            covariance = np.array(config["covariance"])
        else:
            # Use diagonal errors
            errors = np.array([p.error for p in data_points])
            if np.any(errors > 0):
                covariance = np.diag(errors**2)

        return BAODataset(
            name=config.get("name", "Custom BAO Dataset"),
            data_points=data_points,
            covariance=covariance,
            reference=config.get("reference", ""),
            year=config.get("year", 0),
        )


def get_available_datasets() -> List[str]:
    """Get list of available BAO datasets."""
    return [
        "sdss_dr12",
        "sdss_dr16",
        "boss_dr12",
        "desi_2024",
        "desi_dr2",
        "sixdf",
    ]


def create_bao_likelihood(dataset: str, **kwargs) -> BAOLikelihood:
    """
    Factory function to create BAO likelihood from dataset name.

    Args:
        dataset: Name of BAO dataset
        **kwargs: Additional arguments for likelihood

    Returns:
        BAOLikelihood object
    """
    dataset_map = {
        "sdss_dr12": SDSSDR12BAO,
        "sdss_dr16": SDSSDR16BAO,
        "boss_dr12": BOSSDR12BAO,
        "desi_2024": DESI2024BAO,
        "desi_dr2": DESIDR2BAO,
        "sixdf": SixDFBAO,
    }

    if dataset not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {dataset}. " f"Available: {list(dataset_map.keys())}"
        )

    return dataset_map[dataset](**kwargs)
