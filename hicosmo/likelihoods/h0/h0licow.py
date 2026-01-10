"""
H0LiCOW Strong-Lensing Time-Delay Likelihood
=============================================

Clean implementation of H0LiCOW (2019) strong-lensing likelihood.
Following CLAUDE.md design principles: minimal, efficient, and JAX-optimized.

Data source: Wong et al. 2019, MNRAS, 498, 1420
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import logsumexp

from ...models.lcdm import LCDM
from ...utils.logging import get_logger
from ..base import NuisanceList

logger = get_logger(__name__)


def _get_default_data_path() -> Path:
    """Get default data path relative to package location."""
    package_dir = Path(__file__).parent
    # Data is in hicosmo/data/h0licow
    # Go up 2 levels: h0/ -> likelihoods/ -> hicosmo/
    return package_dir.parent.parent / "data" / "h0licow"


@dataclass
class LensConfig:
    """Configuration for a single lens system."""
    name: str
    zlens: float
    zsource: float
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    data_file: Optional[str] = None
    read_csv_kwargs: Optional[Dict[str, Any]] = None
    columns: Optional[Dict[str, str]] = None
    bandwidth: Optional[float] = None
    nbins: Optional[int] = None
    explim: float = 100.0
    max_ddt: Optional[float] = None


@jit
def _skewed_lognormal_logpdf(value: jnp.ndarray, mu: float, sigma: float,
                              lam: float, explim: float) -> jnp.ndarray:
    """Skewed log-normal probability density function."""
    value = jnp.asarray(value, dtype=jnp.float64)
    shifted = value - lam
    valid = shifted > 0.0
    safe_log = jnp.where(valid, jnp.log(shifted), 0.0)
    exponent = -0.5 * ((safe_log - mu) / sigma) ** 2
    valid = jnp.logical_and(valid, (-exponent) <= explim)
    log_norm = jnp.log(jnp.sqrt(2.0 * jnp.pi) * sigma) + jnp.where(valid, jnp.log(shifted), 0.0)
    logpdf = exponent - log_norm
    return jnp.where(valid, logpdf, -jnp.inf)


def _build_kde_logpdf_1d(points: jnp.ndarray, log_weights: jnp.ndarray,
                         log_weight_sum: jnp.ndarray, bandwidth: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build 1D KDE log-PDF evaluator."""
    scalar_points = jnp.asarray(points[:, 0], dtype=jnp.float64)
    bw = jnp.asarray(bandwidth, dtype=jnp.float64)
    log_norm = jnp.log(jnp.sqrt(2.0 * jnp.pi) * bw)

    def _single(value: jnp.ndarray) -> jnp.ndarray:
        diff = (value - scalar_points) / bw
        log_kernel = -0.5 * diff**2 - log_norm
        return logsumexp(log_weights + log_kernel) - log_weight_sum

    batched = jit(vmap(_single))

    def evaluate(values: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.atleast_1d(jnp.asarray(values, dtype=jnp.float64))
        return batched(arr)

    return evaluate


def _build_kde_logpdf_2d(points: jnp.ndarray, log_weights: jnp.ndarray,
                         log_weight_sum: jnp.ndarray, bandwidth: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build 2D KDE log-PDF evaluator."""
    pts = jnp.asarray(points, dtype=jnp.float64)
    bw = jnp.asarray(bandwidth, dtype=jnp.float64)
    log_norm = jnp.log(2.0 * jnp.pi * bw**2)

    def _single(vec: jnp.ndarray) -> jnp.ndarray:
        diff = (vec - pts) / bw
        quad = jnp.sum(diff**2, axis=1)
        log_kernel = -0.5 * quad - log_norm
        return logsumexp(log_weights + log_kernel) - log_weight_sum

    batched = jit(vmap(_single))

    def evaluate(vectors: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(vectors, dtype=jnp.float64)
        arr = jnp.atleast_2d(arr)
        return batched(arr)

    return evaluate


class KDEEstimator:
    """Simple isotropic Gaussian KDE helper implemented with JAX."""

    def __init__(self, points: np.ndarray, weights: Optional[np.ndarray],
                 bandwidth: float) -> None:
        if weights is None:
            weights = np.ones(points.shape[0], dtype=np.float64)
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[:, None]
        self.points = jnp.asarray(pts, dtype=jnp.float64)
        self.weights = jnp.asarray(weights, dtype=jnp.float64)
        self.bandwidth = float(bandwidth)
        self.log_weights = jnp.log(self.weights)
        self.log_weight_sum = jnp.log(jnp.sum(self.weights))
        self._logpdf_1d_fn = None
        self._logpdf_2d_fn = None
        if self.points.shape[1] == 1:
            self._logpdf_1d_fn = _build_kde_logpdf_1d(
                self.points, self.log_weights, self.log_weight_sum, self.bandwidth)
        elif self.points.shape[1] == 2:
            self._logpdf_2d_fn = _build_kde_logpdf_2d(
                self.points, self.log_weights, self.log_weight_sum, self.bandwidth)

    def logpdf_1d(self, value: float) -> jnp.ndarray:
        if self._logpdf_1d_fn is None:
            raise RuntimeError("KDEEstimator is not initialised for 1D evaluations.")
        value = jnp.asarray(value, dtype=jnp.float64)
        result = self._logpdf_1d_fn(value)
        return result[0] if result.shape == (1,) else result

    def logpdf_2d(self, vector: np.ndarray) -> jnp.ndarray:
        if self._logpdf_2d_fn is None:
            raise RuntimeError("KDEEstimator is not initialised for 2D evaluations.")
        vec = jnp.asarray(vector, dtype=jnp.float64)
        result = self._logpdf_2d_fn(vec)
        return result[0] if result.shape == (1,) else result


def _compress_histogram(samples: np.ndarray, weights: np.ndarray,
                        nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compress samples into histogram for efficient KDE."""
    hist, edges = np.histogram(samples, bins=nbins, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = hist > 0
    return centers[mask], hist[mask]


def _compress_histogram_2d(dd: np.ndarray, ddt: np.ndarray, weights: np.ndarray,
                           nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compress 2D samples into histogram for efficient KDE."""
    hist, x_edges, y_edges = np.histogram2d(dd, ddt, bins=nbins, weights=weights)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    flattened = hist.ravel()
    mask = flattened > 0
    points = np.column_stack([grid_x.ravel()[mask], grid_y.ravel()[mask]])
    return points, flattened[mask]


class H0LiCOWLens:
    """Container for a single lens likelihood component."""

    def __init__(self, config: LensConfig, data_directory: Path) -> None:
        self.config = config
        self.name = config.name
        self.zlens = config.zlens
        self.zsource = config.zsource
        self.kind = config.kind
        self.explim = config.explim
        self._kde: Optional[KDEEstimator] = None
        self._uses_dd = False
        self._prepare(data_directory)

    @property
    def uses_dd(self) -> bool:
        return self._uses_dd

    def _prepare(self, root: Path) -> None:
        kind = self.kind
        if kind in {"skewed_lognormal", "gaussian"}:
            return
        if kind in {"skewed_lognormal_dd", "skewed_lognormal_dd_only"}:
            self._uses_dd = True
            return
        if not self.config.data_file:
            raise ValueError(f"Lens {self.name} requires data file for kind {kind}.")
        path = root / self.config.data_file
        if not path.exists():
            raise FileNotFoundError(path)

        cols = self.config.columns or {}
        bw = self.config.bandwidth or 20.0

        if kind == "kde_hist_1d":
            weights_col = cols.get("weight")
            if weights_col:
                kwargs = dict(self.config.read_csv_kwargs or {})
                kwargs.setdefault("header", 0)
                df = pd.read_csv(path, **kwargs)
                samples = df[cols.get("ddt", df.columns[0])].to_numpy(dtype=np.float64)
                weights = df[weights_col].to_numpy(dtype=np.float64)
            else:
                samples = np.loadtxt(path, dtype=np.float64, comments=cols.get("comment", "#"))
                if samples.ndim > 1:
                    samples = samples[:, 0]
                weights = np.ones_like(samples)
            if self.config.max_ddt is not None:
                mask = (samples > 0) & (samples < self.config.max_ddt)
                samples = samples[mask]
                weights = weights[mask]
            centers, hist_weights = _compress_histogram(samples, weights, self.config.nbins or 200)
            self._kde = KDEEstimator(centers.reshape(-1, 1), hist_weights, bw)
        elif kind in {"kde_hist_2d", "kde_full_2d"}:
            if kind == "kde_full_2d":
                kwargs = dict(self.config.read_csv_kwargs or {})
                kwargs.setdefault("header", 0)
                df = pd.read_csv(path, **kwargs)
                points = df[[cols.get("dd", "dd"), cols.get("ddt", "ddt")]].dropna().to_numpy(dtype=np.float64)
                weights = np.ones(points.shape[0], dtype=np.float64)
            else:
                weights_col = cols.get("weight")
                if weights_col:
                    kwargs = dict(self.config.read_csv_kwargs or {})
                    kwargs.setdefault("header", 0)
                    df = pd.read_csv(path, **kwargs)
                    dd = df[cols.get("dd", df.columns[0])].to_numpy(dtype=np.float64)
                    ddt = df[cols.get("ddt", df.columns[1])].to_numpy(dtype=np.float64)
                    weights = df[weights_col].to_numpy(dtype=np.float64)
                else:
                    data = np.loadtxt(path, dtype=np.float64)
                    dd = data[:, 0]
                    ddt = data[:, 1]
                    weights = np.ones_like(dd)
                points, weights = _compress_histogram_2d(dd, ddt, weights, self.config.nbins or 80)
            self._kde = KDEEstimator(points, weights, bw)
            self._uses_dd = True
        else:
            raise ValueError(f"Unsupported lens kind '{kind}'.")

    def log_likelihood(self, dd: jnp.ndarray, ddt: jnp.ndarray) -> jnp.ndarray:
        kind = self.kind
        params = self.config.params
        if kind == "skewed_lognormal":
            return _skewed_lognormal_logpdf(ddt, params["mu"], params["sigma"], params["lam"], self.explim)
        if kind == "skewed_lognormal_dd":
            ll_dt = _skewed_lognormal_logpdf(ddt, params["mu"], params["sigma"], params["lam"], self.explim)
            ll_dd = _skewed_lognormal_logpdf(dd, params["mu_dd"], params["sigma_dd"], params["lam_dd"], self.explim)
            return ll_dt + ll_dd
        if kind == "skewed_lognormal_dd_only":
            return _skewed_lognormal_logpdf(dd, params["mu_dd"], params["sigma_dd"], params["lam_dd"], self.explim)
        if kind == "gaussian":
            residual = (ddt - params["mu"]) / params["sigma"]
            return -0.5 * residual**2 - jnp.log(jnp.sqrt(2.0 * jnp.pi) * params["sigma"])
        if self._kde is None:
            raise RuntimeError(f"Lens {self.name} was not initialised correctly.")
        if kind == "kde_hist_1d":
            return self._kde.logpdf_1d(ddt)
        if kind in {"kde_hist_2d", "kde_full_2d"}:
            vec = jnp.stack([dd.reshape(()), ddt.reshape(())])
            return self._kde.logpdf_2d(vec)
        raise ValueError(f"Unsupported lens kind '{kind}'.")


class H0LiCOWLikelihood:
    """
    H0LiCOW strong-lensing time-delay likelihood.

    Clean implementation following same API as SN_likelihood:
    - Direct callable interface: likelihood(**params)
    - Support for likelihood combination: sne + h0licow
    - Automatic cosmology model construction

    Parameters
    ----------
    data_path : str or Path, optional
        Path to H0LiCOW data directory. If None, uses default package location.
    lens_names : list of str, optional
        Subset of lenses to use. If None, uses all available lenses.
    cosmology_class : type, optional
        Cosmology model class (default: LCDM).

    Examples
    --------
    >>> # Simple usage
    >>> h0licow = H0LiCOWLikelihood()
    >>> log_L = h0licow(H0=70.0, Omega_m=0.3)

    >>> # Combined with other likelihoods
    >>> from hicosmo.likelihoods import SN_likelihood
    >>> joint = SN_likelihood(LCDM, "pantheon+") + H0LiCOWLikelihood()
    >>> MCMC(params, joint, chain_name='joint').run(...)
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        lens_names: Optional[Iterable[str]] = None,
        cosmology_class: type = None,
        verbose: bool = True,
    ) -> None:
        # Resolve data path
        if data_path is None:
            self.data_path = _get_default_data_path()
        else:
            self.data_path = Path(data_path)

        self.lens_names = set(lens_names) if lens_names else None
        self._cosmology_class = cosmology_class if cosmology_class is not None else LCDM
        self.verbose = verbose

        # Load data
        self._load_data()
        self._setup_traced_helpers()
        self._lens_evaluator = self._build_lens_evaluator()

        # Output summary (matching Pantheon style)
        if self.verbose:
            logger.info(f"H0LiCOW loaded: {len(self.lenses)} lenses")
            lens_list = ", ".join(lens.name for lens in self.lenses)
            logger.info(f"Lenses: {lens_list}")
            z_range = (min(lens.zlens for lens in self.lenses),
                       max(lens.zsource for lens in self.lenses))
            logger.info(f"Redshift range: {z_range[0]:.3f} - {z_range[1]:.3f}")

    def _load_data(self) -> None:
        """Load lens configurations and data."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"H0LiCOW data directory not found: {self.data_path}")

        configs = self._get_default_lens_configs()
        selected = [cfg for cfg in configs if self.lens_names is None or cfg.name in self.lens_names]

        if not selected:
            available = ", ".join(cfg.name for cfg in configs)
            raise ValueError(f"No H0LiCOW lenses selected. Available: {available}")

        self.lenses = [H0LiCOWLens(cfg, self.data_path) for cfg in selected]

    def _setup_traced_helpers(self) -> None:
        """Set up JAX-traced distance computation."""
        z_lenses = np.array([lens.zlens for lens in self.lenses], dtype=np.float64)
        z_sources = np.array([lens.zsource for lens in self.lenses], dtype=np.float64)
        self._z_lenses = jnp.asarray(z_lenses, dtype=jnp.float64)
        self._z_sources = jnp.asarray(z_sources, dtype=jnp.float64)

        z_max = float(np.max(z_sources))
        self._z_grid = jnp.linspace(0.0, z_max + 0.5, 4096, dtype=jnp.float64)
        self._distance_interp = self._build_traced_distance_fn()

    def _build_traced_distance_fn(self):
        """Build JIT-compiled distance computation function."""
        z_grid = self._z_grid
        z_l = self._z_lenses
        z_s = self._z_sources
        cosmology_class = self._cosmology_class

        def compute_distances(params: Dict[str, float]):
            cosmo = cosmology_class.compute_grid_traced(z_grid, params)
            d_c_grid = cosmo['d_C']
            d_c_l = jnp.interp(z_l, z_grid, d_c_grid)
            d_c_s = jnp.interp(z_s, z_grid, d_c_grid)
            dd = d_c_l / (1.0 + z_l)
            ds = d_c_s / (1.0 + z_s)
            dds = (d_c_s - d_c_l) / (1.0 + z_s)
            ddt = (1.0 + z_l) * dd * ds / dds
            return dd, ddt

        return jax.jit(compute_distances)

    def _build_lens_evaluator(self):
        """Build JIT-compiled lens log-likelihood evaluator."""
        lens_functions = tuple(lens.log_likelihood for lens in self.lenses)

        def lens_sum(dd_array: jnp.ndarray, ddt_array: jnp.ndarray) -> jnp.ndarray:
            idx = jnp.arange(dd_array.shape[0], dtype=jnp.int32)

            def _single(i, dd, ddt):
                return jax.lax.switch(i, lens_functions, dd, ddt)

            return jnp.sum(vmap(_single)(idx, dd_array, ddt_array))

        return jax.jit(lens_sum)

    def __call__(self, **params) -> float:
        """
        Callable interface for MCMC sampling.

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
        >>> h0licow = H0LiCOWLikelihood()
        >>> log_L = h0licow(H0=70.0, Omega_m=0.3)
        """
        # Ensure required parameters exist with defaults
        cosmo_params = {
            'H0': params.get('H0'),
            'Omega_m': params.get('Omega_m'),
            'Omega_k': params.get('Omega_k', 0.0),
            'Omega_r': params.get('Omega_r', 0.0),
        }
        dd, ddt = self._distance_interp(cosmo_params)
        return self._lens_evaluator(dd, ddt)

    def log_likelihood(self, model, **kwargs) -> float:
        """
        Compute log-likelihood for a cosmology model.

        Parameters
        ----------
        model : CosmologyBase
            Cosmological model instance.

        Returns
        -------
        float
            Log-likelihood value.
        """
        params = self._extract_cosmo_params(model)
        dd, ddt = self._distance_interp(params)
        return self._lens_evaluator(dd, ddt)

    def _extract_cosmo_params(self, model) -> Dict[str, jnp.ndarray]:
        """Convert model.params to JAX-compatible dict."""
        params_dict = model.params.to_dict()
        return {
            k: jnp.asarray(v, dtype=jnp.float64)
            for k, v in params_dict.items()
            if isinstance(v, (int, float, jnp.ndarray, np.ndarray))
        }

    @property
    def nuisance_parameters(self):
        """
        Return nuisance parameters for automatic registry integration.

        H0LiCOW has no nuisance parameters - all lens modeling is internal.

        Returns
        -------
        list
            Empty list (no nuisance parameters).
        """
        return NuisanceList()

    def get_derived_params(self, model) -> Dict[str, float]:
        """Get derived parameters (time-delay distances)."""
        params = self._extract_cosmo_params(model)
        dd_array, ddt_array = self._distance_interp(params)
        derived: Dict[str, float] = {}
        for i, lens in enumerate(self.lenses):
            derived[f"Ddt_{lens.name}"] = float(ddt_array[i])
            if lens.uses_dd:
                derived[f"Dd_{lens.name}"] = float(dd_array[i])
        return derived

    def get_info(self) -> Dict:
        """Get dataset information."""
        return {
            'name': 'H0LiCOW',
            'n_lenses': len(self.lenses),
            'lenses': [lens.name for lens in self.lenses],
            'redshift_range': (
                min(lens.zlens for lens in self.lenses),
                max(lens.zsource for lens in self.lenses)
            ),
        }

    # __add__ and __radd__ inherited from Likelihood base class

    def __repr__(self):
        """String representation."""
        info = self.get_info()
        return (f"H0LiCOWLikelihood({info['n_lenses']} lenses: "
                f"{', '.join(info['lenses'])})")

    @staticmethod
    def _get_default_lens_configs() -> List[LensConfig]:
        """Get default lens configurations (Wong et al. 2019)."""
        return [
            LensConfig(
                name="B1608",
                zlens=0.6304,
                zsource=1.394,
                kind="skewed_lognormal_dd",
                params={
                    "mu": 7.0531390,
                    "sigma": 0.2282395,
                    "lam": 4000.0,
                    "mu_dd": 6.79671,
                    "sigma_dd": 0.1836,
                    "lam_dd": 334.2,
                },
            ),
            LensConfig(
                name="RXJ1131",
                zlens=0.295,
                zsource=0.654,
                kind="kde_hist_2d",
                data_file="h0licow_distance_chains/RXJ1131_AO+HST_Dd_Ddt.dat",
                read_csv_kwargs={"sep": r"\\s+", "comment": "#", "names": ["dd", "ddt"], "header": None, "engine": "python"},
                columns={"dd": "dd", "ddt": "ddt"},
                bandwidth=20.0,
                nbins=80,
            ),
            LensConfig(
                name="HE0435",
                zlens=0.4546,
                zsource=1.693,
                kind="kde_hist_1d",
                data_file="h0licow_distance_chains/HE0435_Ddt_AO+HST.dat",
                read_csv_kwargs={"sep": r"\\s+", "names": ["ddt"], "header": None, "engine": "python"},
                columns={"ddt": "ddt"},
                bandwidth=20.0,
                nbins=400,
            ),
            LensConfig(
                name="J1206",
                zlens=0.745,
                zsource=1.789,
                kind="kde_full_2d",
                data_file="h0licow_distance_chains/J1206_final.csv",
                read_csv_kwargs={"sep": ",", "header": 0},
                columns={"dd": "dd", "ddt": "ddt"},
                bandwidth=80.0,
            ),
            LensConfig(
                name="WFI2033",
                zlens=0.6575,
                zsource=1.662,
                kind="kde_hist_1d",
                data_file="h0licow_distance_chains/wfi2033_dt_bic.dat",
                read_csv_kwargs={"sep": ",", "header": 0},
                columns={"ddt": "Dt", "weight": "weight"},
                bandwidth=20.0,
                nbins=400,
                max_ddt=8000.0,
            ),
            LensConfig(
                name="PG1115",
                zlens=0.311,
                zsource=1.722,
                kind="kde_hist_2d",
                data_file="h0licow_distance_chains/PG1115_AO+HST_Dd_Ddt.dat",
                read_csv_kwargs={"sep": r"\\s+", "comment": "#", "names": ["dd", "ddt"], "header": None, "engine": "python"},
                columns={"dd": "dd", "ddt": "ddt"},
                bandwidth=20.0,
                nbins=80,
            ),
            LensConfig(
                name="DES0408",
                zlens=0.597,
                zsource=2.375,
                kind="gaussian",
                params={"mu": 3382.0, "sigma": 130.5},
            ),
        ]
