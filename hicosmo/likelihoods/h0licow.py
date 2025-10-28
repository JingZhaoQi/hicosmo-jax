"""H0LiCOW strong-lensing time-delay likelihood implemented for HIcosmo."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.special import logsumexp

from .base import Likelihood


@dataclass
class LensConfig:
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


def _skewed_lognormal_logpdf(value: jnp.ndarray, mu: float, sigma: float, lam: float, explim: float) -> jnp.ndarray:
    shifted = value - lam
    valid = shifted > 0.0
    log_term = jnp.log(shifted)
    exponent = -0.5 * ((log_term - mu) / sigma) ** 2
    valid = valid & (-(exponent) <= explim)
    norm = jnp.log(jnp.sqrt(2.0 * jnp.pi)) + jnp.log(sigma) + jnp.log(shifted)
    logpdf = exponent - norm
    return jnp.where(valid, logpdf, jnp.array(-1e12))


class KDEEstimator:
    """Simple isotropic Gaussian KDE helper that works with JAX arrays."""

    def __init__(self, points: np.ndarray, weights: Optional[np.ndarray], bandwidth: float) -> None:
        if weights is None:
            weights = np.ones(points.shape[0], dtype=np.float64)
        self.points = jnp.asarray(points, dtype=jnp.float32)
        self.weights = jnp.asarray(weights, dtype=jnp.float32)
        self.bandwidth = float(bandwidth)
        self.log_norm_1d = jnp.log(jnp.sqrt(2.0 * jnp.pi) * self.bandwidth)
        self.log_norm_2d = jnp.log(2.0 * jnp.pi * self.bandwidth**2)
        self.weight_sum = jnp.sum(self.weights)

    def logpdf_1d(self, value: jnp.ndarray) -> jnp.ndarray:
        diff = (value - self.points) / self.bandwidth
        log_kernel = -0.5 * diff**2 - self.log_norm_1d
        log_weighted = jnp.log(self.weights) + log_kernel
        return logsumexp(log_weighted) - jnp.log(self.weight_sum)

    def logpdf_2d(self, vector: jnp.ndarray) -> jnp.ndarray:
        diff = (vector - self.points) / self.bandwidth
        quad = jnp.sum(diff**2, axis=1)
        log_kernel = -0.5 * quad - self.log_norm_2d
        log_weighted = jnp.log(self.weights) + log_kernel
        return logsumexp(log_weighted) - jnp.log(self.weight_sum)


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
            return self._kde.logpdf_1d(ddt.reshape(()))
        if kind in {"kde_hist_2d", "kde_full_2d"}:
            vec = jnp.stack([dd.reshape(()), ddt.reshape(())])
            return self._kde.logpdf_2d(vec)
        raise ValueError(f"Unsupported lens kind '{kind}'.")


def _compress_histogram(samples: np.ndarray, weights: np.ndarray, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(samples, bins=nbins, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = hist > 0
    return centers[mask], hist[mask]


def _compress_histogram_2d(dd: np.ndarray, ddt: np.ndarray, weights: np.ndarray, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    hist, x_edges, y_edges = np.histogram2d(dd, ddt, bins=nbins, weights=weights)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    flattened = hist.ravel()
    mask = flattened > 0
    points = np.column_stack([grid_x.ravel()[mask], grid_y.ravel()[mask]])
    return points, flattened[mask]


class H0LiCOWLikelihood(Likelihood):
    """H0LiCOW strong-lensing likelihood compatible with HIcosmo."""

    def __init__(
        self,
        dataset_name: str = "wong2019",
        lens_names: Optional[Iterable[str]] = None,
        data_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.dataset_name = dataset_name
        self.lens_names = set(lens_names) if lens_names else None
        super().__init__(name=dataset_name, data_path=data_path, **kwargs)
        self.initialize()

    def _default_dataset_name(self) -> str:
        return self.dataset_name

    def _load_data(self) -> None:
        base_dir = Path(self.data_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "h0licow"))
        if not base_dir.exists():
            raise FileNotFoundError(base_dir)
        configs = _get_default_lens_configs()
        selected = [cfg for cfg in configs if self.lens_names is None or cfg.name in self.lens_names]
        if not selected:
            available = ", ".join(cfg.name for cfg in configs)
            raise ValueError(f"No H0LiCOW lenses selected. Available: {available}")
        self.lenses = [H0LiCOWLens(cfg, base_dir) for cfg in selected]

    def _setup_covariance(self) -> None:
        self.inv_cov = None

    def theory(self, cosmology, **kwargs):
        raise NotImplementedError("H0LiCOWLikelihood does not return a simple theory vector.")

    def log_likelihood(self, cosmology, **kwargs) -> float:
        total = jnp.array(0.0)
        for lens in self.lenses:
            dd = cosmology.angular_diameter_distance(lens.zlens)
            ds = cosmology.angular_diameter_distance(lens.zsource)
            dds = cosmology.angular_diameter_distance_between(lens.zlens, lens.zsource)
            ddt = (1.0 + lens.zlens) * dd * ds / dds
            total = total + lens.log_likelihood(dd, ddt)
        return jnp.asarray(total)

    def get_requirements(self) -> Dict[str, Any]:
        z_single = sorted({lens.zlens for lens in self.lenses} | {lens.zsource for lens in self.lenses})
        z_pairs = sorted({(lens.zlens, lens.zsource) for lens in self.lenses})
        return {
            "angular_diameter_distance": {"z": z_single},
            "angular_diameter_distance_pairs": {"pairs": z_pairs},
        }

    def get_derived_params(self, cosmology) -> Dict[str, float]:
        derived: Dict[str, float] = {}
        for lens in self.lenses:
            dd = float(cosmology.angular_diameter_distance(lens.zlens))
            ds = float(cosmology.angular_diameter_distance(lens.zsource))
            dds = float(cosmology.angular_diameter_distance_between(lens.zlens, lens.zsource))
            ddt = (1.0 + lens.zlens) * dd * ds / dds
            derived[f"Ddt_{lens.name}"] = ddt
            if lens.uses_dd:
                derived[f"Dd_{lens.name}"] = dd
        return derived


def _get_default_lens_configs() -> List[LensConfig]:
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
