"""TDCOSMO hierarchical strong-lensing likelihood components."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import re

import jax
import jax.numpy as jnp
import numpy as np
import pickle

from jax.scipy.special import logsumexp

from .base import Likelihood
from ..utils.constants import c_km_s

_DTYPE = jnp.float32
_TWO_PI = jnp.array(2.0 * jnp.pi, dtype=_DTYPE)
_EPS = jnp.array(1e-12, dtype=_DTYPE)
_NEG_LARGE = jnp.array(-1e12, dtype=_DTYPE)

_HERMITE_NODES = jnp.asarray(
    [-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856],
    dtype=_DTYPE,
)
_HERMITE_WEIGHTS = jnp.asarray(
    [0.01995324205904591, 0.39361932315224116, 0.9453087204829419, 0.39361932315224116, 0.01995324205904591],
    dtype=_DTYPE,
)
_HERMITE_NORM = jnp.sqrt(jnp.pi)



def _gaussian_logpdf(value: jnp.ndarray, mean: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Log-density of a normal distribution with stability guards."""

    sigma = jnp.where(sigma > 0.0, sigma, jnp.nan)
    diff = (value - mean) / sigma
    return -0.5 * diff**2 - jnp.log(sigma * jnp.sqrt(_TWO_PI))


@dataclass
class KappaPrior:
    mean: float = 0.0
    sigma: float = 0.02

    def logpdf(self, value: jnp.ndarray) -> jnp.ndarray:
        if self.sigma <= 0.0:
            return jnp.array(0.0, dtype=_DTYPE)
        diff = (value - self.mean) / self.sigma
        return -0.5 * diff**2 - jnp.log(self.sigma * jnp.sqrt(_TWO_PI))


@dataclass
class TDCOSMOLensData:
    name: str
    z_lens: float
    z_source: float
    lambda_scaling: float
    ddt_centers: jnp.ndarray
    ddt_weights: jnp.ndarray
    bandwidth: float
    sigma_v_obs: jnp.ndarray
    cov_meas: jnp.ndarray
    cov_j_sqrt: jnp.ndarray
    j_model: jnp.ndarray
    ani_params: jnp.ndarray
    ani_scaling: jnp.ndarray  # shape (n_meas, len(ani_params))
    kappa_centers: jnp.ndarray
    kappa_pdf: jnp.ndarray
    kappa_min: float
    kappa_max: float
    ddt_norm_factor: float

    def ddt_logpdf(self, value: jnp.ndarray) -> jnp.ndarray:
        value_exp = jnp.expand_dims(value, axis=-1)
        centers = jnp.asarray(self.ddt_centers, dtype=_DTYPE)
        diff = (value_exp - centers) / self.bandwidth
        log_kernel = -0.5 * diff**2 - jnp.log(self.bandwidth * jnp.sqrt(_TWO_PI))
        log_weights = jnp.log(self.ddt_weights + _EPS)
        return logsumexp(log_kernel + log_weights, axis=-1) - self.ddt_norm_factor

    def kappa_logpdf(self, kappa: jnp.ndarray) -> jnp.ndarray:
        pdf = jnp.interp(kappa, self.kappa_centers, self.kappa_pdf)
        pdf = jnp.maximum(pdf, _EPS)
        return jnp.log(pdf)

    def anisotropy_scaling(self, ani_param: Optional[jnp.ndarray]) -> jnp.ndarray:
        if ani_param is None:
            return jnp.ones(self.sigma_v_obs.shape, dtype=_DTYPE)
        ani_param = jnp.asarray(ani_param, dtype=_DTYPE)
        ani_param = jnp.clip(ani_param, self.ani_params[0], self.ani_params[-1])
        def _interp(scaling_row: jnp.ndarray) -> jnp.ndarray:
            return jnp.interp(ani_param, self.ani_params, scaling_row)
        scaling = jnp.array([_interp(row) for row in self.ani_scaling], dtype=_DTYPE)
        return scaling


def _load_processed_lens(path: Path, lambda_scaling: float) -> TDCOSMOLensData:
    with path.open("rb") as f:
        data = pickle.load(f)

    ddt_samples = data["ddt_samples"].astype("float32")
    weights = data.get("ddt_weights")
    if weights is None:
        weights = np.ones_like(ddt_samples)
    weights = weights.astype("float32")
    nbins = data.get("nbins_hist", 200)
    hist, edges = np.histogram(ddt_samples, bins=nbins, weights=weights, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = edges[1:] - edges[:-1]
    mask = hist > 0.0
    centers = centers[mask]
    hist = hist[mask]
    widths = bin_widths[mask]
    hist_weights = hist * widths
    hist_weights = hist_weights / np.sum(hist_weights)

    ddt_mean = np.average(ddt_samples, weights=weights)
    variance = np.average((ddt_samples - ddt_mean) ** 2, weights=weights)
    ddt_sigma = float(np.sqrt(max(variance, 1e-12)))
    ddt_norm_factor = np.log(1.0 / (ddt_sigma * np.sqrt(2.0 * np.pi)))

    sigma_v_obs = np.asarray(data["sigma_v_measurement"], dtype="float32")
    cov_meas = np.asarray(data["error_cov_measurement"], dtype="float32")
    cov_j_sqrt = np.asarray(data["error_cov_j_sqrt"], dtype="float32")
    if cov_meas.ndim == 0:
        cov_meas = np.array([[cov_meas]], dtype="float32")
    if cov_j_sqrt.ndim == 0:
        cov_j_sqrt = np.array([[cov_j_sqrt]], dtype="float32")
    j_model = np.asarray(data["j_model"], dtype="float32")

    ani_params = np.asarray(data.get("ani_param_array", [0.0]), dtype="float32")
    scaling_list = data.get("ani_scaling_array_list", None)
    if scaling_list is None or len(scaling_list) == 0:
        ani_scaling = np.ones((len(sigma_v_obs), len(ani_params)), dtype="float32")
    else:
        ani_scaling = np.stack(
            [np.asarray(row, dtype="float32") for row in scaling_list], axis=0
        )

    kappa_edges = np.asarray(data["kappa_bin_edges"], dtype="float32")
    kappa_pdf = np.asarray(data["kappa_pdf"], dtype="float32")
    kappa_centers = 0.5 * (kappa_edges[:-1] + kappa_edges[1:])

    return TDCOSMOLensData(
        name=data["name"],
        z_lens=float(data["z_lens"]),
        z_source=float(data["z_source"]),
        lambda_scaling=float(lambda_scaling),
        ddt_centers=jnp.asarray(centers, dtype=_DTYPE),
        ddt_weights=jnp.asarray(hist_weights, dtype=_DTYPE),
        bandwidth=float(data.get("bandwidth", 20.0)),
        sigma_v_obs=jnp.asarray(sigma_v_obs, dtype=_DTYPE),
        cov_meas=jnp.asarray(cov_meas, dtype=_DTYPE),
        cov_j_sqrt=jnp.asarray(cov_j_sqrt, dtype=_DTYPE),
        j_model=jnp.asarray(j_model, dtype=_DTYPE),
        ani_params=jnp.asarray(ani_params, dtype=_DTYPE),
        ani_scaling=jnp.asarray(ani_scaling, dtype=_DTYPE),
        kappa_centers=jnp.asarray(kappa_centers, dtype=_DTYPE),
        kappa_pdf=jnp.asarray(kappa_pdf, dtype=_DTYPE),
        kappa_min=float(kappa_centers.min()),
        kappa_max=float(kappa_centers.max()),
        ddt_norm_factor=ddt_norm_factor,
    )


def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name)


class TDCOSMOLikelihood(Likelihood):
    """TDCOSMO strong-lensing likelihood with external convergence and anisotropy parameters."""

    def __init__(
        self,
        lens_names: Optional[Iterable[str]] = None,
        lens_priors: Optional[Dict[str, KappaPrior]] = None,
        data_path: Optional[str] = None,
        name: Optional[str] = None,
        lambda_bounds: Tuple[float, float] = (0.5, 1.5),
        anisotropy_bounds: Tuple[float, float] = (0.1, 5.0),
        log_scatter_prior: bool = True,
        omega_m_prior: Optional[Tuple[float, float]] = None,
        default_omega_b: float = 0.049,
        **kwargs: Any,
    ) -> None:
        self.data_path = Path(data_path or Path(__file__).resolve().parents[1] / "data" / "tdcosmo" / "processed")
        raw_names = list(lens_names) if lens_names is not None else [
            p.stem.replace("_processed", "") for p in sorted(self.data_path.glob("*_processed.pkl"))
        ]
        if not raw_names:
            raise ValueError("No TDCOSMO lenses found.")
        self.lens_data: Dict[str, TDCOSMOLensData] = {}
        # map raw names to scaling property from CSV
        scaling_map = self._load_lambda_scaling()
        for name in raw_names:
            sanitized = _sanitize(name)
            scaling = scaling_map.get(sanitized, 0.0)
            data = _load_processed_lens(self.data_path / f"{name}_processed.pkl", scaling)
            data = dataclasses.replace(data, name=sanitized)
            self.lens_data[sanitized] = data
        self.lens_names = list(self.lens_data.keys())

        self.lens_priors: Dict[str, KappaPrior] = {}
        custom_priors = lens_priors or {}
        for key, prior in custom_priors.items():
            self.lens_priors[_sanitize(key)] = prior
        for name in self.lens_names:
            self.lens_priors.setdefault(name, KappaPrior())
        if lambda_bounds[0] >= lambda_bounds[1]:
            raise ValueError("lambda_bounds must satisfy min < max")
        if anisotropy_bounds[0] >= anisotropy_bounds[1]:
            raise ValueError("anisotropy_bounds must satisfy min < max")
        self.lambda_bounds = lambda_bounds
        self.anisotropy_bounds = anisotropy_bounds
        self.log_scatter_prior = log_scatter_prior
        self.omega_m_prior = omega_m_prior
        self.default_omega_b = default_omega_b
        super().__init__(name=name or "tdcosmo", data_path=str(self.data_path), **kwargs)
        self.initialize()

    def _distance_tuple(self, cosmology, z_lens: float, z_source: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        fi = getattr(cosmology, "fast_integration", None)
        z_l = jnp.asarray(z_lens, dtype=_DTYPE)
        z_s = jnp.asarray(z_source, dtype=_DTYPE)

        if fi is not None:
            D_C_l = jnp.asarray(fi.comoving_distance(z_l), dtype=_DTYPE)
            D_C_s = jnp.asarray(fi.comoving_distance(z_s), dtype=_DTYPE)
            D_C_between = D_C_s - D_C_l
            dd = D_C_l / (1.0 + z_l)
            ds = D_C_s / (1.0 + z_s)
            dds = D_C_between / (1.0 + z_s)
            return dd, ds, dds

        dd = jnp.asarray(cosmology.angular_diameter_distance(float(z_lens)), dtype=_DTYPE)
        ds = jnp.asarray(cosmology.angular_diameter_distance(float(z_source)), dtype=_DTYPE)
        dds = jnp.asarray(cosmology.angular_diameter_distance_between(float(z_lens), float(z_source)), dtype=_DTYPE)
        return dd, ds, dds

    def _default_dataset_name(self) -> str:
        return "tdcosmo"

    def _load_data(self) -> None:  # type: ignore[override]
        return

    def _setup_covariance(self) -> None:  # type: ignore[override]
        return

    def get_requirements(self) -> Dict[str, Any]:  # type: ignore[override]
        return {}

    def theory(self, cosmology, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def _gaussian_nodes(
        self,
        mean: jnp.ndarray,
        sigma: jnp.ndarray,
        bounds: Tuple[float, float],
        override: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if override is not None:
            node = jnp.asarray(override, dtype=_DTYPE)
            return node[None], jnp.array([1.0], dtype=_DTYPE)

        sigma = jnp.asarray(sigma, dtype=_DTYPE)
        mean = jnp.asarray(mean, dtype=_DTYPE)
        sigma = jnp.maximum(sigma, jnp.array(1e-6, dtype=_DTYPE))

        nodes = mean + jnp.sqrt(2.0) * sigma * _HERMITE_NODES
        nodes = jnp.clip(nodes, bounds[0], bounds[1])
        weights = _HERMITE_WEIGHTS / _HERMITE_NORM
        weights = weights / jnp.maximum(jnp.sum(weights), _EPS)
        return nodes, weights

    def _kappa_nodes(
        self,
        lens: TDCOSMOLensData,
        override: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if override is not None:
            node = jnp.asarray(override, dtype=_DTYPE)
            return node[None], jnp.array([1.0], dtype=_DTYPE)

        weights = lens.kappa_pdf / jnp.maximum(jnp.sum(lens.kappa_pdf), _EPS)
        return lens.kappa_centers, weights

    def _integrated_lens_loglike(
        self,
        lens: TDCOSMOLensData,
        ddt: jnp.ndarray,
        dd: jnp.ndarray,
        lambda_nodes: jnp.ndarray,
        lambda_weights: jnp.ndarray,
        ani_nodes: jnp.ndarray,
        ani_weights: jnp.ndarray,
        kappa_nodes: jnp.ndarray,
        kappa_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        terms = []
        for i in range(lambda_nodes.shape[0]):
            lam = lambda_nodes[i]
            log_w_lam = jnp.log(lambda_weights[i] + _EPS)
            for j in range(ani_nodes.shape[0]):
                ani = ani_nodes[j]
                log_w_ani = jnp.log(ani_weights[j] + _EPS)
                inner = []
                for k in range(kappa_nodes.shape[0]):
                    kap = kappa_nodes[k]
                    log_w_kap = jnp.log(kappa_weights[k] + _EPS) + self.lens_priors[lens.name].logpdf(kap)
                    loglike = self._single_lens_loglike(lens, ddt, dd, kap, lam, ani)
                    inner.append(log_w_kap + loglike)
                inner_total = logsumexp(jnp.stack(inner))
                terms.append(log_w_lam + log_w_ani + inner_total)

        return logsumexp(jnp.stack(terms))

    def log_likelihood(self, cosmology, **kwargs) -> float:
        include_priors = kwargs.pop("include_priors", True)

        lambda_mean = jnp.asarray(kwargs.get("lambda_int_mean", 1.0), dtype=_DTYPE)
        lambda_sigma = jnp.asarray(kwargs.get("lambda_int_sigma", 0.05), dtype=_DTYPE)
        alpha_lambda = jnp.asarray(kwargs.get("alpha_lambda", 0.0), dtype=_DTYPE)
        a_mean = jnp.asarray(kwargs.get("a_ani_mean", 1.0), dtype=_DTYPE)
        a_sigma = jnp.asarray(kwargs.get("a_ani_sigma", 0.1), dtype=_DTYPE)

        lambda_lower = jnp.array(self.lambda_bounds[0], dtype=_DTYPE)
        lambda_upper = jnp.array(self.lambda_bounds[1], dtype=_DTYPE)
        a_lower = jnp.array(self.anisotropy_bounds[0], dtype=_DTYPE)
        a_upper = jnp.array(self.anisotropy_bounds[1], dtype=_DTYPE)

        total = jnp.array(0.0, dtype=_DTYPE)

        # Enforce positive scatters through an immediate penalty when violated.
        total = total + jnp.where(lambda_sigma > 0.0, 0.0, _NEG_LARGE)
        total = total + jnp.where(a_sigma > 0.0, 0.0, _NEG_LARGE)

        lambda_sigma_safe = jnp.where(lambda_sigma > 0.0, lambda_sigma, jnp.array(1.0, dtype=_DTYPE))
        a_sigma_safe = jnp.where(a_sigma > 0.0, a_sigma, jnp.array(1.0, dtype=_DTYPE))

        for name, lens in self.lens_data.items():
            kappa_key = f"kappa_ext_{name}"
            ani_key = f"a_ani_{name}"
            legacy_ani_key = f"ani_param_{name}"
            lambda_key = f"lambda_int_{name}"

            lambda_loc = lambda_mean + alpha_lambda * jnp.asarray(lens.lambda_scaling, dtype=_DTYPE)

            lambda_override = kwargs.get(lambda_key, None)
            ani_override = kwargs.get(ani_key, kwargs.get(legacy_ani_key, None))
            kappa_override = kwargs.get(kappa_key, None)

            if lambda_override is not None:
                total = total + self._bounds_penalty(jnp.asarray(lambda_override, dtype=_DTYPE), lambda_lower, lambda_upper)
            if ani_override is not None:
                total = total + self._bounds_penalty(jnp.asarray(ani_override, dtype=_DTYPE), a_lower, a_upper)
            if kappa_override is not None:
                total = total + self._bounds_penalty(
                    jnp.asarray(kappa_override, dtype=_DTYPE),
                    jnp.array(lens.kappa_min, dtype=_DTYPE),
                    jnp.array(lens.kappa_max, dtype=_DTYPE),
                )

            lambda_nodes, lambda_weights = self._gaussian_nodes(lambda_loc, lambda_sigma, self.lambda_bounds, override=lambda_override)
            ani_nodes, ani_weights = self._gaussian_nodes(a_mean, a_sigma, self.anisotropy_bounds, override=ani_override)
            kappa_nodes, kappa_weights = self._kappa_nodes(lens, override=kappa_override)

            dd, ds, dds = self._distance_tuple(cosmology, lens.z_lens, lens.z_source)
            ddt = (1.0 + lens.z_lens) * dd * ds / dds

            loglike = self._integrated_lens_loglike(
                lens,
                ddt,
                dd,
                lambda_nodes,
                lambda_weights,
                ani_nodes,
                ani_weights,
                kappa_nodes,
                kappa_weights,
            )
            total = total + loglike

        if include_priors:
            if self.log_scatter_prior:
                total = total - jnp.log(lambda_sigma_safe) - jnp.log(a_sigma_safe)
            total = total - jnp.log(jnp.maximum(a_mean, _EPS))
            if self.omega_m_prior is not None:
                om = jnp.asarray(cosmology.params.get("Omega_m"), dtype=_DTYPE)
                mu, sigma = self.omega_m_prior
                total = total + _gaussian_logpdf(
                    om,
                    jnp.array(mu, dtype=_DTYPE),
                    jnp.array(sigma, dtype=_DTYPE),
                )

        return total

    def _single_lens_loglike(
        self,
        lens: TDCOSMOLensData,
        ddt: jnp.ndarray,
        dd: jnp.ndarray,
        kappa: jnp.ndarray,
        lambda_mst: jnp.ndarray,
        ani_param: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        lambda_tot = lambda_mst * (1.0 - kappa)
        invalid = (lambda_mst <= 0.0) | (lambda_tot <= 0.0)
        lambda_tot = jnp.where(invalid, jnp.array(1.0, dtype=_DTYPE), lambda_tot)
        lambda_mst_safe = jnp.asarray(lambda_mst, dtype=_DTYPE)
        ddt_eff = ddt * lambda_tot
        ddt_log = lens.ddt_logpdf(ddt_eff)

        ds_dds_base = jnp.maximum(ddt / dd / (1.0 + lens.z_lens), _EPS)
        kin_scaling = lens.anisotropy_scaling(ani_param)
        sigma_model = jnp.sqrt(lens.j_model * ds_dds_base * kin_scaling * lambda_mst_safe) * c_km_s

        scaling_mat = jnp.outer(jnp.sqrt(kin_scaling), jnp.sqrt(kin_scaling))
        cov_model = lens.cov_j_sqrt * scaling_mat * ds_dds_base * lambda_mst_safe * c_km_s**2
        cov_total = cov_model + lens.cov_meas
        n = lens.sigma_v_obs.shape[0]
        cov_total = cov_total + jnp.eye(n, dtype=_DTYPE) * (1e-4 * c_km_s**2)

        delta = lens.sigma_v_obs - sigma_model
        solve = jnp.linalg.solve(cov_total, delta)
        chi2 = jnp.dot(delta, solve)
        sign, logdet = jnp.linalg.slogdet(cov_total)
        kin_log = -0.5 * (chi2 + logdet + n * jnp.log(_TWO_PI))
        invalid = invalid | (sign <= 0) | (~jnp.isfinite(kin_log))
        kin_log = jnp.where(jnp.isfinite(kin_log), kin_log, _NEG_LARGE)

        loglike = ddt_log + kin_log
        loglike = jnp.where(jnp.isfinite(loglike), loglike, _NEG_LARGE)
        return jnp.where(invalid, _NEG_LARGE, loglike)

    @staticmethod
    def _bounds_penalty(value: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        lower = jnp.asarray(lower, dtype=_DTYPE)
        upper = jnp.asarray(upper, dtype=_DTYPE)
        return jnp.where((value < lower) | (value > upper), _NEG_LARGE, 0.0)

    def default_parameters(self) -> Dict[str, float]:
        """Return a physically motivated default parameter dictionary."""

        defaults: Dict[str, float] = {
            "lambda_int_mean": 1.0,
            "lambda_int_sigma": 0.05,
            "alpha_lambda": 0.0,
            "a_ani_mean": 1.0,
            "a_ani_sigma": 0.1,
        }
        return defaults

    def get_derived_params(self, cosmology) -> Dict[str, float]:  # type: ignore[override]
        derived: Dict[str, float] = {}
        for name, lens in self.lens_data.items():
            dd, ds, dds = self._distance_tuple(cosmology, lens.z_lens, lens.z_source)
            dd = float(dd)
            ds = float(ds)
            dds = float(dds)
            ddt = (1.0 + lens.z_lens) * dd * ds / dds
            derived[f"Ddt_{name}"] = ddt
            derived[f"Dd_{name}"] = dd
        return derived

    def _load_lambda_scaling(self) -> Dict[str, float]:
        csv_path = self.data_path.parents[0] / "tdcosmo_sample.csv"
        if not csv_path.exists():
            return {}
        import csv

        mapping: Dict[str, float] = {}
        with csv_path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw = row.get("name")
                if raw is None:
                    continue
                sanitized = _sanitize(raw)
                try:
                    r_eff = float(row.get("r_eff", 0.0))
                    theta_e = float(row.get("theta_E", 1.0))
                except ValueError:
                    continue
                if theta_e == 0.0:
                    scaling = 0.0
                else:
                    scaling = r_eff / theta_e - 1.0
                mapping[sanitized] = scaling
        return mapping
