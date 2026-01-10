"""
Gravitational Wave Standard Siren Likelihood (JAX-Optimized)
=============================================================

HIcosmo-native implementation of GW cosmology following icarogw methodology,
but fully integrated with HIcosmo's architecture principles.

Design Principles:
- NO hardcoded cosmology: Uses cosmology.luminosity_distance(z)
- Single file implementation: All code in hicosmo/likelihoods/
- Complete functionality: Based on icarogw2.0_tutorial analysis
- JAX acceleration: Only for data processing, not cosmology

References:
- Mastrogiovanni et al. (2023): https://arxiv.org/abs/2305.17973
- icarogw: https://github.com/simone-mastrogiovanni/icarogw
- HIcosmo design: CLAUDE.md principles

Phase 1 (MVP): Basic hierarchical likelihood
Phase 2: Full rate models + selection effects
Phase 3: EM counterparts + sky pixelization
Phase 4: JIT optimization + Fisher gradients
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple, Callable, Sequence, Type, Union
from pathlib import Path
import pickle
import re

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, random as jrandom
import jax.scipy.special

from ..base import Likelihood
from hicosmo.models import LCDM
from .population import (
    MassPrior,
    PowerLawMass,
    PowerLawPeak,
    BNSMassPrior,
    RateEvolution,
    MadauRate,
)


def _build_pe_prior_from_mode(
    d_l: np.ndarray,
    mode: Optional[str],
    *,
    bounds: Optional[Tuple[float, float]] = None,
) -> Optional[np.ndarray]:
    if mode is None:
        return None
    if d_l is None:
        return None
    mode_norm = str(mode).lower()
    if mode_norm in {"auto", "none", "pesummary", "official"}:
        return None
    if mode_norm in {"dl2", "d_l2", "dl^2", "d_l^2"}:
        prior = np.asarray(d_l, dtype=np.float64) ** 2
        if bounds is not None:
            min_val, max_val = bounds
            mask = (d_l >= min_val) & (d_l <= max_val)
            prior = np.where(mask, prior, 0.0)
        return prior
    raise ValueError(f"Unknown pe_prior_mode: {mode}")


def _parse_prior_expression(expr: Union[str, bytes], samples: np.ndarray) -> Optional[np.ndarray]:
    if expr is None:
        return None
    if isinstance(expr, (bytes, np.bytes_)):
        expr = expr.decode(errors="ignore")
    expr = str(expr).strip()
    if not expr:
        return None

    float_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    def _get_float(key: str) -> Optional[float]:
        match = re.search(rf"{key}\s*=\s*({float_pattern})", expr)
        return float(match.group(1)) if match else None

    if "PowerLaw" in expr:
        alpha = _get_float("alpha")
        minimum = _get_float("minimum") or _get_float("min")
        maximum = _get_float("maximum") or _get_float("max")
        if alpha is None or minimum is None or maximum is None:
            return None
        prior = np.zeros_like(samples, dtype=np.float64)
        mask = (samples >= minimum) & (samples <= maximum)
        prior[mask] = np.power(samples[mask], alpha)
        return prior

    if "LogUniform" in expr:
        minimum = _get_float("minimum") or _get_float("min")
        maximum = _get_float("maximum") or _get_float("max")
        if minimum is None or maximum is None:
            return None
        prior = np.zeros_like(samples, dtype=np.float64)
        mask = (samples >= minimum) & (samples <= maximum)
        prior[mask] = 1.0 / np.clip(samples[mask], 1e-300, None)
        return prior

    if "Uniform" in expr:
        minimum = _get_float("minimum") or _get_float("min")
        maximum = _get_float("maximum") or _get_float("max")
        if minimum is None or maximum is None:
            return None
        prior = np.zeros_like(samples, dtype=np.float64)
        mask = (samples >= minimum) & (samples <= maximum)
        prior[mask] = 1.0
        return prior

    return None


def _estimate_joint_prior_kde(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
) -> Optional[np.ndarray]:
    try:
        from scipy.stats import gaussian_kde
    except ImportError:  # pragma: no cover - optional dependency
        return None

    try:
        kde = gaussian_kde(prior_samples)
        return kde(posterior_samples)
    except Exception:
        return None


def _build_joint_pe_prior(
    *,
    prior_samples: Dict[str, np.ndarray],
    posterior: Dict[str, np.ndarray],
    bounds: Optional[Tuple[float, float]] = None,
) -> Optional[np.ndarray]:
    d_l = posterior.get("luminosity_distance")
    m1 = posterior.get("mass_1")
    m2 = posterior.get("mass_2")
    if d_l is None or m1 is None or m2 is None:
        return None

    prior_d_l = prior_samples.get("luminosity_distance")
    prior_m1 = prior_samples.get("mass_1")
    prior_m2 = prior_samples.get("mass_2")
    if prior_d_l is None or prior_m1 is None or prior_m2 is None:
        return None

    prior_matrix = np.vstack(
        [
            np.asarray(prior_m1, dtype=np.float64),
            np.asarray(prior_m2, dtype=np.float64),
            np.asarray(prior_d_l, dtype=np.float64),
        ]
    )
    posterior_matrix = np.vstack(
        [
            np.asarray(m1, dtype=np.float64),
            np.asarray(m2, dtype=np.float64),
            np.asarray(d_l, dtype=np.float64),
        ]
    )

    prior_pdf = _estimate_joint_prior_kde(prior_matrix, posterior_matrix)
    if prior_pdf is None:
        # Fallback: factorized KDEs (m1 * m2 * d_L)
        try:
            from scipy.stats import gaussian_kde
        except ImportError:  # pragma: no cover - optional dependency
            return None
        try:
            kde_m1 = gaussian_kde(np.asarray(prior_m1, dtype=np.float64))
            kde_m2 = gaussian_kde(np.asarray(prior_m2, dtype=np.float64))
            kde_dl = gaussian_kde(np.asarray(prior_d_l, dtype=np.float64))
            prior_pdf = kde_m1(m1) * kde_m2(m2) * kde_dl(d_l)
        except Exception:
            return None

    if bounds is not None:
        min_val, max_val = bounds
        mask = (np.asarray(d_l) >= min_val) & (np.asarray(d_l) <= max_val)
        prior_pdf = np.where(mask, prior_pdf, 0.0)

    return prior_pdf

# ============================================================================
# Phase 1: Data Containers
# ============================================================================

@dataclass
class GWEventData:
    """
    Single gravitational wave event posterior samples.

    Attributes:
        name: Event identifier (e.g., "GW150914")
        posterior_samples: PE posterior (n_samples, n_params)
                          Columns: [d_L, z, m1, m2, ...]
        weights: Importance sampling weights (normalized)
        prior_samples: Optional sampling prior evaluated at posterior samples
        has_em_counterpart: Whether EM counterpart is confirmed
    """
    name: str
    posterior_samples: np.ndarray
    weights: Optional[np.ndarray] = None
    prior_samples: Optional[np.ndarray] = None
    has_em_counterpart: bool = False
    mass_frame: str = "detector"
    snr: Optional[float] = None
    ifar: Optional[float] = None
    em_redshift: Optional[float] = None
    em_redshift_sigma: Optional[float] = None

    # Future: EM counterpart data
    em_z_samples: Optional[np.ndarray] = None
    em_z_weights: Optional[np.ndarray] = None
    sky_location: Optional[Tuple[float, float]] = None  # (RA, Dec)

    def __post_init__(self):
        """Validate and normalize weights."""
        if self.mass_frame not in {"detector", "source"}:
            raise ValueError(f"mass_frame must be 'detector' or 'source', got {self.mass_frame}")
        if self.weights is None:
            n_samples = len(self.posterior_samples)
            self.weights = np.ones(n_samples) / n_samples
        else:
            # Normalize
            self.weights = self.weights / np.sum(self.weights)

        if len(self.weights) != len(self.posterior_samples):
            raise ValueError(
                f"Weights ({len(self.weights)}) must match "
                f"samples ({len(self.posterior_samples)})"
            )

        if (self.em_redshift is None) != (self.em_redshift_sigma is None):
            raise ValueError("em_redshift 与 em_redshift_sigma 必须同时提供或同时为空")

    @property
    def n_samples(self) -> int:
        return len(self.posterior_samples)


@dataclass
class GWInjectionSet:
    """
    GW injection campaign data for selection effects.

    Attributes:
        injections: Simulated events (n_inj, n_params)
        weights: Sampling prior p(d_L, m, q) - NOT normalized!
                 These are used for importance sampling correction.
        n_total: Total injections performed
        V_T: Sensitive volume-time (Gpc^3 yr)
        T_obs: Observation time (years)
    """
    injections: np.ndarray  # [d_L, z, m1, m2, ...]
    weights: np.ndarray     # Sampling prior (NOT final weights!)
    n_total: int           # Total injections
    V_T: float             # Sensitive volume-time
    T_obs: float           # Observation duration
    snr: Optional[np.ndarray] = None
    snr_threshold: float = 0.0
    ifar: Optional[np.ndarray] = None
    ifar_threshold: float = 0.0
    mass_frame: str = "detector"

    def __post_init__(self):
        """Validate that sampling prior与注入条目数量一致。"""
        if self.mass_frame not in {"detector", "source"}:
            raise ValueError(f"mass_frame must be 'detector' or 'source', got {self.mass_frame}")
        if isinstance(self.injections, dict):
            lengths = {len(np.asarray(val)) for val in self.injections.values()}
            if len(lengths) != 1:
                raise ValueError('Injection dict 字段长度必须一致')
            inj_len = lengths.pop()
        else:
            inj_len = len(self.injections)

        if len(self.weights) != inj_len:
            raise ValueError("weights and injections must have same length")
        if self.snr is not None and len(self.snr) != inj_len:
            raise ValueError("snr and injections must have same length")
        if self.ifar is not None and len(self.ifar) != inj_len:
            raise ValueError("ifar and injections must have same length")

    @property
    def n_detected(self) -> int:
        return len(self.injections)

    @property
    def detection_efficiency(self) -> float:
        return self.n_detected / self.n_total


@dataclass
class GWGalaxyCatalog:
    """Galaxy catalog prior (per-pixel redshift densities)."""

    z_grid: jnp.ndarray
    density_table: jnp.ndarray  # shape (npix, nz)
    nside: int
    mask_has_gal: jnp.ndarray
    epsilon: float = 1e-40

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "GWGalaxyCatalog":
        """Load galaxy catalog from supported formats (currently HDF5)."""
        path = Path(path)
        if path.suffix in {".hdf5", ".h5"}:
            return cls.from_hdf5(path, **kwargs)
        raise ValueError(f"Unsupported galaxy catalog format: {path}")

    @classmethod
    def from_hdf5(
        cls,
        path: Union[str, Path],
        *,
        z_max: Optional[float] = None,
        nbins: int = 400,
        min_sigma: float = 5e-4,
        weight_scheme: str = "luminosity",
    ) -> "GWGalaxyCatalog":
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("h5py is required to load galaxy catalogs") from exc

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Galaxy catalog file not found: {path}")

        with h5py.File(path, "r") as handle:
            if "catalog" not in handle:
                raise KeyError("Galaxy catalog HDF5 missing 'catalog' group")
            group = handle["catalog"]
            ra = np.asarray(group["ra"], dtype=np.float64)
            dec = np.asarray(group["dec"], dtype=np.float64)
            z = np.asarray(group["z"], dtype=np.float64)
            sigmaz = np.asarray(group["sigmaz"], dtype=np.float64)
            magnitudes = np.asarray(group.get("m"), dtype=np.float64)
            sky_indices = np.asarray(group.get("sky_indices"), dtype=np.int64)
            attrs = dict(group.attrs)

        if ra.size == 0:
            raise ValueError("Galaxy catalog is empty")

        nside = int(attrs.get("nside", 16))
        npix = int(attrs.get("npixels", 12 * nside * nside))
        if z_max is None:
            z_max = float(np.nanmax(z))

        mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
        mask &= np.isfinite(sigmaz)
        if magnitudes is not None:
            mask &= np.isfinite(magnitudes)
        mask &= z > 0.0
        mask &= z <= z_max
        if not np.any(mask):
            raise ValueError("Galaxy catalog has no valid entries after filtering")

        ra = ra[mask]
        dec = dec[mask]
        z = z[mask]
        sigmaz = np.clip(sigmaz[mask], min_sigma, None)
        magnitudes = magnitudes[mask]
        sky_indices = sky_indices[mask]

        weights = cls._compute_weights(magnitudes, scheme=weight_scheme)
        z_grid = np.linspace(0.0, z_max, nbins, dtype=np.float64)
        density = cls._build_density_table(
            sky_indices,
            z,
            sigmaz,
            weights,
            npix=npix,
            z_grid=z_grid,
        )

        return cls(
            z_grid=jnp.asarray(z_grid, dtype=jnp.float64),
            density_table=jnp.asarray(density, dtype=jnp.float64),
            nside=nside,
            mask_has_gal=jnp.asarray(np.any(density > 0.0, axis=1)),
        )

    @classmethod
    def from_arrays(
        cls,
        ra: np.ndarray,
        dec: np.ndarray,
        z: np.ndarray,
        sigma_z: np.ndarray,
        weights: Optional[np.ndarray] = None,
        *,
        nside: int = 16,
        nbins: int = 200,
        z_max: Optional[float] = None,
    ) -> "GWGalaxyCatalog":
        ra = np.asarray(ra, dtype=np.float64).reshape(-1)
        dec = np.asarray(dec, dtype=np.float64).reshape(-1)
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        sigma_z = np.asarray(sigma_z, dtype=np.float64).reshape(-1)
        if weights is None:
            weights = np.ones_like(z)
        else:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if not (ra.size == dec.size == z.size == sigma_z.size == weights.size):
            raise ValueError("Galaxy arrays must share the same length")
        valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
        valid &= np.isfinite(sigma_z) & np.isfinite(weights)
        valid &= z > 0.0
        if not np.any(valid):
            raise ValueError("No valid galaxies in provided arrays")
        ra = ra[valid]
        dec = dec[valid]
        z = z[valid]
        sigma_z = np.clip(sigma_z[valid], 5e-4, None)
        weights = weights[valid]
        if z_max is None:
            z_max = float(np.max(z))
        z_grid = np.linspace(0.0, z_max, nbins, dtype=np.float64)
        pixels = _compute_healpix_pixels(ra.reshape(1, -1), dec.reshape(1, -1), nside).reshape(-1)
        npix = 12 * nside * nside
        density = cls._build_density_table(
            pixels,
            z,
            sigma_z,
            weights,
            npix=npix,
            z_grid=z_grid,
        )
        return cls(
            z_grid=jnp.asarray(z_grid, dtype=jnp.float64),
            density_table=jnp.asarray(density, dtype=jnp.float64),
            nside=nside,
            mask_has_gal=jnp.asarray(np.any(density > 0.0, axis=1)),
        )

    @staticmethod
    def _compute_weights(mag: np.ndarray, scheme: str = "luminosity") -> np.ndarray:
        if scheme == "uniform" or mag is None:
            return np.ones_like(mag, dtype=np.float64)
        zero_point = np.nanmedian(mag)
        weights = np.power(10.0, -0.4 * (mag - zero_point))
        weights[~np.isfinite(weights)] = 1.0
        return weights.astype(np.float64)

    @staticmethod
    def _build_density_table(
        pixel_indices: np.ndarray,
        z: np.ndarray,
        sigma_z: np.ndarray,
        weights: np.ndarray,
        *,
        npix: int,
        z_grid: np.ndarray,
    ) -> np.ndarray:
        density = np.zeros((npix, z_grid.size), dtype=np.float64)
        order = np.argsort(pixel_indices)
        sorted_pixels = pixel_indices[order]
        sorted_z = z[order]
        sorted_sigma = sigma_z[order]
        sorted_weights = weights[order]

        pixel_ids = np.arange(npix, dtype=np.int64)
        starts = np.searchsorted(sorted_pixels, pixel_ids, side="left")
        stops = np.searchsorted(sorted_pixels, pixel_ids, side="right")

        sqrt_two_pi = np.sqrt(2.0 * np.pi)
        for pix in range(npix):
            start = int(starts[pix])
            stop = int(stops[pix])
            if start == stop:
                continue
            z_chunk = sorted_z[start:stop][:, None]
            sigma_chunk = sorted_sigma[start:stop][:, None]
            weight_chunk = sorted_weights[start:stop][:, None]
            inv_sigma = 1.0 / np.maximum(sigma_chunk, 1e-6)
            norm = weight_chunk * inv_sigma / sqrt_two_pi
            diff = z_grid[None, :] - z_chunk
            kernels = norm * np.exp(-0.5 * np.square(diff * inv_sigma))
            density[pix] = kernels.sum(axis=0)

        min_positive = np.min(density[density > 0.0]) if np.any(density > 0.0) else 1e-40
        density[density <= 0.0] = min_positive
        return density

    def log_density(self, z: jnp.ndarray, pixel_indices: jnp.ndarray) -> jnp.ndarray:
        values = self._interp_density(z, pixel_indices)
        return jnp.log(values)

    def _interp_density(self, z: jnp.ndarray, pixel_indices: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z, dtype=jnp.float64)
        pixels = jnp.asarray(pixel_indices, dtype=jnp.int32)
        npix = self.density_table.shape[0]
        pixels = jnp.clip(pixels, 0, npix - 1)

        z_grid = self.z_grid
        z_clamped = jnp.clip(z, z_grid[0], z_grid[-1])
        idx = jnp.searchsorted(z_grid, z_clamped, side="right") - 1
        idx = jnp.clip(idx, 0, z_grid.size - 2)

        rows = jnp.take(self.density_table, pixels, axis=0, mode="clip")
        z0 = z_grid[idx]
        z1 = z_grid[idx + 1]
        denom = jnp.where(z1 > z0, z1 - z0, 1.0)
        frac = (z_clamped - z0) / denom

        gather_idx = idx[..., None]
        dens0 = jnp.take_along_axis(rows, gather_idx, axis=1).squeeze(axis=-1)
        dens1 = jnp.take_along_axis(rows, gather_idx + 1, axis=1).squeeze(axis=-1)
        dens = dens0 + frac * (dens1 - dens0)
        return jnp.maximum(dens, self.epsilon)


@dataclass
class GWCatalogData:
    """
    Complete GW event catalog.

    Attributes:
        events: List of detected GW events
        injections: Selection effects data (optional)
        selection_function: Custom selection function (optional)
    """
    events: List[GWEventData]
    injections: Optional[GWInjectionSet] = None
    selection_function: Optional[Callable] = None
    galaxy_catalog: Optional[GWGalaxyCatalog] = None

    @property
    def n_events(self) -> int:
        return len(self.events)


# ============================================================================
# Phase 2: Posterior/Injection 管线（JAX 版）
# ============================================================================


@dataclass
class PosteriorSamples:
    """探测器帧 posterior 样本容器，用于 JAX 权重计算。"""

    luminosity_distance: jnp.ndarray
    mass_1_det: jnp.ndarray
    mass_2_det: jnp.ndarray
    prior: jnp.ndarray
    right_ascension: Optional[jnp.ndarray] = None
    declination: Optional[jnp.ndarray] = None
    mass_frame: str = "detector"
    em_redshift: Optional[float] = None
    em_redshift_sigma: Optional[float] = None

    @property
    def n_samples(self) -> int:
        return int(self.luminosity_distance.shape[0])

    @staticmethod
    def _as_device_array(values, field: str, event_name: str) -> jnp.ndarray:
        try:
            arr = jnp.asarray(values, dtype=jnp.float64)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"事件 {event_name} 字段 {field} 无法转换为数值数组") from exc
        if arr.ndim != 1:
            raise ValueError(f"事件 {event_name} 字段 {field} 需要为一维数组")
        return arr

    @classmethod
    def from_event(cls, event: GWEventData) -> 'PosteriorSamples':
        samples = event.posterior_samples
        mass_frame = event.mass_frame

        if isinstance(samples, dict):
            d_l_raw = samples.get('luminosity_distance', samples.get('distance'))
            if 'mass_1_det' in samples or 'mass_2_det' in samples:
                m1_raw = samples.get('mass_1_det')
                m2_raw = samples.get('mass_2_det')
                mass_frame = 'detector'
            elif 'mass_1_source' in samples or 'mass_2_source' in samples:
                m1_raw = samples.get('mass_1_source')
                m2_raw = samples.get('mass_2_source')
                mass_frame = 'source'
            else:
                m1_raw = samples.get('mass_1')
                m2_raw = samples.get('mass_2')
            ra_raw = samples.get('right_ascension')
            dec_raw = samples.get('declination')
            if d_l_raw is None or m1_raw is None or m2_raw is None:
                raise ValueError(
                    f"事件 {event.name} posterior dict 缺少必需字段 'luminosity_distance'/'mass_1_det'/'mass_2_det'"
                )
            d_l = cls._as_device_array(d_l_raw, 'luminosity_distance', event.name)
            m1_det = cls._as_device_array(m1_raw, 'mass_1_det', event.name)
            m2_det = cls._as_device_array(m2_raw, 'mass_2_det', event.name)
            if (ra_raw is None) != (dec_raw is None):
                raise ValueError(f"事件 {event.name} sky 坐标必须同时提供 RA 和 Dec")
            ra_arr = cls._as_device_array(ra_raw, 'right_ascension', event.name) if ra_raw is not None else None
            dec_arr = cls._as_device_array(dec_raw, 'declination', event.name) if dec_raw is not None else None
        else:
            arr = jnp.asarray(samples, dtype=jnp.float64)
            if arr.ndim != 2 or arr.shape[1] < 4:
                raise ValueError(
                    "posterior 数组必须包含 [d_L, z, m1_source, m2_source] 四列"
                )
            d_l = arr[:, 0]
            z = arr[:, 1]
            m1_src = arr[:, 2]
            m2_src = arr[:, 3]
            factor = 1.0 + z
            m1_det = m1_src * factor
            m2_det = m2_src * factor
            ra_arr = None
            dec_arr = None
            mass_frame = 'detector'

        if d_l.shape != m1_det.shape or d_l.shape != m2_det.shape:
            raise ValueError(f'事件 {event.name} posterior 样本长度不一致')

        prior = event.prior_samples
        if prior is None:
            prior = jnp.power(d_l, 2.0)
        prior = cls._as_device_array(prior, 'prior', event.name)

        if prior.shape != d_l.shape:
            raise ValueError(
                f"事件 {event.name} 的 prior 数量 {prior.shape} 与 posterior {d_l.shape} 不匹配"
            )

        return cls(
            luminosity_distance=d_l,
            mass_1_det=m1_det,
            mass_2_det=m2_det,
            prior=prior,
            right_ascension=ra_arr,
            declination=dec_arr,
            mass_frame=mass_frame,
            em_redshift=event.em_redshift,
            em_redshift_sigma=event.em_redshift_sigma,
        )


class PosteriorCatalog:
    """JAX-native posterior采样容器，使用统一矩阵和掩码。"""

    def __init__(
        self,
        events: Sequence[GWEventData],
        nparallel: Optional[int],
        random_seed: Optional[int] = None,
        galaxy_catalog: Optional[GWGalaxyCatalog] = None,
    ) -> None:
        if not events:
            raise ValueError('posterior catalog 至少包含一个事件')

        samples = [PosteriorSamples.from_event(ev) for ev in events]
        self.event_names = [ev.name for ev in events]
        mass_frames = {sample.mass_frame for sample in samples}
        if len(mass_frames) != 1:
            raise ValueError(f"posterior catalog 包含混合 mass_frame: {mass_frames}")
        self.mass_frame = mass_frames.pop()
        target = int(nparallel) if nparallel is not None else max(s.n_samples for s in samples)
        if target <= 0:
            raise ValueError('posterior catalog target 样本数必须大于 0')

        self.nparallel = target
        self.n_events = len(samples)
        seed = 0 if random_seed is None else int(random_seed)
        keys = list(jrandom.split(jrandom.PRNGKey(seed), self.n_events))

        mass1_rows = []
        mass2_rows = []
        distance_rows = []
        prior_rows = []
        mask_rows = []
        Ns = []
        ra_rows: List[jnp.ndarray] = []
        dec_rows: List[jnp.ndarray] = []
        em_mu: List[float] = []
        em_sigma: List[float] = []
        em_mask: List[bool] = []
        needs_pixels = galaxy_catalog is not None

        for idx, sample in enumerate(samples):
            if sample.n_samples == 0:
                raise ValueError(f'事件 {self.event_names[idx]} 不包含 posterior 样本')
            n_take = int(min(target, sample.n_samples))
            key = keys[idx]
            perm = jrandom.permutation(key, sample.n_samples)
            take_idx = perm[:n_take]

            def _select(values: jnp.ndarray) -> jnp.ndarray:
                taken = jnp.take(values, take_idx, axis=0)
                pad = target - n_take
                if pad == 0:
                    return taken
                return jnp.pad(taken, (0, pad), mode='edge')

            mass1_rows.append(_select(sample.mass_1_det))
            mass2_rows.append(_select(sample.mass_2_det))
            distance_rows.append(_select(sample.luminosity_distance))
            prior_rows.append(_select(sample.prior))
            mask_rows.append(jnp.pad(jnp.ones((n_take,), dtype=bool), (0, target - n_take)))
            Ns.append(float(n_take))

            if needs_pixels:
                if sample.right_ascension is None or sample.declination is None:
                    raise ValueError(
                        f"事件 {self.event_names[idx]} 缺少天空坐标，无法使用星系 catalog"
                    )
                ra_rows.append(_select(sample.right_ascension))
                dec_rows.append(_select(sample.declination))

            if sample.em_redshift is None:
                em_mu.append(0.0)
                em_sigma.append(1.0)
                em_mask.append(False)
            else:
                em_mu.append(float(sample.em_redshift))
                em_sigma.append(float(sample.em_redshift_sigma or 1.0))
                em_mask.append(True)

        self.mass_1_det = jnp.stack(mass1_rows, axis=0)
        self.mass_2_det = jnp.stack(mass2_rows, axis=0)
        self.luminosity_distance = jnp.stack(distance_rows, axis=0)
        self.prior = jnp.stack(prior_rows, axis=0)
        self.valid_mask = jnp.stack(mask_rows, axis=0)
        self.Ns_array = jnp.asarray(Ns, dtype=jnp.float64)
        self.sum_weights = jnp.zeros((self.n_events,), dtype=jnp.float64)
        self.sum_weights_squared = jnp.zeros((self.n_events,), dtype=jnp.float64)
        self.log_weights: Optional[jnp.ndarray] = None
        self.pixel_indices: Optional[jnp.ndarray] = None
        self.em_redshift_mu = jnp.asarray(em_mu, dtype=jnp.float64)
        self.em_redshift_sigma = jnp.asarray(em_sigma, dtype=jnp.float64)
        self.em_redshift_mask = jnp.asarray(em_mask, dtype=bool)
        if needs_pixels:
            ra_matrix = np.stack([np.asarray(row) for row in ra_rows], axis=0)
            dec_matrix = np.stack([np.asarray(row) for row in dec_rows], axis=0)
            pixels = _compute_healpix_pixels(ra_matrix, dec_matrix, galaxy_catalog.nside)
            self.pixel_indices = jnp.asarray(pixels, dtype=jnp.int32)

    def update_weights(
        self,
        rate_model,
        *,
        galaxy_catalog: Optional[GWGalaxyCatalog] = None,
        cosmology=None,
        pop_params: Optional[Dict[str, float]] = None,
    ) -> None:
        if cosmology is not None or pop_params is not None:
            try:
                logw = rate_model.log_rate_PE(
                    self.luminosity_distance,
                    self.mass_1_det,
                    self.mass_2_det,
                    self.prior,
                    cosmology,
                    pop_params or {},
                    mass_frame=self.mass_frame,
                )
            except TypeError:
                logw = rate_model.log_rate_PE(
                    self.prior,
                    self.luminosity_distance,
                    self.mass_1_det,
                    self.mass_2_det,
                )
        else:
            logw = rate_model.log_rate_PE(
                self.prior,
                self.luminosity_distance,
                self.mass_1_det,
                self.mass_2_det,
            )
        needs_z = galaxy_catalog is not None or self.em_redshift_mask is not None
        if needs_z:
            if cosmology is None:
                raise ValueError('使用红移先验时需要传入 cosmology 实例')
            z_samples = cosmology.dl_to_z(self.luminosity_distance)
            if galaxy_catalog is not None:
                if self.pixel_indices is None:
                    raise ValueError('Posterior catalog 缺少像素索引，无法应用星系先验')
                log_prior = galaxy_catalog.log_density(
                    z_samples.reshape(-1),
                    self.pixel_indices.reshape(-1),
                )
                log_prior = log_prior.reshape(self.luminosity_distance.shape)
                logw = logw + log_prior
            if self.em_redshift_mask is not None:
                mu = self.em_redshift_mu[:, None]
                sigma = self.em_redshift_sigma[:, None]
                log_em = gaussian_log_pdf(z_samples, mu, sigma)
                logw = jnp.where(self.em_redshift_mask[:, None], logw + log_em, logw)
        logw = jnp.where(self.valid_mask, logw, -jnp.inf)
        self.log_weights = logw
        Ns = self.Ns_array
        logsum = jsp.special.logsumexp(logw, axis=1)
        logsum_sq = jsp.special.logsumexp(2.0 * logw, axis=1)
        sum_weights = jnp.exp(logsum) / Ns
        sum_weights_sq = jnp.exp(logsum_sq) / jnp.square(Ns)
        self.sum_weights = sum_weights
        self.sum_weights_squared = sum_weights_sq

    def get_effective_number_of_PE(self) -> jnp.ndarray:
        denom = self.sum_weights_squared
        valid = denom > 0.0
        safe = jnp.where(valid, denom, 1.0)
        neff = jnp.where(valid, jnp.square(self.sum_weights) / safe, 0.0)
        return neff

    def variance_correction(self, neff: jnp.ndarray, return_jnp: bool = False) -> jnp.ndarray | float:
        Ns = self.Ns_array
        neff = jnp.asarray(neff, dtype=jnp.float64)
        term = jnp.where(
            neff > 0.0,
            (1.0 / neff) * (1.0 - neff / Ns),
            0.0,
        )
        total = jnp.sum(term)
        return total if return_jnp else float(total)


def _compute_healpix_pixels(ra: np.ndarray, dec: np.ndarray, nside: int) -> np.ndarray:
    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "healpy is required for galaxy catalog support; install via pip install healpy"
        ) from exc

    theta = 0.5 * np.pi - dec
    phi = np.mod(ra, 2.0 * np.pi)
    pixels = hp.ang2pix(nside, theta, phi, nest=False)
    return pixels.astype(np.int32)


class InjectionSampler:
    """JAX 版注入处理器，使用向量化数组计算选择效应。"""

    def __init__(self, injections: GWInjectionSet):
        if injections is None:
            raise ValueError('GW 分析必须提供注入数据以评估选择效应')

        data = injections.injections
        if isinstance(data, dict):
            def _extract(field: str, alt: Optional[str] = None):
                if field in data:
                    return data[field]
                if alt and alt in data:
                    return data[alt]
                return None

            d_l_raw = _extract('luminosity_distance', 'distance')
            m1_raw = _extract('mass_1_det', 'mass_1')
            m2_raw = _extract('mass_2_det', 'mass_2')
            if d_l_raw is None or m1_raw is None or m2_raw is None:
                raise ValueError('注入 dict 缺少 luminosity_distance/mass_1/mass_2 字段')
            d_l = jnp.asarray(d_l_raw, dtype=jnp.float64)
            m1 = jnp.asarray(m1_raw, dtype=jnp.float64)
            m2 = jnp.asarray(m2_raw, dtype=jnp.float64)
        else:
            arr = jnp.asarray(data, dtype=jnp.float64)
            if arr.ndim != 2 or arr.shape[1] < 4:
                raise ValueError('注入数组必须包含 [d_L, z, m1_det, m2_det] 四列')
            d_l = arr[:, 0]
            m1 = arr[:, 2]
            m2 = arr[:, 3]

        if d_l.shape != m1.shape or d_l.shape != m2.shape:
            raise ValueError('注入样本字段长度必须一致')

        prior = jnp.asarray(injections.weights, dtype=jnp.float64)
        if prior.shape != d_l.shape:
            raise ValueError('注入 prior 与样本数量不一致')

        self._distance = d_l
        self._mass_1_det = m1
        self._mass_2_det = m2
        self._prior = prior
        self._snr = (
            jnp.asarray(injections.snr, dtype=jnp.float64)
            if injections.snr is not None
            else None
        )
        self._ifar = (
            jnp.asarray(injections.ifar, dtype=jnp.float64)
            if injections.ifar is not None
            else None
        )
        self.snr_threshold = float(injections.snr_threshold or 0.0)
        self.ifar_threshold = float(injections.ifar_threshold or 0.0)
        self.mass_frame = injections.mass_frame
        self.ntotal = float(injections.n_total)
        self.ntotal_device = jnp.asarray(self.ntotal, dtype=jnp.float64)
        self.Tobs = float(injections.T_obs)
        self.Tobs_device = jnp.asarray(self.Tobs, dtype=jnp.float64)

        self.distance: Optional[jnp.ndarray] = None
        self.mass_1_det: Optional[jnp.ndarray] = None
        self.mass_2_det: Optional[jnp.ndarray] = None
        self.prior: Optional[jnp.ndarray] = None
        self.log_weights: Optional[jnp.ndarray] = None
        self.pseudo_rate: float = 0.0
        self._pseudo_rate_jax = jnp.asarray(0.0, dtype=jnp.float64)
        self.neff: float = 0.0
        self._neff_jax = jnp.asarray(0.0, dtype=jnp.float64)
        self._second_moment: float = 0.0
        self._second_moment_jax = jnp.asarray(0.0, dtype=jnp.float64)
        self._valid_indices: Optional[jnp.ndarray] = None
        self.apply_threshold()

    def apply_threshold(self):
        mask = jnp.ones_like(self._prior, dtype=bool)
        if self._snr is not None and self.snr_threshold > 0.0:
            mask = jnp.logical_and(mask, self._snr >= self.snr_threshold)
        if self.ifar_threshold > 0.0:
            if self._ifar is None:
                raise ValueError('设置 ifar_threshold 但注入数据缺少 IFAR')
            mask = jnp.logical_and(mask, self._ifar >= self.ifar_threshold)

        indices = jnp.asarray(jnp.nonzero(mask, size=None)[0])
        if indices.shape[0] == 0:
            raise ValueError('注入筛选后没有检测到的注入事件，无法评估选择效应')

        self.distance = jnp.take(self._distance, indices, axis=0)
        self.mass_1_det = jnp.take(self._mass_1_det, indices, axis=0)
        self.mass_2_det = jnp.take(self._mass_2_det, indices, axis=0)
        self.prior = jnp.take(self._prior, indices, axis=0)
        self._valid_indices = indices

    def update_weights(self, rate_model, *, cosmology=None, pop_params: Optional[Dict[str, float]] = None) -> None:
        if self.prior is None or self.distance is None:
            raise RuntimeError('请先调用 apply_threshold() 以初始化注入样本')

        if cosmology is not None or pop_params is not None:
            try:
                logw = rate_model.log_rate_injections(
                    self.distance,
                    self.mass_1_det,
                    self.mass_2_det,
                    self.prior,
                    cosmology,
                    pop_params or {},
                    mass_frame=self.mass_frame,
                )
            except TypeError:
                logw = rate_model.log_rate_injections(
                    self.prior,
                    self.distance,
                    self.mass_1_det,
                    self.mass_2_det,
                )
        else:
            logw = rate_model.log_rate_injections(
                self.prior,
                self.distance,
                self.mass_1_det,
                self.mass_2_det,
            )
        logw = jnp.asarray(logw, dtype=jnp.float64)
        logsum = jsp.special.logsumexp(logw)
        logsum_sq = jsp.special.logsumexp(2.0 * logw)
        mean = jnp.exp(logsum) / self.ntotal
        second = jnp.exp(logsum_sq) / (self.ntotal ** 2)
        var = second - (mean ** 2) / self.ntotal
        tiny = jnp.finfo(jnp.float64).tiny
        var = jnp.where(var <= 0.0, tiny, var)
        self.log_weights = logw
        self._pseudo_rate_jax = mean
        self._second_moment_jax = jnp.maximum(second, tiny)
        neff_val = (mean ** 2) / var
        self._neff_jax = neff_val
        self.pseudo_rate = _safe_float(mean)
        self._second_moment = _safe_float(self._second_moment_jax)
        self.neff = _safe_float(neff_val)

    def effective_injections_number(self) -> float:
        return float(_safe_float(self.neff))

    def effective_injections_number_jax(self) -> jnp.ndarray:
        return self._neff_jax

    def pseudo_rate_device(self) -> jnp.ndarray:
        return self._pseudo_rate_jax

    def expected_number_detections(self) -> float:
        return float(_safe_float(self.pseudo_rate) * self.Tobs)

    def expected_number_detections_jax(self) -> jnp.ndarray:
        return self._pseudo_rate_jax * self.Tobs_device


# ============================================================================
# Phase 1: JAX Utilities (Data Processing Only)
# ============================================================================

@jit
def effective_sample_size(weights: jnp.ndarray) -> float:
    """
    Compute effective sample size (ESS).

    ESS = (Σ w_i)² / Σ w_i²

    Args:
        weights: Normalized importance weights

    Returns:
        Effective number of independent samples
    """
    return jnp.sum(weights)**2 / jnp.sum(weights**2)


@jit
def gaussian_log_pdf(x: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Gaussian log probability density."""
    return -0.5 * jnp.log(2 * jnp.pi * sigma**2) - 0.5 * ((x - mu) / sigma)**2


def poisson_log_pmf(k: int, lam: float) -> float:
    """Poisson log probability mass function."""
    # log P(k | λ) = k log λ - λ - log(k!)
    # For likelihood, ignore constant log(k!)
    return k * jnp.log(lam) - lam


def _safe_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError, jax.errors.TracerIntegerConversionError, jax.errors.ConcretizationTypeError):
        return value


# ============================================================================
# Phase 2: GW Rate Model (Combination Layer)
# ============================================================================

class GWRateModel:
    """
    Gravitational wave population model combining cosmology, mass, and rate.

    Implements differential merger rate:
        dN/dVc/dt/dm1/dm2 = R(z) * p(m1, m2 | theta) * (1+z)^-1

    Args:
        mass_prior: Binary mass distribution
        rate_evolution: Merger rate evolution R(z)
        scale_free: If True, rate is relative (typical for hierarchical inference)

    Usage:
        >>> mass = PowerLawMass()
        >>> rate = MadauRate()
        >>> model = GWRateModel(mass, rate)
        >>>
        >>> # Compute differential rate
        >>> z = jnp.array([0.1, 0.2, 0.3])
        >>> m1 = jnp.array([30.0, 35.0, 40.0])
        >>> m2 = jnp.array([25.0, 28.0, 32.0])
        >>> pop_params = {'alpha': -2.35, 'gamma': 2.7, ...}
        >>>
        >>> dN = model.differential_rate(z, m1, m2, cosmology, pop_params)
    """

    def __init__(
        self,
        mass_prior: MassPrior,
        rate_evolution: RateEvolution,
        scale_free: bool = True
    ):
        self.mass_prior = mass_prior
        self.rate_evolution = rate_evolution
        self.scale_free = scale_free
        mass_params = list(getattr(self.mass_prior, "parameter_names", ()))
        rate_params = list(getattr(self.rate_evolution, "parameter_names", ()))
        self.population_parameters = mass_params + rate_params
        if not scale_free:
            self.population_parameters.append('R0')

    def log_rate_PE(
        self,
        d_L: jnp.ndarray,
        m1_detector: jnp.ndarray,
        m2_detector: jnp.ndarray,
        pe_prior: jnp.ndarray,
        cosmology,
        pop_params: Dict[str, float],
        *,
        mass_frame: str = "detector",
    ) -> jnp.ndarray:
        """
        Calculate importance sampling weights for PE posterior samples.

        This is the KEY METHOD for hierarchical Bayesian inference!

        CORRECT FORMULA (from icarogw rates.py:71-99):
            log_weights = log[p(m_source)] + log[R(z)] + log[dVc/dz]
                          - log[PE_prior] - log[Jacobian] - log(1+z)

        Where:
            z = z(d_L, cosmology)  ← CRITICAL: recalculate from d_L!
            m_source = m_detector / (1+z)
            dVc/dz = cosmology.dVc_dz(z)  [Gpc³]
            Jacobian = (1+z) * ddL/dz
            PE_prior = d_L^2

        Args:
            d_L: Luminosity distance samples [Mpc]
            m1_detector: Primary mass (detector frame) [M_sun]
            m2_detector: Secondary mass (detector frame) [M_sun]
            pe_prior: PE prior weights (d_L^2)
            cosmology: HIcosmo cosmology object (LCDM)
            pop_params: Population parameters (alpha, beta, gamma, etc.)

        Returns:
            log_weights: shape same as input arrays

        Mathematical Explanation:
            The key insight is that z MUST be recalculated from d_L using the
            trial cosmology. When we test H0=20 vs H0=70, the SAME d_L corresponds
            to DIFFERENT redshifts!

            At H0=20: d_L=500 Mpc → z ≈ 0.6
            At H0=70: d_L=500 Mpc → z ≈ 0.12

            This z-dependence in log_R_z(z), dVc/dz, and Jacobian is what
            provides H0 sensitivity!

        References:
            icarogw/icarogw/rates.py lines 71-99 (log_rate_PE method)
            icarogw/icarogw/conversions.py line 918 (detector2source_jacobian_q)
        """
        # Step 1: Recalculate z from d_L using trial cosmology (CRITICAL!)
        z = cosmology.dl_to_z(d_L)

        # Step 2: Convert masses to source frame
        if mass_frame == "detector":
            m1_source = m1_detector / (1 + z)
            m2_source = m2_detector / (1 + z)
            jacobian_mass = (1 + z) ** 2
        elif mass_frame == "source":
            m1_source = m1_detector
            m2_source = m2_detector
            jacobian_mass = 1.0
        else:
            raise ValueError(f"mass_frame must be 'detector' or 'source', got {mass_frame}")

        # Step 3: Mass prior evaluation
        log_p_mass = self.mass_prior.log_prob(m1_source, m2_source, **pop_params)

        # Step 4: Rate evolution evaluation
        log_R_z = self.rate_evolution.log_rate(z, **pop_params)

        # Step 5: Differential comoving volume (cosmology-dependent!)
        # This is dVc/dz = 4π × D_H × D_M² / E(z) in Gpc³
        dVc_dz = cosmology.dVc_dz(z)  # Gpc³
        log_dVc_dz = jnp.log(dVc_dz)

        # Step 6: Jacobian for detector<->source frame transformation
        # For transformation (m1_det, m2_det, d_L) → (m1_src, m2_src, z):
        #   |J| = (1+z)² × ddL/dz
        # If masses are already in source frame, the mass Jacobian is 1.
        ddL_dz = cosmology.ddL_dz(z)  # Mpc
        jacobian = jacobian_mass * ddL_dz
        log_jacobian = jnp.log(jacobian)

        # Step 7: PE prior
        log_pe_prior = jnp.log(pe_prior)  # pe_prior = d_L^2

        # Step 8: Combine all terms (importance sampling weight formula!)
        #
        # CORRECT FORMULA (matching icarogw CBC_vanilla_rate):
        #   w = [p(m_src) × R(z) × dVc/dz / (1+z)] / [π_PE × |Jacobian|]
        #
        # This is the exact formula from icarogw.rates.CBC_vanilla_rate.log_rate_PE
        log_weights = (
            log_p_mass          # Mass prior p(m1, m2)
            + log_R_z           # Rate evolution R(z)
            + log_dVc_dz        # Differential comoving volume
            - jnp.log1p(z)      # Time dilation factor 1/(1+z)
            - log_pe_prior      # PE prior (typically d_L²)
            - log_jacobian      # Detector<->source frame Jacobian
        )
        if not self.scale_free:
            R0 = pop_params.get("R0", 1.0)
            log_weights = log_weights + jnp.log(R0)

        return log_weights

    def log_rate_injections(
        self,
        d_L: jnp.ndarray,
        m1_detector: jnp.ndarray,
        m2_detector: jnp.ndarray,
        pe_prior: jnp.ndarray,
        cosmology,
        pop_params: Dict[str, float],
        *,
        mass_frame: str = "detector",
    ) -> jnp.ndarray:
        return self.log_rate_PE(
            d_L,
            m1_detector,
            m2_detector,
            pe_prior,
            cosmology,
            pop_params,
            mass_frame=mass_frame,
        )

    def differential_rate(
        self,
        z: jnp.ndarray,
        m1: jnp.ndarray,
        m2: jnp.ndarray,
        cosmology,
        pop_params: Dict[str, float]
    ) -> jnp.ndarray:
        """
        Compute differential merger rate density.

        Args:
            z: Source redshift(s)
            m1: Primary mass(es) [M_sun]
            m2: Secondary mass(es) [M_sun]
            cosmology: HIcosmo cosmology object
            pop_params: Population parameters (alpha, beta, gamma, etc.)

        Returns:
            dN/dVc/dt/dm1/dm2 [Gpc^-3 yr^-1 M_sun^-2]

        Formula:
            dN = R(z; theta_rate) * p(m1, m2 | theta_mass) * dVc/dz / (1+z)

        Note:
            The (1+z)^-1 factor converts source-frame to observer-frame time.
        """
        # Merger rate evolution (intrinsic rate per comoving volume per source time)
        # R(z): [Gpc^-3 yr^-1] in source frame
        R_z = self.rate_evolution.rate(z, **pop_params)

        # Mass distribution (probability density per M_sun^2)
        # p(m1,m2): [M_sun^-2]
        log_p_mass = self.mass_prior.log_prob(m1, m2, **pop_params)
        p_mass = jnp.exp(log_p_mass)

        # Time dilation factor: dt_source = dt_obs / (1+z)
        time_dilation = 1.0 / (1.0 + z)

        # Differential merger rate density:
        # dN/(dVc dt_obs dm1 dm2) = R(z) * p(m1,m2) / (1+z)
        # Units: [Gpc^-3 yr^-1 M_sun^-2] * [M_sun^-2] * [dimensionless]
        #      = [Gpc^-3 yr^-1 M_sun^-2]
        dN_rate = R_z * p_mass * time_dilation

        return dN_rate

    def expected_detections(
        self,
        injections: GWInjectionSet,
        cosmology,
        pop_params: Dict[str, float],
        scale_free: bool = False
    ) -> float:
        """
        Calculate expected detections N_exp(θ) using correct physics formula.
        
        This implementation uses the SAME formula as log_rate_PE to ensure consistency.
        
        Formula:
            N_exp = T_obs × (Σ exp(log_weights)) / N_total
            
        where log_weights contains:
            - log p(m1, m2): mass distribution
            - log R(z): rate evolution
            - log dVc/dz: comoving volume element
            - log(1+z): time dilation
            - log π_inj: injection sampling prior
            - log |J_d→s|: Jacobian (detector → source frame)
        
        Args:
            injections: GW injection campaign data
            cosmology: Cosmology model
            pop_params: Population parameters
            scale_free: If True, return Vdet [Gpc³]; if False, return N_exp
        
        Returns:
            Expected number of detections or explorable volume
        """
        # Step 1: Extract injection data (assumes column order: [d_L, z, m1_det, m2_det])
        d_L_inj = injections.injections[:, 0]
        m1_det_inj = injections.injections[:, 2]
        m2_det_inj = injections.injections[:, 3]
        
        # Step 2: Compute redshift from luminosity distance
        z_inj = cosmology.dl_to_z(d_L_inj)
        
        # Step 3: Transform to source frame masses
        mass_frame = getattr(injections, "mass_frame", "detector")
        if mass_frame == "detector":
            m1_source_inj = m1_det_inj / (1 + z_inj)
            m2_source_inj = m2_det_inj / (1 + z_inj)
            jacobian_mass = (1 + z_inj) ** 2
        elif mass_frame == "source":
            m1_source_inj = m1_det_inj
            m2_source_inj = m2_det_inj
            jacobian_mass = 1.0
        else:
            raise ValueError(f"mass_frame must be 'detector' or 'source', got {mass_frame}")
        
        # Step 4: Calculate mass distribution log p(m1, m2)
        log_p_mass = self.mass_prior.log_prob(
            m1_source_inj, m2_source_inj, **pop_params
        )
        
        # Step 5: Calculate rate evolution log R(z)
        log_R_z = self.rate_evolution.log_rate(z_inj, **pop_params)
        
        # Step 6: Calculate dVc/dz
        dVc_dz = cosmology.dVc_dz(z_inj)  # Units: Gpc³
        log_dVc_dz = jnp.log(dVc_dz)
        
        # Step 7: Calculate Jacobian |J_d→s| = (1+z)² × ddL/dz (or ddL/dz for source-frame masses)
        ddL_dz = cosmology.ddL_dz(z_inj)
        jacobian = jacobian_mass * ddL_dz
        log_jacobian = jnp.log(jacobian)
        
        # Step 8: Injection prior (sampling prior)
        pe_prior_inj = injections.weights
        log_pe_prior = jnp.log(pe_prior_inj + 1e-300)  # Avoid log(0)
        
        # Step 9: Combine all terms (SAME as log_rate_PE formula)
        log_weights = (
            log_p_mass          # p(m1, m2)
            + log_R_z           # R(z)
            + log_dVc_dz        # dVc/dz
            - jnp.log1p(z_inj)  # 1/(1+z) time dilation
            - log_pe_prior      # 1/π_inj
            - log_jacobian      # 1/|J|
        )
        if not self.scale_free:
            R0 = pop_params.get("R0", 1.0)
            log_weights = log_weights + jnp.log(R0)
        
        # Step 10: Numerically stable summation
        import jax.scipy.special
        log_sum = jax.scipy.special.logsumexp(log_weights)
        sum_weights = jnp.exp(log_sum)
        
        # Step 11: Pseudo-rate
        pseudo_rate = sum_weights / injections.n_total
        
        # Step 12: Expected detections
        N_exp = pseudo_rate * injections.T_obs
        
        return float(N_exp)


class _LegacyGWStandardSirenLikelihood(Likelihood):
    """
    GW Standard Siren Hierarchical Likelihood.

    Correctly integrates with HIcosmo by:
    1. Using cosmology.luminosity_distance(z) - NO hardcoding
    2. Single file implementation in hicosmo/likelihoods/
    3. Complete icarogw functionality via modular wrappers

    Phase 1 (MVP): Basic event likelihood + simplified selection
    Phase 2: Full rate models + injection-based selection
    Phase 3: EM counterparts + sky pixelization
    Phase 4: JIT optimization + autodiff gradients

    Usage:
        >>> catalog = create_mock_gw_catalog(n_events=10, include_injections=True)
        >>> gw_like = GWStandardSirenLikelihood(catalog=catalog)
        >>>
        >>> from hicosmo.models import LCDM
        >>> model = LCDM(H0=70, Omega_m=0.3)
        >>> log_L = gw_like.log_likelihood(model)
        >>>
        >>> # Use in MCMC
        >>> from hicosmo import hicosmo
        >>> inference = hicosmo(
        ...     cosmology='LCDM',
        ...     likelihood=gw_like,
        ...     free_params=['H0', 'Omega_m']
        ... )
    """

    def __init__(
        self,
        catalog: Optional[GWCatalogData] = None,
        data_path: Optional[str] = None,
        name: Optional[str] = None,
        apply_selection_bias: bool = True,
        min_eff_samples: int = 50,
        # Phase 2 parameters
        mass_prior: Optional[MassPrior] = None,
        rate_evolution: Optional[RateEvolution] = None,
        use_galaxy_catalog: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize GW standard siren likelihood.

        Args:
            catalog: GWCatalogData with events and injections
            data_path: Path to pickled catalog (alternative)
            name: Likelihood identifier
            apply_selection_bias: Include selection effects
            min_eff_samples: Minimum ESS per event
            mass_prior: Binary mass distribution (Phase 2)
            rate_evolution: Merger rate R(z) (Phase 2)
        """
        super().__init__(name=name or "gw_standard_siren", data_path=data_path, **kwargs)

        self.catalog = catalog
        self.apply_selection_bias = apply_selection_bias
        self.min_eff_samples = min_eff_samples
        self.use_galaxy_catalog = bool(use_galaxy_catalog)

        # Phase 2 components (default if not provided)
        self.mass_prior = mass_prior or PowerLawMass()
        self.rate_evolution = rate_evolution or MadauRate()

        # Phase 3: Create combined rate model
        self.rate_model = GWRateModel(self.mass_prior, self.rate_evolution)

        # Storage for JAX arrays
        self._posterior_samples_jax: Optional[jnp.ndarray] = None
        self._weights_jax: Optional[jnp.ndarray] = None
        self._n_events: int = 0
        self._galaxy_catalog: Optional[GWGalaxyCatalog] = None

        self.initialize()

    @classmethod
    def from_gwtc3(
        cls,
        *,
        data_root: Union[str, Path] = "data/gwtc-3",
        population: str = "bbh",
        events: Optional[Sequence[str]] = None,
        events_path: Optional[Union[str, Path]] = None,
        max_events: Optional[int] = None,
        injections_path: Optional[Union[str, Path]] = None,
        injections_meta: Optional[Dict[str, Any]] = None,
        galaxy_catalog_path: Optional[Union[str, Path]] = None,
        selection: Optional[str] = None,
        snr_threshold: Optional[float] = None,
        ifar_threshold: Optional[float] = None,
        mass_frame: Optional[str] = None,
        pe_prior_mode: Optional[str] = None,
        pe_prior_bounds: Optional[Tuple[float, float]] = None,
        pe_data_root: Optional[Union[str, Path]] = None,
        pe_data_prefer: str = "cosmo",
        pe_data_dataset: Optional[str] = None,
        em_redshift_map: Optional[Dict[str, Dict[str, float]]] = None,
        extra_events: Optional[Sequence[GWEventData]] = None,
        **kwargs: Any,
    ) -> "GWStandardSirenLikelihood":
        """Construct GWStandardSirenLikelihood from GWTC-3 data."""
        catalog = load_gwtc3_catalog(
            data_root=data_root,
            population=population,
            events=events,
            events_path=events_path,
            max_events=max_events,
            injections_path=injections_path,
            injections_meta=injections_meta,
            galaxy_catalog_path=galaxy_catalog_path,
            selection=selection,
            snr_threshold=snr_threshold,
            ifar_threshold=ifar_threshold,
            mass_frame=mass_frame,
            pe_prior_mode=pe_prior_mode,
            pe_prior_bounds=pe_prior_bounds,
            pe_data_root=pe_data_root,
            pe_data_prefer=pe_data_prefer,
            pe_data_dataset=pe_data_dataset,
            em_redshift_map=em_redshift_map,
            extra_events=extra_events,
        )
        kwargs.setdefault("population", population)
        return cls(catalog=catalog, **kwargs)

    def _default_dataset_name(self) -> str:
        return "gwtc3"

    def _load_data(self) -> None:
        """Load catalog from file or validate provided data."""
        if self.catalog is None and self.data_path is not None:
            data_file = Path(self.data_path)
            if not data_file.exists():
                raise FileNotFoundError(f"GW catalog not found: {self.data_path}")

            with open(data_file, 'rb') as f:
                self.catalog = pickle.load(f)

        if self.catalog is None:
            raise ValueError(
                "No GW catalog provided. Pass 'catalog' or 'data_path'."
            )

        self._n_events = self.catalog.n_events

        # Validate ESS
        for event in self.catalog.events:
            ess = float(effective_sample_size(jnp.array(event.weights)))
            if ess < self.min_eff_samples:
                import warnings
                warnings.warn(
                    f"{event.name} has low ESS: {ess:.1f} < {self.min_eff_samples}"
                )

    def _setup_covariance(self) -> None:
        """
        Convert posterior samples to JAX arrays.

        Creates uniform-shape arrays by padding events with fewer samples.
        """
        max_samples = max(event.n_samples for event in self.catalog.events)

        posterior_list = []
        weights_list = []

        for event in self.catalog.events:
            samples = event.posterior_samples
            weights = event.weights

            # Pad to max_samples
            n_pad = max_samples - event.n_samples
            if n_pad > 0:
                samples_padded = np.vstack([
                    samples,
                    np.zeros((n_pad, samples.shape[1]))
                ])
                weights_padded = np.concatenate([
                    weights,
                    np.zeros(n_pad)
                ])
            else:
                samples_padded = samples
                weights_padded = weights

            posterior_list.append(samples_padded)
            weights_list.append(weights_padded)

        # Convert to JAX
        self._posterior_samples_jax = jnp.array(np.stack(posterior_list))
        self._weights_jax = jnp.array(np.stack(weights_list))

        # No covariance matrix for hierarchical likelihood
        self.inv_cov = None

    def get_requirements(self) -> Dict[str, Any]:
        """
        Specify required cosmological quantities.

        GW sirens primarily constrain H0 via d_L(z).
        """
        # Extract redshift range from posteriors
        all_z = np.concatenate([
            event.posterior_samples[:, 1]
            for event in self.catalog.events
        ])
        z_min = float(np.min(all_z))
        z_max = float(np.max(all_z))

        return {
            'luminosity_distance': {
                'z_range': (z_min, z_max),
                'description': 'GW distance ladder'
            }
        }

    def theory(self, cosmology, **kwargs) -> jnp.ndarray:
        """
        Compute theoretical prediction for GW luminosity distances.

        This method is required by Likelihood base class but is not used directly
        in hierarchical likelihood. Instead, cosmology.luminosity_distance(z) is
        called within log_likelihood() for each posterior sample.

        Args:
            cosmology: HIcosmo cosmology model
            **kwargs: Additional arguments (unused)

        Returns:
            Placeholder array (not used in hierarchical inference)
        """
        # For hierarchical likelihood, theory is computed per-sample in log_likelihood()
        # This method exists to satisfy Likelihood ABC interface
        return jnp.array([0.0])

    def log_likelihood(self, cosmology, **kwargs) -> float:
        """
        Compute log-likelihood for given cosmology.

        KEY DESIGN: Uses cosmology.luminosity_distance(z) - NO hardcoding!

        Args:
            cosmology: HIcosmo cosmology model (LCDM, wCDM, etc.)
            **kwargs: Additional parameters (future: mass/rate params)

        Returns:
            Log-likelihood value

        Implementation:
            Phase 1: Sum over events + simplified selection
            Phase 2: Full hierarchical with rate models
            Phase 3: EM counterpart handling
            Phase 4: JIT-compiled for speed
        """
        log_L_total = 0.0

        # ============================================================================
        # HIERARCHICAL BAYESIAN INFERENCE (Correct Implementation)
        # ============================================================================
        # Based on icarogw hierarchical_likelihood.py:94-100
        #
        # Mathematical Background:
        #   PE posterior: p(θ | data, π_PE) where π_PE ∝ d_L^2
        #   Population model: p(θ | Λ, cosmology) = p(m1,m2 | Λ_mass) * R(z | Λ_rate)
        #
        #   Importance sampling weight:
        #       w_j = p(θ_j | Λ, cosmology) / π_PE(θ_j)
        #
        #   Event likelihood:
        #       L_i = (1/N_PE) * Σ_j w_j
        #
        #   Total data likelihood:
        #       log L_data = Σ_i log L_i
        #
        # Reference: icarogw2.0_tutorial/CBC_vanilla_rate/vanilla_rate.ipynb Cell 11
        # ============================================================================

        for i, event in enumerate(self.catalog.events):
            # Extract PE posterior samples [d_L, z, m1_source, m2_source]
            samples = self._posterior_samples_jax[i]

            d_L_obs = samples[:, 0]  # Luminosity distance [Mpc]
            z_obs_pe = samples[:, 1]  # Redshift (from PE, DO NOT use for cosmology!)
            m1_source = samples[:, 2]  # Primary mass (source frame) [M_sun]
            m2_source = samples[:, 3]  # Secondary mass (source frame) [M_sun]

            # CRITICAL: Convert masses back to detector frame!
            # PE samples have source-frame masses, but log_rate_PE needs detector-frame
            # so it can recalculate z from d_L and convert back to source frame
            m1_detector = m1_source * (1 + z_obs_pe)
            m2_detector = m2_source * (1 + z_obs_pe)

            # PE prior: uniform in comoving volume → π_PE ∝ d_L^2
            # This is the KEY that was missing in the Gaussian approximation!
            pe_prior = d_L_obs**2

            # Extract population parameters from kwargs
            pop_params = kwargs.get('pop_params', None)
            if pop_params is None and len(kwargs) > 0:
                pop_params = kwargs

            # Calculate importance sampling weights using CORRECTED log_rate_PE method
            # Now includes: dVc/dz, Jacobian, log(1+z), and z recalculation!
            log_weights = self.rate_model.log_rate_PE(
                d_L_obs, m1_detector, m2_detector,
                pe_prior, cosmology, pop_params
            )

            # Marginalize over PE samples: L_event = (1/N_PE) * Σ exp(log_weights)
            # Use logsumexp for numerical stability
            n_samples = len(samples)
            log_L_event = jax.scipy.special.logsumexp(log_weights) - jnp.log(n_samples)

            log_L_total += float(log_L_event)

        # Phase 3: Selection effects with full rate model
        if self.apply_selection_bias and self.catalog.injections is not None:
            # Extract population parameters from kwargs
            # Support both {'pop_params': {...}} and direct {key: value, ...}
            pop_params = kwargs.get('pop_params', None)
            if pop_params is None and len(kwargs) > 0:
                # If pop_params not provided but kwargs has parameters,
                # use kwargs directly as pop_params
                pop_params = kwargs

            # Expected number of detections
            N_exp = self._compute_expected_detections(cosmology, pop_params)

            # Poisson term: -N_exp + N_det * log(N_exp)
            log_L_selection = poisson_log_pmf(self._n_events, N_exp)

            log_L_total += float(log_L_selection)

        return float(log_L_total)

    def _compute_expected_detections(self, cosmology, pop_params: Optional[Dict[str, float]] = None) -> float:
        """
        Compute expected number of detections using full rate model.

        Phase 3: Uses GWRateModel for accurate expected detections

        Args:
            cosmology: Cosmology model
            pop_params: Population hyperparameters. If None, uses defaults

        Returns:
            Expected number N_exp
        """
        if self.catalog.injections is None:
            # No selection effects
            return float(self._n_events)

        # Use default population parameters if not provided
        if pop_params is None:
            pop_params = {
                'alpha': -2.35,  # Salpeter-like
                'beta': 1.0,
                'mmin': 5.0,
                'mmax': 100.0,
                'gamma': 2.7,
                'kappa': 2.9,
                'zp': 2.47
            }

        # Phase 3: Use full rate model for expected detections
        N_exp = self.rate_model.expected_detections(
            self.catalog.injections,
            cosmology,
            pop_params
        )

        return float(N_exp)

    def get_derived_params(self, cosmology) -> Dict[str, float]:
        """Derived parameters from this likelihood."""
        derived = {'n_gw_events': self._n_events}
        if hasattr(cosmology, "params") and "H0" in cosmology.params:
            derived['H0_gw'] = float(cosmology.params['H0'])
        return derived



class GWStandardSirenLikelihood(Likelihood):
    """HIcosmo-native hierarchical likelihood with JAX acceleration."""

    def __init__(
        self,
        catalog: Optional[GWCatalogData] = None,
        data_path: Optional[str] = None,
        name: Optional[str] = None,
        apply_selection_bias: bool = True,
        nparallel: Optional[int] = 2048,
        neffPE: int = 20,
        neffINJ: Optional[int] = None,
        likelihood_variance_thr: Optional[float] = None,
        scale_free: bool = True,
        zmax: float = 10.0,
        cosmology_model: Optional[Type] = None,
        random_seed: Optional[int] = None,
        use_galaxy_catalog: bool = False,
        population: Optional[str] = None,
        mass_prior: Optional[MassPrior] = None,
        rate_evolution: Optional[RateEvolution] = None,
        population_params: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name or "gw_standard_siren", data_path=data_path, **kwargs)
        self.catalog = catalog
        self.apply_selection_bias = bool(apply_selection_bias)
        self.nparallel = nparallel
        self.neffPE = neffPE
        self.neffINJ = neffINJ
        self.likelihood_variance_thr = likelihood_variance_thr
        self.scale_free = scale_free
        self.zmax = zmax
        self.use_galaxy_catalog = bool(use_galaxy_catalog)
        self.population = population
        self._mass_prior_override = mass_prior
        self._rate_evolution_override = rate_evolution
        self._population_prior_overrides = dict(population_params) if population_params else None
        self._population_params_cached: Optional[Dict[str, float]] = None

        self.cosmology_model = cosmology_model or LCDM
        self._cosmology_class = self.cosmology_model
        self.random_seed = random_seed

        self._posterior_catalog = None
        self._injections = None
        self._rate_model = None
        self._galaxy_catalog: Optional[GWGalaxyCatalog] = None

        self.initialize()

    @classmethod
    def from_gwtc3(
        cls,
        *,
        data_root: Union[str, Path] = "data/gwtc-3",
        population: str = "bbh",
        events: Optional[Sequence[str]] = None,
        events_path: Optional[Union[str, Path]] = None,
        max_events: Optional[int] = None,
        injections_path: Optional[Union[str, Path]] = None,
        injections_meta: Optional[Dict[str, Any]] = None,
        galaxy_catalog_path: Optional[Union[str, Path]] = None,
        selection: Optional[str] = None,
        snr_threshold: Optional[float] = None,
        ifar_threshold: Optional[float] = None,
        mass_frame: Optional[str] = None,
        pe_prior_mode: Optional[str] = None,
        pe_prior_bounds: Optional[Tuple[float, float]] = None,
        pe_data_root: Optional[Union[str, Path]] = None,
        pe_data_prefer: str = "cosmo",
        pe_data_dataset: Optional[str] = None,
        em_redshift_map: Optional[Dict[str, Dict[str, float]]] = None,
        extra_events: Optional[Sequence[GWEventData]] = None,
        **kwargs: Any,
    ) -> "GWStandardSirenLikelihood":
        """Construct GWStandardSirenLikelihood from GWTC-3 data."""
        catalog = load_gwtc3_catalog(
            data_root=data_root,
            population=population,
            events=events,
            events_path=events_path,
            max_events=max_events,
            injections_path=injections_path,
            injections_meta=injections_meta,
            galaxy_catalog_path=galaxy_catalog_path,
            selection=selection,
            snr_threshold=snr_threshold,
            ifar_threshold=ifar_threshold,
            mass_frame=mass_frame,
            pe_prior_mode=pe_prior_mode,
            pe_prior_bounds=pe_prior_bounds,
            pe_data_root=pe_data_root,
            pe_data_prefer=pe_data_prefer,
            pe_data_dataset=pe_data_dataset,
            em_redshift_map=em_redshift_map,
            extra_events=extra_events,
        )
        return cls(catalog=catalog, **kwargs)

    @property
    def nuisance_parameters(self):
        """Register population hyperparameters as nuisance parameters for sampling."""
        from ...parameters import Parameter

        specs = self._population_prior_specs()
        params = []
        for name, spec in specs.items():
            try:
                params.append(Parameter.from_simple_config(name, spec))
            except Exception:
                continue
        return self._wrap_nuisance(params)

    def _population_prior_specs(self) -> Dict[str, Dict[str, Any]]:
        """Build prior specs for population parameters (can be overridden)."""
        if (self.population or "").lower() == "bbh":
            specs = {
                "alpha": {"prior": {"dist": "uniform", "min": 1.5, "max": 12.0}, "ref": 3.78},
                "alpha_1": {"prior": {"dist": "uniform", "min": 1.0, "max": 12.0}, "ref": 3.0},
                "alpha_2": {"prior": {"dist": "uniform", "min": 1.0, "max": 12.0}, "ref": 6.0},
                "beta": {"prior": {"dist": "uniform", "min": -4.0, "max": 12.0}, "ref": 0.81},
                "mmin": {"prior": {"dist": "uniform", "min": 2.0, "max": 10.0}, "ref": 4.98},
                "mmax": {"prior": {"dist": "uniform", "min": 50.0, "max": 200.0}, "ref": 112.5},
                "b": {"prior": {"dist": "uniform", "min": 0.0, "max": 1.0}, "ref": 0.5},
                "delta_m": {"prior": {"dist": "uniform", "min": 0.0, "max": 10.0}, "ref": 4.8},
                "mu_g": {"prior": {"dist": "uniform", "min": 20.0, "max": 50.0}, "ref": 32.27},
                "sigma_g": {"prior": {"dist": "uniform", "min": 0.4, "max": 10.0}, "ref": 3.88},
                "lambda_peak": {"prior": {"dist": "uniform", "min": 0.0, "max": 1.0}, "ref": 0.03},
                "gamma": {"prior": {"dist": "uniform", "min": 0.0, "max": 12.0}, "ref": 4.59},
                "kappa": {"prior": {"dist": "uniform", "min": 0.0, "max": 12.0}, "ref": 2.86},
                "zp": {"prior": {"dist": "uniform", "min": 0.0, "max": 4.0}, "ref": 2.47},
            }
        else:
            specs = {
                "alpha": {"prior": {"dist": "uniform", "min": 0.0, "max": 8.0}, "ref": 3.0},
                "beta": {"prior": {"dist": "uniform", "min": 0.0, "max": 8.0}, "ref": 1.0},
                "mmin": {"prior": {"dist": "uniform", "min": 2.0, "max": 10.0}, "ref": 5.0},
                "mmax": {"prior": {"dist": "uniform", "min": 20.0, "max": 150.0}, "ref": 80.0},
                "delta_m": {"prior": {"dist": "uniform", "min": 0.0, "max": 10.0}, "ref": 2.0},
                "mu_g": {"prior": {"dist": "uniform", "min": 20.0, "max": 50.0}, "ref": 32.0},
                "sigma_g": {"prior": {"dist": "uniform", "min": 1.0, "max": 10.0}, "ref": 4.0},
                "lambda_peak": {"prior": {"dist": "uniform", "min": 0.0, "max": 0.5}, "ref": 0.03},
                "gamma": {"prior": {"dist": "uniform", "min": 0.0, "max": 10.0}, "ref": 4.0},
                "kappa": {"prior": {"dist": "uniform", "min": 0.0, "max": 10.0}, "ref": 3.0},
                "zp": {"prior": {"dist": "uniform", "min": 0.1, "max": 6.0}, "ref": 2.0},
                "mminns": {"prior": {"dist": "uniform", "min": 0.5, "max": 2.0}, "ref": 1.0},
                "mmaxns": {"prior": {"dist": "uniform", "min": 2.0, "max": 4.0}, "ref": 3.0},
                "alphans": {"prior": {"dist": "uniform", "min": -2.0, "max": 4.0}, "ref": 0.0},
            }

        if self._population_prior_overrides:
            for name, value in self._population_prior_overrides.items():
                if isinstance(value, dict):
                    specs[name] = dict(value)
                else:
                    specs[name] = {"value": value, "free": False}

        if not self.scale_free and "R0" not in specs:
            specs["R0"] = {"prior": {"dist": "uniform", "min": 0.0, "max": 200.0}, "ref": 30.0}
        # Filter to only those parameters used by the current model.
        active = set(self._rate_model.population_parameters)
        return {name: spec for name, spec in specs.items() if name in active}

    # ------------------------------------------------------------------
    # Likelihood API
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        if self.catalog is None:
            raise ValueError(
                "GWStandardSirenLikelihood requires a GWCatalogData instance."
            )

    def _setup_covariance(self) -> None:
        if not isinstance(self.catalog, GWCatalogData):
            raise TypeError(
                "catalog must be a GWCatalogData object; got "
                f"{type(self.catalog)}"
            )

        if not self.catalog.events:
            raise ValueError("catalog contains no GW events.")

        self._setup_backend()

    def _setup_backend(self) -> None:
        galaxy_catalog = None
        if self.use_galaxy_catalog:
            if self.catalog.galaxy_catalog is None:
                raise ValueError('use_galaxy_catalog=True 但 catalog 中没有星系数据')
            galaxy_catalog = self.catalog.galaxy_catalog
        self._galaxy_catalog = galaxy_catalog
        self._posterior_catalog = PosteriorCatalog(
            self.catalog.events,
            nparallel=self.nparallel,
            random_seed=self.random_seed,
            galaxy_catalog=galaxy_catalog,
        )
        if self.apply_selection_bias:
            if self.catalog.injections is None:
                raise ValueError("apply_selection_bias=True 但 catalog 中没有注入数据")
            self._injections = InjectionSampler(self.catalog.injections)
            self._injections.apply_threshold()
        else:
            self._injections = None
        if self._mass_prior_override is not None:
            mass_prior = self._mass_prior_override
        else:
            if self.population == "bns":
                mass_prior = BNSMassPrior()
            else:
                mass_prior = PowerLawPeak()
        rate_evolution = self._rate_evolution_override or MadauRate()
        self._rate_model = GWRateModel(
            mass_prior=mass_prior,
            rate_evolution=rate_evolution,
            scale_free=self.scale_free,
        )
        self.parameters = {name: None for name in self._rate_model.population_parameters}

    def _initialize_nuisance(self) -> None:
        # No additional nuisance parameters
        return

    def get_requirements(self) -> Dict[str, Any]:
        return {
            'luminosity_distance': {
                'z_range': (0.0, self.zmax),
                'description': 'Required to evaluate GW sirens'
            }
        }

    def theory(self, **kwargs):  # pragma: no cover - theoretic vector unused
        raise NotImplementedError(
            "GWStandardSirenLikelihood directly evaluates the hierarchical "
            "log-likelihood and does not expose a theory vector."
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_likelihood(self, cosmology, **kwargs) -> float | jnp.ndarray:
        return_jnp = self._contains_jax_values(kwargs) or self._cosmology_has_jax(cosmology)
        return self._log_likelihood_core(cosmology, return_jnp=return_jnp, **kwargs)

    def log_likelihood_traced(self, cosmo_params: Dict[str, Any], **kwargs) -> jnp.ndarray:
        cosmology = self.cosmology_model(**cosmo_params)
        return self._log_likelihood_core(cosmology, return_jnp=True, **kwargs)

    @staticmethod
    def _contains_jax_values(values: Dict[str, Any]) -> bool:
        if not values:
            return False
        for val in values.values():
            if isinstance(val, jax.core.Tracer) or isinstance(val, jax.Array):
                return True
        return False

    @staticmethod
    def _cosmology_has_jax(cosmology) -> bool:
        params = getattr(cosmology, "params", {})
        for val in params.values():
            if isinstance(val, jax.core.Tracer) or isinstance(val, jax.Array):
                return True
        return False

    def _prepare_population_parameters(self, cosmology, kwargs) -> Dict[str, float]:
        pop_params: Dict[str, float] = {}
        pop_params.update(self._population_default_values())
        pop_params.update(kwargs.pop('pop_params', {}))
        for name in self._rate_model.population_parameters:
            if name in kwargs:
                pop_params[name] = kwargs[name]

        if "kappa" not in pop_params:
            alias = pop_params.pop("Madau_k", pop_params.pop("k", None))
            if alias is not None:
                pop_params["kappa"] = alias
        if "zp" not in pop_params:
            alias = pop_params.pop("Madau_zp", None)
            if alias is not None:
                pop_params["zp"] = alias

        missing = [
            key for key in self._rate_model.population_parameters if key not in pop_params
        ]
        if missing:
            raise ValueError(
                f"缺少族群参数: {missing}。请通过 population_params 或 pop_params 传入。"
            )

        return pop_params

    def _population_default_values(self) -> Dict[str, float]:
        defaults: Dict[str, float] = {}
        specs = self._population_prior_specs()
        for name, spec in specs.items():
            if not isinstance(spec, dict):
                continue
            if "value" in spec:
                defaults[name] = spec["value"]
            elif "ref" in spec:
                defaults[name] = spec["ref"]
        return defaults

    def _log_likelihood_core(self, cosmology, *, return_jnp: bool, **kwargs):
        if self._posterior_catalog is None:
            raise RuntimeError('JAX backend 尚未初始化')
        if self.apply_selection_bias and self._injections is None:
            raise RuntimeError('未初始化注入数据，无法评估选择效应')

        pop_params = self._prepare_population_parameters(cosmology, kwargs)

        self._posterior_catalog.update_weights(
            self._rate_model,
            galaxy_catalog=self._galaxy_catalog,
            cosmology=cosmology,
            pop_params=pop_params,
        )
        neff_pe = self._posterior_catalog.get_effective_number_of_PE()
        invalid_mask = jnp.any(neff_pe <= 0.0)

        if self.likelihood_variance_thr is None:
            if self.neffPE is not None:
                neff_pe_thr = jnp.asarray(float(self.neffPE), dtype=jnp.float64)
                invalid_mask = jnp.logical_or(invalid_mask, jnp.any(neff_pe < neff_pe_thr))

        sum_weights = self._posterior_catalog.sum_weights
        invalid_mask = jnp.logical_or(invalid_mask, jnp.any(sum_weights <= 0.0))

        if self._injections is None:
            if not self.scale_free:
                raise ValueError("缺少注入数据时必须使用 scale_free=True")
            likelihood_variance = jnp.asarray(0.0, dtype=jnp.float64)
            log_likeli = jnp.sum(jnp.log(sum_weights))
        else:
            self._injections.update_weights(
                self._rate_model, cosmology=cosmology, pop_params=pop_params
            )
            neff_inj = jnp.asarray(
                self._injections.effective_injections_number_jax(), dtype=jnp.float64
            )
            invalid_mask = jnp.logical_or(invalid_mask, neff_inj <= 0.0)
            n_events = jnp.asarray(float(self._posterior_catalog.n_events), dtype=jnp.float64)
            ntotal = self._injections.ntotal_device

            if self.likelihood_variance_thr is None:
                neff_inj_thr = self.neffINJ
                if neff_inj_thr is None:
                    neff_inj_thr = 4 * self._posterior_catalog.n_events
                neff_inj_thr = jnp.asarray(float(neff_inj_thr), dtype=jnp.float64)
                invalid_mask = jnp.logical_or(invalid_mask, neff_inj < neff_inj_thr)

            variance_corr = self._posterior_catalog.variance_correction(
                neff_pe, return_jnp=True
            )
            likelihood_variance = (
                (jnp.square(n_events) / neff_inj) * (1.0 - neff_inj / ntotal) + variance_corr
            )

            if self.likelihood_variance_thr is not None:
                thr = jnp.asarray(float(self.likelihood_variance_thr), dtype=jnp.float64)
                invalid_mask = jnp.logical_or(invalid_mask, likelihood_variance > thr)

            if self.scale_free:
                pseudo_rate = self._injections.pseudo_rate_device()
                invalid_mask = jnp.logical_or(invalid_mask, pseudo_rate <= 0.0)
                log_likeli = jnp.sum(jnp.log(sum_weights)) - n_events * jnp.log(pseudo_rate)
            else:
                Nexp = self._injections.expected_number_detections_jax()
                invalid_mask = jnp.logical_or(invalid_mask, Nexp <= 0.0)
                log_likeli = (
                    -Nexp
                    + n_events * jnp.log(self._injections.Tobs_device)
                    + jnp.sum(jnp.log(sum_weights))
                )

        log_likeli = jnp.where(invalid_mask, -jnp.inf, log_likeli)
        if return_jnp:
            return log_likeli

        self.likelihood_variance = float(likelihood_variance)
        return float(log_likeli)


# ============================================================================
# Utility Functions
# ============================================================================

def load_gwtc3_events(
    data_path: Union[str, Path],
    *,
    population: str = "bbh",
    events: Optional[Sequence[str]] = None,
    max_events: Optional[int] = None,
    mass_frame: Optional[str] = None,
    snr_threshold: Optional[float] = None,
    ifar_threshold: Optional[float] = None,
    pe_prior_mode: Optional[str] = None,
    pe_prior_bounds: Optional[Tuple[float, float]] = None,
    pe_data_root: Optional[Union[str, Path]] = None,
    pe_data_prefer: str = "cosmo",
    pe_data_dataset: Optional[str] = None,
    em_redshift_map: Optional[Dict[str, Dict[str, float]]] = None,
    event_selector: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
) -> List[GWEventData]:
    """
    Load GWTC-3 posterior samples into GWEventData objects.

    Parameters
    ----------
    data_path : str or Path
        Path to gwtc1234_pe_data_v3.pkl or a pre-filtered event pickle.
    population : {"bbh", "bns", "nsbh"}
        Population subset to load.
    events : list[str], optional
        Specific event names to load (order preserved).
    max_events : int, optional
        Limit number of events for quick tests.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到 GWTC-3 数据文件: {data_path}")

    with data_path.open("rb") as handle:
        data = pickle.load(handle)

    if isinstance(data, dict):
        if population in data and isinstance(data[population], dict):
            pool = data[population]
        else:
            known_pops = {"bbh", "bns", "nsbh"}
            if any(key in data for key in known_pops):
                raise ValueError(
                    f"population={population!r} 不在数据中，可选: {list(data.keys())}"
                )
            pool = data
    else:
        raise ValueError("GWTC-3 数据格式异常，期望 dict 结构")
    if not isinstance(pool, dict):
        raise ValueError("GWTC-3 数据格式异常，期望 dict 结构")

    if events is None:
        names = list(pool.keys())
        if max_events is not None:
            names = names[: int(max_events)]
    else:
        names = list(events)

    gw_events: List[GWEventData] = []
    def _scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 0:
            return float(arr)
        if arr.size == 0:
            return None
        return float(np.median(arr))

    path_hint = str(data_path)
    default_frame = mass_frame
    if default_frame is None:
        default_frame = "source" if "gwtc1234_pe_data_v3" in path_hint else "detector"

    pe_data_root_path = Path(pe_data_root) if pe_data_root is not None else None

    def _resolve_pe_file(event_name: str) -> Optional[Path]:
        if pe_data_root_path is None or not pe_data_root_path.exists():
            return None
        prefer = (pe_data_prefer or "").lower()
        patterns = [
            f"*{event_name}*PEDataRelease*{prefer}*.h5",
            f"*{event_name}*PEDataRelease*{prefer}*.hdf5",
        ]
        for pattern in patterns:
            matches = sorted(pe_data_root_path.glob(pattern))
            if matches:
                return matches[-1]
        if prefer and prefer != "nocosmo":
            for pattern in (
                f"*{event_name}*PEDataRelease*nocosmo*.h5",
                f"*{event_name}*PEDataRelease*nocosmo*.hdf5",
            ):
                matches = sorted(pe_data_root_path.glob(pattern))
                if matches:
                    return matches[-1]
        return None

    for name in names:
        if name not in pool:
            raise KeyError(f"事件 {name} 不在 GWTC-3 数据中")
        entry = pool[name]
        if not isinstance(entry, dict):
            raise ValueError(f"事件 {name} 数据格式异常，期望 dict")

        if event_selector is not None and not event_selector(name, entry):
            continue

        snr_val = _scalar(entry.get("snr") or entry.get("snr_network") or entry.get("network_snr"))
        ifar_val = _scalar(entry.get("ifar") or entry.get("IFAR"))
        if snr_threshold is not None:
            if snr_val is None:
                raise ValueError(f"事件 {name} 缺少 SNR 信息，无法应用筛选")
            if snr_val < snr_threshold:
                continue
        if ifar_threshold is not None:
            if ifar_val is None:
                raise ValueError(f"事件 {name} 缺少 IFAR 信息，无法应用筛选")
            if ifar_val < ifar_threshold:
                continue

        em_info = em_redshift_map.get(name) if em_redshift_map else None
        pe_file = _resolve_pe_file(name)
        if pe_file is not None:
            event = load_gw_event_hdf5(
                pe_file,
                name=name,
                dataset=pe_data_dataset,
                mass_frame=mass_frame or default_frame,
                em_redshift=em_info.get("z") if em_info else None,
                em_redshift_sigma=em_info.get("sigma") if em_info else None,
                pe_prior_mode=pe_prior_mode,
                pe_prior_bounds=pe_prior_bounds,
            )
            event.snr = snr_val
            event.ifar = ifar_val
            gw_events.append(event)
            continue

        frame = default_frame
        if "mass_1_det" in entry or "mass_2_det" in entry:
            frame = "detector"
        elif "mass_1_source" in entry or "mass_2_source" in entry:
            frame = "source"
        if mass_frame is not None and frame != mass_frame:
            raise ValueError(
                f"事件 {name} mass_frame 推断为 {frame}，但调用方指定 {mass_frame}"
            )

        posterior = {
            "luminosity_distance": entry.get("luminosity_distance", entry.get("distance")),
        }
        if "mass_1_det" in entry:
            posterior["mass_1_det"] = entry.get("mass_1_det")
        elif "mass_1_source" in entry:
            posterior["mass_1_source"] = entry.get("mass_1_source")
        else:
            posterior["mass_1"] = entry.get("mass_1")

        if "mass_2_det" in entry:
            posterior["mass_2_det"] = entry.get("mass_2_det")
        elif "mass_2_source" in entry:
            posterior["mass_2_source"] = entry.get("mass_2_source")
        else:
            posterior["mass_2"] = entry.get("mass_2")
        def _first_non_none(*vals):
            for val in vals:
                if val is not None:
                    return val
            return None

        m1_check = _first_non_none(
            posterior.get("mass_1_det"),
            posterior.get("mass_1_source"),
            posterior.get("mass_1"),
        )
        m2_check = _first_non_none(
            posterior.get("mass_2_det"),
            posterior.get("mass_2_source"),
            posterior.get("mass_2"),
        )
        if posterior.get("luminosity_distance") is None or m1_check is None or m2_check is None:
            raise ValueError(f"事件 {name} posterior 缺少 luminosity_distance/mass_1/mass_2")

        if "right_ascension" in entry and "declination" in entry:
            posterior["right_ascension"] = entry["right_ascension"]
            posterior["declination"] = entry["declination"]

        prior = (
            entry.get("prior")
            or entry.get("weights")
            or entry.get("prior_samples")
            or entry.get("pe_prior")
        )
        if prior is None and pe_prior_mode is not None:
            mode = pe_prior_mode
            if str(mode).lower() in {"official", "auto"}:
                mode = "dl2"
            prior = _build_pe_prior_from_mode(
                posterior.get("luminosity_distance"),
                mode,
                bounds=pe_prior_bounds,
            )
        gw_events.append(
            GWEventData(
                name=name,
                posterior_samples=posterior,
                weights=None,
                prior_samples=prior,
                has_em_counterpart=em_info is not None,
                mass_frame=frame,
                snr=snr_val,
                ifar=ifar_val,
                em_redshift=em_info.get("z") if em_info else None,
                em_redshift_sigma=em_info.get("sigma") if em_info else None,
            )
        )

    return gw_events


def load_gw_event_hdf5(
    path: Union[str, Path],
    *,
    name: Optional[str] = None,
    dataset: Optional[str] = None,
    mass_frame: Optional[str] = None,
    em_redshift: Optional[float] = None,
    em_redshift_sigma: Optional[float] = None,
    pe_prior_mode: Optional[str] = None,
    pe_prior_bounds: Optional[Tuple[float, float]] = None,
) -> GWEventData:
    """Load a single GW event posterior from an HDF5 file."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required to load GW HDF5 posterior files") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到 GW posterior 文件: {path}")

    with h5py.File(path, "r") as handle:
        group = None
        posterior_parent = None
        if dataset:
            group = handle[dataset]
        else:
            candidates = [
                "IMRPhenomPv2NRT_lowSpin_posterior",
                "IMRPhenomPv2NRT_highSpin_posterior",
                "IMRPhenomPv2_posterior",
                "PublicationSamples",
                "C01:IMRPhenomXPHM",
                "C01:SEOBNRv4PHM",
                "C01:Mixed",
            ]
            for key in candidates:
                if key in handle:
                    group = handle[key]
                    break
        if group is None:
            group = handle
        if isinstance(group, h5py.Group) and "posterior_samples" in group:
            posterior_parent = group
            group = group["posterior_samples"]
        elif isinstance(group, h5py.Dataset):
            posterior_parent = group.parent

        prior_samples = None
        data_fields: set = set()
        if isinstance(group, h5py.Dataset):
            data = group[()]
            data_fields = set(data.dtype.names or [])

            def _get_ds(*keys):
                if not data.dtype.names:
                    return None
                for key in keys:
                    if key in data.dtype.names:
                        return data[key]
                return None

            d_l = _get_ds("luminosity_distance", "luminosity_distance_Mpc")
            ra = _get_ds("right_ascension", "ra")
            dec = _get_ds("declination", "dec")
            m1 = _get_ds("m1_detector_frame_Msun", "mass_1", "mass_1_source")
            m2 = _get_ds("m2_detector_frame_Msun", "mass_2", "mass_2_source")
        else:
            def _get(*keys):
                for key in keys:
                    if key in group:
                        return group[key][()]
                return None

            d_l = _get("luminosity_distance", "luminosity_distance_Mpc")
            ra = _get("right_ascension", "ra")
            dec = _get("declination", "dec")
            m1 = _get("m1_detector_frame_Msun", "mass_1", "mass_1_source")
            m2 = _get("m2_detector_frame_Msun", "mass_2", "mass_2_source")

        if d_l is None or m1 is None or m2 is None:
            raise ValueError("posterior 文件缺少 luminosity_distance 或 mass 字段")

        frame = mass_frame
        if frame is None:
            if data_fields:
                if "m1_detector_frame_Msun" in data_fields or "mass_1" in data_fields:
                    frame = "detector"
                elif "mass_1_source" in data_fields:
                    frame = "source"
                else:
                    frame = "detector"
            elif isinstance(group, h5py.Group) and "m1_detector_frame_Msun" in group:
                frame = "detector"
            elif isinstance(group, h5py.Group) and "mass_1_source" in group:
                frame = "source"
            else:
                frame = "detector"

        if pe_prior_mode is None or str(pe_prior_mode).lower() in {"auto", "pesummary", "official"}:
            if isinstance(posterior_parent, h5py.Group):
                prior_group = None
                if "priors/samples" in posterior_parent:
                    prior_group = posterior_parent["priors/samples"]
                if prior_group is not None:
                    def _get_prior(*keys):
                        for key in keys:
                            if key in prior_group:
                                return prior_group[key][()]
                        return None

                    if frame == "source":
                        prior_m1 = _get_prior("mass_1_source", "mass_1")
                        prior_m2 = _get_prior("mass_2_source", "mass_2")
                    else:
                        prior_m1 = _get_prior("mass_1", "mass_1_detector_frame_Msun")
                        prior_m2 = _get_prior("mass_2", "mass_2_detector_frame_Msun")
                    prior_dl = _get_prior("luminosity_distance", "luminosity_distance_Mpc")
                    prior_dict = {
                        "mass_1": prior_m1,
                        "mass_2": prior_m2,
                        "luminosity_distance": prior_dl,
                    }
                    prior_samples = _build_joint_pe_prior(
                        prior_samples=prior_dict,
                        posterior={
                            "mass_1": np.asarray(m1, dtype=np.float64),
                            "mass_2": np.asarray(m2, dtype=np.float64),
                            "luminosity_distance": np.asarray(d_l, dtype=np.float64),
                        },
                        bounds=pe_prior_bounds,
                    )

            if prior_samples is None:
                prior_expr = None
                if isinstance(posterior_parent, h5py.Group):
                    for key in (
                        "priors/analytic/luminosity_distance",
                        "priors/analytic/luminosity_distance_Mpc",
                    ):
                        if key in posterior_parent:
                            prior_expr = posterior_parent[key][0]
                            break
                if prior_expr is not None:
                    prior_samples = _parse_prior_expression(prior_expr, np.asarray(d_l, dtype=np.float64))

        if prior_samples is None:
            prior_samples = _build_pe_prior_from_mode(
                np.asarray(d_l, dtype=np.float64),
                pe_prior_mode,
                bounds=pe_prior_bounds,
            )

    posterior = {
        "luminosity_distance": d_l,
    }
    if frame == "detector":
        posterior["mass_1_det"] = m1
        posterior["mass_2_det"] = m2
    else:
        posterior["mass_1_source"] = m1
        posterior["mass_2_source"] = m2
    if ra is not None and dec is not None:
        posterior["right_ascension"] = ra
        posterior["declination"] = dec

    return GWEventData(
        name=name or path.stem,
        posterior_samples=posterior,
        prior_samples=prior_samples,
        mass_frame=frame,
        em_redshift=em_redshift,
        em_redshift_sigma=em_redshift_sigma,
        has_em_counterpart=em_redshift is not None,
    )

def load_gw_injections(
    injections_path: Union[str, Path],
    *,
    n_total: Optional[int] = None,
    T_obs: Optional[float] = None,
    V_T: Optional[float] = None,
    snr_threshold: float = 0.0,
    ifar_threshold: float = 0.0,
    ifar_proxy_mode: Optional[str] = None,
    ifar_proxy_power: float = 4.0,
    snr_ref: Optional[float] = None,
    ifar_ref: Optional[float] = None,
    mass_frame: Optional[str] = None,
) -> GWInjectionSet:
    """
    Load GW injection campaign from a pickle file.

    Expected dict keys (best-effort): injections, weights, n_total, T_obs, V_T, snr.
    Provide n_total/T_obs/V_T explicitly if missing in file.
    """
    injections_path = Path(injections_path)
    if not injections_path.exists():
        raise FileNotFoundError(f"未找到注入数据文件: {injections_path}")

    def _load_pickle(path: Path):
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except Exception:
            class _Fallback:
                pass

            class _FallbackUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == "icarogw.injections" and name == "injections_at_detector":
                        return _Fallback
                    return super().find_class(module, name)

            with path.open("rb") as handle:
                try:
                    return _FallbackUnpickler(handle).load()
                except Exception:
                    handle.seek(0)
                    raise

    data = _load_pickle(injections_path)

    def _first_non_none(*values):
        for value in values:
            if value is not None:
                return value
        return None

    if isinstance(data, GWInjectionSet):
        return data

    if hasattr(data, "m1det") and hasattr(data, "m2det") and hasattr(data, "dldet"):
        injections = {
            "mass_1": np.asarray(getattr(data, "m1det")),
            "mass_2": np.asarray(getattr(data, "m2det")),
            "luminosity_distance": np.asarray(getattr(data, "dldet")),
        }
        weights = _first_non_none(
            getattr(data, "ini_prior", None),
            getattr(data, "ini_prior_original", None),
            getattr(data, "prior", None),
        )
        snr = _first_non_none(
            getattr(data, "snrdet", None),
            getattr(data, "snr_original", None),
            getattr(data, "snr", None),
        )
        ifar = _first_non_none(
            getattr(data, "ifar", None),
            getattr(data, "IFAR", None),
        )
        n_total = (
            n_total
            or getattr(data, "ntotal", None)
            or getattr(data, "ntotal_original", None)
            or getattr(data, "n_total", None)
        )
        T_obs = T_obs or getattr(data, "Tobs", None) or getattr(data, "T_obs", None)
        V_T = V_T or getattr(data, "V_T", None) or getattr(data, "VT", None)
        mass_frame = mass_frame or "detector"
    elif isinstance(data, dict):
        if "injections" in data:
            injections = data["injections"]
        else:
            injections = {
                key: data[key]
                for key in (
                    "luminosity_distance",
                    "distance",
                    "mass_1",
                    "mass_1_source",
                    "mass_1_det",
                    "mass_2",
                    "mass_2_source",
                    "mass_2_det",
                )
                if key in data
            }
            if not injections:
                injections = data
        weights = _first_non_none(
            data.get("weights"),
            data.get("prior"),
            data.get("sampling_prior"),
        )
        snr = _first_non_none(data.get("snr"), data.get("snr_network"))
        ifar = _first_non_none(data.get("ifar"), data.get("IFAR"))
        n_total = (
            n_total
            or data.get("n_total")
            or data.get("n_injections")
            or data.get("Ntotal")
            or data.get("ntotal")
        )
        T_obs = T_obs or data.get("T_obs") or data.get("Tobs")
        V_T = V_T or data.get("V_T") or data.get("VT")
        if mass_frame is None:
            mass_frame = data.get("mass_frame")
        if mass_frame is None:
            if "mass_1_source" in data or "mass_2_source" in data:
                mass_frame = "source"
            else:
                mass_frame = "detector"
    else:
        injections = data
        weights = None
        snr = None
        ifar = None

    if ifar is not None:
        ifar_arr = np.asarray(ifar)
        if not np.isfinite(ifar_arr).any():
            ifar = None

    if weights is None:
        raise ValueError("注入数据缺少 weights/prior；请提供采样权重")
    if n_total is None or T_obs is None:
        raise ValueError("注入数据缺少 n_total/T_obs；请显式传入")
    if ifar is None and ifar_threshold and ifar_threshold > 0.0:
        if snr is None:
            raise ValueError("注入数据缺少 IFAR/SNR，无法应用 IFAR 近似筛选")
        mode = (ifar_proxy_mode or "snr_powerlaw").lower()
        if mode in {"snr_powerlaw", "snr"}:
            snr_ref_val = float(snr_ref) if snr_ref is not None else None
            if snr_ref_val is None or snr_ref_val <= 0.0:
                snr_ref_val = float(snr_threshold) if snr_threshold else float(np.median(snr))
            ifar_ref_val = float(ifar_ref) if ifar_ref is not None else float(ifar_threshold)
            ifar = ifar_ref_val * (np.asarray(snr, dtype=np.float64) / snr_ref_val) ** float(ifar_proxy_power)
        else:
            raise ValueError(f"Unknown ifar_proxy_mode: {ifar_proxy_mode}")

    if mass_frame is None:
        mass_frame = "detector"

    return GWInjectionSet(
        injections=injections,
        weights=weights,
        n_total=int(n_total),
        V_T=float(V_T) if V_T is not None else 0.0,
        T_obs=float(T_obs),
        snr=snr,
        snr_threshold=snr_threshold,
        ifar=ifar,
        ifar_threshold=ifar_threshold,
        mass_frame=mass_frame,
    )


def load_gwtc3_catalog(
    *,
    data_root: Union[str, Path] = "data/gwtc-3",
    population: str = "bbh",
    events: Optional[Sequence[str]] = None,
    events_path: Optional[Union[str, Path]] = None,
    max_events: Optional[int] = None,
    injections_path: Optional[Union[str, Path]] = None,
    injections_meta: Optional[Dict[str, Any]] = None,
    galaxy_catalog_path: Optional[Union[str, Path]] = None,
    selection: Optional[str] = None,
    snr_threshold: Optional[float] = None,
    ifar_threshold: Optional[float] = None,
    mass_frame: Optional[str] = None,
    pe_prior_mode: Optional[str] = None,
    pe_prior_bounds: Optional[Tuple[float, float]] = None,
    pe_data_root: Optional[Union[str, Path]] = None,
    pe_data_prefer: str = "cosmo",
    pe_data_dataset: Optional[str] = None,
    em_redshift_map: Optional[Dict[str, Dict[str, float]]] = None,
    extra_events: Optional[Sequence[GWEventData]] = None,
    prefer_ifar_injections: bool = True,
) -> GWCatalogData:
    """
    Convenience loader for GWTC-3 catalog with optional injections/galaxies.
    """
    data_root = Path(data_root)
    pe_path = data_root / "gwtc1234_pe_data_v3.pkl"
    if events_path is None:
        if selection in {"snr11_ifar4", "gwtc3_bbh_snr11_ifar4"}:
            if population != "bbh":
                raise ValueError("SNR/IFAR 事件文件仅适用于 BBH")
            candidate = data_root / "GWTC3_BBH_SNR_11_IFAR_4.p"
            if candidate.exists():
                events_path = candidate
        if events_path is None and pe_path.exists():
            events_path = pe_path
        if events_path is None:
            fallback = data_root / "GW_events" / "GWTC3_BBH_SNR_11_IFAR_4.p"
            if not fallback.exists():
                raise FileNotFoundError(
                    f"未找到 GWTC-3 事件文件: {pe_path} 或 {fallback}"
                )
            events_path = fallback

    gw_events = load_gwtc3_events(
        events_path,
        population=population,
        events=events,
        max_events=max_events,
        mass_frame=mass_frame,
        snr_threshold=snr_threshold,
        ifar_threshold=ifar_threshold,
        pe_prior_mode=pe_prior_mode,
        pe_prior_bounds=pe_prior_bounds,
        pe_data_root=pe_data_root,
        pe_data_prefer=pe_data_prefer,
        pe_data_dataset=pe_data_dataset,
        em_redshift_map=em_redshift_map,
    )
    if extra_events:
        gw_events.extend(list(extra_events))

    injections = None
    if injections_path is None:
        candidates = [
            data_root / "official" / "O1_O2_O3_det_frame_SNR9.inj",
            data_root / "injections" / "GWTC3_cosmo_paper_injections.p",
            data_root / "GWTC3_cosmo_paper_injections.p",
            data_root / "gwtc3_cosmo_paper_injections.p",
            data_root / "gwtc3_injections.pkl",
        ]

        def _candidate_has_ifar(path: Path) -> bool:
            try:
                with path.open("rb") as handle:
                    data = pickle.load(handle)
            except Exception:
                return False
            if isinstance(data, dict):
                return ("ifar" in data) or ("IFAR" in data)
            return False

        prefer_ifar = False
        if prefer_ifar_injections:
            meta = injections_meta or {}
            ifar_thr = meta.get("ifar_threshold", 0.0) or 0.0
            prefer_ifar = ifar_thr > 0.0

        if prefer_ifar:
            for candidate in candidates:
                if candidate.exists() and _candidate_has_ifar(candidate):
                    injections_path = candidate
                    break

        if injections_path is None:
            for candidate in candidates:
                if candidate.exists():
                    injections_path = candidate
                    break
    if injections_path is not None:
        injections_meta = injections_meta or {}
        injections = load_gw_injections(injections_path, **injections_meta)

    galaxy_catalog = None
    if galaxy_catalog_path is not None:
        galaxy_catalog = GWGalaxyCatalog.load(galaxy_catalog_path)

    return GWCatalogData(
        events=gw_events,
        injections=injections,
        galaxy_catalog=galaxy_catalog,
    )

def create_mock_gw_catalog(
    n_events: int = 10,
    z_max: float = 0.5,
    H0_true: float = 70.0,
    Omega_m_true: float = 0.3,
    seed: int = 42,
    include_injections: bool = False
) -> GWCatalogData:
    """
    Create mock GW catalog for testing and forecasting.

    Args:
        n_events: Number of GW events
        z_max: Maximum redshift
        H0_true: True Hubble constant
        Omega_m_true: True matter density
        seed: Random seed
        include_injections: Add injection set for selection effects

    Returns:
        GWCatalogData object

    Example:
        >>> catalog = create_mock_gw_catalog(n_events=50, z_max=1.0, include_injections=True)
        >>> gw_like = GWStandardSirenLikelihood(catalog=catalog)
    """
    from hicosmo.models import LCDM

    rng = np.random.RandomState(seed)
    model_true = LCDM(H0=H0_true, Omega_m=Omega_m_true)

    events = []

    for i in range(n_events):
        # Simulate redshift (uniform in comoving volume)
        z_event = rng.uniform(0.01, z_max)

        # True distance
        d_L_true = model_true.luminosity_distance(z_event)

        # GW measurement uncertainty (~15% typical)
        sigma_d_L = 0.15 * d_L_true

        # Mock posterior samples
        n_samples = 1000
        d_L_samples = rng.normal(d_L_true, sigma_d_L, n_samples)
        z_samples = np.full(n_samples, z_event)  # Known redshift (bright siren)
        m1_samples = rng.uniform(20, 50, n_samples)
        m2_samples = rng.uniform(10, 30, n_samples)

        posterior = np.column_stack([d_L_samples, z_samples, m1_samples, m2_samples])

        event = GWEventData(
            name=f"GW_MOCK_{i:03d}",
            posterior_samples=posterior,
            weights=None,  # Uniform
            prior_samples=np.power(d_L_samples, 2.0),
            has_em_counterpart=True
        )
        events.append(event)

    # Create injection set if requested
    injections = None
    if include_injections:
        n_inj = n_events * 100  # Typical 1% detection rate

        z_inj = rng.uniform(0.01, z_max, n_inj)
        d_L_inj = np.array([model_true.luminosity_distance(z) for z in z_inj])
        m1_inj = rng.uniform(20, 50, n_inj)
        m2_inj = rng.uniform(10, 30, n_inj)

        injection_samples = np.column_stack([d_L_inj, z_inj, m1_inj, m2_inj])

        # Monte Carlo weights for integration over (z, m1, m2)
        # Weight = (sampling volume) / (number of samples)
        V_z = z_max - 0.01  # Redshift range
        V_m1 = 50 - 20      # m1 range (M_sun)
        V_m2 = 30 - 10      # m2 range (M_sun)
        mc_weight = (V_z * V_m1 * V_m2) / n_inj
        detection_weights = np.ones(n_inj) * mc_weight

        injections = GWInjectionSet(
            injections=injection_samples,
            weights=detection_weights,
            n_total=n_inj * 100,
            V_T=1000.0,  # Gpc^3 yr (O3-like)
            T_obs=1.0    # years
        )

    catalog = GWCatalogData(
        events=events,
        injections=injections
    )

    return catalog


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Data containers
    'GWEventData',
    'GWInjectionSet',
    'GWCatalogData',
    'GWGalaxyCatalog',
    # Wrappers (Phase 2)
    'MassPrior',
    'PowerLawMass',
    'RateEvolution',
    'MadauRate',
    # Rate model (Phase 2)
    'GWRateModel',
    # Main likelihood
    'GWStandardSirenLikelihood',
    # Utilities
    'create_mock_gw_catalog',
    'load_gwtc3_events',
    'load_gw_injections',
    'load_gwtc3_catalog',
    'load_gw_event_hdf5',
]
