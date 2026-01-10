"""Nested sampling support (Dynesty backend)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NestedResults:
    samples: Dict[str, np.ndarray]
    logz: float
    logz_err: float
    metadata: Dict[str, Any]


def _build_prior_transform(free_params: Dict[str, Any]):
    transforms = []
    names = []
    for name, param in free_params.items():
        prior = param.prior or {}
        dist = prior.get("dist", "uniform")
        if dist == "uniform":
            lo = prior.get("min")
            hi = prior.get("max")
            if lo is None or hi is None:
                raise ValueError(f"Uniform prior for '{name}' requires min/max")
            transforms.append(("uniform", float(lo), float(hi)))
        else:
            raise ValueError(
                f"Dynesty backend currently supports only uniform priors. '{name}' has '{dist}'."
            )
        names.append(name)

    def prior_transform(u):
        vals = []
        for idx, (kind, lo, hi) in enumerate(transforms):
            if kind == "uniform":
                vals.append(lo + u[idx] * (hi - lo))
        return np.array(vals, dtype=float)

    return names, prior_transform


def run_nested(loglike_func, free_params: Dict[str, Any], options: Dict[str, Any]) -> NestedResults:
    try:
        import dynesty
        from dynesty.utils import resample_equal
    except ImportError as exc:
        raise ImportError("Dynesty is required for nested sampling. Install via pip install dynesty") from exc

    names, prior_transform = _build_prior_transform(free_params)

    def loglike(theta):
        kwargs = {name: float(theta[idx]) for idx, name in enumerate(names)}
        return float(loglike_func(**kwargs))

    ndim = len(names)
    sampler = dynesty.NestedSampler(loglike, prior_transform, ndim, **options)
    sampler.run_nested()

    results = sampler.results
    if hasattr(results, "weights"):
        weights = results.weights
    else:
        logz = results.logz[-1] if hasattr(results, "logz") else None
        if logz is None:
            raise ValueError("Dynesty results missing logz; cannot compute weights.")
        weights = np.exp(results.logwt - logz)

    samples_equal = resample_equal(results.samples, weights)
    samples_dict = {name: samples_equal[:, idx] for idx, name in enumerate(names)}

    def _as_int(value, *, total: bool = False) -> int:
        arr = np.asarray(value)
        if arr.shape == ():
            return int(arr)
        if total:
            return int(arr.sum())
        return int(arr[-1])

    metadata = {
        "sampler": "dynesty",
        "ncall": _as_int(results.ncall, total=True),
        "niter": _as_int(results.niter),
    }

    return NestedResults(
        samples=samples_dict,
        logz=float(results.logz[-1]),
        logz_err=float(results.logzerr[-1]),
        metadata=metadata,
    )


def save_nested_results(results: NestedResults, filename: str, format: str = "npz") -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npz":
        save_dict = {f"samples_{k}": v for k, v in results.samples.items()}
        save_dict["_logz"] = np.array(results.logz)
        save_dict["_logz_err"] = np.array(results.logz_err)
        save_dict["_metadata"] = np.array(json.dumps(results.metadata))
        np.savez_compressed(path, **save_dict)
        return

    if format == "json":
        payload = {
            "samples": {k: v.tolist() for k, v in results.samples.items()},
            "logz": results.logz,
            "logz_err": results.logz_err,
            "metadata": results.metadata,
        }
        path.write_text(json.dumps(payload, indent=2))
        return

    raise ValueError("Unsupported format. Use 'npz' or 'json'.")
