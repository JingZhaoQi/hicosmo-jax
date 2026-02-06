#!/usr/bin/env python3
"""
Information criteria for MCMC results (AIC, BIC, Bayesian evidence).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import json

import numpy as np

from ..utils.logging import get_logger
from .plotting import _load_chain_simple

logger = get_logger(__name__)

DEFAULT_LOG_LIKE_KEYS = (
    "log_likelihood",
    "loglike",
    "log_like",
    "lnlike",
)
DEFAULT_EXCLUDE_KEYS = DEFAULT_LOG_LIKE_KEYS + (
    "log_prob",
    "log_posterior",
)


def aic(
    log_likelihood_max: float,
    num_params: int,
) -> float:
    """
    Compute Akaike Information Criterion.

    Parameters
    ----------
    log_likelihood_max : float
        Maximum log-likelihood value from MCMC samples.
        Convention: log_likelihood = -χ²/2 (no normalization included).
    num_params : int
        Number of free parameters (k).

    Returns
    -------
    float
        AIC = 2k - 2*log_likelihood_max = 2k + χ²_min

    Notes
    -----
    HIcosmo convention: all likelihoods return log(L) = -χ²/2 without normalization.
    Normalization factors are available via get_normalization_constant() but not
    used in AIC/BIC since they cancel in model comparison.
    """
    # log_likelihood = -χ²/2, so χ² = -2 * log_likelihood
    # AIC = 2k + χ² = 2k - 2*log_likelihood
    return 2.0 * float(num_params) - 2.0 * float(log_likelihood_max)


def bic(
    log_likelihood_max: float,
    num_params: int,
    num_data: int,
) -> float:
    """
    Compute Bayesian Information Criterion.

    Parameters
    ----------
    log_likelihood_max : float
        Maximum log-likelihood value from MCMC samples.
        Convention: log_likelihood = -χ²/2 (no normalization included).
    num_params : int
        Number of free parameters (k).
    num_data : int
        Number of data points (N).

    Returns
    -------
    float
        BIC = k*ln(N) - 2*log_likelihood_max = k*ln(N) + χ²_min

    Notes
    -----
    HIcosmo convention: all likelihoods return log(L) = -χ²/2 without normalization.
    Normalization factors are available via get_normalization_constant() but not
    used in AIC/BIC since they cancel in model comparison.
    """
    if num_data <= 0:
        raise ValueError("num_data must be positive for BIC")
    # log_likelihood = -χ²/2, so χ² = -2 * log_likelihood
    # BIC = k*ln(N) + χ² = k*ln(N) - 2*log_likelihood
    return float(num_params) * np.log(float(num_data)) - 2.0 * float(log_likelihood_max)


def log_evidence_harmonic_mean(
    log_likelihoods: Union[np.ndarray, Sequence[float]],
) -> float:
    """Estimate log evidence via the harmonic-mean estimator."""
    values = _ensure_finite(log_likelihoods)
    return float(np.log(values.size) - _logsumexp(-values))


def information_criteria(
    samples: Optional[Union[str, Path, Mapping[str, Any], np.ndarray]] = None,
    *,
    log_likelihoods: Optional[Union[np.ndarray, Sequence[float]]] = None,
    log_likelihood_fn: Optional[Callable[..., float]] = None,
    num_params: Optional[int] = None,
    num_data: Optional[int] = None,
    param_names: Optional[Sequence[str]] = None,
    log_likelihood_key: Optional[str] = None,
    burnin_frac: float = 0.0,
    thin: int = 1,
    max_samples: Optional[int] = None,
    evidence_method: Optional[str] = "harmonic_mean",
    parameter_config: Optional[Any] = None,
    data_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute AIC, BIC, and log evidence from MCMC results.

    Parameters
    ----------
    samples : various, optional
        MCMC samples (dict, array, SamplerResults/MCMCState, chain name, or file
        path).
    log_likelihoods : array-like, optional
        Log-likelihood values for samples. Required if log_likelihood_fn is not
        provided.
    log_likelihood_fn : callable, optional
        Function that evaluates log-likelihood. Accepts kwargs or a dict.
    num_params : int, optional
        Number of free parameters. If None, inferred from parameter_config or samples.
    num_data : int, optional
        Number of data points for BIC. If None, BIC is omitted.
    param_names : sequence of str, optional
        Parameter names to use when evaluating log_likelihood_fn.
    log_likelihood_key : str, optional
        Key for log-likelihood stored in samples dict.
    burnin_frac : float, optional
        Fraction of samples to discard as burn-in.
    thin : int, optional
        Thinning factor.
    max_samples : int, optional
        Cap on the number of samples used for evaluation.
    evidence_method : str or None, optional
        Evidence estimator ("harmonic_mean"). Set to None to skip.
    parameter_config : optional
        ParameterConfig or dict to infer num_params and free parameter names.
    data_info : mapping, optional
        Data metadata to infer num_data.

    Notes
    -----
    If samples include derived parameters, pass num_params, param_names,
    or parameter_config to avoid over-counting parameters.
    """
    if thin <= 0:
        raise ValueError("thin must be positive")
    if not 0.0 <= burnin_frac < 1.0:
        raise ValueError("burnin_frac must be in [0, 1)")

    samples_dict, metadata = _coerce_samples(samples, param_names)

    if parameter_config is None:
        parameter_config = metadata.get("parameter_config")
    if data_info is None:
        data_info = metadata.get("data_info")

    log_likelihoods = _resolve_log_likelihoods(
        log_likelihoods=log_likelihoods,
        samples_dict=samples_dict,
        log_likelihood_fn=log_likelihood_fn,
        param_names=param_names,
        log_likelihood_key=log_likelihood_key,
        burnin_frac=burnin_frac,
        thin=thin,
        max_samples=max_samples,
        parameter_config=parameter_config,
    )

    log_likelihoods = _ensure_finite(log_likelihoods)
    log_likelihood_max = float(np.max(log_likelihoods))

    inferred_params = _infer_num_params(
        parameter_config=parameter_config,
        param_names=param_names,
        samples=samples_dict,
        log_likelihood_key=log_likelihood_key,
    )
    if num_params is None:
        num_params = inferred_params
    if num_params is None:
        raise ValueError(
            "num_params could not be inferred; pass num_params or parameter_config"
        )

    if num_data is None:
        num_data = _infer_num_data(data_info)

    # Get normalization constant (stored for reference, not used in AIC/BIC)
    # HIcosmo convention: log_likelihood = -χ²/2 (no normalization included)
    norm_const = _infer_normalization_constant(data_info, log_likelihood_fn)

    # Compute chi-squared minimum from log-likelihood
    # log_likelihood = -χ²/2, so χ² = -2 * log_likelihood
    chi2_min = -2.0 * log_likelihood_max

    aic_value = aic(log_likelihood_max, num_params)
    bic_value = (
        bic(log_likelihood_max, num_params, num_data) if num_data is not None else None
    )

    log_evidence = None
    if evidence_method:
        method = evidence_method.lower()
        if method in {"harmonic_mean", "hme"}:
            log_evidence = log_evidence_harmonic_mean(log_likelihoods)
        else:
            raise ValueError(f"Unsupported evidence_method: {evidence_method}")

    return {
        "num_params": int(num_params) if num_params is not None else None,
        "num_data": int(num_data) if num_data is not None else None,
        "n_samples": int(log_likelihoods.size),
        "log_likelihood_max": log_likelihood_max,
        "normalization_constant": norm_const,
        "chi2_min": chi2_min,
        "aic": aic_value,
        "bic": bic_value,
        "log_evidence": log_evidence,
        "log_evidence_method": evidence_method,
    }


def _coerce_samples(
    samples: Optional[Union[str, Path, Mapping[str, Any], np.ndarray]],
    param_names: Optional[Sequence[str]],
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    if samples is None:
        return None, metadata

    if hasattr(samples, "samples"):
        if hasattr(samples, "parameter_config"):
            metadata["parameter_config"] = getattr(samples, "parameter_config")
        if hasattr(samples, "data_info"):
            metadata["data_info"] = getattr(samples, "data_info")
        samples = getattr(samples, "samples")

    if isinstance(samples, (str, Path)):
        samples, metadata = _load_chain_with_metadata(Path(samples))
        return samples, metadata

    if isinstance(samples, np.ndarray):
        if samples.ndim != 2:
            raise ValueError("samples array must be 2D (n_samples, n_params)")
        if param_names is not None:
            names = list(param_names)
        else:
            names = [f"param_{i}" for i in range(samples.shape[1])]
        return (
            {name: samples[:, idx] for idx, name in enumerate(names)},
            metadata,
        )

    if isinstance(samples, Mapping):
        return {k: np.asarray(v) for k, v in samples.items()}, metadata

    raise TypeError("Unsupported samples type")


def _load_chain_with_metadata(
    path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    import h5py

    if not path.suffix:
        path = Path("mcmc_chains") / f"{path}.h5"
    if not path.exists():
        raise FileNotFoundError(f"Chain file not found: {path}")

    metadata: Dict[str, Any] = {}

    if path.suffix in {".h5", ".hdf5"}:
        samples: Dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            group = f["samples"] if "samples" in f else f.get("chains")
            if group is None:
                raise ValueError(f"No samples group found in {path}")
            for key in group.keys():
                samples[str(key)] = np.array(group[key])

            if "metadata" in f and "parameter_config" in f["metadata"]:
                try:
                    config_str = f["metadata"]["parameter_config"][()].decode("utf-8")
                    metadata["parameter_config"] = json.loads(config_str)
                except Exception:
                    logger.warning("Failed to parse parameter_config in %s", path)

            # Load data_info (normalization_constant, num_data) for AIC/BIC
            if "data" in f and "data_info" in f["data"]:
                try:
                    data_bytes = f["data"]["data_info"][()]
                    data_str = (
                        data_bytes.decode("utf-8")
                        if isinstance(data_bytes, bytes)
                        else str(data_bytes)
                    )
                    metadata["data_info"] = json.loads(data_str)
                except Exception:
                    logger.warning("Failed to parse data_info in %s", path)

        return samples, metadata

    loaded = _load_chain_simple(str(path))
    if isinstance(loaded, tuple) and len(loaded) == 2:
        return {k: np.asarray(v) for k, v in loaded[0].items()}, metadata
    if isinstance(loaded, Mapping):
        return {k: np.asarray(v) for k, v in loaded.items()}, metadata
    loaded_array = np.asarray(loaded)
    if loaded_array.ndim == 1:
        return {"param_0": loaded_array}, metadata
    if loaded_array.ndim == 2:
        return (
            {f"param_{i}": loaded_array[:, i] for i in range(loaded_array.shape[1])},
            metadata,
        )
    raise ValueError("Loaded chain data must be 1D or 2D array")


def _resolve_log_likelihoods(
    *,
    log_likelihoods: Optional[Union[np.ndarray, Sequence[float]]],
    samples_dict: Optional[Mapping[str, Any]],
    log_likelihood_fn: Optional[Callable[..., float]],
    param_names: Optional[Sequence[str]],
    log_likelihood_key: Optional[str],
    burnin_frac: float,
    thin: int,
    max_samples: Optional[int],
    parameter_config: Optional[Any],
) -> np.ndarray:
    if log_likelihoods is not None:
        values = _apply_burnin_and_thin(log_likelihoods, burnin_frac, thin)
        return _downsample(values, max_samples)

    if samples_dict is None:
        raise ValueError("samples required when log_likelihoods is not provided")

    inferred_key = log_likelihood_key or _find_log_likelihood_key(samples_dict)
    if inferred_key and inferred_key in samples_dict:
        values = _apply_burnin_and_thin(samples_dict[inferred_key], burnin_frac, thin)
        return _downsample(values, max_samples)

    if log_likelihood_fn is None:
        raise ValueError(
            "log_likelihood_fn required when log_likelihoods not present in samples"
        )

    names = _infer_param_names(
        parameter_config, samples_dict, param_names, inferred_key
    )
    if not names:
        raise ValueError(
            "No parameter names available for log_likelihood_fn evaluation"
        )
    missing = [name for name in names if name not in samples_dict]
    if missing:
        raise ValueError(f"Missing samples for parameters: {missing}")
    param_arrays = {
        name: _apply_burnin_and_thin(samples_dict[name], burnin_frac, thin)
        for name in names
    }

    length = _validate_sample_lengths(param_arrays)
    indices = _select_indices(length, max_samples)

    log_like_values = np.empty(indices.size, dtype=float)
    for i, idx in enumerate(indices):
        params = {name: float(param_arrays[name][idx]) for name in names}
        log_like_values[i] = _call_log_likelihood(log_likelihood_fn, params)

    return log_like_values


def _apply_burnin_and_thin(
    values: Union[np.ndarray, Sequence[float]],
    burnin_frac: float,
    thin: int,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim >= 2:
        n_samples = arr.shape[1]
        burnin = int(n_samples * burnin_frac)
        arr = arr[:, burnin:]
        if thin > 1:
            arr = arr[:, ::thin]
    else:
        n_samples = arr.shape[0]
        burnin = int(n_samples * burnin_frac)
        arr = arr[burnin:]
        if thin > 1:
            arr = arr[::thin]
    return np.asarray(arr).reshape(-1)


def _downsample(values: np.ndarray, max_samples: Optional[int]) -> np.ndarray:
    if max_samples is None:
        return values
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if values.size <= max_samples:
        return values
    indices = np.linspace(0, values.size - 1, max_samples, dtype=int)
    return values[indices]


def _select_indices(n_samples: int, max_samples: Optional[int]) -> np.ndarray:
    if max_samples is None or max_samples >= n_samples:
        return np.arange(n_samples, dtype=int)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    return np.linspace(0, n_samples - 1, max_samples, dtype=int)


def _validate_sample_lengths(param_arrays: Mapping[str, np.ndarray]) -> int:
    lengths = {name: len(values) for name, values in param_arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Parameter sample lengths mismatch: {lengths}")
    return unique_lengths.pop()


def _call_log_likelihood(
    log_likelihood_fn: Callable[..., float],
    params: Dict[str, float],
) -> float:
    try:
        value = log_likelihood_fn(**params)
    except TypeError:
        value = log_likelihood_fn(params)
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _find_log_likelihood_key(samples_dict: Mapping[str, Any]) -> Optional[str]:
    for key in DEFAULT_LOG_LIKE_KEYS:
        if key in samples_dict:
            return key
    return None


def _infer_param_names(
    parameter_config: Optional[Any],
    samples_dict: Mapping[str, Any],
    param_names: Optional[Sequence[str]],
    log_likelihood_key: Optional[str],
) -> Sequence[str]:
    if param_names is not None:
        return list(param_names)

    if parameter_config is not None:
        params = _parameter_config_params(parameter_config)
        if params:
            return list(params)

    names = [name for name in samples_dict.keys() if name not in DEFAULT_EXCLUDE_KEYS]
    if log_likelihood_key in names:
        names.remove(log_likelihood_key)
    return names


def _infer_num_params(
    *,
    parameter_config: Optional[Any],
    param_names: Optional[Sequence[str]],
    samples: Optional[Mapping[str, Any]],
    log_likelihood_key: Optional[str],
) -> Optional[int]:
    if param_names is not None:
        return len(param_names)

    if parameter_config is not None:
        params = _parameter_config_params(parameter_config)
        if params:
            return len(params)

    if samples is None:
        return None

    names = [name for name in samples.keys() if name not in DEFAULT_EXCLUDE_KEYS]
    if log_likelihood_key in names:
        names.remove(log_likelihood_key)
    return len(names) if names else None


def _parameter_config_params(parameter_config: Any) -> Optional[Sequence[str]]:
    if hasattr(parameter_config, "parameters"):
        params = getattr(parameter_config, "parameters")
        return [name for name, param in params.items() if getattr(param, "free", True)]

    if isinstance(parameter_config, Mapping):
        if "parameters" in parameter_config:
            params = parameter_config.get("parameters", {})
        else:
            params = parameter_config

        if isinstance(params, Mapping):
            return [
                name
                for name, info in params.items()
                if not isinstance(info, Mapping) or info.get("free", True)
            ]

    return None


def _infer_num_data(data_info: Optional[Mapping[str, Any]]) -> Optional[int]:
    if not data_info:
        return None
    if "num_data" in data_info:
        try:
            num_data = int(data_info["num_data"])
            if num_data > 0:
                return num_data
        except (TypeError, ValueError):
            pass
    keys = data_info.get("data_keys", [])
    if not keys:
        return None
    total = 0
    for key in keys:
        size_key = f"{key}_size"
        if size_key in data_info:
            total += int(data_info[size_key])
    return total if total > 0 else None


def _infer_normalization_constant(
    data_info: Optional[Mapping[str, Any]],
    log_likelihood_fn: Optional[Callable[..., float]],
) -> float:
    """
    Infer log-likelihood normalization constant.

    The normalization constant is the portion of log-likelihood that does NOT
    depend on model parameters. For Gaussian likelihoods:
        log(L) = -χ²/2 + normalization_constant

    Priority:
    1. From data_info metadata (stored during MCMC)
    2. From likelihood function's get_normalization_constant() method
    3. Default to 0.0 (no normalization)

    Parameters
    ----------
    data_info : mapping, optional
        Data metadata from MCMC chain file.
    log_likelihood_fn : callable, optional
        Likelihood function, may have get_normalization_constant() method.

    Returns
    -------
    float
        Normalization constant (default 0.0 if not available).
    """
    # Priority 1: From data_info metadata
    if data_info and "normalization_constant" in data_info:
        try:
            return float(data_info["normalization_constant"])
        except (TypeError, ValueError):
            pass

    # Priority 2: From likelihood function
    if log_likelihood_fn is not None:
        norm_const = 0.0
        has_norm = False

        # Check if it's a CombinedLikelihood
        if hasattr(log_likelihood_fn, "_likelihoods"):
            for lik in log_likelihood_fn._likelihoods:
                if hasattr(lik, "get_normalization_constant"):
                    try:
                        norm_const += float(lik.get_normalization_constant())
                        has_norm = True
                    except Exception:
                        pass
        elif hasattr(log_likelihood_fn, "get_normalization_constant"):
            try:
                norm_const = float(log_likelihood_fn.get_normalization_constant())
                has_norm = True
            except Exception:
                pass

        if has_norm:
            return norm_const

    # Default: no normalization
    return 0.0


def _ensure_finite(values: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite log-likelihood values available")
    return arr


def _logsumexp(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    return vmax + float(np.log(np.sum(np.exp(values - vmax))))


__all__ = [
    "aic",
    "bic",
    "log_evidence_harmonic_mean",
    "information_criteria",
]
