#!/usr/bin/env python3
"""
Parameter Validation Utilities
================================

Centralized validation logic for parameter management system.
"""

from typing import Dict, List, Any, Optional
from difflib import get_close_matches


def _normalize_prior_dict(prior: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized copy of a prior dict (dist aliases + key aliases)."""
    normalized = dict(prior)
    dist_type = normalized.get("dist")
    if dist_type:
        dist_lower = str(dist_type).lower()
        dist_aliases = {
            "log_normal": "lognormal",
            "truncated_normal": "truncnorm",
            "half_normal": "halfnormal",
            "half_cauchy": "halfcauchy",
        }
        normalized["dist"] = dist_aliases.get(dist_lower, dist_lower)

    dist_norm = normalized.get("dist")

    if dist_norm == "truncnorm":
        if "low" not in normalized and "min" in normalized:
            normalized["low"] = normalized["min"]
        if "high" not in normalized and "max" in normalized:
            normalized["high"] = normalized["max"]

    if dist_norm == "beta":
        if "alpha" not in normalized and "a" in normalized:
            normalized["alpha"] = normalized["a"]
        if "beta" not in normalized and "b" in normalized:
            normalized["beta"] = normalized["b"]

    if dist_norm == "gamma":
        if "concentration" not in normalized and "shape" in normalized:
            normalized["concentration"] = normalized["shape"]

    return normalized


def validate_prior_dict(prior: Dict[str, Any], param_name: str = "") -> None:
    """
    Validate prior distribution dictionary format.

    Parameters
    ----------
    prior : dict
        Prior distribution specification.
        Required keys: 'dist', plus distribution-specific parameters.
    param_name : str, optional
        Parameter name (for error messages).

    Raises
    ------
    ValueError
        If prior format is invalid.

    Examples
    --------
    >>> validate_prior_dict({'dist': 'uniform', 'min': 0, 'max': 1})
    >>> validate_prior_dict({'dist': 'normal', 'loc': 0, 'scale': 1})
    """
    if not isinstance(prior, dict):
        raise ValueError(
            f"Prior for '{param_name}' must be a dict, got {type(prior).__name__}"
        )

    if "dist" not in prior:
        raise ValueError(
            f"Prior for '{param_name}' must have 'dist' key specifying distribution type"
        )

    normalized = _normalize_prior_dict(prior)
    dist_type = normalized["dist"].lower()

    # Validate distribution-specific parameters
    required_params = {
        "uniform": ["min", "max"],
        "normal": ["loc", "scale"],
        "truncnorm": ["loc", "scale", "low", "high"],
        "lognormal": ["loc", "scale"],
        "halfnormal": ["scale"],
        "halfcauchy": ["scale"],
        "beta": ["alpha", "beta"],
        "gamma": ["concentration", "rate"],
        "exponential": ["rate"],
    }

    if dist_type in required_params:
        missing = [p for p in required_params[dist_type] if p not in normalized]
        if missing:
            raise ValueError(
                f"Prior '{dist_type}' for '{param_name}' missing required parameters: {missing}"
            )
    else:
        supported = list(required_params.keys())
        raise ValueError(
            f"Distribution '{dist_type}' not supported for '{param_name}'. "
            f"Supported: {supported}"
        )


def validate_parameter_name(name: str) -> None:
    """
    Validate parameter name format.

    Rules:
    - Must be non-empty string
    - No leading/trailing whitespace
    - No special characters except underscore
    - Start with letter or underscore

    Parameters
    ----------
    name : str
        Parameter name to validate.

    Raises
    ------
    ValueError
        If name format is invalid.

    Examples
    --------
    >>> validate_parameter_name('H0')
    >>> validate_parameter_name('Omega_m')
    >>> validate_parameter_name('f_R0')
    """
    if not isinstance(name, str):
        raise ValueError(f"Parameter name must be str, got {type(name).__name__}")

    if not name:
        raise ValueError("Parameter name cannot be empty")

    if name != name.strip():
        raise ValueError(f"Parameter name '{name}' has leading/trailing whitespace")

    if not (name[0].isalpha() or name[0] == "_"):
        raise ValueError(
            f"Parameter name '{name}' must start with letter or underscore"
        )

    if not all(c.isalnum() or c == "_" for c in name):
        raise ValueError(
            f"Parameter name '{name}' contains invalid characters. "
            f"Only alphanumeric and underscore allowed."
        )


def suggest_similar_names(
    name: str, valid_names: List[str], cutoff: float = 0.6, n: int = 3
) -> List[str]:
    """
    Suggest similar valid names for a potentially misspelled parameter.

    Uses difflib for fuzzy string matching.

    Parameters
    ----------
    name : str
        The potentially misspelled name.
    valid_names : list of str
        List of valid parameter names.
    cutoff : float, default 0.6
        Similarity threshold (0-1). Higher = more strict.
    n : int, default 3
        Maximum number of suggestions.

    Returns
    -------
    list of str
        Suggested valid names (sorted by similarity).

    Examples
    --------
    >>> valid = ['H0', 'Omega_m', 'Omega_b', 'Omega_Lambda']
    >>> suggest_similar_names('Omege_m', valid)
    ['Omega_m']
    >>> suggest_similar_names('H_0', valid)
    ['H0']
    """
    if not valid_names:
        return []

    matches = get_close_matches(name, valid_names, n=n, cutoff=cutoff)
    return matches


def validate_bounds(bounds: Optional[tuple], param_name: str = "") -> None:
    """
    Validate parameter bounds format.

    Parameters
    ----------
    bounds : tuple or None
        (min, max) bounds for parameter.
    param_name : str, optional
        Parameter name (for error messages).

    Raises
    ------
    ValueError
        If bounds format is invalid.

    Examples
    --------
    >>> validate_bounds((0, 1), 'Omega_m')
    >>> validate_bounds(None, 'H0')  # OK, bounds optional
    """
    if bounds is None:
        return

    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError(
            f"Bounds for '{param_name}' must be (min, max) tuple, got {bounds}"
        )

    min_val, max_val = bounds

    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        raise ValueError(
            f"Bounds for '{param_name}' must be numeric, got {type(min_val).__name__}, {type(max_val).__name__}"
        )

    if min_val >= max_val:
        raise ValueError(
            f"Bounds for '{param_name}' invalid: min ({min_val}) >= max ({max_val})"
        )


def build_parameter_error_message(
    invalid_params: set, model_name: str, model_params: list, provided_params: list
) -> str:
    """
    Build user-friendly error message for parameter validation failures.

    Parameters
    ----------
    invalid_params : set
        Parameters that failed validation.
    model_name : str
        Name of the cosmology model class.
    model_params : list
        Valid parameter names from model.
    provided_params : list
        Parameters provided by user.

    Returns
    -------
    str
        Formatted error message with suggestions.

    Examples
    --------
    >>> msg = build_parameter_error_message(
    ...     invalid_params={'Omege_m'},
    ...     model_name='LCDM',
    ...     model_params=['H0', 'Omega_m', 'Omega_Lambda'],
    ...     provided_params=['H0', 'Omege_m']
    ... )
    """
    lines = [
        f"❌ Invalid parameters for model '{model_name}': {sorted(invalid_params)}\n"
    ]

    # For each invalid parameter, provide spelling suggestions
    for param in sorted(invalid_params):
        matches = suggest_similar_names(param, model_params, cutoff=0.6, n=2)
        if matches:
            lines.append(f"   '{param}' not found. Did you mean: {matches}?")
        else:
            lines.append(f"   '{param}' not found in model.")

    lines.extend(
        [
            f"\n📋 Model '{model_name}' accepts these parameters:",
            f"   {sorted(model_params)}",
            f"\n🔍 You provided:",
            f"   {sorted(provided_params)}",
        ]
    )

    return "\n".join(lines)
