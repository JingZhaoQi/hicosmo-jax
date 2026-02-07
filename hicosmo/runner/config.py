"""Configuration schema and adapters for HIcosmo."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from ..parameters import Parameter, ParameterRegistry


class ConfigError(ValueError):
    """Raised when a configuration dictionary is invalid."""


DEFAULT_CONFIG: Dict[str, Any] = {
    "name": None,
    "params": {},
    "prior": {},
    "derived": [],
    "likelihood": [],
    "theory": None,
    "sampler": {"name": "numpyro"},
    "output": {"root": "results", "chain_name": None},
    "data": {"root": None, "datasets": []},
    "preset": None,
}


def load_config(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Load a config from path or return a deep copy of a dict."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        raise ConfigError("Unsupported config file type. Use .yaml/.yml/.json")

    if isinstance(source, dict):
        return deepcopy(source)

    raise ConfigError("Config source must be a dict or path string.")


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config keys to the canonical schema."""
    cfg = deepcopy(config)

    # Legacy format: parameters + mcmc
    if "parameters" in cfg or "mcmc" in cfg:
        params = cfg.get("parameters", {})
        sampler = cfg.get("sampler")
        mcmc = cfg.get("mcmc", {})
        if sampler is None:
            sampler = {}
        if isinstance(sampler, str):
            sampler = {"name": sampler}
        if isinstance(sampler, dict):
            sampler = {**sampler, **mcmc}
        cfg = {
            "name": cfg.get("name"),
            "params": params,
            "prior": cfg.get("prior", {}),
            "derived": cfg.get("derived", []),
            "likelihood": cfg.get("likelihood", cfg.get("likelihoods", [])),
            "theory": cfg.get("theory", cfg.get("cosmology")),
            "sampler": sampler,
            "output": cfg.get("output", {}),
            "data": cfg.get("data", {}),
            "preset": cfg.get("preset"),
            "free": cfg.get("free", []),
            "fixed": cfg.get("fixed", []),
        }

    if "likelihoods" in cfg and "likelihood" not in cfg:
        cfg["likelihood"] = cfg["likelihoods"]
    if "cosmology" in cfg and "theory" not in cfg:
        cfg["theory"] = cfg["cosmology"]

    normalized = deepcopy(DEFAULT_CONFIG)
    normalized.update({k: v for k, v in cfg.items() if v is not None})

    sampler = normalized.get("sampler", {})
    if isinstance(sampler, str):
        sampler = {"name": sampler}
    if sampler is None:
        sampler = {"name": "numpyro"}
    sampler.setdefault("name", "numpyro")
    normalized["sampler"] = sampler

    normalized["likelihood"] = _ensure_list(normalized.get("likelihood"))
    normalized["params"] = normalized.get("params", {}) or {}
    normalized["prior"] = normalized.get("prior", {}) or {}
    normalized["derived"] = normalized.get("derived", []) or []

    output = normalized.get("output") or {}
    if isinstance(output, str):
        output = {"root": output}
    output.setdefault("root", "results")
    output.setdefault("chain_name", None)
    normalized["output"] = output

    data = normalized.get("data") or {}
    if isinstance(data, str):
        data = {"root": data}
    data.setdefault("root", None)
    data.setdefault("datasets", [])
    normalized["data"] = data

    normalized["free"] = normalized.get("free", []) or []
    normalized["fixed"] = normalized.get("fixed", []) or []
    return normalized


def validate_config(config: Dict[str, Any]) -> None:
    """Validate a normalized configuration dict."""
    cfg = normalize_config(config)

    if not cfg.get("params") and not cfg.get("preset"):
        raise ConfigError("Config must provide 'params' or a 'preset'.")
    if not cfg.get("likelihood"):
        raise ConfigError("Config must include at least one likelihood.")
    if not cfg.get("theory"):
        raise ConfigError("Config must specify 'theory' (e.g. 'LCDM').")
    if not isinstance(cfg.get("params"), dict):
        raise ConfigError("'params' must be a dictionary.")
    if cfg.get("prior") and not isinstance(cfg["prior"], dict):
        raise ConfigError("'prior' must be a dictionary when provided.")
    if not isinstance(cfg.get("sampler"), dict):
        raise ConfigError("'sampler' must be a dictionary.")


def _merge_prior(param_name: str, param_spec: Any, prior_map: Dict[str, Any]) -> Any:
    if not prior_map or param_name not in prior_map:
        return param_spec
    prior_spec = prior_map[param_name]
    if isinstance(param_spec, dict):
        merged = dict(param_spec)
        merged.setdefault("prior", prior_spec)
        return merged
    if isinstance(param_spec, (int, float)):
        return {"value": param_spec, "free": False, "prior": prior_spec}
    return param_spec


def build_parameter_registry(config: Dict[str, Any]) -> ParameterRegistry:
    """Build a ParameterRegistry from a normalized config."""
    cfg = normalize_config(config)
    prior_map = cfg.get("prior", {})

    if cfg.get("preset"):
        registry = ParameterRegistry.from_defaults(cfg["preset"])
    else:
        registry = ParameterRegistry(name=cfg.get("name", "default"))

    params = cfg.get("params", {})

    for name, spec in params.items():
        merged_spec = _merge_prior(name, spec, prior_map)

        if isinstance(merged_spec, Parameter):
            param = merged_spec
        elif isinstance(merged_spec, (tuple, list, set)):
            param = Parameter.from_tuple(name, merged_spec)
        elif isinstance(merged_spec, dict):
            param = Parameter.from_simple_config(name, merged_spec)
        elif isinstance(merged_spec, (int, float)):
            param = Parameter.from_simple_config(
                name, {"value": merged_spec, "free": False}
            )
        else:
            raise ConfigError(
                f"Unsupported parameter spec for '{name}': {type(merged_spec)}"
            )

        if name in registry:
            existing = registry.get(name)
            existing.value = param.value
            existing.prior = param.prior
            existing.free = param.free
            existing.bounds = param.bounds
            existing.latex_label = param.latex_label
            existing.description = param.description
        else:
            registry.add(
                name,
                value=param.value,
                free=param.free,
                prior=param.prior,
                bounds=param.bounds,
                latex_label=param.latex_label,
                description=param.description,
            )

    free_list = cfg.get("free") or []
    fixed_list = cfg.get("fixed") or []
    if free_list:
        registry.set_free(list(free_list))
    if fixed_list:
        registry.set_fixed(list(fixed_list))

    return registry
