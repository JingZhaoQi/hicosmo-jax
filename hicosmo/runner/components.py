"""Component registries for HIcosmo YAML configuration."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ComponentRegistry:
    """Minimal registry for named factories."""

    def __init__(self, kind: str):
        self.kind = kind
        self._factories: Dict[str, Callable[..., object]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., object],
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        key = name.lower()
        self._factories[key] = factory
        if aliases:
            for alias in aliases:
                self._factories[alias.lower()] = factory

    def get(self, name: str) -> Callable[..., object]:
        key = name.lower()
        if key not in self._factories:
            available = ", ".join(sorted(self._factories.keys()))
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {available}")
        return self._factories[key]

    def create(self, name: str, **kwargs) -> object:
        factory = self.get(name)
        return factory(**kwargs)

    def list(self) -> List[str]:
        return sorted(set(self._factories.keys()))


# =============================================================================
# Theory (Cosmology Models) Registry
# =============================================================================

THEORY_REGISTRY = ComponentRegistry("theory")


def _lazy_class(module_path: str, attr: str) -> Callable[..., object]:
    def factory(**_kwargs):
        module = importlib.import_module(module_path)
        return getattr(module, attr)

    factory._hicosmo_target = (module_path, attr)  # type: ignore[attr-defined]
    return factory


THEORY_REGISTRY.register(
    "lcdm", _lazy_class("hicosmo.models.lcdm", "LCDM"), aliases=["LCDM"]
)
THEORY_REGISTRY.register(
    "wcdm", _lazy_class("hicosmo.models.wcdm", "wCDM"), aliases=["wCDM"]
)
THEORY_REGISTRY.register(
    "cpl", _lazy_class("hicosmo.models.cpl", "CPL"), aliases=["CPL"]
)
THEORY_REGISTRY.register(
    "ilcdm", _lazy_class("hicosmo.models.ilcdm", "ILCDM"), aliases=["ILCDM"]
)


def resolve_theory(spec: Any) -> Any:
    """Resolve theory spec to a cosmology class."""
    if spec is None:
        raise KeyError("Theory spec is required.")
    if isinstance(spec, str):
        return THEORY_REGISTRY.create(spec)
    if isinstance(spec, dict):
        name = spec.get("name") or spec.get("theory") or spec.get("model")
        if not name:
            raise KeyError("Theory spec dict must include 'name'.")
        return THEORY_REGISTRY.create(name)
    return spec


# =============================================================================
# Sampler Registry
# =============================================================================

SAMPLER_REGISTRY = ComponentRegistry("sampler")
SAMPLER_REGISTRY.register("numpyro", lambda **kwargs: ("numpyro", kwargs))
SAMPLER_REGISTRY.register("emcee", lambda **kwargs: ("emcee", kwargs))
SAMPLER_REGISTRY.register("dynesty", lambda **kwargs: ("dynesty", kwargs))
SAMPLER_REGISTRY.register("nested", lambda **kwargs: ("nested", kwargs))


def resolve_sampler(spec: Any) -> Tuple[str, Dict[str, Any]]:
    """Resolve sampler config into (sampler_name, options)."""
    if spec is None:
        return "numpyro", {}
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        name = spec.get("name", "numpyro")
        options = {k: v for k, v in spec.items() if k != "name"}
        return name, options
    return "numpyro", {}


# =============================================================================
# Likelihood Registry
# =============================================================================

LIKELIHOOD_REGISTRY = ComponentRegistry("likelihood")


def _lazy_factory(module_path: str, attr: str) -> Callable[..., object]:
    def factory(**kwargs):
        module = importlib.import_module(module_path)
        target = getattr(module, attr)
        return target(**kwargs)

    factory._hicosmo_target = (module_path, attr)  # type: ignore[attr-defined]
    return factory


LIKELIHOOD_REGISTRY.register(
    "sn", _lazy_factory("hicosmo.likelihoods", "SN_likelihood"), aliases=["sne"]
)
LIKELIHOOD_REGISTRY.register(
    "sn_shoes", _lazy_factory("hicosmo.likelihoods", "SN_likelihood")
)
LIKELIHOOD_REGISTRY.register(
    "bao", _lazy_factory("hicosmo.likelihoods", "BAO_likelihood")
)
LIKELIHOOD_REGISTRY.register(
    "h0licow",
    _lazy_factory("hicosmo.likelihoods", "H0LiCOWLikelihood"),
    aliases=["h0liCOW"],
)
LIKELIHOOD_REGISTRY.register(
    "planck2018",
    _lazy_factory("hicosmo.likelihoods", "Planck2018DistancePriorsLikelihood"),
    aliases=["planck", "cmb", "planck2018_distance"],
)
LIKELIHOOD_REGISTRY.register(
    "sh0es",
    _lazy_factory("hicosmo.likelihoods", "SH0ESLikelihood"),
    aliases=["shoes", "h0"],
)
LIKELIHOOD_REGISTRY.register(
    "tdcosmo", _lazy_factory("hicosmo.likelihoods", "TDCOSMOLikelihood")
)
LIKELIHOOD_REGISTRY.register(
    "gw_standard_siren",
    _lazy_factory("hicosmo.likelihoods.gw", "GWStandardSirenLikelihood"),
    aliases=["gw", "gwsiren"],
)

DEFAULT_DATA_HINTS = {
    "sn": "sne",
    "sne": "sne",
    "sn_shoes": "sne",
    "bao": "bao_data",
    "h0licow": "h0licow",
    "tdcosmo": "tdcosmo",
    "gw_standard_siren": "gwtc-3",
}

DEFAULT_OPTIONS = {
    "sn": {"dataset": "pantheon+", "M_B": "marginalize"},
    "sn_shoes": {"dataset": "pantheon+shoes", "M_B": "marginalize"},
    "bao": {"dataset": "desi2024"},
}


def _infer_name_and_options(spec: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        name = spec.get("name") or spec.get("likelihood")
        if not name:
            raise ValueError("Likelihood spec dict must include 'name'.")
        options = {k: v for k, v in spec.items() if k not in {"name", "likelihood"}}
        if "options" in options and isinstance(options["options"], dict):
            merged = dict(options.pop("options"))
            merged.update(options)
            options = merged
        return str(name), options
    raise ValueError(f"Unsupported likelihood spec: {spec}")


def _apply_data_root(
    name: str,
    options: Dict[str, Any],
    data_root: Optional[Union[str, Path]],
) -> Dict[str, Any]:
    if not data_root:
        return options
    if "data_path" in options:
        return options
    hint = DEFAULT_DATA_HINTS.get(name.lower())
    if not hint:
        return options
    with_data_path = dict(options)
    with_data_path["data_path"] = str(Path(data_root) / hint)
    return with_data_path


def _apply_default_options(name: str, options: Dict[str, Any]) -> Dict[str, Any]:
    defaults = DEFAULT_OPTIONS.get(name.lower())
    if not defaults:
        return options
    merged = dict(defaults)
    merged.update(options)
    return merged


def _infer_signature_target(
    factory: Callable[..., object],
) -> Optional[Callable[..., object]]:
    target = getattr(factory, "_hicosmo_target", None)
    if target is None:
        return factory
    module_path, attr = target
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    except Exception:
        return None


def _inject_theory_if_needed(
    factory: Callable[..., object],
    options: Dict[str, Any],
    theory_class: Any,
) -> Dict[str, Any]:
    if theory_class is None:
        return options

    target = _infer_signature_target(factory)
    if target is None:
        return options

    try:
        signature = inspect.signature(
            target.__init__ if inspect.isclass(target) else target
        )
    except (TypeError, ValueError):
        return options

    params = signature.parameters
    updated = dict(options)
    if "cosmology_class" in params and "cosmology_class" not in updated:
        updated["cosmology_class"] = theory_class
    if "cosmology_model" in params and "cosmology_model" not in updated:
        updated["cosmology_model"] = theory_class
    return updated


def build_likelihoods(
    specs: Iterable[Any],
    *,
    theory_class: Any = None,
    data_root: Optional[Union[str, Path]] = None,
) -> List[Any]:
    """Instantiate likelihoods from configuration specs."""
    likelihoods: List[Any] = []
    for spec in specs:
        if not isinstance(spec, (str, dict)) and callable(spec):
            likelihoods.append(spec)
            continue

        name, options = _infer_name_and_options(spec)
        options = _apply_default_options(name, options)
        options = _apply_data_root(name, options, data_root)
        factory = LIKELIHOOD_REGISTRY.get(name)
        options = _inject_theory_if_needed(factory, options, theory_class)
        likelihoods.append(factory(**options))
    return likelihoods
