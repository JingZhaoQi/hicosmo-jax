"""Supernova likelihood factory and dataset registry."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .pantheonplus import PantheonPlusLikelihood
from .union3 import Union3Likelihood
from .desy5 import DESY5Likelihood
from ..base import Likelihood, NuisanceList
from ...models import LCDM


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


class _FixedMBLikelihood(Likelihood):
    """Wrapper to fix M_B while preserving likelihood interface."""

    def __init__(self, base, m_b: float):
        super().__init__(name=getattr(base, "name", base.__class__.__name__))
        self._base = base
        self._fixed_m_b = float(m_b)
        self.name = getattr(base, "name", base.__class__.__name__)

    def _load_data(self) -> None:
        return

    def _setup_covariance(self) -> None:
        return

    def get_requirements(self):
        if hasattr(self._base, "get_requirements"):
            return self._base.get_requirements()
        return {}

    def theory(self, **kwargs):
        if hasattr(self._base, "theory"):
            return self._base.theory(**kwargs)
        raise NotImplementedError("Wrapped likelihood does not provide theory()")

    def __getattr__(self, name):
        return getattr(self._base, name)

    def log_likelihood(self, model, **kwargs) -> float:
        return self._base.log_likelihood(model, M_B=self._fixed_m_b, **kwargs)

    def __call__(self, **params) -> float:
        params = dict(params)
        params.setdefault("M_B", self._fixed_m_b)
        return self._base(**params)

    def _loglike_from_grid(self, cosmo_grid, grid_z, params):
        """Shared-grid path with the fixed M_B injected.

        Without this override, CombinedLikelihood would fetch the base
        closure via __getattr__ and the fixed value would be silently
        replaced by the closure's built-in default.
        """
        params = dict(params)
        params.setdefault("M_B", self._fixed_m_b)
        return self._base._loglike_from_grid(cosmo_grid, grid_z, params)

    @property
    def nuisance_parameters(self):
        """Return nuisance parameters excluding M_B (which is fixed)."""
        nuisance = getattr(self._base, "nuisance_parameters", None)
        if nuisance is None:
            return NuisanceList()
        nuisance = nuisance() if callable(nuisance) else nuisance
        # Always return NuisanceList for consistency
        if isinstance(nuisance, dict):
            filtered = [v for k, v in nuisance.items() if k != "M_B"]
        else:
            filtered = [p for p in nuisance if getattr(p, "name", None) != "M_B"]
        return NuisanceList(filtered)


class _FixedMLikelihood(Likelihood):
    """Wrapper to fix M (magnitude offset) while preserving likelihood interface.

    Used for DESY5 and other datasets that use 'M' instead of 'M_B'.
    """

    def __init__(self, base, m: float):
        super().__init__(name=getattr(base, "name", base.__class__.__name__))
        self._base = base
        self._fixed_m = float(m)
        self.name = getattr(base, "name", base.__class__.__name__)

    def _load_data(self) -> None:
        return

    def _setup_covariance(self) -> None:
        return

    def get_requirements(self):
        if hasattr(self._base, "get_requirements"):
            return self._base.get_requirements()
        return {}

    def theory(self, **kwargs):
        if hasattr(self._base, "theory"):
            return self._base.theory(**kwargs)
        raise NotImplementedError("Wrapped likelihood does not provide theory()")

    def __getattr__(self, name):
        return getattr(self._base, name)

    def log_likelihood(self, model, **kwargs) -> float:
        return self._base.log_likelihood(model, M=self._fixed_m, **kwargs)

    def __call__(self, **params) -> float:
        params = dict(params)
        params.setdefault("M", self._fixed_m)
        return self._base(**params)

    def _loglike_from_grid(self, cosmo_grid, grid_z, params):
        """Shared-grid path with the fixed M injected (see _FixedMBLikelihood)."""
        params = dict(params)
        params.setdefault("M", self._fixed_m)
        return self._base._loglike_from_grid(cosmo_grid, grid_z, params)

    @property
    def nuisance_parameters(self):
        """Return nuisance parameters excluding M (which is fixed)."""
        nuisance = getattr(self._base, "nuisance_parameters", None)
        if nuisance is None:
            return NuisanceList()
        nuisance = nuisance() if callable(nuisance) else nuisance
        if isinstance(nuisance, dict):
            filtered = [v for k, v in nuisance.items() if k != "M"]
        else:
            filtered = [p for p in nuisance if getattr(p, "name", None) != "M"]
        return NuisanceList(filtered)


_SN_DATASETS: Dict[str, Callable[..., Any]] = {
    "pantheon+": PantheonPlusLikelihood,
    "pantheon+shoes": PantheonPlusLikelihood,
    "union3": Union3Likelihood,
    "desy5": DESY5Likelihood,
}

_SN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "pantheon+": {"include_shoes": False},
    "pantheon+shoes": {"include_shoes": True},
    "union3": {},
    "desy5": {},
}

# Datasets where M_B is already handled (no M_B parameter needed)
_SN_NO_MB: set = {"union3"}

# Datasets that use 'marginalize_M' instead of 'marginalize_M_B'
# and 'M' instead of 'M_B' as the nuisance parameter name.
_SN_M_PARAM: Dict[str, str] = {
    "desy5": "M",
}

_SN_ALIASES: Dict[str, str] = {
    "pantheon": "pantheon+",
    "pantheonplus": "pantheon+",
    "pantheonplusshoes": "pantheon+shoes",
    "pantheonshoes": "pantheon+shoes",
    "pantheonsh0es": "pantheon+shoes",
    "pantheonplussh0es": "pantheon+shoes",
    "sn": "pantheon+",
    "union31": "union3",
    "union3.1": "union3",
    "desy5": "desy5",
    "dessn5yr": "desy5",
    "desyr5": "desy5",
    "desy5sn": "desy5",
    "des5yr": "desy5",
}


def register_sn_dataset(
    name: str,
    builder: Callable[..., Any],
    *,
    defaults: Optional[Dict[str, Any]] = None,
    aliases: Optional[Iterable[str]] = None,
) -> None:
    """Register a new SN dataset builder."""
    canonical = str(name)
    _SN_DATASETS[canonical] = builder
    _SN_DEFAULTS.setdefault(canonical, {})
    if defaults:
        _SN_DEFAULTS[canonical].update(defaults)
    _SN_ALIASES[_normalize_key(canonical)] = canonical
    for alias in aliases or []:
        _SN_ALIASES[_normalize_key(alias)] = canonical


def available_sn_datasets() -> List[str]:
    """
    Return available SN dataset identifiers.

    This function auto-discovers datasets by scanning the data directory.
    Registered datasets are combined with discovered data files.

    Returns
    -------
    List[str]
        Sorted list of available SN dataset identifiers.

    Examples
    --------
    >>> from hicosmo.likelihoods import available_sn_datasets
    >>> available_sn_datasets()
    ['pantheon+', 'pantheon+shoes']
    """
    # Start with registered datasets
    registered = set(_SN_DATASETS.keys())

    # Add discovered datasets from data directory
    try:
        from ...data_registry import DataRegistry

        registry = DataRegistry()
        discovered = set(registry.sn())
        all_datasets = registered | discovered
    except ImportError:
        all_datasets = registered

    return sorted(all_datasets)


def _resolve_sn_dataset(dataset: str) -> Tuple[str, Callable[..., Any]]:
    if not isinstance(dataset, str):
        raise TypeError("SN dataset must be a string identifier.")
    if dataset in _SN_DATASETS:
        return dataset, _SN_DATASETS[dataset]
    key = _normalize_key(dataset)
    canonical = _SN_ALIASES.get(key)
    if canonical is None or canonical not in _SN_DATASETS:
        raise ValueError(
            f"Unknown SN dataset '{dataset}'. Available: {available_sn_datasets()}"
        )
    return canonical, _SN_DATASETS[canonical]


def SN_likelihood(
    cosmology_class: Optional[type] = None,
    dataset: str = "pantheon+",
    M_B: Union[str, float, None] = "marginalize",
    **kwargs,
):
    """
    Construct an SN likelihood with a dataset selector.

    Parameters
    ----------
    cosmology_class : type, optional
        Cosmology model class (LCDM, wCDM, CPL, etc.). Defaults to LCDM.
    dataset : str, default "pantheon+"
        Dataset identifier (e.g., "pantheon+", "pantheon+shoes", "union3", "desy5").
    M_B : {"marginalize", "free"} or float, default "marginalize"
        M_B handling. Use "free" to sample, "marginalize" for analytic marginalization,
        or a float to fix M_B. Ignored for datasets where M_B is already handled
        (e.g., Union3 with UNITY1.5 standardization).
    **kwargs
        Passed to the underlying likelihood builder.
    """
    cosmology_class = cosmology_class or LCDM
    canonical, builder = _resolve_sn_dataset(dataset)
    options = dict(_SN_DEFAULTS.get(canonical, {}))
    options.update(kwargs)
    options.setdefault("cosmology_class", cosmology_class)

    # Datasets with M_B already handled (e.g., Union3/UNITY1.5)
    if canonical in _SN_NO_MB:
        return builder(**options)

    # Determine the marginalization keyword for this dataset
    # DESY5 uses 'marginalize_M'; Pantheon+ uses 'marginalize_M_B'
    m_param_name = _SN_M_PARAM.get(canonical)
    marg_key = "marginalize_M" if m_param_name == "M" else "marginalize_M_B"

    if isinstance(M_B, str):
        mb_mode = M_B.strip().lower()
        if mb_mode in {"free", "sample", "fit"}:
            options[marg_key] = False
        elif mb_mode in {"marginalize", "marginalized", "auto", "analytic"}:
            options[marg_key] = True
        else:
            raise ValueError("M_B must be 'free', 'marginalize', or a float value.")
        return builder(**options)

    if M_B is None:
        options.setdefault(marg_key, True)
        return builder(**options)

    # Fixed M_B/M value
    options[marg_key] = False
    base = builder(**options)
    if m_param_name == "M":
        return _FixedMLikelihood(base, float(M_B))
    return _FixedMBLikelihood(base, float(M_B))


SNLikelihood = SN_likelihood
