"""BAO likelihood factory and dataset registry."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .datasets import (
    BOSSDR12BAO,
    DESI2024BAO,
    DESIDR2BAO,
    SDSSDR12BAO,
    SDSSDR16BAO,
    SixDFBAO,
)
from ...models import LCDM


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


_BAO_DATASETS: Dict[str, Callable[..., Any]] = {
    "desi2024": DESI2024BAO,
    "desi_dr2": DESIDR2BAO,
    "sdss_dr12": SDSSDR12BAO,
    "sdss_dr16": SDSSDR16BAO,
    "boss_dr12": BOSSDR12BAO,
    "sixdf": SixDFBAO,
}

_BAO_ALIASES: Dict[str, str] = {
    "desi": "desi2024",
    "desi2024": "desi2024",
    "desidr1": "desi2024",
    "desi2024dr1": "desi2024",
    "desidr2": "desi_dr2",
    "desi2025": "desi_dr2",
    "desiy3": "desi_dr2",
    "desidr2bao": "desi_dr2",
    "sdss": "sdss_dr16",
    "sdssdr16": "sdss_dr16",
    "sdss16": "sdss_dr16",
    "sdssdr12": "sdss_dr12",
    "sdss12": "sdss_dr12",
    "boss": "boss_dr12",
    "bossdr12": "boss_dr12",
    "6df": "sixdf",
    "sixdf": "sixdf",
}


def register_bao_dataset(
    name: str,
    builder: Callable[..., Any],
    *,
    aliases: Optional[Iterable[str]] = None,
) -> None:
    """Register a new BAO dataset builder."""
    canonical = str(name)
    _BAO_DATASETS[canonical] = builder
    _BAO_ALIASES[_normalize_key(canonical)] = canonical
    for alias in aliases or []:
        _BAO_ALIASES[_normalize_key(alias)] = canonical


def available_bao_datasets() -> List[str]:
    """
    Return available BAO dataset identifiers.

    This function auto-discovers datasets by scanning the data directory.
    Registered datasets are combined with discovered data files.

    Returns
    -------
    List[str]
        Sorted list of available BAO dataset identifiers.

    Examples
    --------
    >>> from hicosmo.likelihoods import available_bao_datasets
    >>> available_bao_datasets()
    ['boss_dr12', 'desi2024', 'sdss_dr12', 'sdss_dr16', 'sixdf']
    """
    # Start with registered datasets
    registered = set(_BAO_DATASETS.keys())

    # Add discovered datasets from data directory
    try:
        from ...data_registry import DataRegistry

        registry = DataRegistry()
        discovered = set(registry.bao())
        # Merge registered and discovered (registered takes priority for naming)
        all_datasets = registered | discovered
    except ImportError:
        all_datasets = registered

    return sorted(all_datasets)


def _resolve_bao_dataset(dataset: str) -> Tuple[str, Callable[..., Any]]:
    if not isinstance(dataset, str):
        raise TypeError("BAO dataset must be a string identifier.")
    if dataset in _BAO_DATASETS:
        return dataset, _BAO_DATASETS[dataset]
    key = _normalize_key(dataset)
    canonical = _BAO_ALIASES.get(key)
    if canonical is None or canonical not in _BAO_DATASETS:
        raise ValueError(
            f"Unknown BAO dataset '{dataset}'. Available: {available_bao_datasets()}"
        )
    return canonical, _BAO_DATASETS[canonical]


def BAO_likelihood(
    cosmology_class: Optional[type] = None,
    dataset: str = "desi2024",
    **kwargs,
):
    """
    Construct a BAO likelihood with a dataset selector.

    Parameters
    ----------
    cosmology_class : type, optional
        Cosmology model class (LCDM, wCDM, CPL, etc.). Defaults to LCDM.
    dataset : str, default "desi2024"
        Dataset identifier (e.g., "desi2024", "sdss_dr12", "sixdf").
    **kwargs
        Passed to the underlying BAO likelihood builder.
    """
    cosmology_class = cosmology_class or LCDM
    _, builder = _resolve_bao_dataset(dataset)
    options = dict(kwargs)
    options.setdefault("cosmology_class", cosmology_class)

    try:
        bao = builder(**options)
    except TypeError:
        options.pop("cosmology_class", None)
        bao = builder(**options)
        try:
            bao._cosmology_class = cosmology_class
        except Exception:
            pass
    else:
        if getattr(bao, "_cosmology_class", None) is None:
            try:
                bao._cosmology_class = cosmology_class
            except Exception:
                pass

    return bao


BAOLikelihoodFactory = BAO_likelihood
