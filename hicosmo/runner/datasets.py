"""Dataset registry and installer utilities."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlretrieve

from ..utils.logging import get_logger

logger = get_logger(__name__)


DATASETS: Dict[str, Dict[str, Any]] = {
    "pantheon_plus": {
        "relative_path": "sne",
        "version": "1.0",
        "url": None,
        "sha256": None,
    },
    "pantheon_plus_shoes": {
        "relative_path": "sne",
        "version": "1.0",
        "url": None,
        "sha256": None,
    },
    "bao": {
        "relative_path": "bao_data",
        "version": "1.0",
        "url": None,
        "sha256": None,
    },
    "desi2024": {
        "relative_path": "bao_data",
        "version": "1.0",
        "url": None,
        "sha256": None,
    },
    "sdss_dr16": {
        "relative_path": "bao_data",
        "version": "1.0",
        "url": None,
        "sha256": None,
    },
    "h0licow": {
        "relative_path": "h0licow",
        "version": "1.0",
        "url": None,
        "sha256": None,
    },
    "tdcosmo": {
        "relative_path": "tdcosmo",
        "version": "1.0",
        "url": None,
        "sha256": None,
    },
    "tdcosmo2025": {
        "relative_path": "tdcosmo/tdcosmo2025",
        "version": "1.0",
        "url": None,
        "sha256": None,
    },
    "planck2018_distance": {
        "relative_path": "cmb",
        "version": "1.0",
        "url": None,
        "sha256": None,
    },
    "gwtc3": {
        "relative_path": "gwtc-3",
        "version": "1.0",
        "url": None,
        "sha256": None,
    },
}

_ALIASES = {
    "pantheon+": "pantheon_plus",
    "pantheon_plus": "pantheon_plus",
    "pantheonplus": "pantheon_plus",
    "pantheon+shoes": "pantheon_plus_shoes",
    "pantheon_plus_shoes": "pantheon_plus_shoes",
    "desi_2024": "desi2024",
    "gwtc-3": "gwtc3",
    "gwtc_3": "gwtc3",
}


def get_dataset(name: str) -> Dict[str, Any]:
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASETS.keys())}"
        )
    return DATASETS[key]


def resolve_data_root(data_root: Optional[str] = None) -> Path:
    if data_root:
        return Path(data_root)
    env_root = os.getenv("HICOSMO_DATA")
    if env_root:
        return Path(env_root)
    # Default to package data directory: hicosmo/data
    return Path(__file__).resolve().parents[1] / "data"


def resolve_dataset_path(name: str, data_root: Optional[str] = None) -> Path:
    meta = get_dataset(name)
    root = resolve_data_root(data_root)
    return root / meta["relative_path"]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset(
    name: str,
    data_root: Optional[str] = None,
    download: bool = True,
) -> Path:
    """Ensure dataset exists locally; download if URL is configured."""
    meta = get_dataset(name)
    path = resolve_dataset_path(name, data_root)
    if path.exists():
        return path

    url = meta.get("url")
    if not url:
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {path}. "
            "No download URL configured; please install manually or set HICOSMO_DATA."
        )
    if not download:
        raise FileNotFoundError(f"Dataset '{name}' missing at {path}.")

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset %s from %s", name, url)
    urlretrieve(url, path)

    expected_sha = meta.get("sha256")
    if expected_sha:
        got_sha = _hash_file(path)
        if got_sha != expected_sha:
            raise ValueError(
                f"Dataset '{name}' checksum mismatch: {got_sha} != {expected_sha}"
            )

    return path
