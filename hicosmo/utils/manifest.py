"""Run manifest writer for reproducibility."""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import jax
import numpy
import numpyro

from .logging import get_logger

logger = get_logger(__name__)


def _collect_versions() -> Dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": getattr(numpy, "__version__", "unknown"),
        "jax": getattr(jax, "__version__", "unknown"),
        "numpyro": getattr(numpyro, "__version__", "unknown"),
    }
    return versions


def write_run_manifest(output_dir: Path, config: Dict[str, Any], *, chain_name: str) -> Path:
    """Write a run manifest JSON file for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "chain_name": chain_name,
        "config": config,
        "versions": _collect_versions(),
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    }

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Run manifest written to %s", manifest_path)
    return manifest_path
