"""
HIcosmo Data Registry - Auto-discovery of installed datasets.

This module provides automatic discovery of available datasets by scanning
the installed data directories. No hardcoding required - new datasets are
automatically detected when added to the data folder.

Usage:
    >>> from hicosmo.data_registry import DataRegistry
    >>> registry = DataRegistry()
    >>> print(registry.list_all())  # Show all available datasets
    >>> print(registry.bao())       # Show BAO datasets only
    >>> print(registry.sn())        # Show SN datasets only
"""

from .registry import DataRegistry, list_all_datasets, show_available_datasets

__all__ = ["DataRegistry", "list_all_datasets", "show_available_datasets"]
