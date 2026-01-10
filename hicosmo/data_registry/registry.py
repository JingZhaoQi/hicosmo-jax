"""
Data Registry - Auto-discovery of installed datasets.

Scans the hicosmo/data directory to find available datasets without hardcoding.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json


@dataclass
class DatasetInfo:
    """Information about an available dataset."""
    name: str
    category: str  # bao, sn, cmb, h0, gw, lensing
    path: Path
    description: str = ""
    files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"DatasetInfo(name='{self.name}', category='{self.category}', files={len(self.files)})"


class DataRegistry:
    """
    Auto-discovery registry for HIcosmo datasets.

    Scans the data directory to find available datasets without hardcoding.
    New datasets are automatically detected when added to the data folder.

    Examples
    --------
    >>> from hicosmo.data_registry import DataRegistry
    >>> registry = DataRegistry()

    >>> # List all available datasets
    >>> registry.list_all()
    {'bao': ['desi_2024', 'boss_dr12', 'sdss_dr12', 'sdss_dr16', 'sixdf'],
     'sn': ['pantheon+shoes'],
     'h0': ['h0licow'],
     'lensing': ['tdcosmo'],
     'gw': ['gwtc-3']}

    >>> # Get datasets by category
    >>> registry.bao()
    ['desi_2024', 'boss_dr12', 'sdss_dr12', 'sdss_dr16', 'sixdf']

    >>> # Get detailed info
    >>> registry.get_info('bao', 'desi_2024')
    DatasetInfo(name='desi_2024', category='bao', files=2)

    >>> # Pretty print all available datasets
    >>> registry.show()
    ╔══════════════════════════════════════════════════════════════╗
    ║                   HIcosmo Available Datasets                  ║
    ╠══════════════════════════════════════════════════════════════╣
    ║ BAO (Baryon Acoustic Oscillations)                           ║
    ║   • desi_2024      : DESI 2024 BAO measurements              ║
    ║   • boss_dr12      : BOSS DR12 LRG BAO                       ║
    ║   ...                                                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    # Map data directory names to categories
    _DIR_TO_CATEGORY = {
        "bao_data": "bao",
        "sne": "sn",
        "h0licow": "h0",
        "tdcosmo": "lensing",
        "gwtc-3": "gw",
        "planck": "cmb",
    }

    # Dataset descriptions (auto-generated from directory names where not specified)
    _DESCRIPTIONS = {
        # BAO
        "desi_2024": "DESI 2024 DR1 BAO measurements (z=0.3-2.3)",
        "boss_dr12": "BOSS DR12 Luminous Red Galaxy BAO",
        "sdss_dr12": "SDSS DR12 consensus BAO",
        "sdss_dr16": "SDSS DR16 LRG and QSO BAO",
        "sixdf": "6dF Galaxy Survey BAO (z=0.1)",
        # SN
        "pantheon+shoes": "Pantheon+ with SH0ES Cepheid calibration (1701 SNe Ia)",
        "pantheon+": "Pantheon+ SNe Ia compilation (1701 SNe Ia)",
        # H0
        "h0licow": "H0LiCOW strong lensing time delays (6 lenses)",
        # Lensing
        "tdcosmo": "TDCOSMO hierarchical strong lensing (7 lenses)",
        "tdcosmo2025": "TDCOSMO 2025 updated analysis",
        # GW
        "gwtc-3": "GWTC-3 gravitational wave catalog",
        # CMB
        "planck2018": "Planck 2018 distance priors",
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data registry.

        Parameters
        ----------
        data_dir : Path, optional
            Custom data directory. Defaults to hicosmo/data/.
        """
        if data_dir is None:
            # Default: hicosmo/data relative to this file
            self._data_dir = Path(__file__).parent.parent / "data"
        else:
            self._data_dir = Path(data_dir)

        self._cache: Optional[Dict[str, List[DatasetInfo]]] = None

    def _scan_datasets(self) -> Dict[str, List[DatasetInfo]]:
        """Scan data directory and discover available datasets."""
        if self._cache is not None:
            return self._cache

        result: Dict[str, List[DatasetInfo]] = {
            "bao": [],
            "sn": [],
            "cmb": [],
            "h0": [],
            "gw": [],
            "lensing": [],
        }

        if not self._data_dir.exists():
            self._cache = result
            return result

        for subdir in self._data_dir.iterdir():
            if not subdir.is_dir():
                continue

            dirname = subdir.name

            # Handle bao_data directory (contains subdirectories)
            if dirname == "bao_data":
                for bao_subdir in subdir.iterdir():
                    if bao_subdir.is_dir() and not bao_subdir.name.startswith("."):
                        files = [f.name for f in bao_subdir.iterdir() if f.is_file()]
                        if files:  # Only add if has data files
                            info = DatasetInfo(
                                name=bao_subdir.name,
                                category="bao",
                                path=bao_subdir,
                                description=self._DESCRIPTIONS.get(
                                    bao_subdir.name, f"BAO data from {bao_subdir.name}"
                                ),
                                files=files,
                            )
                            result["bao"].append(info)

            # Handle SNe data
            elif dirname == "sne":
                files = [f.name for f in subdir.iterdir() if f.is_file() and not f.name.startswith(".")]
                if files:
                    # Check for Pantheon+SH0ES specifically
                    has_shoes = any("SH0ES" in f for f in files)
                    name = "pantheon+shoes" if has_shoes else "pantheon+"
                    info = DatasetInfo(
                        name=name,
                        category="sn",
                        path=subdir,
                        description=self._DESCRIPTIONS.get(name, "Type Ia Supernova data"),
                        files=files,
                    )
                    result["sn"].append(info)

            # Handle H0LiCOW data
            elif dirname == "h0licow":
                files = self._collect_files_recursive(subdir)
                if files:
                    info = DatasetInfo(
                        name="h0licow",
                        category="h0",
                        path=subdir,
                        description=self._DESCRIPTIONS.get("h0licow", "H0LiCOW lensing data"),
                        files=files,
                    )
                    result["h0"].append(info)

            # Handle TDCOSMO data
            elif dirname == "tdcosmo":
                files = self._collect_files_recursive(subdir)
                if files:
                    info = DatasetInfo(
                        name="tdcosmo",
                        category="lensing",
                        path=subdir,
                        description=self._DESCRIPTIONS.get("tdcosmo", "TDCOSMO lensing data"),
                        files=files,
                    )
                    result["lensing"].append(info)

                    # Check for TDCOSMO2025 subdirectory
                    tdcosmo2025 = subdir / "tdcosmo2025"
                    if tdcosmo2025.exists():
                        files_2025 = [f.name for f in tdcosmo2025.iterdir() if f.is_file()]
                        if files_2025:
                            info_2025 = DatasetInfo(
                                name="tdcosmo2025",
                                category="lensing",
                                path=tdcosmo2025,
                                description=self._DESCRIPTIONS.get("tdcosmo2025", "TDCOSMO 2025 data"),
                                files=files_2025,
                            )
                            result["lensing"].append(info_2025)

            # Handle GWTC data
            elif dirname == "gwtc-3":
                files = self._collect_files_recursive(subdir)
                if files:
                    info = DatasetInfo(
                        name="gwtc-3",
                        category="gw",
                        path=subdir,
                        description=self._DESCRIPTIONS.get("gwtc-3", "GWTC-3 GW catalog"),
                        files=files,
                    )
                    result["gw"].append(info)

        # Check for Planck data (may be in different locations)
        planck_indicators = ["planck", "cmb"]
        for indicator in planck_indicators:
            planck_dir = self._data_dir / indicator
            if planck_dir.exists():
                files = [f.name for f in planck_dir.iterdir() if f.is_file()]
                if files:
                    info = DatasetInfo(
                        name="planck2018",
                        category="cmb",
                        path=planck_dir,
                        description=self._DESCRIPTIONS.get("planck2018", "Planck 2018 CMB data"),
                        files=files,
                    )
                    result["cmb"].append(info)
                    break

        # If no explicit Planck data dir, check if distance priors are available
        # (they're often hardcoded in the likelihood)
        if not result["cmb"]:
            # Planck 2018 distance priors are always available (hardcoded values)
            info = DatasetInfo(
                name="planck2018_distance",
                category="cmb",
                path=self._data_dir,  # Not from file
                description="Planck 2018 distance priors (built-in)",
                files=[],
                metadata={"builtin": True},
            )
            result["cmb"].append(info)

        self._cache = result
        return result

    def _collect_files_recursive(self, directory: Path, max_depth: int = 3) -> List[str]:
        """Collect file names from directory recursively."""
        files = []
        for item in directory.rglob("*"):
            if item.is_file() and not item.name.startswith("."):
                # Return relative path from the dataset root
                try:
                    rel_path = item.relative_to(directory)
                    files.append(str(rel_path))
                except ValueError:
                    files.append(item.name)
        return files[:50]  # Limit to first 50 files for display

    def refresh(self) -> None:
        """Clear cache and rescan datasets."""
        self._cache = None
        self._scan_datasets()

    def list_all(self) -> Dict[str, List[str]]:
        """
        List all available datasets grouped by category.

        Returns
        -------
        dict
            Dictionary mapping category to list of dataset names.
        """
        datasets = self._scan_datasets()
        return {
            category: [d.name for d in infos]
            for category, infos in datasets.items()
            if infos
        }

    def bao(self) -> List[str]:
        """Return list of available BAO dataset names."""
        return [d.name for d in self._scan_datasets().get("bao", [])]

    def sn(self) -> List[str]:
        """Return list of available SN dataset names."""
        return [d.name for d in self._scan_datasets().get("sn", [])]

    def cmb(self) -> List[str]:
        """Return list of available CMB dataset names."""
        return [d.name for d in self._scan_datasets().get("cmb", [])]

    def h0(self) -> List[str]:
        """Return list of available H0 dataset names."""
        return [d.name for d in self._scan_datasets().get("h0", [])]

    def gw(self) -> List[str]:
        """Return list of available GW dataset names."""
        return [d.name for d in self._scan_datasets().get("gw", [])]

    def lensing(self) -> List[str]:
        """Return list of available lensing dataset names."""
        return [d.name for d in self._scan_datasets().get("lensing", [])]

    def get_info(self, category: str, name: str) -> Optional[DatasetInfo]:
        """
        Get detailed information about a specific dataset.

        Parameters
        ----------
        category : str
            Dataset category (bao, sn, cmb, h0, gw, lensing).
        name : str
            Dataset name.

        Returns
        -------
        DatasetInfo or None
            Dataset information if found.
        """
        datasets = self._scan_datasets()
        for info in datasets.get(category, []):
            if info.name == name:
                return info
        return None

    def show(self, category: Optional[str] = None) -> None:
        """
        Pretty print available datasets.

        Parameters
        ----------
        category : str, optional
            If specified, show only this category.
        """
        datasets = self._scan_datasets()

        # Category display names
        category_names = {
            "bao": "BAO (Baryon Acoustic Oscillations)",
            "sn": "SN (Type Ia Supernovae)",
            "cmb": "CMB (Cosmic Microwave Background)",
            "h0": "H0 (Direct Hubble Measurements)",
            "gw": "GW (Gravitational Waves)",
            "lensing": "Lensing (Strong Gravitational Lensing)",
        }

        print("\n" + "=" * 70)
        print("                    HIcosmo Available Datasets")
        print("=" * 70)

        categories_to_show = [category] if category else datasets.keys()

        for cat in categories_to_show:
            infos = datasets.get(cat, [])
            if not infos:
                continue

            print(f"\n  {category_names.get(cat, cat.upper())}")
            print("  " + "-" * 50)

            for info in infos:
                file_count = len(info.files)
                builtin = info.metadata.get("builtin", False)
                status = "(built-in)" if builtin else f"({file_count} files)"
                print(f"    • {info.name:18s} : {info.description[:40]}")

        print("\n" + "=" * 70)
        print("  Use: from hicosmo.likelihoods import BAO_likelihood, SN_likelihood")
        print("       bao = BAO_likelihood(LCDM, 'desi_2024')")
        print("=" * 70 + "\n")

    def to_dict(self) -> Dict[str, Any]:
        """Export registry as dictionary (for JSON serialization)."""
        datasets = self._scan_datasets()
        result = {}
        for category, infos in datasets.items():
            if infos:
                result[category] = [
                    {
                        "name": info.name,
                        "description": info.description,
                        "files_count": len(info.files),
                        "path": str(info.path),
                    }
                    for info in infos
                ]
        return result


# Singleton instance for convenience
_registry: Optional[DataRegistry] = None


def list_all_datasets(refresh: bool = False) -> Dict[str, List[str]]:
    """
    List all available datasets (convenience function).

    Parameters
    ----------
    refresh : bool
        If True, rescan the data directory.

    Returns
    -------
    dict
        Dictionary mapping category to list of dataset names.

    Examples
    --------
    >>> from hicosmo.data_registry import list_all_datasets
    >>> list_all_datasets()
    {'bao': ['desi_2024', 'boss_dr12', ...],
     'sn': ['pantheon+shoes'],
     ...}
    """
    global _registry
    if _registry is None:
        _registry = DataRegistry()
    if refresh:
        _registry.refresh()
    return _registry.list_all()


def show_available_datasets(category: Optional[str] = None) -> None:
    """
    Pretty print all available datasets.

    Parameters
    ----------
    category : str, optional
        If specified, show only this category.
    """
    global _registry
    if _registry is None:
        _registry = DataRegistry()
    _registry.show(category)
