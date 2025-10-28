"""
Specific BAO dataset implementations for HIcosmo.

This module contains implementations for various BAO surveys:
- SDSS DR12 and DR16
- BOSS DR12
- DESI 2024
- 6dFGS
"""

import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import yaml

from .bao_base import BAOLikelihood, BAODataset, BAODataPoint


# Mapping between different dataset naming conventions and HIcosmo observables
_OBSERVABLE_MAP = {
    'DM_over_rs': 'DM_over_rd',
    'DM_over_rd': 'DM_over_rd',
    'bao_DM_over_rs': 'DM_over_rd',
    'DH_over_rs': 'DH_over_rd',
    'DH_over_rd': 'DH_over_rd',
    'bao_DH_over_rs': 'DH_over_rd',
    'DV_over_rs': 'DV_over_rd',
    'DV_over_rd': 'DV_over_rd',
    'bao_DV_over_rs': 'DV_over_rd',
    'rs_over_DV': 'rs_over_DV',
    'bao_rs_over_DV': 'rs_over_DV',
    'Hz_rs': 'Hz_rs',
    'bao_Hz_rs': 'Hz_rs',
    'f_sigma8': 'fsigma8',
}


def _normalize_observable(raw: str) -> str:
    """Map observable labels from external datasets into HIcosmo conventions."""
    observable = raw.strip()
    if observable.startswith('bao_') and observable not in _OBSERVABLE_MAP:
        observable = observable[4:]
    if observable not in _OBSERVABLE_MAP:
        raise ValueError(f"Unsupported BAO observable label '{raw}'.")
    return _OBSERVABLE_MAP[observable]


def _read_data_file(file_path: Path) -> List[Tuple[float, float, Optional[float], str]]:
    """Parse a BAO data file containing rows of measurements."""
    entries: List[Tuple[float, float, Optional[float], str]] = []
    with open(file_path, 'r') as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue

            z = float(parts[0])
            value = float(parts[1])

            if len(parts) >= 4:
                error = float(parts[2])
                observable_raw = parts[3]
            else:
                error = None
                observable_raw = parts[2]

            observable = _normalize_observable(observable_raw)
            entries.append((z, value, error, observable))

    if not entries:
        raise ValueError(f"No BAO data found in '{file_path}'.")
    return entries


def _build_dataset(
    name: str,
    entries: Sequence[Tuple[float, float, Optional[float], str]],
    covariance: Optional[np.ndarray],
    reference: str,
    year: int,
) -> BAODataset:
    """Create a BAODataset from parsed entries and covariance information."""
    num_points = len(entries)
    if covariance is not None:
        covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
        if covariance.shape != (num_points, num_points):
            raise ValueError(
                f"Covariance shape {covariance.shape} does not match number of data points {num_points}."
            )
        errors = np.sqrt(np.diag(covariance))
    else:
        errors = np.array([
            entry[2] if entry[2] is not None else None
            for entry in entries
        ], dtype=float)
        if np.any(np.isnan(errors)):
            raise ValueError(
                f"Covariance not provided and some entries missing errors for dataset '{name}'."
            )
        covariance = np.diag(errors**2)

    data_points: List[BAODataPoint] = []
    for idx, (z, value, error_override, observable) in enumerate(entries):
        error = float(errors[idx])
        data_points.append(
            BAODataPoint(
                z=float(z),
                value=float(value),
                error=error,
                observable=observable,
            )
        )

    return BAODataset(
        name=name,
        data_points=data_points,
        covariance=covariance,
        reference=reference,
        year=year,
    )


class SDSSDR12BAO(BAOLikelihood):
    """
    SDSS DR12 consensus BAO measurements.

    Reference: Alam et al. 2017, MNRAS 470, 2617
    https://arxiv.org/abs/1607.03155
    """

    def _default_dataset_name(self) -> str:
        return 'sdss_dr12_consensus'

    def _load_dataset(self) -> BAODataset:
        """Load SDSS DR12 consensus BAO data from official files."""
        base_path = Path(self.data_path) / 'sdss_dr12'
        data_file = base_path / 'sdss_DR12Consensus_bao.dat'
        cov_file = base_path / 'BAO_consensus_covtot_dM_Hz.txt'

        entries = _read_data_file(data_file)
        covariance = np.loadtxt(cov_file)

        return _build_dataset(
            name="SDSS DR12 Consensus BAO",
            entries=entries,
            covariance=covariance,
            reference="Alam et al. 2017, MNRAS 470, 2617",
            year=2017,
        )


class SDSSDR16BAO(BAOLikelihood):
    """
    SDSS DR16 BAO measurements from eBOSS.

    Reference: eBOSS Collaboration 2020
    """

    def _default_dataset_name(self) -> str:
        return 'sdss_dr16_bao'

    def _load_dataset(self) -> BAODataset:
        """Load SDSS DR16 BAO data."""
        base_path = Path(self.data_path) / 'sdss_dr16'
        data_file = base_path / 'eboss_bao_data.txt'

        entries = _read_data_file(data_file)

        return _build_dataset(
            name="SDSS DR16 eBOSS BAO",
            entries=entries,
            covariance=None,
            reference="eBOSS Collaboration 2020",
            year=2020,
        )


class BOSSDR12BAO(BAOLikelihood):
    """
    BOSS DR12 BAO measurements with anisotropic clustering.

    Reference: Alam et al. 2017
    """

    def _default_dataset_name(self) -> str:
        return 'boss_dr12'

    def _load_dataset(self) -> BAODataset:
        """Load BOSS DR12 BAO data with DM and DH separately."""
        base_path = Path(self.data_path) / 'boss_dr12'
        data_file = base_path / 'sdss_DR12_LRG_BAO_DMDH.dat'
        cov_file = base_path / 'sdss_DR12_LRG_BAO_DMDH_covtot.txt'

        entries = _read_data_file(data_file)
        covariance = np.loadtxt(cov_file)

        return _build_dataset(
            name="BOSS DR12 Anisotropic BAO",
            entries=entries,
            covariance=covariance,
            reference="Alam et al. 2017, MNRAS 470, 2617",
            year=2017,
        )


class DESI2024BAO(BAOLikelihood):
    """
    DESI 2024 BAO measurements (Data Release 1).

    Reference: DESI Collaboration 2024 (preliminary)
    """

    def _default_dataset_name(self) -> str:
        return 'desi_2024_dr1'

    def _load_dataset(self) -> BAODataset:
        """Load DESI 2024 DR1 BAO data."""
        base_path = Path(self.data_path) / 'desi_2024'
        data_file = base_path / 'desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
        cov_file = base_path / 'desi_2024_gaussian_bao_ALL_GCcomb_cov.txt'

        entries = _read_data_file(data_file)
        covariance = np.loadtxt(cov_file)

        return _build_dataset(
            name="DESI 2024 DR1 BAO",
            entries=entries,
            covariance=covariance,
            reference="DESI Collaboration 2024",
            year=2024,
        )


class SixDFBAO(BAOLikelihood):
    """
    6dF Galaxy Survey BAO measurement.

    Low-redshift anchor for BAO scale.
    Reference: Beutler et al. 2011, MNRAS 416, 3017
    """

    def _default_dataset_name(self) -> str:
        return 'sixdf_2011'

    def _load_dataset(self) -> BAODataset:
        """Load 6dFGS BAO data."""
        base_path = Path(self.data_path) / 'sixdf'
        data_file = base_path / 'sixdf_bao_data.txt'

        entries = _read_data_file(data_file)

        return _build_dataset(
            name="6dF Galaxy Survey",
            entries=entries,
            covariance=None,
            reference="Beutler et al. 2011, MNRAS 416, 3017",
            year=2011,
        )


class CustomBAO(BAOLikelihood):
    """
    Custom BAO dataset loaded from user-provided files.

    Expects YAML configuration file with data points and covariance.
    """

    def __init__(self, config_file: str, **kwargs):
        """
        Initialize custom BAO dataset.

        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = config_file
        super().__init__(**kwargs)

    def _default_dataset_name(self) -> str:
        return 'custom_bao'

    def _load_dataset(self) -> BAODataset:
        """Load custom BAO data from configuration file."""
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Parse data points
        data_points = []
        for point in config['data_points']:
            data_points.append(BAODataPoint(
                z=point['z'],
                value=point['value'],
                error=point.get('error', 0.0),
                observable=point['observable']
            ))

        # Load covariance if provided
        covariance = None
        if 'covariance_file' in config:
            cov_path = os.path.join(
                os.path.dirname(self.config_file),
                config['covariance_file']
            )
            covariance = np.loadtxt(cov_path)
        elif 'covariance' in config:
            covariance = np.array(config['covariance'])
        else:
            # Use diagonal errors
            errors = np.array([p.error for p in data_points])
            if np.any(errors > 0):
                covariance = np.diag(errors**2)

        return BAODataset(
            name=config.get('name', 'Custom BAO Dataset'),
            data_points=data_points,
            covariance=covariance,
            reference=config.get('reference', ''),
            year=config.get('year', 0)
        )


def get_available_datasets() -> List[str]:
    """Get list of available BAO datasets."""
    return [
        'sdss_dr12',
        'sdss_dr16',
        'boss_dr12',
        'desi_2024',
        'sixdf',
    ]


def create_bao_likelihood(dataset: str, **kwargs) -> BAOLikelihood:
    """
    Factory function to create BAO likelihood from dataset name.

    Args:
        dataset: Name of BAO dataset
        **kwargs: Additional arguments for likelihood

    Returns:
        BAOLikelihood object
    """
    dataset_map = {
        'sdss_dr12': SDSSDR12BAO,
        'sdss_dr16': SDSSDR16BAO,
        'boss_dr12': BOSSDR12BAO,
        'desi_2024': DESI2024BAO,
        'sixdf': SixDFBAO,
    }

    if dataset not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset}. "
                        f"Available: {list(dataset_map.keys())}")

    return dataset_map[dataset](**kwargs)
