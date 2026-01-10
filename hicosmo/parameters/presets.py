#!/usr/bin/env python3
"""
Cosmological Parameter Presets
===============================

Standard parameter sets from major observational constraints.

This module provides fiducial cosmological parameters from:
- Planck 2018 (TT,TE,EE+lowE+lensing)
- WMAP9 (WMAP9+eCMB+BAO+H0)

Single Source of Truth - All parameter values should reference this module.
"""

from typing import Dict, Any


# Planck 2018 cosmological parameters (TT,TE,EE+lowE+lensing)
# Source: Planck Collaboration VI (2020), Table 2
# DOI: 10.1051/0004-6361/201833910
PLANCK_2018_PARAMS = {
    'H0': 67.36,  # Hubble constant [km/s/Mpc]
    'Omega_m': 0.3153,  # Total matter density
    'Omega_b': 0.0493,  # Baryon density
    # Omega_Lambda is DERIVED: 1 - Omega_m - Omega_k - Omega_r (do NOT store here!)
    'Omega_k': 0.0,  # Curvature density (flat universe)
    'Omega_r': 9.209e-05,  # Radiation density (photons + neutrinos)
    'sigma8': 0.8111,  # Matter fluctuation amplitude at 8 Mpc/h
    'n_s': 0.9649,  # Scalar spectral index
    'T_cmb': 2.7255,  # CMB temperature [K]
    'N_eff': 3.046,  # Effective number of relativistic species
    'w0': -1.0,  # Dark energy equation of state (present)
    'wa': 0.0,  # Dark energy equation of state (evolution)
}

# Parameter metadata for Planck 2018
PLANCK_2018_METADATA = {
    'H0': {'bounds': (20.0, 200.0), 'latex': r'$H_0$',
           'description': 'Hubble constant [km/s/Mpc]'},
    'Omega_m': {'bounds': (0.01, 1.0), 'latex': r'$\Omega_m$',
                'description': 'Total matter density'},
    'Omega_b': {'bounds': (0.005, 0.1), 'latex': r'$\Omega_b$',
                'description': 'Baryon density'},
    # Omega_Lambda is DERIVED - not a free parameter
    'Omega_k': {'bounds': (-0.1, 0.1), 'latex': r'$\Omega_k$',
                'description': 'Curvature density'},
    'Omega_r': {'bounds': (1e-6, 1e-3), 'latex': r'$\Omega_r$',
                'description': 'Radiation density (photons + neutrinos)'},
    'sigma8': {'bounds': (0.5, 1.5), 'latex': r'$\sigma_8$',
               'description': 'Matter fluctuation amplitude at 8 Mpc/h'},
    'n_s': {'bounds': (0.8, 1.2), 'latex': r'$n_s$',
            'description': 'Scalar spectral index'},
    'T_cmb': {'bounds': (2.7, 2.8), 'latex': r'$T_{CMB}$',
              'description': 'CMB temperature [K]'},
    'N_eff': {'bounds': (2.0, 4.0), 'latex': r'$N_{eff}$',
              'description': 'Effective number of relativistic species'},
    'w0': {'bounds': (-3.0, 0.0), 'latex': r'$w_0$',
           'description': 'Dark energy equation of state (present)'},
    'wa': {'bounds': (-3.0, 3.0), 'latex': r'$w_a$',
           'description': 'Dark energy equation of state (evolution)'},
}

# WMAP9 cosmological parameters (WMAP9+eCMB+BAO+H0)
# Source: Hinshaw et al. (2013), Table 1
# DOI: 10.1088/0067-0049/208/2/19
WMAP9_PARAMS = {
    'H0': 69.32,  # Hubble constant [km/s/Mpc]
    'Omega_m': 0.2865,  # Total matter density
    'Omega_b': 0.0463,  # Baryon density
    # Omega_Lambda is DERIVED: 1 - Omega_m - Omega_k - Omega_r (do NOT store here!)
    'Omega_k': 0.0,  # Curvature density (flat universe)
    'Omega_r': 8.4e-05,  # Radiation density (photons + neutrinos)
    'sigma8': 0.820,  # Matter fluctuation amplitude at 8 Mpc/h
    'n_s': 0.9608,  # Scalar spectral index
    'T_cmb': 2.725,  # CMB temperature [K]
    'N_eff': 3.046,  # Effective number of relativistic species
    'w0': -1.0,  # Dark energy equation of state (present)
    'wa': 0.0,  # Dark energy equation of state (evolution)
}

# Parameter metadata for WMAP9 (same structure as Planck 2018)
WMAP9_METADATA = {
    'H0': {'bounds': (20.0, 200.0), 'latex': r'$H_0$',
           'description': 'Hubble constant [km/s/Mpc]'},
    'Omega_m': {'bounds': (0.01, 1.0), 'latex': r'$\Omega_m$',
                'description': 'Total matter density'},
    'Omega_b': {'bounds': (0.005, 0.1), 'latex': r'$\Omega_b$',
                'description': 'Baryon density'},
    # Omega_Lambda is DERIVED - not a free parameter
    'Omega_k': {'bounds': (-0.1, 0.1), 'latex': r'$\Omega_k$',
                'description': 'Curvature density'},
    'Omega_r': {'bounds': (1e-6, 1e-3), 'latex': r'$\Omega_r$',
                'description': 'Radiation density (photons + neutrinos)'},
    'sigma8': {'bounds': (0.5, 1.5), 'latex': r'$\sigma_8$',
               'description': 'Matter fluctuation amplitude at 8 Mpc/h'},
    'n_s': {'bounds': (0.8, 1.2), 'latex': r'$n_s$',
            'description': 'Scalar spectral index'},
    'T_cmb': {'bounds': (2.7, 2.8), 'latex': r'$T_{CMB}$',
              'description': 'CMB temperature [K]'},
    'N_eff': {'bounds': (2.0, 4.0), 'latex': r'$N_{eff}$',
              'description': 'Effective number of relativistic species'},
    'w0': {'bounds': (-3.0, 0.0), 'latex': r'$w_0$',
           'description': 'Dark energy equation of state (present)'},
    'wa': {'bounds': (-3.0, 3.0), 'latex': r'$w_a$',
           'description': 'Dark energy equation of state (evolution)'},
}

# Combined parameters + metadata for registry compatibility
PLANCK_2018_DEFAULTS = {
    param: {'value': PLANCK_2018_PARAMS[param], **PLANCK_2018_METADATA[param]}
    for param in PLANCK_2018_PARAMS
}

WMAP9_DEFAULTS = {
    param: {'value': WMAP9_PARAMS[param], **WMAP9_METADATA[param]}
    for param in WMAP9_PARAMS
}

# Preset aliases
PRESET_CATALOG = {
    'planck2018': PLANCK_2018_DEFAULTS,
    'planck_2018': PLANCK_2018_DEFAULTS,
    'planck': PLANCK_2018_DEFAULTS,  # Default to latest
    'wmap9': WMAP9_DEFAULTS,
    'wmap': WMAP9_DEFAULTS,
}


def get_preset(name: str) -> Dict[str, Dict[str, Any]]:
    """
    Get parameter preset by name.

    Parameters
    ----------
    name : str
        Preset name (case-insensitive). Options:
        - 'planck2018', 'planck_2018', 'planck'
        - 'wmap9', 'wmap'

    Returns
    -------
    dict
        Dictionary mapping parameter names to their specifications.
        Format: {param_name: {'value': ..., 'bounds': ..., 'latex': ..., 'description': ...}}

    Raises
    ------
    ValueError
        If preset name is not recognized.

    Examples
    --------
    >>> preset = get_preset('planck2018')
    >>> print(preset['H0']['value'])
    67.36
    """
    name_lower = name.lower()
    if name_lower not in PRESET_CATALOG:
        available = ', '.join(sorted(set(PRESET_CATALOG.keys())))
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")
    return PRESET_CATALOG[name_lower]


def get_fiducial_values(name: str) -> Dict[str, float]:
    """
    Get fiducial parameter values for a preset.

    Parameters
    ----------
    name : str
        Preset name.

    Returns
    -------
    dict
        Dictionary mapping parameter names to their fiducial values.

    Examples
    --------
    >>> fid = get_fiducial_values('planck2018')
    >>> print(fid['H0'])
    67.36
    """
    preset = get_preset(name)
    return {param: spec['value'] for param, spec in preset.items()}
