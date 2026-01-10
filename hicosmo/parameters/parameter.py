#!/usr/bin/env python3
"""
Parameter Definition for HIcosmo Parameter Registry.

This module provides the core Parameter class that unifies cosmological and
nuisance parameter management across MCMC sampling and Fisher forecasting.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpyro.distributions as dist
from .validation import validate_parameter_name, validate_prior_dict, validate_bounds

# Standard LaTeX labels for cosmological parameters
# Used when user doesn't specify a custom label
DEFAULT_LATEX_LABELS: Dict[str, str] = {
    # Basic cosmological parameters
    'H0': r'H_0~[\mathrm{km~s^{-1}~Mpc^{-1}}]',
    'Omega_m': r'\Omega_m',
    'Omega_b': r'\Omega_b',
    'Omega_k': r'\Omega_k',
    'Omega_Lambda': r'\Omega_\Lambda',
    'Omega_r': r'\Omega_r',
    'sigma8': r'\sigma_8',
    'n_s': r'n_s',
    # Dark energy parameters
    'w': r'w',
    'w0': r'w_0',
    'wa': r'w_a',
    # Nuisance parameters
    'M_B': r'M_B',
    'alpha': r'\alpha',
    'beta': r'\beta',
    # Derived parameters
    'rd': r'r_d~[\mathrm{Mpc}]',
    'rd_h': r'r_d h~[\mathrm{Mpc}]',
    'H0_rd': r'H_0 r_d~[\mathrm{km~s^{-1}}]',
    # CMB parameters
    'T_cmb': r'T_{\mathrm{CMB}}',
    'N_eff': r'N_{\mathrm{eff}}',
}


@dataclass
class Parameter:
    r"""
    Unified parameter definition for HIcosmo.

    This class represents a single parameter (cosmological or nuisance) with all
    information needed for MCMC sampling, Fisher forecasting, and result visualization.

    Parameters
    ----------
    name : str
        Parameter name (must be valid Python identifier).
    value : Optional[float]
        Current value or fiducial value for this parameter.
    free : bool, default=True
        Whether this parameter should be sampled (free) or held fixed.
    prior : Optional[Dict[str, Any]]
        Prior distribution configuration. If None, parameter must be fixed.
        Format: {'dist': 'uniform', 'min': 50, 'max': 100}
        Supported distributions: uniform, normal, truncnorm, lognormal, etc.
    latex_label : Optional[str]
        LaTeX representation for plotting (e.g., r'$H_0$').
    description : Optional[str]
        Human-readable description.
    bounds : Optional[Tuple[float, float]]
        Hard bounds for parameter (min, max). Automatically inferred from
        uniform priors if not specified.

    Examples
    --------
    >>> # Free cosmological parameter with uniform prior
    >>> H0 = Parameter(
    ...     name='H0',
    ...     value=70.0,
    ...     free=True,
    ...     prior={'dist': 'uniform', 'min': 50, 'max': 100},
    ...     latex_label=r'$H_0$'
    ... )

    >>> # Fixed parameter
    >>> Omega_b = Parameter(
    ...     name='Omega_b',
    ...     value=0.049,
    ...     free=False,
    ...     description='Baryon density (fixed to Planck 2018)'
    ... )

    >>> # Nuisance parameter from likelihood
    >>> M_B = Parameter(
    ...     name='M_B',
    ...     value=-19.23,
    ...     free=True,
    ...     prior={'dist': 'uniform', 'min': -20, 'max': -18},
    ...     latex_label=r'$\mathcal{M}_B$',
    ...     description='SNe Ia absolute magnitude'
    ... )
    """

    name: str
    value: Optional[float] = None
    free: bool = True
    prior: Optional[Dict[str, Any]] = None
    latex_label: Optional[str] = None
    description: Optional[str] = None
    bounds: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        """Validate parameter configuration after initialization."""
        self._validate_name()
        self._normalize_prior()
        self._validate_free_prior_consistency()
        self._validate_bounds()
        self._infer_bounds_from_prior()
        self._set_defaults()

    def _normalize_prior(self) -> None:
        """Normalize prior naming and parameter aliases for internal consistency."""
        if self.prior is None:
            return

        # Normalize distribution name
        dist_type = self.prior.get('dist')
        if dist_type:
            dist_lower = dist_type.lower()
            dist_aliases = {
                'log_normal': 'lognormal',
                'truncated_normal': 'truncnorm',
                'half_normal': 'halfnormal',
                'half_cauchy': 'halfcauchy',
            }
            self.prior['dist'] = dist_aliases.get(dist_lower, dist_lower)

        dist_norm = self.prior.get('dist')

        # Normalize truncnorm bounds
        if dist_norm == 'truncnorm':
            if 'low' not in self.prior and 'min' in self.prior:
                self.prior['low'] = self.prior['min']
            if 'high' not in self.prior and 'max' in self.prior:
                self.prior['high'] = self.prior['max']

        # Normalize beta parameter names
        if dist_norm == 'beta':
            if 'alpha' not in self.prior and 'a' in self.prior:
                self.prior['alpha'] = self.prior['a']
            if 'beta' not in self.prior and 'b' in self.prior:
                self.prior['beta'] = self.prior['b']

        # Normalize gamma parameter names
        if dist_norm == 'gamma':
            if 'concentration' not in self.prior and 'shape' in self.prior:
                self.prior['concentration'] = self.prior['shape']

    def _validate_name(self):
        """Validate parameter name using centralized validation."""
        validate_parameter_name(self.name)

    def _validate_free_prior_consistency(self):
        """Validate that free parameters have priors and fixed parameters have values."""
        if self.free:
            if self.prior is None:
                raise ValueError(
                    f"Free parameter '{self.name}' must have a prior distribution"
                )
            # Use centralized prior validation
            validate_prior_dict(self.prior, self.name)
        else:
            # Fixed parameters should have a value
            if self.value is None:
                raise ValueError(
                    f"Fixed parameter '{self.name}' must have a value"
                )

    def _validate_bounds(self):
        """Validate bounds format using centralized validation."""
        if self.bounds is not None:
            validate_bounds(self.bounds, self.name)

    def _infer_bounds_from_prior(self):
        """Automatically infer bounds from uniform prior if not specified."""
        if self.bounds is None and self.prior is not None:
            if self.prior.get('dist') == 'uniform':
                self.bounds = (self.prior['min'], self.prior['max'])
            elif self.prior.get('dist') == 'truncnorm':
                self.bounds = (self.prior['low'], self.prior['high'])

    def _set_defaults(self):
        """Set default values for optional fields."""
        # Use standard LaTeX label if not specified
        if self.latex_label is None:
            self.latex_label = DEFAULT_LATEX_LABELS.get(self.name, self.name)

        # If value is None and prior has ref, use it
        if self.value is None and self.prior is not None:
            if 'ref' in self.prior:
                self.value = self.prior['ref']
            elif self.prior.get('dist') == 'uniform':
                # Default to middle of uniform prior
                self.value = (self.prior['min'] + self.prior['max']) / 2.0
            elif self.prior.get('dist') in ['normal', 'truncnorm']:
                # Default to mean of normal-like distribution
                self.value = self.prior['loc']

    def to_numpyro_dist(self) -> dist.Distribution:
        """
        Convert prior configuration to NumPyro distribution object.

        Returns
        -------
        numpyro.distributions.Distribution
            NumPyro distribution object for MCMC sampling.

        Raises
        ------
        ValueError
            If parameter is fixed (no prior) or prior type is unknown.

        Examples
        --------
        >>> H0 = Parameter(name='H0', value=70, free=True,
        ...                prior={'dist': 'uniform', 'min': 50, 'max': 100})
        >>> numpyro_dist = H0.to_numpyro_dist()
        >>> # numpyro_dist is Uniform(50, 100)
        """
        if not self.free or self.prior is None:
            raise ValueError(f"Cannot convert fixed parameter '{self.name}' to NumPyro distribution")

        dist_type = self.prior['dist'].lower()

        # Map distribution types to NumPyro constructors
        if dist_type == 'uniform':
            return dist.Uniform(self.prior['min'], self.prior['max'])

        elif dist_type == 'normal':
            return dist.Normal(self.prior['loc'], self.prior['scale'])

        elif dist_type == 'truncnorm':
            return dist.TruncatedNormal(
                loc=self.prior['loc'],
                scale=self.prior['scale'],
                low=self.prior['low'],
                high=self.prior['high']
            )

        elif dist_type == 'lognormal':
            return dist.LogNormal(self.prior['loc'], self.prior['scale'])

        elif dist_type == 'beta':
            return dist.Beta(self.prior['alpha'], self.prior['beta'])

        elif dist_type == 'gamma':
            return dist.Gamma(self.prior['concentration'], self.prior['rate'])

        elif dist_type == 'halfnormal':
            return dist.HalfNormal(self.prior['scale'])

        elif dist_type == 'halfcauchy':
            return dist.HalfCauchy(self.prior['scale'])

        elif dist_type == 'exponential':
            return dist.Exponential(self.prior['rate'])

        else:
            raise ValueError(
                f"Unknown prior type '{dist_type}' for parameter '{self.name}'"
            )

    def is_free(self) -> bool:
        """Check if parameter is free (to be sampled)."""
        return self.free

    def is_fixed(self) -> bool:
        """Check if parameter is fixed (held constant)."""
        return not self.free

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter to dictionary representation.

        Returns
        -------
        dict
            Dictionary containing all parameter information.
        """
        return {
            'name': self.name,
            'value': self.value,
            'free': self.free,
            'prior': self.prior,
            'latex_label': self.latex_label,
            'description': self.description,
            'bounds': self.bounds
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Parameter':
        """
        Create Parameter from dictionary representation.

        Parameters
        ----------
        data : dict
            Dictionary with parameter information.

        Returns
        -------
        Parameter
            Parameter instance.
        """
        return cls(**data)

    @classmethod
    def from_tuple(
        cls,
        name: str,
        spec: tuple,
        latex: Optional[str] = None
    ) -> 'Parameter':
        """
        Create Parameter from (initial, min, max, latex) tuple.

        Migrated from AutoParameter to unify parameter creation API.

        Parameters
        ----------
        name : str
            Parameter name.
        spec : tuple
            Parameter specification: (initial, min, max) or (initial, min, max, latex).
        latex : str, optional
            LaTeX label (overrides latex from tuple if provided).

        Returns
        -------
        Parameter
            Configured parameter instance.

        Examples
        --------
        >>> H0 = Parameter.from_tuple('H0', (70.0, 50.0, 100.0, r'$H_0$'))
        >>> Omega_m = Parameter.from_tuple('Omega_m', (0.3, 0.1, 0.5))
        """
        if len(spec) < 3:
            raise ValueError(f"Parameter '{name}' needs at least (initial, min, max)")

        initial = float(spec[0])
        min_val = float(spec[1])
        max_val = float(spec[2])
        latex_from_tuple = spec[3] if len(spec) > 3 else None

        # Use provided latex or fall back to tuple latex
        final_latex = latex if latex is not None else latex_from_tuple

        return cls(
            name=name,
            value=initial,
            free=True,
            prior={'dist': 'uniform', 'min': min_val, 'max': max_val},
            bounds=(min_val, max_val),
            latex_label=final_latex
        )

    @classmethod
    def from_simple_config(cls, name: str, config: dict) -> 'Parameter':
        """
        Create Parameter from simplified configuration dict.

        Migrated from AutoParameter to unify parameter creation API.
        Supports both AutoParameter's dict format and simplified format.

        Parameters
        ----------
        name : str
            Parameter name.
        config : dict
            Simplified parameter configuration. Supports:
            - {'prior': {...}, 'ref': value, 'latex': '...', 'bounds': (...)}
            - {'value': 70.0, 'free': False}
            - {'dist': 'uniform', 'min': 50, 'max': 100}

        Returns
        -------
        Parameter
            Configured parameter instance.

        Examples
        --------
        >>> # AutoParameter-style config with prior
        >>> H0 = Parameter.from_simple_config('H0', {
        ...     'prior': {'dist': 'uniform', 'min': 50, 'max': 100},
        ...     'ref': 70.0,
        ...     'latex': r'$H_0$'
        ... })

        >>> # Fixed parameter
        >>> Omega_b = Parameter.from_simple_config('Omega_b', {
        ...     'value': 0.049,
        ...     'free': False
        ... })

        >>> # Inline prior format
        >>> Omega_m = Parameter.from_simple_config('Omega_m', {
        ...     'dist': 'uniform',
        ...     'min': 0.1,
        ...     'max': 0.5
        ... })
        """
        # Case 1: Full config with 'prior' key (AutoParameter format)
        if 'prior' in config:
            prior = config['prior']
            value = config.get('ref', config.get('value'))
            latex = config.get('latex', config.get('latex_label'))
            bounds = config.get('bounds')
            description = config.get('description')
            free = config.get('free', True)

            return cls(
                name=name,
                value=value,
                free=free,
                prior=prior if free else None,
                latex_label=latex,
                description=description,
                bounds=bounds
            )

        # Case 2: Fixed parameter with 'value' and 'free': False
        elif 'value' in config and not config.get('free', True):
            return cls(
                name=name,
                value=config['value'],
                free=False,
                latex_label=config.get('latex', config.get('latex_label')),
                description=config.get('description')
            )

        # Case 3: Inline prior format (dist at top level)
        elif 'dist' in config:
            value = config.get('ref', config.get('value'))
            latex = config.get('latex', config.get('latex_label'))
            bounds = config.get('bounds')
            description = config.get('description')

            return cls(
                name=name,
                value=value,
                free=True,
                prior=config,
                latex_label=latex,
                description=description,
                bounds=bounds
            )

        # Case 4: User-friendly format with init/min/max
        # e.g., {'init': 0.0, 'min': -5.0, 'max': 5.0, 'latex': '$b$'}
        elif 'min' in config and 'max' in config:
            min_val = config['min']
            max_val = config['max']
            init_val = config.get('init', config.get('value', (min_val + max_val) / 2.0))
            latex = config.get('latex', config.get('latex_label'))
            description = config.get('description')

            return cls(
                name=name,
                value=init_val,
                free=True,
                prior={'dist': 'uniform', 'min': min_val, 'max': max_val},
                latex_label=latex,
                description=description,
                bounds=(min_val, max_val)
            )

        else:
            raise ValueError(
                f"Cannot interpret config for parameter '{name}'. "
                "Expected 'prior' key, 'value' with 'free': False, 'dist' key, "
                "or 'min'/'max' keys."
            )

    def __repr__(self) -> str:
        """String representation of parameter."""
        status = "free" if self.free else "fixed"
        value_str = f"={self.value}" if self.value is not None else ""
        return f"Parameter('{self.name}'{value_str}, {status})"

    # ==================== Backward Compatibility Aliases ====================

    @property
    def ref(self) -> Optional[float]:
        """Backward-compatible alias for value."""
        return self.value

    @ref.setter
    def ref(self, new_value: Optional[float]) -> None:
        self.value = new_value

    @property
    def latex(self) -> Optional[str]:
        """Backward-compatible alias for latex_label."""
        return self.latex_label

    @latex.setter
    def latex(self, new_value: Optional[str]) -> None:
        self.latex_label = new_value
