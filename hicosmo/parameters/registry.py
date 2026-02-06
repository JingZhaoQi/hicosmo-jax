#!/usr/bin/env python3
"""
Parameter Registry for HIcosmo.

This module provides the central parameter management system that unifies
cosmological and nuisance parameters across MCMC sampling, Fisher forecasting,
and visualization.
"""

from typing import Dict, List, Optional, Any, Union
from .parameter import Parameter
from .presets import get_preset, PRESET_CATALOG
import copy
import warnings

# Import preset defaults from centralized presets module
# This ensures Single Source of Truth for all parameter values
from .presets import (
    PLANCK_2018_DEFAULTS,
    WMAP9_DEFAULTS,
)

PRESET_DEFAULTS = PRESET_CATALOG  # Use centralized preset catalog


class ParameterRegistry:
    """
    Central parameter registry for cosmological and nuisance parameters.

    This class provides a unified interface for managing all parameters in
    HIcosmo analyses, including:
    - Cosmological parameters (H0, Omega_m, w0, wa, etc.)
    - Nuisance parameters (M_B, alpha, beta for SNe, etc.)
    - Automatic merging of parameters from likelihood objects
    - Conversion to NumPyro models, Cobaya configs, etc.

    Parameters
    ----------
    name : str, optional
        Registry name for identification.

    Examples
    --------
    >>> # Create registry with Planck 2018 defaults
    >>> registry = ParameterRegistry.from_defaults('planck2018')
    >>>
    >>> # Set which parameters to sample
    >>> registry.set_free(['H0', 'Omega_m'])
    >>>
    >>> # Add a nuisance parameter
    >>> registry.add('M_B', value=-19.23, free=True,
    ...              prior={'dist': 'uniform', 'min': -20, 'max': -18})
    >>>
    >>> # Get free parameters for MCMC
    >>> free_params = registry.get_free()
    >>> # {'H0': Parameter(...), 'Omega_m': Parameter(...), 'M_B': Parameter(...)}

    >>> # Automatic nuisance parameter merging (Phase 1.2)
    >>> pantheon = SN_likelihood(LCDM, "pantheon+", M_B="free")
    >>> registry.add_from_likelihood(pantheon)  # Auto-adds M_B if needed
    """

    def __init__(self, name: str = "default"):
        """Initialize empty parameter registry."""
        self.name = name
        self._params: Dict[str, Parameter] = {}

    def add(
        self,
        name: str,
        *,
        value: Optional[float] = None,
        free: bool = True,
        prior: Optional[Dict[str, Any]] = None,
        latex_label: Optional[str] = None,
        description: Optional[str] = None,
        bounds: Optional[tuple] = None,
        **kwargs,
    ) -> None:
        """
        Add a parameter to the registry.

        Parameters
        ----------
        name : str
            Parameter name (must be valid Python identifier).
        value : float, optional
            Fiducial value or current value.
        free : bool, default=True
            Whether this parameter should be sampled (True) or held fixed (False).
        prior : dict, optional
            Prior distribution configuration. Required if free=True.
            Format: {'dist': 'uniform', 'min': 50, 'max': 100}
        latex_label : str, optional
            LaTeX representation for plotting.
        description : str, optional
            Human-readable description.
        bounds : tuple, optional
            Hard bounds (min, max). Auto-inferred from uniform prior if not provided.
        **kwargs
            Additional keyword arguments passed to Parameter.

        Raises
        ------
        ValueError
            If parameter name already exists or validation fails.

        Examples
        --------
        >>> registry = ParameterRegistry()
        >>> registry.add('H0', value=70, free=True,
        ...              prior={'dist': 'uniform', 'min': 50, 'max': 100})
        >>> registry.add('Omega_b', value=0.049, free=False)  # Fixed parameter
        """
        if name in self._params:
            raise ValueError(f"Parameter '{name}' already exists in registry")

        param = Parameter(
            name=name,
            value=value,
            free=free,
            prior=prior,
            latex_label=latex_label,
            description=description,
            bounds=bounds,
            **kwargs,
        )

        self._params[name] = param

    def remove(self, name: str) -> None:
        """
        Remove a parameter from the registry.

        Parameters
        ----------
        name : str
            Parameter name to remove.

        Raises
        ------
        KeyError
            If parameter does not exist.
        """
        if name not in self._params:
            raise KeyError(f"Parameter '{name}' not found in registry")
        del self._params[name]

    def get(self, name: str) -> Parameter:
        """
        Get a parameter from the registry.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        Parameter
            Parameter object.

        Raises
        ------
        KeyError
            If parameter does not exist.
        """
        if name not in self._params:
            raise KeyError(f"Parameter '{name}' not found in registry")
        return self._params[name]

    def set_free(self, param_names: List[str]) -> None:
        """
        Set specified parameters as free (to be sampled), others as fixed.

        Parameters
        ----------
        param_names : list of str
            Names of parameters to set as free.

        Raises
        ------
        KeyError
            If any parameter name does not exist.
        ValueError
            If a free parameter lacks a prior distribution.

        Examples
        --------
        >>> registry = ParameterRegistry.from_defaults('planck2018')
        >>> registry.set_free(['H0', 'Omega_m'])  # Only these two will be sampled
        """
        # Validate all parameters exist
        for name in param_names:
            if name not in self._params:
                raise KeyError(f"Parameter '{name}' not found in registry")

        # Set all parameters to fixed first
        for param in self._params.values():
            param.free = False

        # Set specified parameters to free
        for name in param_names:
            param = self._params[name]
            param.free = True

            # Validate free parameters have priors
            if param.prior is None:
                raise ValueError(
                    f"Cannot set '{name}' as free: no prior distribution defined. "
                    f"Use add() with prior argument to add a prior."
                )

    def set_fixed(self, param_names: List[str]) -> None:
        """
        Set specified parameters as fixed (held constant).

        Parameters
        ----------
        param_names : list of str
            Names of parameters to fix.

        Raises
        ------
        KeyError
            If any parameter name does not exist.

        Examples
        --------
        >>> registry.set_fixed(['Omega_k'])  # Fix curvature to zero
        """
        for name in param_names:
            if name not in self._params:
                raise KeyError(f"Parameter '{name}' not found in registry")
            self._params[name].free = False

    def get_free(self) -> Dict[str, Parameter]:
        """
        Get all free (sampled) parameters.

        Returns
        -------
        dict
            Dictionary of {name: Parameter} for free parameters.

        Examples
        --------
        >>> free_params = registry.get_free()
        >>> print(list(free_params.keys()))
        ['H0', 'Omega_m']
        """
        return {name: param for name, param in self._params.items() if param.is_free()}

    def get_fixed(self) -> Dict[str, Parameter]:
        """
        Get all fixed (constant) parameters.

        Returns
        -------
        dict
            Dictionary of {name: Parameter} for fixed parameters.
        """
        return {name: param for name, param in self._params.items() if param.is_fixed()}

    def get_all(self) -> Dict[str, Parameter]:
        """
        Get all parameters (free and fixed).

        Returns
        -------
        dict
            Dictionary of {name: Parameter} for all parameters.
        """
        return self._params.copy()

    def get_parameter_names(self) -> List[str]:
        """Get names of all parameters."""
        return list(self._params.keys())

    def get_free_names(self) -> List[str]:
        """Get names of free parameters."""
        return [name for name, param in self._params.items() if param.is_free()]

    def get_fixed_names(self) -> List[str]:
        """Get names of fixed parameters."""
        return [name for name, param in self._params.items() if param.is_fixed()]

    @classmethod
    def from_defaults(cls, preset: str = "planck2018") -> "ParameterRegistry":
        """
        Create a parameter registry from preset cosmological parameters.

        Parameters
        ----------
        preset : str, default='planck2018'
            Preset name. Options:
            - 'planck2018' or 'planck_2018' or 'planck': Planck 2018 results
            - 'wmap9' or 'wmap': WMAP 9-year results

        Returns
        -------
        ParameterRegistry
            Registry initialized with preset parameters (all fixed by default).

        Raises
        ------
        ValueError
            If preset name is unknown.

        Examples
        --------
        >>> # Load Planck 2018 defaults
        >>> registry = ParameterRegistry.from_defaults('planck2018')
        >>> registry.set_free(['H0', 'Omega_m'])  # Choose which to sample
        >>>
        >>> # Load WMAP9 defaults
        >>> registry_wmap = ParameterRegistry.from_defaults('wmap9')
        """
        preset_lower = preset.lower()
        if preset_lower not in PRESET_DEFAULTS:
            available = list(PRESET_DEFAULTS.keys())
            raise ValueError(
                f"Unknown preset '{preset}'. Available presets: {available}"
            )

        defaults = PRESET_DEFAULTS[preset_lower]
        registry = cls(name=f"preset_{preset}")

        # Add all preset parameters as fixed (with priors so they can be set_free later)
        for name, config in defaults.items():
            # Create uniform prior from bounds for flexibility
            bounds = config.get("bounds")
            prior = None
            if bounds is not None:
                prior = {"dist": "uniform", "min": bounds[0], "max": bounds[1]}

            registry.add(
                name,
                value=config["value"],
                free=False,  # Start as fixed (users call set_free() later)
                prior=prior,  # Add prior so set_free() works
                latex_label=config.get("latex"),
                description=config.get("description"),
                bounds=bounds,
            )

        return registry

    @classmethod
    def from_cosmological_parameters(
        cls, cosmo_params, name: str = "cosmo_params"
    ) -> "ParameterRegistry":
        """
        Create a ParameterRegistry from CosmologicalParameters.

        Parameters
        ----------
        cosmo_params : CosmologicalParameters
            Unified parameter set.
        name : str
            Registry name.
        """
        from ..models.unified_parameters import CosmologicalParameters

        if not isinstance(cosmo_params, CosmologicalParameters):
            raise TypeError("cosmo_params must be a CosmologicalParameters instance")

        registry = cls(name=name)
        all_params = cosmo_params.get_all_parameters()

        for param_name, value in all_params.items():
            spec = CosmologicalParameters.PARAM_SPECS.get(param_name)
            bounds = None
            prior = None
            if spec and spec.min_val is not None and spec.max_val is not None:
                bounds = (spec.min_val, spec.max_val)
                prior = {"dist": "uniform", "min": spec.min_val, "max": spec.max_val}

            registry.add(
                param_name,
                value=float(value) if isinstance(value, (int, float)) else value,
                free=False,
                prior=prior,
                bounds=bounds,
            )

        return registry

    def add_from_likelihood(self, likelihood) -> None:
        """
        Automatically add nuisance parameters from a likelihood object.

        This method extracts nuisance parameters (e.g., M_B for SNe) from
        a likelihood object and adds them to the registry if not already present.

        Parameters
        ----------
        likelihood : BaseLikelihood
            Likelihood object with `nuisance_parameters` property.

        Raises
        ------
        AttributeError
            If likelihood does not have `nuisance_parameters` attribute.
        ValueError
            If parameter already exists with different configuration.

        Examples
        --------
        >>> registry = ParameterRegistry.from_defaults('planck2018')
        >>> registry.set_free(['H0', 'Omega_m'])
        >>>
        >>> pantheon = SN_likelihood(LCDM, "pantheon+", M_B="free")
        >>> registry.add_from_likelihood(pantheon)  # Auto-adds M_B
        >>>
        >>> print(registry.get_free_names())
        ['H0', 'Omega_m', 'M_B', 'alpha', 'beta']  # Nuisance params added

        Notes
        -----
        This feature will be fully implemented in Phase 1.2 when BaseLikelihood
        interface is established.
        """
        # Phase 1.2 implementation: check if likelihood has nuisance_parameters
        if not hasattr(likelihood, "nuisance_parameters"):
            # Silently skip if old-style likelihood without this interface
            return

        nuisance_attr = likelihood.nuisance_parameters
        nuisance_params = nuisance_attr() if callable(nuisance_attr) else nuisance_attr

        if not nuisance_params:
            return  # No nuisance parameters to add

        # Support legacy dict-style nuisance parameters:
        # {'M_B': {'prior': {...}, 'ref': ...}, ...}
        if isinstance(nuisance_params, dict):
            converted = []
            for name, spec in nuisance_params.items():
                try:
                    converted.append(Parameter.from_simple_config(name, spec))
                except Exception as exc:
                    warnings.warn(
                        f"Skipping nuisance parameter '{name}' (invalid spec): {exc}"
                    )
            nuisance_params = converted

        for param in nuisance_params:
            if not isinstance(param, Parameter):
                warnings.warn(
                    f"Skipping non-Parameter nuisance object from "
                    f"{likelihood.__class__.__name__}: {param}"
                )
                continue

            if param.name not in self._params:
                # Add new nuisance parameter
                self._params[param.name] = copy.deepcopy(param)
            else:
                # Parameter exists - validate consistency
                existing = self._params[param.name]
                if existing.free != param.free:
                    raise ValueError(
                        f"Parameter '{param.name}' already exists with different "
                        f"free status (existing: {existing.free}, likelihood: {param.free})"
                    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert registry to dictionary representation.

        Returns
        -------
        dict
            Dictionary with all parameters.
        """
        return {
            "name": self.name,
            "parameters": {
                name: param.to_dict() for name, param in self._params.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterRegistry":
        """
        Create registry from dictionary representation.

        Parameters
        ----------
        data : dict
            Dictionary with registry data.

        Returns
        -------
        ParameterRegistry
            Registry instance.
        """
        registry = cls(name=data.get("name", "default"))
        for name, param_data in data["parameters"].items():
            param = Parameter.from_dict(param_data)
            registry._params[name] = param
        return registry

    def __repr__(self) -> str:
        """String representation of registry."""
        n_free = len(self.get_free())
        n_fixed = len(self.get_fixed())
        return f"ParameterRegistry('{self.name}', {n_free} free, {n_fixed} fixed)"

    def __contains__(self, name: str) -> bool:
        """Check if parameter exists in registry."""
        return name in self._params

    def __len__(self) -> int:
        """Return number of parameters in registry."""
        return len(self._params)

    def to_cosmological_parameters(self):
        """
        Convert registry values to CosmologicalParameters.

        Returns
        -------
        CosmologicalParameters
            Unified parameter set for model evaluation.
        """
        from ..models.unified_parameters import CosmologicalParameters

        params = {
            name: param.value
            for name, param in self._params.items()
            if param.value is not None
        }
        return CosmologicalParameters(**params)
