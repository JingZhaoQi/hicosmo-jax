#!/usr/bin/env python3
"""
Unified Parameter Collector for HIcosmo.

This module provides a lightweight aggregator that collects parameters from multiple
sources (cosmology models, likelihoods, user definitions) and produces a unified
ParameterRegistry for MCMC sampling.

Author: HIcosmo Development Team
"""

from typing import Dict, List, Optional, Any, Union, Type
import copy
import warnings

from .parameter import Parameter
from .registry import ParameterRegistry


class UnifiedParameterCollector:
    """
    Unified parameter collector that aggregates parameters from multiple sources.

    This class provides a clean interface for collecting parameters from:
    - Cosmological models (LCDM, wCDM, CPL, etc.)
    - Likelihood objects (nuisance parameters like M_B, alpha, beta)
    - User-defined custom parameters

    The collector automatically handles:
    - Parameter name conflict detection
    - Prior validation
    - Merging of parameters from different sources

    Parameters
    ----------
    name : str, optional
        Name for this parameter collection.
    preset : str, optional
        Initialize with a preset ('planck2018', 'wmap9'). If None, starts empty.

    Examples
    --------
    >>> from hicosmo.parameters import UnifiedParameterCollector
    >>> from hicosmo.models import LCDM
    >>> from hicosmo.likelihoods import PantheonPlusLikelihood, DESI2024BAO
    >>>
    >>> # Create collector
    >>> collector = UnifiedParameterCollector()
    >>>
    >>> # Add cosmology model parameters (specify which to sample)
    >>> collector.add_from_model(LCDM, free_params=['H0', 'Omega_m'])
    >>>
    >>> # Auto-collect nuisance parameters from likelihoods
    >>> pantheon = PantheonPlusLikelihood()
    >>> collector.add_from_likelihood(pantheon)  # Adds M_B automatically
    >>>
    >>> bao = DESI2024BAO(cosmology_class=LCDM)
    >>> collector.add_from_likelihood(bao)  # No nuisance params, skipped
    >>>
    >>> # Get unified registry for MCMC
    >>> registry = collector.to_registry()
    >>> print(registry.get_free_names())
    ['H0', 'Omega_m', 'M_B']

    Notes
    -----
    This class is designed to work with existing HIcosmo infrastructure:
    - Uses `Likelihood.nuisance_parameters` property for auto-discovery
    - Uses `Model.get_parameters()` classmethod for model parameters
    - Produces standard `ParameterRegistry` for MCMC compatibility
    """

    def __init__(self, name: str = "collected", preset: Optional[str] = None):
        """
        Initialize parameter collector.

        Parameters
        ----------
        name : str
            Name for this collection.
        preset : str, optional
            Load preset cosmological parameters ('planck2018', 'wmap9').
        """
        self.name = name
        self._params: Dict[str, Parameter] = {}
        self._sources: Dict[str, str] = {}  # Track parameter source for debugging

        # Load preset if specified
        if preset is not None:
            self._load_preset(preset)

    def _load_preset(self, preset: str) -> None:
        """Load cosmological parameter preset."""
        try:
            registry = ParameterRegistry.from_defaults(preset)
            for name, param in registry.get_all().items():
                self._params[name] = copy.deepcopy(param)
                self._sources[name] = f"preset:{preset}"
        except ValueError as e:
            raise ValueError(f"Failed to load preset '{preset}': {e}")

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
        source: str = "user",
    ) -> "UnifiedParameterCollector":
        """
        Add a custom parameter to the collection.

        Parameters
        ----------
        name : str
            Parameter name.
        value : float, optional
            Initial/fiducial value.
        free : bool
            Whether to sample this parameter.
        prior : dict, optional
            Prior distribution config. Required if free=True.
        latex_label : str, optional
            LaTeX label for plotting.
        description : str, optional
            Human-readable description.
        bounds : tuple, optional
            Hard bounds (min, max).
        source : str
            Source identifier for debugging.

        Returns
        -------
        UnifiedParameterCollector
            Self for method chaining.

        Raises
        ------
        ValueError
            If parameter already exists with different configuration.
        """
        if name in self._params:
            existing = self._params[name]
            # Check for conflict
            if existing.free != free or existing.value != value:
                raise ValueError(
                    f"Parameter '{name}' already exists from source '{self._sources[name]}' "
                    f"with different configuration. Cannot add from '{source}'."
                )
            # Same parameter, skip silently
            return self

        param = Parameter(
            name=name,
            value=value,
            free=free,
            prior=prior,
            latex_label=latex_label,
            description=description,
            bounds=bounds,
        )

        self._params[name] = param
        self._sources[name] = source
        return self

    def add_from_likelihood(
        self, likelihood, override: bool = False
    ) -> "UnifiedParameterCollector":
        """
        Automatically collect nuisance parameters from a likelihood object.

        Parameters
        ----------
        likelihood : Likelihood
            Likelihood object with `nuisance_parameters` property.
        override : bool
            If True, override existing parameters. If False, raise on conflict.

        Returns
        -------
        UnifiedParameterCollector
            Self for method chaining.

        Notes
        -----
        This method looks for the `nuisance_parameters` property on the likelihood.
        If the property doesn't exist or returns empty, this is a no-op.

        Examples
        --------
        >>> pantheon = PantheonPlusLikelihood()
        >>> collector.add_from_likelihood(pantheon)
        >>> # Now collector has M_B parameter (if defined in pantheon.nuisance_parameters)
        """
        # Check if likelihood has nuisance_parameters property
        if not hasattr(likelihood, "nuisance_parameters"):
            return self

        nuisance_attr = likelihood.nuisance_parameters
        nuisance_params = nuisance_attr() if callable(nuisance_attr) else nuisance_attr

        if not nuisance_params:
            return self  # No nuisance parameters

        # Support legacy dict-style nuisance parameters
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

        source = f"likelihood:{likelihood.__class__.__name__}"

        for param in nuisance_params:
            if not isinstance(param, Parameter):
                warnings.warn(
                    f"Skipping non-Parameter object in {likelihood.__class__.__name__}.nuisance_parameters"
                )
                continue

            if param.name in self._params:
                if override:
                    self._params[param.name] = copy.deepcopy(param)
                    self._sources[param.name] = source
                else:
                    existing = self._params[param.name]
                    # Silent skip if same configuration
                    if existing.free == param.free and existing.value == param.value:
                        continue
                    raise ValueError(
                        f"Parameter '{param.name}' already exists from '{self._sources[param.name]}'. "
                        f"Cannot add from '{source}'. Use override=True to replace."
                    )
            else:
                self._params[param.name] = copy.deepcopy(param)
                self._sources[param.name] = source

        return self

    def add_from_model(
        self,
        model_class: Type,
        free_params: Optional[List[str]] = None,
        fixed_params: Optional[Dict[str, float]] = None,
    ) -> "UnifiedParameterCollector":
        """
        Collect parameters from a cosmology model class.

        Parameters
        ----------
        model_class : Type
            Cosmology model class (e.g., LCDM, wCDM) with `get_parameters()` classmethod.
        free_params : List[str], optional
            Names of parameters to set as free (sampled). Others will be fixed.
        fixed_params : Dict[str, float], optional
            Explicit values for fixed parameters.

        Returns
        -------
        UnifiedParameterCollector
            Self for method chaining.

        Examples
        --------
        >>> from hicosmo.models import LCDM
        >>> collector.add_from_model(LCDM, free_params=['H0', 'Omega_m'])
        """
        if not hasattr(model_class, "get_parameters"):
            warnings.warn(
                f"Model {model_class.__name__} does not have get_parameters() method. "
                f"Cannot auto-discover parameters."
            )
            return self

        model_params = model_class.get_parameters()
        source = f"model:{model_class.__name__}"
        free_set = set(free_params) if free_params else set()
        fixed_dict = fixed_params or {}

        for param in model_params:
            if not isinstance(param, Parameter):
                continue

            # Determine if this parameter should be free
            is_free = param.name in free_set

            # Get fixed value if specified
            value = fixed_dict.get(param.name, param.value)

            # Create modified parameter
            new_param = Parameter(
                name=param.name,
                value=value,
                free=is_free,
                prior=param.prior if is_free else None,
                latex_label=param.latex_label,
                description=param.description,
                bounds=param.bounds,
            )

            if param.name not in self._params:
                self._params[param.name] = new_param
                self._sources[param.name] = source

        return self

    def add_from_dict(
        self, params_dict: Dict[str, Dict[str, Any]]
    ) -> "UnifiedParameterCollector":
        """
        Add parameters from a user-friendly dictionary format.

        Parameters
        ----------
        params_dict : dict
            Parameter configuration in simplified format:
            {'H0': {'init': 70, 'min': 60, 'max': 80}, ...}

        Returns
        -------
        UnifiedParameterCollector
            Self for method chaining.

        Examples
        --------
        >>> params = {
        ...     'H0': {'init': 70, 'min': 60, 'max': 80},
        ...     'Omega_m': {'init': 0.3, 'min': 0.1, 'max': 0.5}
        ... }
        >>> collector.add_from_dict(params)
        """
        for name, config in params_dict.items():
            param = Parameter.from_simple_config(name, config)
            if name not in self._params:
                self._params[name] = param
                self._sources[name] = "user:dict"

        return self

    def set_free(self, param_names: List[str]) -> "UnifiedParameterCollector":
        """
        Set specified parameters as free (to be sampled).

        Parameters
        ----------
        param_names : List[str]
            Names of parameters to set as free.

        Returns
        -------
        UnifiedParameterCollector
            Self for method chaining.
        """
        for name in param_names:
            if name not in self._params:
                raise KeyError(f"Parameter '{name}' not found in collection")
            param = self._params[name]
            if param.prior is None:
                raise ValueError(
                    f"Cannot set '{name}' as free: no prior defined. "
                    f"Add a prior first using add() method."
                )
            param.free = True

        return self

    def set_fixed(self, param_names: List[str]) -> "UnifiedParameterCollector":
        """
        Set specified parameters as fixed.

        Parameters
        ----------
        param_names : List[str]
            Names of parameters to fix.

        Returns
        -------
        UnifiedParameterCollector
            Self for method chaining.
        """
        for name in param_names:
            if name not in self._params:
                raise KeyError(f"Parameter '{name}' not found in collection")
            self._params[name].free = False

        return self

    def get_free_names(self) -> List[str]:
        """Get names of all free parameters."""
        return [name for name, param in self._params.items() if param.is_free()]

    def get_fixed_names(self) -> List[str]:
        """Get names of all fixed parameters."""
        return [name for name, param in self._params.items() if param.is_fixed()]

    def get_all_names(self) -> List[str]:
        """Get names of all parameters."""
        return list(self._params.keys())

    def get_sources(self) -> Dict[str, str]:
        """Get parameter sources for debugging."""
        return self._sources.copy()

    def to_registry(self) -> ParameterRegistry:
        """
        Convert collected parameters to a ParameterRegistry.

        Returns
        -------
        ParameterRegistry
            Registry ready for MCMC sampling.

        Examples
        --------
        >>> registry = collector.to_registry()
        >>> mcmc = MCMC(registry, log_likelihood, ...)
        """
        registry = ParameterRegistry(name=self.name)
        for name, param in self._params.items():
            registry._params[name] = copy.deepcopy(param)
        return registry

    def summary(self) -> str:
        """
        Generate a summary of collected parameters.

        Returns
        -------
        str
            Human-readable summary.
        """
        lines = [f"UnifiedParameterCollector: {self.name}", "=" * 50]

        free_params = [(n, p) for n, p in self._params.items() if p.is_free()]
        fixed_params = [(n, p) for n, p in self._params.items() if p.is_fixed()]

        if free_params:
            lines.append(f"\nFree Parameters ({len(free_params)}):")
            for name, param in free_params:
                source = self._sources.get(name, "unknown")
                lines.append(f"  - {name}: value={param.value}, source={source}")

        if fixed_params:
            lines.append(f"\nFixed Parameters ({len(fixed_params)}):")
            for name, param in fixed_params:
                source = self._sources.get(name, "unknown")
                lines.append(f"  - {name}: value={param.value}, source={source}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        n_free = len(self.get_free_names())
        n_fixed = len(self.get_fixed_names())
        return (
            f"UnifiedParameterCollector('{self.name}', {n_free} free, {n_fixed} fixed)"
        )

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __len__(self) -> int:
        return len(self._params)
