"""
Unified Parameter Management for HIcosmo
========================================

Single source of truth for all cosmological parameter handling.
Features:
- Bidirectional parameter relations (forward and reverse derivation)
- Flexible parameter basis (CMB-style vs background-style)
- JAX tracer compatible
- Automatic parameter solving

Example usage:
    # Background cosmology style (H0, Omega_m)
    params = CosmologicalParameters(H0=70, Omega_m=0.3, Omega_b=0.05)
    print(params['h'])        # 0.7
    print(params['omega_m'])  # 0.147

    # CMB style (h, omega_b_h2, omega_c_h2)
    params = CosmologicalParameters(h=0.7, omega_b_h2=0.0224, omega_c_h2=0.120)
    print(params['H0'])       # 70.0
    print(params['Omega_m'])  # ~0.29
"""

from typing import Dict, Optional, List, Callable, Any, Union
from dataclasses import dataclass
import warnings
import jax.numpy as jnp
import jax
import jax.errors
from jax import tree_util
from ..parameters.presets import PLANCK_2018_PARAMS, WMAP9_PARAMS


@dataclass
class ParameterSpec:
    """Specification for a cosmological parameter."""
    default: float
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    description: str = ""
    unit: str = ""


@dataclass
class FreeParameter:
    """Lightweight container describing a free parameter."""
    name: str
    value: float


class ParameterRelations:
    """
    Bidirectional parameter relation definitions.

    This class defines how cosmological parameters relate to each other,
    enabling automatic derivation in both directions.
    """

    # ==================== Forward Relations ====================
    # Compute derived parameters from base parameters
    FORWARD: Dict[str, Callable[[Dict], Any]] = {
        # Hubble parameter
        'h': lambda p: p['H0'] / 100.0,

        # Physical densities (omega = Omega * h^2)
        'omega_m': lambda p: p['Omega_m'] * (p['H0'] / 100.0)**2,
        'omega_b': lambda p: p['Omega_b'] * (p['H0'] / 100.0)**2,
        'omega_c': lambda p: (p['Omega_m'] - p['Omega_b']) * (p['H0'] / 100.0)**2,

        # Aliases for CMB conventions
        'omega_b_h2': lambda p: p['Omega_b'] * (p['H0'] / 100.0)**2,
        'omega_c_h2': lambda p: (p['Omega_m'] - p['Omega_b']) * (p['H0'] / 100.0)**2,
        'omega_m_h2': lambda p: p['Omega_m'] * (p['H0'] / 100.0)**2,

        # CDM density
        'Omega_c': lambda p: p['Omega_m'] - p['Omega_b'],

        # NOTE: Omega_Lambda is computed in _compute_additional_derived() AFTER Omega_r
        # to ensure proper closure relation. Do NOT add it here.

        # Structure formation
        'S8': lambda p: p['sigma8'] * jnp.sqrt(p['Omega_m'] / 0.3),
    }

    # ==================== Reverse Relations ====================
    # Compute base parameters from derived parameters
    REVERSE: Dict[str, Callable[[Dict], Any]] = {
        # From h to H0
        'H0': lambda p: p['h'] * 100.0,

        # From physical densities to density parameters
        'Omega_m': lambda p: p.get('omega_m', p.get('omega_m_h2', 0)) / p['h']**2 if 'omega_m' in p or 'omega_m_h2' in p else (p['omega_b_h2'] + p['omega_c_h2']) / p['h']**2,
        'Omega_b': lambda p: p.get('omega_b', p.get('omega_b_h2', 0)) / p['h']**2,
        'Omega_c': lambda p: p.get('omega_c', p.get('omega_c_h2', 0)) / p['h']**2,

        # From S8 to sigma8
        'sigma8': lambda p: p['S8'] / jnp.sqrt(p['Omega_m'] / 0.3),
    }

    # ==================== Dependencies ====================
    # Maps each derived parameter to its required inputs
    FORWARD_DEPS: Dict[str, List[str]] = {
        'h': ['H0'],
        'omega_m': ['Omega_m', 'H0'],
        'omega_b': ['Omega_b', 'H0'],
        'omega_c': ['Omega_m', 'Omega_b', 'H0'],
        'omega_b_h2': ['Omega_b', 'H0'],
        'omega_c_h2': ['Omega_m', 'Omega_b', 'H0'],
        'omega_m_h2': ['Omega_m', 'H0'],
        'Omega_c': ['Omega_m', 'Omega_b'],
        # NOTE: Omega_Lambda computed in _compute_additional_derived(), not here
        'S8': ['sigma8', 'Omega_m'],
    }

    REVERSE_DEPS: Dict[str, List[List[str]]] = {
        # Each entry is a list of alternative dependency sets (OR logic)
        'H0': [['h']],
        'Omega_m': [['omega_m', 'h'], ['omega_m_h2', 'h'], ['omega_b_h2', 'omega_c_h2', 'h']],
        'Omega_b': [['omega_b', 'h'], ['omega_b_h2', 'h']],
        'Omega_c': [['omega_c', 'h'], ['omega_c_h2', 'h']],
        'sigma8': [['S8', 'Omega_m']],
    }


@tree_util.register_pytree_node_class
class CosmologicalParameters:
    """
    Unified cosmological parameter manager with bidirectional derivation.

    This class handles parameter validation, defaults, and automatic
    bidirectional parameter derivation. Users can provide parameters
    in any supported basis (CMB-style or background-style), and the
    system will automatically compute all related parameters.

    Examples
    --------
    >>> # Background cosmology style
    >>> params = CosmologicalParameters(H0=70, Omega_m=0.3, Omega_b=0.05)
    >>> print(params['h'])       # 0.7
    >>> print(params['omega_m']) # 0.147

    >>> # CMB style (Planck-like)
    >>> params = CosmologicalParameters(h=0.7, omega_b_h2=0.0224, omega_c_h2=0.120)
    >>> print(params['H0'])      # 70.0
    >>> print(params['Omega_m']) # ~0.29
    """

    # Parameter specifications for validation
    _PLANCK = PLANCK_2018_PARAMS

    PARAM_SPECS = {
        # Primary parameters
        'H0': ParameterSpec(_PLANCK['H0'], 20.0, 200.0, "Hubble constant", "km/s/Mpc"),
        'h': ParameterSpec(_PLANCK['H0'] / 100.0, 0.2, 2.0, "Reduced Hubble constant", ""),
        'Omega_m': ParameterSpec(_PLANCK['Omega_m'], 0.01, 1.0, "Total matter density parameter", ""),
        'Omega_b': ParameterSpec(_PLANCK['Omega_b'], 0.005, 0.1, "Baryon density parameter", ""),
        'Omega_c': ParameterSpec(_PLANCK['Omega_m'] - _PLANCK['Omega_b'], 0.01, 1.0, "CDM density parameter", ""),
        'Omega_k': ParameterSpec(0.0, -0.5, 0.5, "Curvature density parameter", ""),
        'sigma8': ParameterSpec(_PLANCK['sigma8'], 0.1, 2.0, "Matter fluctuation amplitude", ""),
        'n_s': ParameterSpec(_PLANCK['n_s'], 0.8, 1.2, "Scalar spectral index", ""),

        # Physical densities (CMB conventions)
        'omega_b_h2': ParameterSpec(0.0224, 0.01, 0.04, "Physical baryon density", ""),
        'omega_c_h2': ParameterSpec(0.120, 0.05, 0.2, "Physical CDM density", ""),
        'omega_m_h2': ParameterSpec(0.142, 0.05, 0.25, "Physical matter density", ""),

        # CMB and recombination
        'T_cmb': ParameterSpec(_PLANCK['T_cmb'], 2.0, 3.0, "CMB temperature", "K"),
        'N_eff': ParameterSpec(3.046, 1.0, 10.0, "Effective neutrino species", ""),
        'Y_p': ParameterSpec(0.2453, 0.2, 0.3, "Primordial helium fraction", ""),

        # Dark energy
        'w': ParameterSpec(-1.0, -3.0, 0.0, "Dark energy EoS (constant)", ""),
        'w0': ParameterSpec(-1.0, -3.0, 0.0, "Dark energy EoS today", ""),
        'wa': ParameterSpec(0.0, -3.0, 3.0, "Dark energy EoS evolution", ""),

        # Neutrinos
        'mnu': ParameterSpec(0.06, 0.0, 2.0, "Sum of neutrino masses", "eV"),

        # Modified gravity
        'mu_0': ParameterSpec(0.0, -1.0, 1.0, "Modified gravity parameter", ""),
        'sigma_0': ParameterSpec(0.0, -1.0, 1.0, "Modified gravity parameter", ""),
        'beta': ParameterSpec(0.0, -1.0, 1.0, "Interaction coupling parameter", ""),
    }

    def __init__(self, **params):
        """
        Initialize parameters with automatic bidirectional derivation.

        Parameters
        ----------
        **params : dict
            Cosmological parameters to set. Can use any supported
            parameter basis (H0/Omega_m or h/omega_b_h2/omega_c_h2).
        """
        self._given: Dict[str, Any] = {}       # User-provided parameters
        self._computed: Dict[str, Any] = {}    # System-computed parameters
        self._sampled: List[str] = []          # Parameters marked for MCMC sampling

        # Store user-provided values
        for name, value in params.items():
            self._validate_and_store(name, value)

        # Solve for all derivable parameters
        self._solve_parameters()

    # -------------------- PyTree support (JAX-friendly) --------------------

    def __tree_flatten__(self):
        # Flatten user-provided parameters only; derived params are recomputed.
        items = tuple(sorted(self._given.items()))
        keys = tuple(k for k, _ in items)
        values = tuple(v for _, v in items)
        return values, keys

    @classmethod
    def __tree_unflatten__(cls, aux_data, children):
        keys = aux_data
        params = {k: v for k, v in zip(keys, children)}
        return cls(**params)

    # JAX expects tree_flatten/tree_unflatten (no leading underscores)
    def tree_flatten(self):
        return self.__tree_flatten__()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.__tree_unflatten__(aux_data, children)

    def _validate_and_store(self, name: str, value: Any) -> None:
        """Validate and store a user-provided parameter."""
        # Range validation only for concrete values
        if name in self.PARAM_SPECS:
            spec = self.PARAM_SPECS[name]
            try:
                concrete_value = float(value)
                if spec.min_val is not None and concrete_value < spec.min_val:
                    raise ValueError(f"{name} = {concrete_value} below minimum {spec.min_val}")
                if spec.max_val is not None and concrete_value > spec.max_val:
                    raise ValueError(f"{name} = {concrete_value} above maximum {spec.max_val}")
            except (TypeError, jax.errors.TracerIntegerConversionError, jax.errors.ConcretizationTypeError):
                pass  # JAX tracer, skip validation

        self._given[name] = value

    def _solve_parameters(self) -> None:
        """
        Iteratively solve for all derivable parameters.

        Uses both forward and reverse relations to compute parameters
        from whatever basis the user provided.
        """
        all_params = dict(self._given)

        # Iterative solving (handles dependency chains)
        max_iterations = 15
        for iteration in range(max_iterations):
            changed = False

            # Try forward derivations
            for name, func in ParameterRelations.FORWARD.items():
                if name not in all_params:
                    deps = ParameterRelations.FORWARD_DEPS.get(name, [])
                    # Check if all required deps are available
                    if all(d in all_params for d in deps if d not in ['Omega_k', 'Omega_r']):
                        try:
                            all_params[name] = func(all_params)
                            changed = True
                        except KeyError:
                            pass

            # Try reverse derivations
            for name, func in ParameterRelations.REVERSE.items():
                if name not in all_params:
                    dep_sets = ParameterRelations.REVERSE_DEPS.get(name, [])
                    # Try each alternative dependency set
                    for deps in dep_sets:
                        if all(d in all_params for d in deps):
                            try:
                                all_params[name] = func(all_params)
                                changed = True
                                break
                            except KeyError:
                                pass

            if not changed:
                break

        # Compute additional derived quantities
        self._compute_additional_derived(all_params)

        # Separate given from computed
        self._computed = {k: v for k, v in all_params.items() if k not in self._given}

    def _compute_additional_derived(self, params: Dict) -> None:
        """Compute additional derived parameters that aren't part of the relation graph."""
        # Ensure h exists
        if 'h' not in params and 'H0' in params:
            params['h'] = params['H0'] / 100.0
        elif 'H0' not in params and 'h' in params:
            params['H0'] = params['h'] * 100.0

        h = params.get('h', 0.6736)
        h2 = h**2

        # Radiation density from CMB temperature
        T_cmb = params.get('T_cmb', 2.7255)
        N_eff = params.get('N_eff', 3.046)
        Omega_gamma_h2 = 2.47e-5 * (T_cmb / 2.7255)**4
        params['Omega_gamma'] = Omega_gamma_h2 / h2
        params['Omega_r'] = params['Omega_gamma'] * (1 + 0.2271 * N_eff)
        params['omega_r'] = params['Omega_r'] * h2

        # Dark energy from closure (if not already set)
        if 'Omega_Lambda' not in params and 'Omega_m' in params:
            Omega_k = params.get('Omega_k', 0.0)
            params['Omega_Lambda'] = 1.0 - params['Omega_m'] - Omega_k - params['Omega_r']

        # Hubble distance and time
        if 'H0' in params:
            from ..utils.constants import c_km_s
            params['D_H'] = c_km_s / params['H0']  # Mpc
            params['t_H'] = 9.777952 / h  # Gyr

        # Age approximation
        if 'Omega_Lambda' in params:
            OL = params['Omega_Lambda']
            t_H = params.get('t_H', 9.777952 / h)
            params['age'] = jnp.where(OL > 0, (2.0/3.0) * t_H / jnp.sqrt(OL), (2.0/3.0) * t_H)

        # Critical density
        params['rho_crit'] = 2.77536627e11 * h2  # M_sun/Mpc^3

        # Ensure omega aliases exist
        if 'Omega_b' in params:
            params['omega_b'] = params['Omega_b'] * h2
            params['Omega_b_h2'] = params['omega_b']  # Alias
        if 'Omega_c' not in params and 'Omega_m' in params and 'Omega_b' in params:
            params['Omega_c'] = params['Omega_m'] - params['Omega_b']
        if 'Omega_c' in params:
            params['omega_c'] = params['Omega_c'] * h2
            params['Omega_c_h2'] = params['omega_c']  # Alias
        if 'Omega_m' in params:
            params['omega_m'] = params['Omega_m'] * h2
            params['Omega_m_h2'] = params['omega_m']  # Alias

    def set_sampled(self, param_names: List[str]) -> None:
        """
        Declare which parameters are sampled (free parameters for MCMC).

        Parameters
        ----------
        param_names : List[str]
            Names of parameters to sample.
        """
        self._sampled = list(param_names)

    def get_sampled_names(self) -> List[str]:
        """Return the list of sampled parameter names."""
        return self._sampled

    def update(self, **params) -> None:
        """
        Update parameters and re-solve.

        Parameters
        ----------
        **params : dict
            Parameters to update.
        """
        for name, value in params.items():
            self._validate_and_store(name, value)
        self._solve_parameters()

    # ==================== Backward Compatibility ====================

    @property
    def _params(self) -> Dict[str, Any]:
        """Backward compatible: return all parameters as if they were primary."""
        return {**self._given, **self._computed}

    @property
    def _derived(self) -> Dict[str, Any]:
        """Backward compatible: return computed parameters."""
        return self._computed

    def get_parameter(self, name: str) -> Any:
        """Get parameter value."""
        if name in self._given:
            return self._given[name]
        if name in self._computed:
            return self._computed[name]
        raise KeyError(f"Parameter '{name}' not found")

    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all parameters (given + computed)."""
        return {**self._given, **self._computed}

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter and re-solve."""
        self._validate_and_store(name, value)
        self._solve_parameters()

    def validate_closure(self, tolerance: float = 1e-4) -> None:
        """Validate closure relation."""
        all_params = self.get_all_parameters()
        total = (
            all_params.get('Omega_m', 0) +
            all_params.get('Omega_k', 0) +
            all_params.get('Omega_r', 0) +
            all_params.get('Omega_Lambda', 0)
        )
        try:
            concrete_total = float(total)
            if abs(concrete_total - 1.0) > tolerance:
                raise ValueError(f"Closure relation violated: Ω_total = {concrete_total:.6f} ≠ 1")
        except (TypeError, jax.errors.TracerIntegerConversionError, jax.errors.ConcretizationTypeError):
            pass  # JAX tracer

    def validate_physics(self) -> None:
        """Check for unphysical parameter combinations."""
        try:
            OL = float(self.get('Omega_Lambda', 0))
            if OL < -1e-6:
                raise ValueError(f"Omega_Lambda = {OL:.6f} < 0 unphysical")
            Oc = float(self.get('Omega_c', 0))
            if Oc < -1e-6:
                raise ValueError(f"Omega_c = {Oc:.6f} < 0 unphysical")
            Ob = float(self.get('Omega_b', 0))
            Om = float(self.get('Omega_m', 0))
            if Ob >= Om and Om > 0:
                raise ValueError("Omega_b >= Omega_m unphysical")
        except (TypeError, jax.errors.TracerIntegerConversionError, jax.errors.ConcretizationTypeError):
            pass

    def set_free_parameters(self, parameter_names: List[str]) -> None:
        """Alias for set_sampled (backward compatibility)."""
        self.set_sampled(parameter_names)

    def get_free_parameters(self) -> List[FreeParameter]:
        """Return the list of currently free parameters."""
        return [FreeParameter(name, self[name]) for name in self._sampled]

    def update_and_compute_derived(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Return parameter dictionary after applying updates."""
        temp = CosmologicalParameters(**{**self._given, **updates})
        return temp.get_all_parameters()

    def summary(self) -> str:
        """Generate parameter summary."""
        lines = ["Cosmological Parameters Summary", "=" * 40]
        all_params = self.get_all_parameters()

        lines.append("\nGiven Parameters:")
        for name, value in sorted(self._given.items()):
            try:
                lines.append(f"  {name:<15} = {float(value):10.6f}")
            except (TypeError, ValueError):
                lines.append(f"  {name:<15} = {value}")

        lines.append("\nDerived Parameters:")
        derived_show = ['h', 'H0', 'Omega_m', 'Omega_b', 'Omega_c', 'Omega_Lambda',
                       'omega_m', 'omega_b', 'omega_c', 'Omega_r']
        for name in derived_show:
            if name in self._computed:
                try:
                    lines.append(f"  {name:<15} = {float(self._computed[name]):10.6f}")
                except (TypeError, ValueError):
                    lines.append(f"  {name:<15} = {self._computed[name]}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.get_all_parameters()

    def copy(self) -> 'CosmologicalParameters':
        """Create a copy of this parameter set."""
        return CosmologicalParameters(**self._given)

    @classmethod
    def from_registry(cls, registry: Any) -> 'CosmologicalParameters':
        """
        Create CosmologicalParameters from a ParameterRegistry.

        Parameters
        ----------
        registry : ParameterRegistry
            Registry with Parameter objects (free/fixed).
        """
        params = {}
        for name, param in registry.get_all().items():
            if param.value is not None:
                params[name] = param.value
        return cls(**params)

    # ==================== Dictionary-like Interface ====================

    def __getitem__(self, key: str) -> Any:
        return self.get_parameter(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set_parameter(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.get_parameter(key)
        except KeyError:
            return default

    def keys(self):
        return self.get_all_parameters().keys()

    def values(self):
        return self.get_all_parameters().values()

    def items(self):
        return self.get_all_parameters().items()

    def __contains__(self, key: str) -> bool:
        return key in self._given or key in self._computed

    def __repr__(self) -> str:
        given = ", ".join(f"{k}={v}" for k, v in self._given.items())
        return f"CosmologicalParameters({given})"


# ==================== Default Parameter Sets ====================

PLANCK_2018 = CosmologicalParameters(**PLANCK_2018_PARAMS)

PLANCK_2015 = CosmologicalParameters(
    H0=67.74, Omega_m=0.3089, Omega_b=0.0486, Omega_k=0.0,
    sigma8=0.8159, n_s=0.9667, T_cmb=2.7255
)

WMAP9 = CosmologicalParameters(**WMAP9_PARAMS)

# CMB-style preset (using physical densities)
PLANCK_2018_CMB = CosmologicalParameters(
    h=0.6736, omega_b_h2=0.02237, omega_c_h2=0.1200,
    sigma8=0.8111, n_s=0.9649, T_cmb=2.7255
)
