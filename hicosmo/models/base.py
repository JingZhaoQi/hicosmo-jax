"""
HIcosmo Base Classes - Simplified and Clean
===========================================

Minimal base classes for all cosmological models in HIcosmo.
This replaces the bloated cosmology.py with a clean, focused interface.

Key principles:
- Single responsibility
- No implementation details in base class
- Clear interface contracts
- No redundant methods
"""

from abc import ABC, abstractmethod
from typing import Dict, Union, Any
import jax.numpy as jnp

from ..utils.jax_tools import cumulative_trapezoid, gradient_1d
from ..utils.constants import c_km_s


def compute_omega_r(params: Dict) -> float:
    """
    Compute radiation density parameter Omega_r from physical parameters.

    This is a shared utility for all cosmological models. Omega_r MUST always
    be computed from T_cmb, H0, N_eff - it should NEVER default to 0.

    At z_star ≈ 1090, radiation contributes ~24% of E²(z). Setting Omega_r = 0
    causes catastrophic errors in CMB distance calculations.

    Parameters
    ----------
    params : Dict
        Must contain 'H0'. Optional: 'T_cmb' (default 2.7255 K),
        'N_eff' (default 3.046).

    Returns
    -------
    float
        Omega_r = Omega_gamma * (1 + 0.22711 * N_eff)

    Notes
    -----
    Formula: Omega_gamma * h^2 = 2.4728e-5 * (T_cmb / 2.7255)^4
    (photon density from sigma_B T^4; value as in Planck 2018 conventions).
    The previous 3-digit constant 2.47e-5 underestimated the radiation
    density by 0.10%, which propagated into a 0.09% bias of D_M(z_*) —
    about 6 sigma of the Planck distance-prior l_a covariance.
    The factor 0.22711 = (7/8)(4/11)^{4/3} accounts for relativistic
    neutrinos.
    """
    if "Omega_r" in params:
        return params["Omega_r"]

    from ..utils.constants import NEUTRINO_RADIATION_FACTOR, Omega_gamma_h2

    H0 = params["H0"]
    T_cmb = params.get("T_cmb", 2.7255)
    N_eff = params.get("N_eff", 3.046)
    h = H0 / 100.0
    theta = T_cmb / 2.7255
    omega_gamma_h2 = Omega_gamma_h2 * theta**4
    return (omega_gamma_h2 / h**2) * (1.0 + NEUTRINO_RADIATION_FACTOR * N_eff)


class CosmologyBase(ABC):
    """
    Minimal abstract base class for all cosmological models.

    Defines the essential interface that all models must implement,
    without any concrete implementations to avoid code duplication.
    """

    def __init__(self, **params):
        """
        Initialize cosmological model with parameters.

        Note: Subclasses should create self.cosmology_params (CosmologicalParameters)
        before calling super().__init__() to enable unified parameter management.
        """
        # If subclass hasn't created cosmology_params, fall back to simple dict
        if not hasattr(self, "cosmology_params"):
            from .unified_parameters import CosmologicalParameters

            self.cosmology_params = CosmologicalParameters(**params)

        # Make params a reference to cosmology_params for backward compatibility
        # This allows Fisher code to use .params['H0'] and .params.get('sigma8', default)
        self.params = self.cosmology_params

        self._warn_if_traced_methods_inherit_other_physics()

        self._validate_params()

    def _warn_if_traced_methods_inherit_other_physics(self) -> None:
        """Warn when a subclass defines its own E(z) but was never registered.

        Without @register_cosmology_model (or manual wiring), the inherited
        compute_grid_traced closure is still bound to the PARENT's E_z kernel:
        plotting via instance methods would use the new physics while MCMC
        silently samples the parent model.
        """
        cls = type(self)
        defines_physics = "E_z" in cls.__dict__ or "_E_z_kernel" in cls.__dict__
        has_own_traced = "compute_grid_traced" in cls.__dict__
        if defines_physics and not has_own_traced:
            import warnings

            warnings.warn(
                f"{cls.__name__} defines its own E_z but is not decorated with "
                "@register_cosmology_model: compute_grid_traced and the other "
                "traced methods are inherited and still evaluate the PARENT "
                "model's physics. MCMC sampling would silently use the wrong "
                "E(z). Add the decorator to wire this class's physics.",
                UserWarning,
                stacklevel=3,
            )

    @abstractmethod
    def _validate_params(self) -> None:
        """Validate input parameters for physical consistency."""
        pass

    # ==================== Core Interface ====================

    @abstractmethod
    def E_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Dimensionless Hubble parameter E(z) = H(z)/H0.

        This is the fundamental quantity that defines the cosmological model.
        All other quantities are derived from this.
        """
        pass

    @abstractmethod
    def comoving_distance(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """
        Comoving distance in Mpc.

        Must be implemented with high performance integration.
        """
        pass

    @abstractmethod
    def angular_diameter_distance(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """Angular diameter distance in Mpc."""
        pass

    @abstractmethod
    def luminosity_distance(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """Luminosity distance in Mpc."""
        pass

    @abstractmethod
    def distance_modulus(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """Distance modulus in magnitudes."""
        pass

    # ==================== Optional Interface ====================

    def w_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Dark energy equation of state w(z).

        Default implementation for ΛCDM (w = -1).
        Can be overridden by extensions.
        """
        return jnp.full_like(jnp.asarray(z), -1.0)

    # ==================== Derived Parameters Interface ====================

    def derived_parameters(self) -> Dict[str, float]:
        """
        Compute derived parameters from this cosmological model.

        Derived parameters are quantities computed from the sampled parameters,
        such as the sound horizon r_d, age of the universe, etc.

        Returns
        -------
        Dict[str, float]
            Dictionary of derived parameter names and values.
            Default implementation returns empty dict.

        Notes
        -----
        Subclasses should override this to provide model-specific derived
        parameters. For example, LCDM provides r_d, r_d*h, H0*r_d.

        This method is called by post-processing tools (not by the sampler)
        to compute derived quantities from MCMC samples.

        Examples
        --------
        >>> model = LCDM(H0=70, Omega_m=0.3, Omega_b=0.05)
        >>> derived = model.derived_parameters()
        >>> print(derived['rd_h'])  # r_d * h in Mpc
        """
        return {}

    @classmethod
    def derived_param_labels(cls) -> Dict[str, str]:
        """
        Return LaTeX labels for derived parameters.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping parameter names to LaTeX labels.
            Default implementation returns empty dict.

        Notes
        -----
        Subclasses should override this to provide labels for their
        derived parameters. Labels should be valid LaTeX strings
        without surrounding $ signs.

        Examples
        --------
        >>> labels = LCDM.derived_param_labels()
        >>> print(labels['rd_h'])  # 'r_d h\\,[\\mathrm{Mpc}]'
        """
        return {}

    def get_parameters(self) -> Dict[str, float]:
        """Get copy of model parameters."""
        return self.params.copy()

    def update_parameters(self, new_params: Dict[str, float]) -> None:
        """
        Update model parameters.

        Subclasses should override this to handle parameter updates properly.
        """
        self.params.update(new_params)
        self._validate_params()

    @staticmethod
    def normalize_params(
        params: Dict[str, Any], *, dtype: Any = None
    ) -> Dict[str, Any]:
        """
        Normalize common parameter bases (h, omega_b_h2, omega_m_h2) into H0/Omega_*.

        This keeps all cosmology-related parameter conversions inside the model layer.
        """
        params = dict(params)

        def _to_array(value):
            if isinstance(value, (int, float, jnp.ndarray)):
                return (
                    jnp.asarray(value, dtype=dtype)
                    if dtype is not None
                    else jnp.asarray(value)
                )
            return value

        if "H0" not in params and "h" in params:
            params["H0"] = _to_array(params["h"]) * 100.0

        if "Omega_m" not in params and "H0" in params:
            if "omega_m_h2" in params:
                h = _to_array(params["H0"]) / 100.0
                params["Omega_m"] = _to_array(params["omega_m_h2"]) / (h**2)
            elif "Omega_m_h2" in params:
                h = _to_array(params["H0"]) / 100.0
                params["Omega_m"] = _to_array(params["Omega_m_h2"]) / (h**2)

        if "Omega_b" not in params and "H0" in params:
            if "omega_b_h2" in params:
                h = _to_array(params["H0"]) / 100.0
                params["Omega_b"] = _to_array(params["omega_b_h2"]) / (h**2)
            elif "Omega_b_h2" in params:
                h = _to_array(params["H0"]) / 100.0
                params["Omega_b"] = _to_array(params["Omega_b_h2"]) / (h**2)

        # Ensure numeric values are JAX arrays with requested dtype
        normalized = {}
        for key, value in params.items():
            normalized[key] = _to_array(value)
        return normalized

    # ==================== Traced-Aware Interface (for NumPyro MCMC) ====================

    # Note: Not marked as @abstractmethod because subclasses use factory pattern
    # to generate this method after class definition. See make_compute_grid_traced().
    @staticmethod
    def compute_grid_traced(
        z_grid: jnp.ndarray, params: Dict[str, float]
    ) -> Dict[str, jnp.ndarray]:
        """
        **NumPyro traced-aware** batch cosmology computation.

        This is a pure functional interface designed for NumPyro MCMC sampling,
        where parameters may be traced values (JAX tracer objects) during
        automatic differentiation.

        Design Requirements (JAX Pure Function Standard):
        --------------------------------------------------
        1. Pure function: All inputs via parameters, no self dependency
        2. Fully JAX-ified: Only use jnp.* operations for JIT compilation
        3. Traced-aware: Accept traced parameters (H0, Omega_m, etc.)
        4. No side effects: Cannot modify any global state
        5. Static shapes: All array shapes must be compile-time determinable

        Parameters
        ----------
        z_grid : jnp.ndarray, shape (N,)
            Redshift grid for cosmology computation.
            Must be uniformly spaced 1D array.
        params : Dict[str, float]
            Model parameters dictionary. Keys depend on the model:
            - LCDM:  {'H0': float, 'Omega_m': float, 'Omega_k': float (optional)}
            - wCDM:  {'H0': float, 'Omega_m': float, 'w0': float}
            - CPL:   {'H0': float, 'Omega_m': float, 'w0': float, 'wa': float}

            Parameters can be traced values (JAX tracer objects during MCMC).

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing cosmology quantities, all shape (N,)::

                {
                    'd_L': Luminosity distance [Mpc],
                    'dVc_dz': Differential comoving volume [Mpc^3],
                    'ddL_dz': Derivative of d_L w.r.t. z [Mpc],
                    'E_z': Hubble parameter E(z) = H(z)/H0,
                    'd_C': Comoving distance [Mpc] (optional)
                }

        Notes
        -----
        - This method MUST be JIT-compilable (no Python control flow)
        - Numerical integration should use JAX native functions (jnp.cumsum, etc.)
        - All parameters may be traced values - do NOT convert to NumPy
        - This is called inside NumPyro model, potentially thousands of times

        Examples
        --------
        >>> # Usage in NumPyro MCMC likelihood
        >>> def log_likelihood(H0, Omega_m, pop_params):
        ...     z_grid = jnp.linspace(0.001, 5.0, 5000)
        ...     # H0, Omega_m are traced values from NumPyro
        ...     cosmo_grid = LCDM.compute_grid_traced(
        ...         z_grid, {'H0': H0, 'Omega_m': Omega_m}
        ...     )
        ...     # cosmo_grid['d_L'] is also a traced array
        ...     return compute_gw_likelihood(cosmo_grid, pop_params)

        See Also
        --------
        LCDM.compute_grid_traced : Reference implementation
        """
        raise NotImplementedError(
            "Subclasses must implement compute_grid_traced using "
            "make_compute_grid_traced() factory function"
        )

    @staticmethod
    def compute_background_grid_traced(
        z_grid: jnp.ndarray, params: Dict[str, float]
    ) -> Dict[str, jnp.ndarray]:
        """Return the lightweight distance grid needed by SN/BAO-like probes."""
        raise NotImplementedError(
            "Subclasses must implement compute_background_grid_traced using "
            "make_compute_background_grid_traced() factory function"
        )

    @staticmethod
    def sound_horizon_traced(z: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
        """Return sound horizon at redshift z from traced params."""
        raise NotImplementedError(
            "Subclasses must implement sound_horizon_traced using "
            "make_sound_horizon_traced() factory function"
        )

    @staticmethod
    def sound_horizon_drag_traced(params: Dict[str, float]) -> jnp.ndarray:
        """Return sound horizon at drag epoch from traced params."""
        raise NotImplementedError(
            "Subclasses must implement sound_horizon_drag_traced using "
            "make_sound_horizon_drag_traced() factory function"
        )

    @staticmethod
    def recombination_redshift_traced(params: Dict[str, float]) -> jnp.ndarray:
        """Return recombination redshift z_* from traced params."""
        raise NotImplementedError(
            "Subclasses must implement recombination_redshift_traced using "
            "make_recombination_redshift_traced() factory function"
        )

    def recombination_redshift(self) -> float:
        """Return recombination redshift z_* using model parameters."""
        params = (
            self.params.to_dict()
            if hasattr(self.params, "to_dict")
            else dict(self.params)
        )
        return float(self.__class__.recombination_redshift_traced(params))


# ==================== Shared Distance Computation (for traced-aware interface) ====================

from jax import jit, lax
from typing import Callable

# Use unified constant from utils.constants
_C_KM_S = c_km_s


def make_compute_grid_traced(E_z_func: Callable) -> Callable:
    """
    Factory function to create model-specific compute_grid_traced.

    This factory pattern allows each cosmological model to define only its
    E(z) computation while sharing the distance calculation code.

    The closure captures E_z_func at module load time, making the returned
    function fully JIT-compilable with no performance penalty.

    Parameters
    ----------
    E_z_func : Callable
        Static function (z_grid, params) -> E_z_grid that computes
        the dimensionless Hubble parameter E(z) = H(z)/H0.

    Returns
    -------
    Callable
        JIT-compiled compute_grid_traced function with signature:
        (z_grid: jnp.ndarray, params: Dict) -> Dict[str, jnp.ndarray]

    Examples
    --------
    >>> # In model definition
    >>> @staticmethod
    >>> @jit
    >>> def _E_z_static(z_grid, params):
    ...     Omega_m = params['Omega_m']
    ...     Omega_Lambda = 1.0 - Omega_m
    ...     return jnp.sqrt(Omega_m * (1+z_grid)**3 + Omega_Lambda)
    >>>
    >>> # After class definition
    >>> LCDM.compute_grid_traced = staticmethod(
    ...     make_compute_grid_traced(LCDM._E_z_static)
    ... )
    """

    @jit
    def compute_grid_traced(
        z_grid: jnp.ndarray, params: Dict[str, float]
    ) -> Dict[str, jnp.ndarray]:
        """
        NumPyro traced-aware batch cosmology computation.

        Generated by make_compute_grid_traced factory function.
        See model-specific docstrings for parameter details.
        """
        # Call model-specific E(z) function (captured via closure)
        E_z_grid = E_z_func(z_grid, params)

        # Extract cosmological parameters for distance calculations
        H0 = params["H0"]
        Omega_k = params.get("Omega_k", 0.0)

        # Use shared distance computation (now with curvature support)
        distances = compute_distances_from_E_z(z_grid, E_z_grid, H0, Omega_k)

        return {
            "d_L": distances["d_L"],
            "D_M": distances["D_M"],  # Transverse comoving distance (for BAO)
            "D_H": distances["D_H"],  # Hubble distance c/H(z) (for BAO)
            "dVc_dz": distances["dVc_dz"],
            "ddL_dz": distances["ddL_dz"],
            "E_z": E_z_grid,
            "d_C": distances["d_C"],
        }

    return compute_grid_traced


def make_compute_background_grid_traced(E_z_func: Callable) -> Callable:
    """Create a lightweight traced background grid for distance-only likelihoods."""

    @jit
    def compute_background_grid_traced(
        z_grid: jnp.ndarray, params: Dict[str, float]
    ) -> Dict[str, jnp.ndarray]:
        E_z_grid = E_z_func(z_grid, params)
        H0 = params["H0"]
        Omega_k = params.get("Omega_k", 0.0)
        return compute_distances_core(z_grid, E_z_grid, H0, Omega_k)

    return compute_background_grid_traced


def compute_background_grid_for_model(
    cosmology_class: Any, z_grid: jnp.ndarray, params: Dict[str, float]
) -> Dict[str, jnp.ndarray]:
    """Compute the lightest traced distance grid a cosmology class provides."""
    compute_grid = getattr(cosmology_class, "compute_background_grid_traced", None)
    if compute_grid is None:
        compute_grid = cosmology_class.compute_grid_traced
    return compute_grid(z_grid, params)


def make_sound_horizon_traced() -> Callable:
    """
    Factory for traced sound horizon r_s(z) using EH98 fitting formula.

    Note: Uses Eisenstein & Hu 1998 analytical approximation,
    which only depends on (H0, Omega_m, Omega_b, T_cmb).
    """
    from ..utils.jax_tools import sound_horizon_eh98

    @jit
    def sound_horizon_traced(
        z: jnp.ndarray,
        params: Dict[str, float],
    ) -> jnp.ndarray:
        return sound_horizon_eh98(
            params["H0"],
            params["Omega_m"],
            params["Omega_b"],
            params.get("T_cmb", 2.7255),
            z,
        )

    return sound_horizon_traced


def make_sound_horizon_drag_traced() -> Callable:
    """
    Factory for traced sound horizon at drag epoch r_d using EH98 formula.
    """
    from ..utils.jax_tools import sound_horizon_drag_eh98

    @jit
    def sound_horizon_drag_traced(params: Dict[str, float]) -> jnp.ndarray:
        T_cmb = params.get("T_cmb", 2.7255)
        return sound_horizon_drag_eh98(
            params["H0"],
            params["Omega_m"],
            params["Omega_b"],
            T_cmb,
        )

    return sound_horizon_drag_traced


def make_recombination_redshift_traced() -> Callable:
    """
    Factory for recombination redshift z_* (Hu & Sugiyama 1996 fitting formula).
    """

    @jit
    def recombination_redshift_traced(params: Dict[str, float]) -> jnp.ndarray:
        H0 = params["H0"]
        Omega_m = params["Omega_m"]
        Omega_b = params["Omega_b"]

        h = H0 / 100.0
        omega_b_h2 = Omega_b * h**2
        omega_m_h2 = Omega_m * h**2

        g1 = 0.0783 * omega_b_h2 ** (-0.238) / (1.0 + 39.5 * omega_b_h2**0.763)
        g2 = 0.560 / (1.0 + 21.1 * omega_b_h2**1.81)
        z_star = (
            1048.0
            * (1.0 + 0.00124 * omega_b_h2 ** (-0.738))
            * (1.0 + g1 * omega_m_h2**g2)
        )
        return z_star

    return recombination_redshift_traced


@jit
def compute_distances_from_E_z(
    z_grid: jnp.ndarray, E_z_grid: jnp.ndarray, H0: float, Omega_k: float = 0.0
) -> Dict[str, jnp.ndarray]:
    """
    Compute cosmological distances from E(z) grid.

    This is the shared implementation for all cosmological models.
    Each model only needs to compute E(z), then call this function
    for the rest of the distance calculations.

    Parameters
    ----------
    z_grid : jnp.ndarray, shape (N,)
        Redshift grid (uniformly spaced).
    E_z_grid : jnp.ndarray, shape (N,)
        Hubble parameter E(z) = H(z)/H0 on the grid.
    H0 : float
        Hubble constant in km/s/Mpc.
    Omega_k : float, optional
        Curvature density parameter. Default is 0.0 (flat universe).

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary containing distance quantities.
    """
    one_plus_z = 1.0 + z_grid
    D_H_0 = _C_KM_S / H0

    integrand = 1.0 / E_z_grid
    integral_values = cumulative_trapezoid(integrand, z_grid)
    d_C_grid = D_H_0 * integral_values

    def flat_DM(_):
        return d_C_grid

    def curved_DM(_):
        delta = d_C_grid / D_H_0

        def open_DM(_):
            sqrt_ok = jnp.sqrt(Omega_k)
            return D_H_0 / sqrt_ok * jnp.sinh(sqrt_ok * delta)

        def closed_DM(_):
            sqrt_abs_ok = jnp.sqrt(-Omega_k)
            return D_H_0 / sqrt_abs_ok * jnp.sin(sqrt_abs_ok * delta)

        return lax.cond(Omega_k > 0, open_DM, closed_DM, operand=None)

    D_M_grid = lax.cond(jnp.abs(Omega_k) < 1e-10, flat_DM, curved_DM, operand=None)
    D_H_grid = D_H_0 / E_z_grid
    d_L_grid = one_plus_z * D_M_grid
    dVc_dz_grid = 4.0 * jnp.pi * D_H_0 * D_M_grid**2 / E_z_grid
    ddL_dz_grid = gradient_1d(d_L_grid, z_grid)

    return {
        "d_C": d_C_grid,
        "D_M": D_M_grid,
        "d_L": d_L_grid,
        "D_H": D_H_grid,
        "dVc_dz": dVc_dz_grid,
        "ddL_dz": ddL_dz_grid,
    }


@jit
def compute_distances_core(
    z_grid: jnp.ndarray, E_z_grid: jnp.ndarray, H0: float, Omega_k: float = 0.0
) -> Dict[str, jnp.ndarray]:
    """
    Lightweight distance computation — only d_C, D_M, D_H, d_L, E_z.

    Skips dVc_dz and ddL_dz (not needed by SN, BAO, or CMB likelihoods).
    """
    D_H_0 = _C_KM_S / H0

    integrand = 1.0 / E_z_grid
    integral_values = cumulative_trapezoid(integrand, z_grid)
    d_C_grid = D_H_0 * integral_values

    def flat_DM(_):
        return d_C_grid

    def curved_DM(_):
        delta = d_C_grid / D_H_0

        def open_DM(_):
            sqrt_ok = jnp.sqrt(Omega_k)
            return D_H_0 / sqrt_ok * jnp.sinh(sqrt_ok * delta)

        def closed_DM(_):
            sqrt_abs_ok = jnp.sqrt(-Omega_k)
            return D_H_0 / sqrt_abs_ok * jnp.sin(sqrt_abs_ok * delta)

        return lax.cond(Omega_k > 0, open_DM, closed_DM, operand=None)

    D_M_grid = lax.cond(jnp.abs(Omega_k) < 1e-10, flat_DM, curved_DM, operand=None)
    D_H_grid = D_H_0 / E_z_grid
    d_L_grid = (1.0 + z_grid) * D_M_grid

    return {
        "d_C": d_C_grid,
        "D_M": D_M_grid,
        "d_L": d_L_grid,
        "D_H": D_H_grid,
        "E_z": E_z_grid,
    }


# Type aliases for clarity
Redshift = Union[float, jnp.ndarray]
Distance = Union[float, jnp.ndarray]
Parameters = Dict[str, float]


# ==================== Model Registration Decorator ====================


def register_cosmology_model(cls):
    """
    Decorator to auto-generate traced methods for cosmological models.

    This decorator enables minimal model definitions. Users define a static
    `E_z(z, params)` method, and this decorator automatically:
    - Converts it to instance method (calls with self.params)
    - Generates _E_z_static (JIT-compiled version)
    - Generates compute_grid_traced, sound_horizon_traced, etc.

    Example
    -------
    >>> @register_cosmology_model
    ... class wCDM(LCDM):
    ...     def __init__(self, w=-1.0, **kwargs):
    ...         super().__init__(w=w, **kwargs)
    ...
    ...     @staticmethod
    ...     def E_z(z, params):
    ...         # Define core physics - most natural API!
    ...         Omega_m = params['Omega_m']
    ...         w = params.get('w', -1.0)
    ...         ...
    ...
    ...     def w_z(self, z):
    ...         return jnp.full_like(jnp.asarray(z), self.params['w'])
    """
    # Check if user defined E_z as staticmethod (new API)
    if "E_z" in cls.__dict__ and isinstance(cls.__dict__["E_z"], staticmethod):
        # User defined static E_z(z, params) - convert to kernel
        E_z_static_func = cls.__dict__["E_z"].__func__
        cls._E_z_kernel = staticmethod(E_z_static_func)

        # Create instance method that calls static version
        def E_z_instance(self, z):
            return self.__class__._E_z_kernel(z, self.params)

        cls.E_z = E_z_instance

    elif "E_z" in cls.__dict__:
        # A plain function here is the classic typo (missing @staticmethod).
        # Falling through would silently wire the PARENT's physics into all
        # traced methods while the subclass believes its E_z is in use.
        raise TypeError(
            f"{cls.__name__}.E_z must be a @staticmethod taking (z, params). "
            "Did you forget the @staticmethod decorator?"
        )

    # Fall back to _E_z_kernel if defined ON THIS CLASS (backward compatibility).
    # An inherited kernel must not silently satisfy this check: registering a
    # model that defines no physics of its own is almost certainly a mistake.
    elif "_E_z_kernel" in cls.__dict__:
        E_z_static_func = cls.__dict__["_E_z_kernel"]
    else:
        raise TypeError(
            f"{cls.__name__} must define either E_z(z, params) or _E_z_kernel(z, params). "
            "If this subclass intentionally reuses the parent physics, drop the "
            "@register_cosmology_model decorator and inherit the parent's traced methods."
        )

    # Generate JIT-compiled _E_z_static
    kernel = (
        cls._E_z_kernel
        if isinstance(cls._E_z_kernel, staticmethod)
        else cls._E_z_kernel
    )

    @jit
    def _E_z_static(z_grid: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
        if isinstance(kernel, staticmethod):
            return kernel.__func__(z_grid, params)
        return kernel(z_grid, params)

    cls._E_z_static = staticmethod(_E_z_static)

    # Generate factory methods
    _cgt = make_compute_grid_traced(_E_z_static)
    cls.compute_grid_traced = staticmethod(_cgt)
    _background_cgt = make_compute_background_grid_traced(_E_z_static)
    cls.compute_background_grid_traced = staticmethod(_background_cgt)
    cls.sound_horizon_traced = staticmethod(make_sound_horizon_traced())
    cls.sound_horizon_drag_traced = staticmethod(make_sound_horizon_drag_traced())
    cls.recombination_redshift_traced = staticmethod(
        make_recombination_redshift_traced()
    )

    return cls
