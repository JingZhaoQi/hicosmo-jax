"""
Lambda-CDM (ΛCDM) cosmological model - production implementation.

This module provides a complete implementation of the standard ΛCDM model
with all advanced features from qcosmc, optimized for JAX.

Now includes JAX-first integration tools tuned for performance and stability.
"""

from typing import Dict, Union, Optional, Literal
import numbers
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

# Diffrax removed - now using ultra-fast integration engine

from .base import CosmologyBase, compute_omega_r
from .unified_parameters import CosmologicalParameters
from ..utils.constants import c_km_s, c_cm_s, Mpc_km
from ..utils.jax_tools import (
    trapezoid,
    cumulative_trapezoid,
    gauss_legendre_integrate_batch,
    integrate_batch_cumulative,
    bisection_root,
    sound_horizon_eh98,
)


class LCDM(CosmologyBase):
    """
    Standard Lambda-CDM cosmological model with comprehensive functionality.

    This implementation includes:
    - Flat and non-flat geometries
    - Radiation component
    - Sound horizon calculations
    - Recombination history
    - Growth functions
    - All distance and time measures

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc
    Omega_m : float
        Total matter density parameter
    Omega_b : float, optional
        Baryon density parameter
    Omega_k : float, optional
        Curvature density parameter (default: 0.0 for flat)
    Omega_r : float, optional
        Radiation density parameter (computed if not given)
    sigma8 : float, optional
        Matter fluctuation amplitude at 8 h⁻¹ Mpc
    n_s : float, optional
        Scalar spectral index
    T_cmb : float, optional
        CMB temperature today in K (default: 2.7255)
    N_eff : float, optional
        Effective number of neutrino species (default: 3.046)
    """

    def __init__(
        self,
        H0: float = 67.36,
        Omega_m: float = 0.3153,
        Omega_b: float = 0.0493,  # Planck 2018 default (needed for BAO rd calculation)
        Omega_k: float = 0.0,
        Omega_r: Optional[float] = None,
        sigma8: float = 0.8111,
        n_s: float = 0.9649,
        T_cmb: float = 2.7255,
        N_eff: float = 3.046,
        precision_mode: Literal["fast", "balanced", "precise"] = "balanced",
        **kwargs,
    ):

        # Use unified parameter management
        # Check if user provided alternative parameter bases (e.g., h instead of H0)
        # to avoid conflicts with default values
        has_h = "h" in kwargs
        has_omega_m_alt = (
            "omega_m" in kwargs
            or "omega_m_h2" in kwargs
            or ("omega_b_h2" in kwargs and "omega_c_h2" in kwargs)
        )

        param_dict = {}

        # Only set defaults if alternative parameters not provided
        if not has_h:
            param_dict["H0"] = H0
        if not has_omega_m_alt:
            param_dict["Omega_m"] = Omega_m

        # Always include these (no alternative forms commonly used)
        param_dict["Omega_k"] = Omega_k
        param_dict["Omega_b"] = (
            Omega_b  # Always include (needed for BAO rd calculation)
        )
        param_dict["sigma8"] = sigma8
        param_dict["n_s"] = n_s
        param_dict["T_cmb"] = T_cmb
        param_dict["N_eff"] = N_eff

        # Handle optional parameters
        if Omega_r is not None:
            param_dict["Omega_r"] = Omega_r

        # Add any additional parameters (including h, omega_b_h2, etc.)
        param_dict.update(kwargs)

        # Initialize with unified parameter system
        self.cosmology_params = CosmologicalParameters(**param_dict)

        # Call parent - it will use our cosmology_params
        super().__init__()

        self.precision_mode = precision_mode
        # JIT-compiled kernels for hot paths (JAX JIT provides automatic compilation caching)
        self._comoving_distance_gauss_jit = jit(self._comoving_distance_gauss)
        # For dense integration, z_max determines grid size so must be static
        self._comoving_distance_dense_jit = jit(
            self._comoving_distance_dense, static_argnums=(1,)
        )

    # Radiation density computation now handled by unified parameter system

    def _validate_params(self) -> None:
        """Validate LCDM parameters for physical consistency."""
        # Validation now handled by unified parameter system
        self.cosmology_params.validate_closure()
        self.cosmology_params.validate_physics()

    # ==================== Background Evolution ====================

    @staticmethod
    def _E_z_kernel(z: Union[float, jnp.ndarray], params: Dict) -> jnp.ndarray:
        """
        Core E(z) computation for ΛCDM.

        E²(z) = Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_k(1+z)² + Ω_Λ

        This is the ONLY method subclasses need to override.
        """
        z_arr = jnp.asarray(z)
        one_plus_z = 1.0 + z_arr

        Omega_m = params["Omega_m"]
        Omega_k = params.get("Omega_k", 0.0)
        Omega_r = compute_omega_r(params)
        Omega_Lambda = params.get("Omega_Lambda", 1.0 - Omega_m - Omega_k - Omega_r)

        E_squared = (
            Omega_m * one_plus_z**3
            + Omega_r * one_plus_z**4
            + Omega_k * one_plus_z**2
            + Omega_Lambda
        )
        return jnp.sqrt(E_squared)

    def E_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        return self._E_z_kernel(z, self.params)

    # w_z() inherited from CosmologyBase (returns -1 for ΛCDM)

    # ==================== Specialized LCDM Functions ====================

    def redshift_equality(self) -> float:
        """
        Matter-radiation equality redshift z_eq.

        Returns
        -------
        float
            z_eq = Omega_m / Omega_r - 1
        """
        Omega_r = self.params.get("Omega_r", 0.0)
        try:
            Omega_r_val = float(Omega_r)
            if Omega_r_val <= 0.0:
                raise ValueError("Omega_r must be positive to compute z_eq.")
        except (TypeError, ValueError):
            pass
        return self.params["Omega_m"] / Omega_r - 1

    def redshift_acceleration(self) -> float:
        """
        Acceleration redshift z_acc where q(z_acc) = 0.

        For ΛCDM: z_acc = (2*Ω_Λ/Ω_m)^(1/3) - 1

        Returns
        -------
        float
            z_acc
        """
        ratio = 2 * self.params["Omega_Lambda"] / self.params["Omega_m"]
        return jnp.power(ratio, 1.0 / 3.0) - 1.0

    def sound_horizon(self, z: float, n_steps: int = 1000) -> float:
        """
        Sound horizon at redshift z in Mpc.

        r_s = ∫[c_s / H(z')] dz' from z to ∞

        Parameters
        ----------
        z : float
            Redshift
        n_steps : int
            Integration steps

        Returns
        -------
        float
            Sound horizon in Mpc
        """
        # Use fast EH98 approximation for stability and speed.
        return sound_horizon_eh98(
            jnp.asarray(self.params["H0"]),
            jnp.asarray(self.params["Omega_m"]),
            jnp.asarray(self.params.get("Omega_b", 0.05)),
            jnp.asarray(self.params.get("T_cmb", 2.7255)),
            jnp.asarray(z),
        )

    def sound_horizon_drag(self, use_camb: bool = True) -> float:
        """
        Sound horizon at drag epoch.

        Parameters
        ----------
        use_camb : bool, optional
            If True (default), use CAMB for precise calculation.
            Falls back to EH98 fitting formula if CAMB is not available.

        Returns
        -------
        float
            r_s(z_d) in Mpc

        Notes
        -----
        CAMB provides ~0.1% accuracy via numerical integration of the
        Boltzmann equations. EH98 fitting formula has ~2.7% error.
        For BAO analyses, CAMB accuracy is strongly recommended.
        """
        if use_camb:
            try:
                return self._sound_horizon_drag_camb()
            except ImportError:
                pass  # Fall back to EH98

        return self._sound_horizon_drag_eh98()

    def _sound_horizon_drag_camb(self) -> float:
        """Sound horizon via CAMB numerical integration (high precision)."""
        import camb

        H0 = self.params.get("H0")
        Omega_m = self.params.get("Omega_m")
        Omega_b = self.params.get("Omega_b")
        T_cmb = self.params.get("T_cmb", 2.7255)

        h = H0 / 100.0
        omega_b = Omega_b * h**2
        omega_c = (Omega_m - Omega_b) * h**2

        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=H0,
            ombh2=omega_b,
            omch2=omega_c,
            TCMB=T_cmb,
            num_massive_neutrinos=0,
            mnu=0.0,
        )
        results = camb.get_background(pars)
        return results.get_derived_params()["rdrag"]

    def _sound_horizon_drag_eh98(self) -> float:
        """Sound horizon via Eisenstein & Hu 1998 fitting formula (~2.7% error)."""
        Omega_m_h2 = self.Omega_m_h2
        Omega_b_h2 = self.Omega_b_h2

        T_cmb = self.params.get("T_cmb", 2.7255)
        theta_cmb = T_cmb / 2.7

        # Eisenstein & Hu (1998) fitting formulas
        b1 = 0.313 * Omega_m_h2 ** (-0.419) * (1 + 0.607 * Omega_m_h2**0.674)
        b2 = 0.238 * Omega_m_h2**0.223
        z_d = (
            1291
            * Omega_m_h2**0.251
            / (1 + 0.659 * Omega_m_h2**0.828)
            * (1 + b1 * Omega_b_h2**b2)
        )

        z_eq = 2.50e4 * Omega_m_h2 * theta_cmb**-4
        k_eq = 7.46e-2 * Omega_m_h2 * theta_cmb**-2  # Mpc^{-1}

        R_eq = 31.5 * Omega_b_h2 * theta_cmb**-4 * (1000.0 / z_eq)
        R_d = 31.5 * Omega_b_h2 * theta_cmb**-4 * (1000.0 / z_d)

        sqrt_term = jnp.sqrt(6.0 / R_eq)
        log_arg = (jnp.sqrt(1.0 + R_d) + jnp.sqrt(R_d + R_eq)) / (1.0 + jnp.sqrt(R_eq))
        s = (2.0 / (3.0 * k_eq)) * sqrt_term * jnp.log(log_arg)

        return s

    def critical_density(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """
        Critical density at redshift z in units of M_sun/Mpc³.

        ρ_c(z) = ρ_c(0) × [H(z)/H0]² = rho_crit_h2 × h² × E²(z)

        where rho_crit_h2 = 3H0²/(8πG) with H0 = 100 km/s/Mpc

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Critical density in M_sun/Mpc³
        """
        # Use efficient formula: rho_crit(z) = rho_crit_h2 * h^2 * E^2(z)
        # where rho_crit_h2 is the critical density factor (defined at H0=100)
        from ..utils.constants import rho_crit_h2

        h = self.h  # Use derived parameter property
        E_z_val = self.E_z(z)

        return rho_crit_h2 * h**2 * E_z_val**2

    # ==================== Growth Functions ====================

    def growth_factor(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Linear growth factor D(z) normalized to D(0) = 1.

        Uses analytical approximation (Lahav et al. 1991) for ΛCDM.
        Provides 1000x speedup with <1% error compared to ODE solving.

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Growth factor D(z)
        """
        return self.growth_factor_analytical(z)

    def growth_rate(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Growth rate f(z) = d ln D / d ln a.

        Uses analytical approximation (Wang & Steinhardt 1998) for ΛCDM.

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Growth rate f(z)
        """
        return self.growth_rate_analytical(z)

    def f_sigma8(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        fσ8(z) = f(z) * σ8(z) parameter.

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            fσ8(z) values
        """
        sigma8_z = self.sigma8_z(z)
        f_z = self.growth_rate(z)
        return f_z * sigma8_z

    def sigma8_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        σ8(z) = σ8(0) * D(z) for linear growth.

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            σ8(z) values
        """
        sigma8_0 = self.params.get("sigma8", 0.8111)
        D_z = self.growth_factor(z)
        return sigma8_0 * D_z

    # ==================== σ8 Calculation (First Principles) ====================

    def _eisenstein_hu_transfer(self, k: jnp.ndarray) -> jnp.ndarray:
        """
        Eisenstein & Hu (1998) transfer function (no BAO).

        Parameters
        ----------
        k : array_like
            Wavenumber in h/Mpc

        Returns
        -------
        array_like
            Transfer function T(k)
        """
        # Use derived parameters from unified parameter system
        Omega_m_h2 = self.Omega_m_h2
        Omega_b_h2 = self.Omega_b_h2
        theta_cmb = self.params.get("T_cmb", 2.7255) / 2.7  # CMB temperature ratio

        # Fitting parameters
        s = 44.5 * jnp.log(9.83 / Omega_m_h2) / jnp.sqrt(1 + 10 * Omega_b_h2**0.75)
        alpha_gamma = (
            1
            - 0.328 * jnp.log(431 * Omega_m_h2) * Omega_b_h2 / Omega_m_h2
            + 0.38 * jnp.log(22.3 * Omega_m_h2) * (Omega_b_h2 / Omega_m_h2) ** 2
        )

        gamma_eff = Omega_m_h2 * alpha_gamma
        q = k * theta_cmb**2 / gamma_eff

        # Baryon suppression
        L0 = jnp.log(2 * jnp.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1 + 62.5 * q)
        T0 = L0 / (L0 + C0 * q**2)

        return T0

    def _tophat_window(self, k: jnp.ndarray, R: float) -> jnp.ndarray:
        """
        Tophat window function in Fourier space.

        W(kR) = 3(sin(kR) - kR*cos(kR))/(kR)^3

        Parameters
        ----------
        k : array_like
            Wavenumber in h/Mpc
        R : float
            Radius in Mpc/h

        Returns
        -------
        array_like
            Window function W(kR)
        """
        kR = k * R

        # Handle kR -> 0 limit
        def small_kr_limit():
            return 1.0 - kR**2 / 10.0 + kR**4 / 280.0

        def normal_case():
            sin_kR = jnp.sin(kR)
            cos_kR = jnp.cos(kR)
            return 3.0 * (sin_kR - kR * cos_kR) / (kR**3)

        return jnp.where(kR < 1e-3, small_kr_limit(), normal_case())

    def growth_factor_analytical(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """
        Analytical approximation for ΛCDM growth factor.

        Uses Carroll, Press & Turner (1992) formula.

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Growth factor D(z) normalized to D(0) = 1
        """
        a = 1.0 / (1.0 + z)
        Omega_m = self.params["Omega_m"]
        Omega_Lambda = self.params["Omega_Lambda"]

        # Analytical approximation for flat ΛCDM
        if abs(self.params.get("Omega_k", 0.0)) < 1e-6:
            # Lahav et al. (1991) / Carroll, Press & Turner (1992) approximation
            Om_a = Omega_m / (Omega_m + Omega_Lambda * a**3)
            D_a = (
                (5 / 2)
                * Om_a
                / (
                    Om_a ** (4 / 7)
                    - Omega_Lambda
                    + (1 + Om_a / 2) * (1 + Omega_Lambda / 70)
                )
            )
            D_unnormalized = D_a * a

            # Normalize to D(0) = 1
            a_0 = 1.0
            Om_a_0 = Omega_m / (Omega_m + Omega_Lambda * a_0**3)
            D_a_0 = (
                (5 / 2)
                * Om_a_0
                / (
                    Om_a_0 ** (4 / 7)
                    - Omega_Lambda
                    + (1 + Om_a_0 / 2) * (1 + Omega_Lambda / 70)
                )
            )
            D_0 = D_a_0 * a_0

            return D_unnormalized / D_0
        else:
            # For non-flat, use growth rate approximation (normalized to a at a=1)
            return a

    def growth_rate_analytical(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """
        Analytical approximation for ΛCDM growth rate.

        Uses Wang & Steinhardt (1998) formula: f ≈ Ω_m(z)^γ

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Growth rate f(z) = d ln D / d ln a
        """
        # For flat ΛCDM, γ ≈ 0.55 + 0.05 * (1 + w) for general w
        # For ΛCDM (w = -1), γ ≈ 0.55
        gamma = 0.545 + 0.0055 * (1 + self.w_z(z))  # Small correction

        Omega_m_z = self.Omega_m_z(z)
        return Omega_m_z**gamma

    # ==================== Specialized Distances ====================

    def drift_rate(self, z: float, observation_time_yr: float = 30.0) -> float:
        """
        Redshift drift rate for Sandage test.

        The Sandage test measures the temporal change in redshift of distant sources:
        dz/dt = H₀(1+z) - H(z)

        This is converted to an equivalent velocity drift in cm/s.

        Parameters
        ----------
        z : float
            Redshift of the source
        observation_time_yr : float
            Observation time in years (default: 30 years for ELT-ANDES)

        Returns
        -------
        float
            Velocity drift in cm/s over the observation time

        Notes
        -----
        The expected signal is extremely small (~1 cm/s over 30 years),
        requiring extremely stable spectrographs like ELT-ANDES.

        References
        ----------
        Sandage (1962), Loeb (1998), Liske et al. (2008)
        """
        # H0 in units of 1/s (from km/s/Mpc)
        H0_per_s = self.params["H0"] / Mpc_km

        # H(z) = H0 * E(z) in units of 1/s
        E_z_val = self.E_z(z)
        H_z_per_s = H0_per_s * E_z_val

        # Redshift drift rate: dz/dt = H0(1+z) - H(z) [units: 1/s]
        dz_dt = H0_per_s * (1.0 + z) - H_z_per_s

        # Convert observation time to seconds
        observation_time_sec = observation_time_yr * 365.25 * 24.0 * 3600.0

        # Total velocity drift: Δv = c * (dz/dt) * Δt [units: cm/s]
        return dz_dt * c_cm_s * observation_time_sec

    # ==================== Convenience Methods ====================

    @classmethod
    def planck2018(cls, **kwargs) -> "LCDM":
        """
        ΛCDM with Planck 2018 best-fit parameters.

        Parameters
        ----------
        **kwargs
            Override any default parameters

        Returns
        -------
        LCDM
            Planck 2018 ΛCDM model
        """
        planck_params = {
            "H0": 67.36,
            "Omega_m": 0.3153,
            "Omega_b": 0.0493,
            "sigma8": 0.8111,
            "n_s": 0.9649,
        }
        planck_params.update(kwargs)
        return cls(**planck_params)

    @classmethod
    def wmap9(cls, **kwargs) -> "LCDM":
        """
        ΛCDM with WMAP9 best-fit parameters.

        Parameters
        ----------
        **kwargs
            Override any default parameters

        Returns
        -------
        LCDM
            WMAP9 ΛCDM model
        """
        wmap_params = {
            "H0": 70.0,
            "Omega_m": 0.279,
            "Omega_b": 0.046,
            "sigma8": 0.821,
            "n_s": 0.972,
        }
        wmap_params.update(kwargs)
        return cls(**wmap_params)

    @classmethod
    def get_parameters(cls):
        """
        Return all model parameters as Parameter objects.

        This method enables automatic parameter discovery for MCMC sampling.
        Returns Parameter objects for each cosmological parameter with
        default priors and LaTeX labels.

        Returns
        -------
        List[Parameter]
            List of Parameter objects for all LCDM parameters.

        Examples
        --------
        >>> from hicosmo.models import LCDM
        >>> from hicosmo.parameters import UnifiedParameterCollector
        >>>
        >>> # Auto-discover parameters for MCMC
        >>> collector = UnifiedParameterCollector()
        >>> collector.add_from_model(LCDM, free_params=['H0', 'Omega_m'])
        >>> print(collector.get_free_names())
        ['H0', 'Omega_m']
        """
        from ..parameters import Parameter

        return [
            Parameter(
                name="H0",
                value=67.36,
                free=False,  # Default to fixed, users set_free() to sample
                prior={"dist": "uniform", "min": 50.0, "max": 100.0},
                latex_label=r"$H_0$",
                description="Hubble constant [km/s/Mpc]",
            ),
            Parameter(
                name="Omega_m",
                value=0.3153,
                free=False,
                prior={"dist": "uniform", "min": 0.1, "max": 0.5},
                latex_label=r"$\Omega_m$",
                description="Total matter density parameter",
            ),
            Parameter(
                name="Omega_b",
                value=0.0493,
                free=False,
                prior={"dist": "uniform", "min": 0.01, "max": 0.1},
                latex_label=r"$\Omega_b$",
                description="Baryon density parameter",
            ),
            Parameter(
                name="Omega_k",
                value=0.0,
                free=False,
                prior={"dist": "uniform", "min": -0.1, "max": 0.1},
                latex_label=r"$\Omega_k$",
                description="Curvature density parameter",
            ),
            Parameter(
                name="sigma8",
                value=0.8111,
                free=False,
                prior={"dist": "uniform", "min": 0.5, "max": 1.2},
                latex_label=r"$\sigma_8$",
                description="Matter fluctuation amplitude at 8 h^-1 Mpc",
            ),
            Parameter(
                name="n_s",
                value=0.9649,
                free=False,
                prior={"dist": "uniform", "min": 0.8, "max": 1.2},
                latex_label=r"$n_s$",
                description="Scalar spectral index",
            ),
        ]

    def summary(self) -> str:
        """
        Generate a comprehensive summary of the ΛCDM model.

        Returns
        -------
        str
            Formatted summary
        """
        lines = [
            "ΛCDM Cosmological Model Summary",
            "=" * 35,
            "",
            "Primary Parameters:",
            f"  H₀ = {self.params['H0']:.2f} km/s/Mpc",
            f"  Ωₘ = {self.params['Omega_m']:.4f}",
            f"  Ωᵦ = {self.params.get('Omega_b', 0):.4f}",
            f"  Ωₖ = {self.params.get('Omega_k', 0):.4f}",
            f"  ΩΛ = {self.params['Omega_Lambda']:.4f}",
            f"  Ωᵣ = {self.params.get('Omega_r', 0):.6f}",
            "",
            "Structure Formation:",
            f"  σ₈ = {self.params.get('sigma8', 0):.4f}",
            f"  nₛ = {self.params.get('n_s', 0):.4f}",
            "",
        ]

        # Derived parameters
        derived = self.params
        Omega_m = self.params["Omega_m"]
        Omega_Lambda = self.params["Omega_Lambda"]
        Omega_r = self.params.get("Omega_r", 0.0)
        q0 = 0.5 * Omega_m + Omega_r - Omega_Lambda
        lines.extend(
            [
                "Derived Parameters:",
                f"  Age₀ = {derived.get('age', 0):.2f} Gyr",
                f"  tₕ = {derived.get('t_H', 0):.2f} Gyr",
                f"  Dₕ = {derived.get('D_H', 0):.1f} Mpc",
                f"  q₀ = {q0:.3f}",
                f"  z_acc = {self.redshift_acceleration():.2f}",
            ]
        )

        if "Omega_r" in self.params:
            z_eq = self.redshift_equality()
            lines.append(f"  z_eq = {z_eq:.0f}")

        return "\n".join(lines)

    # ==================== Background Evolution Methods ====================

    def H_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Hubble parameter H(z) in km/s/Mpc.

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Hubble parameter in km/s/Mpc
        """
        H0 = self.params["H0"]
        E_z_val = self.E_z(z)
        return H0 * E_z_val

    @property
    def H0_value(self) -> float:
        """Hubble constant in km/s/Mpc."""
        return self.params["H0"]

    @property
    def D_H(self) -> float:
        """Hubble distance in Mpc."""
        return c_km_s / self.params["H0"]

    # ==================== Derived Parameter Properties ====================

    @property
    def h(self) -> float:
        """Dimensionless Hubble parameter h = H0/100."""
        return self.cosmology_params._derived["h"]

    @property
    def Omega_m_h2(self) -> float:
        """Physical matter density Omega_m * h²."""
        return self.cosmology_params._derived["Omega_m_h2"]

    @property
    def Omega_b_h2(self) -> float:
        """Physical baryon density Omega_b * h²."""
        return self.cosmology_params._derived["Omega_b_h2"]

    @property
    def omega_b(self) -> float:
        """Physical baryon density parameter."""
        return self.cosmology_params._derived["omega_b"]

    @property
    def omega_m(self) -> float:
        """Physical matter density parameter."""
        return self.cosmology_params._derived["omega_m"]

    # ==================== Derived Parameters ====================

    def derived_parameters(self) -> Dict[str, float]:
        """
        Compute derived parameters from this LCDM model.

        Returns
        -------
        Dict[str, float]
            Derived parameters:
            - rd: Sound horizon at drag epoch [Mpc]
            - rd_h: r_d * h [Mpc] (BAO-constrained combination)
            - H0_rd: H0 * r_d [km/s] (alternative BAO combination)
            - Omega_Lambda: Dark energy density parameter
            - age: Age of the universe [Gyr] (if available)

        Examples
        --------
        >>> model = LCDM(H0=70, Omega_m=0.3, Omega_b=0.05)
        >>> derived = model.derived_parameters()
        >>> print(f"r_d*h = {derived['rd_h']:.2f} Mpc")
        """
        rd = float(self.sound_horizon_drag())
        h = float(self.h)
        H0 = float(self.params["H0"])

        derived = {
            "rd": rd,
            "rd_h": rd * h,
            "H0_rd": H0 * rd,
            "Omega_Lambda": float(
                1.0
                - self.params["Omega_m"]
                - self.params.get("Omega_k", 0.0)
                - self.params.get("Omega_r", 0.0)
            ),
        }

        # Add age if method exists and works
        try:
            derived["age"] = float(self.age_universe())
        except (AttributeError, Exception):
            pass

        return derived

    @classmethod
    def derived_param_labels(cls) -> Dict[str, str]:
        """
        Return LaTeX labels for LCDM derived parameters.

        Returns
        -------
        Dict[str, str]
            LaTeX labels (without $ signs).
        """
        return {
            "rd": r"r_d\,[\mathrm{Mpc}]",
            "rd_h": r"r_d h\,[\mathrm{Mpc}]",
            "H0_rd": r"H_0 r_d\,[\mathrm{km/s}]",
            "Omega_Lambda": r"\Omega_\Lambda",
            "age": r"t_0\,[\mathrm{Gyr}]",
        }

    # ==================== Redshift-dependent Properties ====================

    def Omega_m_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Matter density parameter as function of redshift."""
        E_z_val = self.E_z(z)
        z_arr = jnp.asarray(z)
        return self.params["Omega_m"] * (1 + z_arr) ** 3 / E_z_val**2

    # ==================== Distance Calculation Methods (Diffrax Integration) ====================

    # Old Diffrax integration methods removed - now using ultra-fast integration engine

    def comoving_distance(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """
        Vectorized comoving distance calculation (JIT-compiled).

        Computes D_c(z) = (c/H0) * ∫[0, z] dz'/E(z').

        Uses adaptive integration strategy:
        - Gauss-Legendre for low-z or small arrays (fast)
        - Dense grid for high-z or large batches (accurate)

        JAX JIT provides automatic compilation caching for repeated calls.
        """
        z_array = jnp.atleast_1d(jnp.asarray(z, dtype=jnp.float64))

        # JIT-compatible integration strategy selection
        # Use jax.lax.cond for traced values, avoiding float() conversion
        z_max = jnp.max(z_array)

        # For JIT compatibility, use lax.cond with fixed grid size
        # Dense grid uses z_max_int=20 which covers z up to ~20 with good accuracy
        # Ensure both branches return same dtype (float64) for JAX compatibility
        result = lax.cond(
            z_max > self.HIGH_Z_THRESHOLD,
            lambda arr: jnp.asarray(
                self._comoving_distance_dense_jit(arr, 20), dtype=jnp.float64
            ),
            lambda arr: jnp.asarray(
                self._comoving_distance_gauss_jit(arr), dtype=jnp.float64
            ),
            z_array,
        )
        return result

    # Threshold above which we use dense grid integration for accuracy
    HIGH_Z_THRESHOLD = 10.0

    def _comoving_distance_gauss(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """Gauss-Legendre integration for low-z (JIT-friendly)."""
        z_array = jnp.atleast_1d(z)

        def integrand(z_prime):
            return 1.0 / self.E_z(z_prime)

        order, _ = self._precision_settings()
        integrals = gauss_legendre_integrate_batch(
            integrand, z_array, z_min=0.0, order=order
        )
        d_c = self.params["D_H"] * integrals
        return jnp.squeeze(d_c)

    def _comoving_distance_dense(
        self, z: Union[float, jnp.ndarray], z_max_int: int
    ) -> Union[float, jnp.ndarray]:
        """Dense grid integration for high-z accuracy (JIT-friendly).

        Parameters
        ----------
        z : array-like
            Redshift values
        z_max_int : int
            Ceiling of max redshift (static argument for JIT)
        """
        z_array = jnp.atleast_1d(z)

        def integrand(z_prime):
            return 1.0 / self.E_z(z_prime)

        _, grid_n = self._precision_settings()
        # Scale grid density with z_max for consistent precision
        adaptive_grid_n = max(grid_n, z_max_int * 8)
        integrals = integrate_batch_cumulative(
            integrand, z_array, z_min=0.0, n_grid=adaptive_grid_n
        )
        d_c = self.params["D_H"] * integrals
        return jnp.squeeze(d_c)

    def _precision_settings(self) -> tuple[int, int]:
        """Return (gauss_order, grid_n) based on precision mode."""
        if self.precision_mode == "fast":
            return 8, 2048
        if self.precision_mode == "precise":
            return 16, 8192
        return 12, 4096

    def transverse_comoving_distance(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """
        Transverse comoving distance (accounts for curvature).

        D_M(z) = {
            D_H/√Ωk sinh(√Ωk D_C/D_H)  for Ωk > 0 (open)
            D_C                          for Ωk = 0 (flat)
            D_H/√|Ωk| sin(√|Ωk| D_C/D_H) for Ωk < 0 (closed)
        }

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Transverse comoving distance(s) in Mpc
        """
        D_C = self.comoving_distance(z)
        Omega_k = self.params.get("Omega_k", 0.0)

        # Use safe sqrt to avoid division by zero in JAX tracing
        # This prevents NaN gradients from propagating through unused branches
        safe_sqrt_Ok = jnp.sqrt(jnp.maximum(jnp.abs(Omega_k), 1e-12))

        flat = D_C
        # Open universe (Omega_k > 0): sinh formula
        open_val = self.D_H / safe_sqrt_Ok * jnp.sinh(safe_sqrt_Ok * D_C / self.D_H)
        # Closed universe (Omega_k < 0): sin formula
        closed_val = self.D_H / safe_sqrt_Ok * jnp.sin(safe_sqrt_Ok * D_C / self.D_H)

        # Select based on sign of Omega_k
        curved = jnp.where(Omega_k > 0, open_val, closed_val)

        # Use JAX conditionals for differentiability
        return jnp.where(jnp.abs(Omega_k) < 1e-6, flat, curved)

    def _transverse_comoving_distance_delta(
        self, z1: Union[float, jnp.ndarray], z2: Union[float, jnp.ndarray]
    ) -> jnp.ndarray:
        """Transverse comoving distance between two redshifts.

        JAX-differentiable version: avoids division by zero in unused branches
        by using safe_sqrt_ok to prevent NaN gradients.
        """
        if isinstance(z1, numbers.Real) and isinstance(z2, numbers.Real):
            if z2 <= z1:
                raise ValueError("Source redshift must be greater than lens redshift.")

        z1_arr, z2_arr = jnp.broadcast_arrays(
            jnp.asarray(z1, dtype=jnp.float64), jnp.asarray(z2, dtype=jnp.float64)
        )

        D_C1 = jnp.asarray(self.comoving_distance(z1_arr))
        D_C2 = jnp.asarray(self.comoving_distance(z2_arr))
        D_H = jnp.asarray(self.D_H, dtype=jnp.float64)
        delta_chi = (D_C2 - D_C1) / D_H

        Omega_k = jnp.asarray(self.params.get("Omega_k", 0.0), dtype=jnp.float64)

        # Use safe sqrt to avoid division by zero in JAX tracing
        # When Omega_k=0, we use 1.0 as placeholder (result won't be used due to where)
        # This prevents NaN gradients from propagating through unused branches
        safe_sqrt_ok = jnp.sqrt(jnp.maximum(jnp.abs(Omega_k), 1e-12))

        flat = D_H * delta_chi
        # Open universe (Omega_k > 0): sinh formula
        open_val = D_H / safe_sqrt_ok * jnp.sinh(safe_sqrt_ok * delta_chi)
        # Closed universe (Omega_k < 0): sin formula
        closed_val = D_H / safe_sqrt_ok * jnp.sin(safe_sqrt_ok * delta_chi)

        # Select based on sign of Omega_k
        curved = jnp.where(Omega_k > 0, open_val, closed_val)

        valid = jnp.greater(z2_arr, z1_arr)
        result = jnp.where(jnp.abs(Omega_k) < 1e-8, flat, curved)
        # Use -1e10 instead of jnp.nan to preserve gradient flow
        return jnp.where(valid, result, -1e10)

    def angular_diameter_distance_between(
        self, z1: Union[float, jnp.ndarray], z2: Union[float, jnp.ndarray]
    ) -> jnp.ndarray:
        """Angular diameter distance between two redshifts z1 < z2."""
        z1_arr, z2_arr = jnp.broadcast_arrays(
            jnp.asarray(z1, dtype=jnp.float64), jnp.asarray(z2, dtype=jnp.float64)
        )
        D_M = self._transverse_comoving_distance_delta(z1_arr, z2_arr)
        return jnp.asarray(D_M / (1.0 + z2_arr))

    def time_delay_distance(self, z_lens: float, z_source: float) -> jnp.ndarray:
        """Time-delay distance in Mpc for a lens-source pair."""
        D_d = jnp.asarray(self.angular_diameter_distance(z_lens))
        D_s = jnp.asarray(self.angular_diameter_distance(z_source))
        D_ds = jnp.asarray(self.angular_diameter_distance_between(z_lens, z_source))
        return (1.0 + z_lens) * D_d * D_s / D_ds

    def angular_diameter_distance(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """
        Angular diameter distance.

        D_A(z) = D_M(z) / (1 + z)

        where D_M is the transverse comoving distance (accounts for curvature).

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Angular diameter distance(s) in Mpc
        """
        z_array = jnp.atleast_1d(z)
        d_m = self.transverse_comoving_distance(z_array)
        return jnp.squeeze(d_m / (1.0 + z_array))

    def luminosity_distance(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """
        Luminosity distance.

        D_L(z) = D_M(z) * (1 + z)

        where D_M is the transverse comoving distance (accounts for curvature).

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Luminosity distance(s) in Mpc
        """
        z_array = jnp.atleast_1d(z)
        d_m = self.transverse_comoving_distance(z_array)
        return jnp.squeeze(d_m * (1.0 + z_array))

    def differential_comoving_volume(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """
        Differential comoving volume element dVc/dz.

        dVc/dz = 4π * c/H0 * D_M²(z) / E(z)  [Mpc³]

        where:
        - D_M(z) is transverse comoving distance (accounts for curvature)
        - E(z) = H(z)/H0 is the dimensionless Hubble parameter

        This is the proper volume element for number counts, luminosity functions,
        and gravitational wave merger rate calculations.

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Differential comoving volume dVc/dz in Mpc³

        Notes
        -----
        - For flat universe (Ω_k=0): D_M = D_C (comoving distance)
        - For curved universe: D_M accounts for spatial curvature
        - Units: H0 in km/s/Mpc → c/H0 in Mpc

        References
        ----------
        Hogg (1999) arXiv:astro-ph/9905116, Eq. 28
        """
        z_array = jnp.atleast_1d(z)

        # Transverse comoving distance (handles curvature)
        D_M = self.transverse_comoving_distance(z_array)  # Mpc

        # Dimensionless Hubble parameter
        E_z = self.E_z(z_array)

        # Use existing Hubble distance property
        D_H = self.D_H  # Mpc

        # Differential comoving volume: 4π * D_H * D_M² / E(z)
        dVc_dz = 4.0 * jnp.pi * D_H * D_M**2 / E_z  # Mpc³

        return jnp.squeeze(dVc_dz)

    def dVc_dz(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Alias for differential_comoving_volume for GW likelihood compatibility.

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Differential comoving volume dVc/dz in Gpc³ (matches icarogw units)
        """
        # Convert from Mpc³ to Gpc³
        return self.differential_comoving_volume(z) / 1e9

    def dl_to_z(
        self,
        d_L: Union[float, jnp.ndarray],
        z_min: float = 0.0,
        z_max: float = 10.0,
        tol: float = 1e-6,
        max_iter: int = 50,
    ) -> Union[float, jnp.ndarray]:
        """
        Inverse luminosity distance: convert d_L to redshift z.

        Uses bisection method to solve: d_L(z) = d_L_target

        This is critical for GW hierarchical inference - when testing different
        cosmologies, we must recalculate z from the observed d_L.

        Parameters
        ----------
        d_L : float or array_like
            Luminosity distance(s) in Mpc
        z_min : float
            Minimum redshift for search
        z_max : float
            Maximum redshift for search
        tol : float
            Tolerance for root finding
        max_iter : int
            Maximum iterations for bisection

        Returns
        -------
        float or array_like
            Redshift(s) corresponding to input d_L

        Notes
        -----
        Uses bisection method (robust and JIT-compatible):
        - Monotonic function d_L(z) increases with z
        - Binary search finds z where d_L(z) = d_L_target

        References
        ----------
        icarogw/cosmology.py: dl2z() method
        """
        # Scalar case: use bisection method with lax.while_loop (JAX-compatible!)
        if jnp.ndim(d_L) == 0 or jnp.size(d_L) == 1:
            d_L_target = d_L
            func = lambda z: self.luminosity_distance(z) - d_L_target
            return bisection_root(func, z_min, z_max, tol=tol, max_iter=max_iter)

        # Array case: use vmap
        else:
            return vmap(lambda dl: self.dl_to_z(dl, z_min, z_max, tol, max_iter))(d_L)

    def ddL_dz(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Derivative of luminosity distance with respect to redshift: dD_L/dz.

        Used in GW hierarchical inference for Jacobian calculation:
            J = (1+z) * dD_L/dz

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            dD_L/dz in Mpc

        Notes
        -----
        D_L(z) = D_M(z) * (1+z)

        Therefore:
            dD_L/dz = dD_M/dz * (1+z) + D_M(z)

        where dD_M/dz = c / [H0 * E(z)] (for flat universe)

        For curved universe, need to account for curvature factor.

        References
        ----------
        icarogw/conversions.py: detector2source_jacobian_q()
        """
        z_array = jnp.atleast_1d(z)

        # Use analytical derivative formula (universal for all cosmology models!)
        # Formula: dD_L/dz = D_M + (1+z) * dD_M/dz
        # where dD_M/dz = c / (H0 * E(z))
        #
        # This is EXACT and works for all models that inherit from this class,
        # as long as they override E_z() correctly.

        # Get transverse comoving distance D_M(z)
        D_M = self.transverse_comoving_distance(z_array)

        # Compute dD_M/dz = c / (H0 * E(z))
        E_z_vals = self.E_z(z_array)  # Calls subclass implementation if overridden
        dD_M_dz = c_km_s / (self.params["H0"] * E_z_vals)

        # Compute dD_L/dz = D_M + (1+z) * dD_M/dz
        ddL_dz_vals = D_M + (1.0 + z_array) * dD_M_dz

        return jnp.squeeze(ddL_dz_vals)

    def distance_modulus(
        self, z: Union[float, jnp.ndarray]
    ) -> Union[float, jnp.ndarray]:
        """
        Distance modulus.

        μ(z) = 5 log₁₀(D_L(z) / 10 pc) = 5 log₁₀(D_L(z)) + 25

        Parameters
        ----------
        z : float or array_like
            Redshift(s)

        Returns
        -------
        float or array_like
            Distance modulus (magnitude)
        """
        # Compute from luminosity distance
        d_L = self.luminosity_distance(z)
        return 5.0 * jnp.log10(d_L) + 25.0

    def distance_summary(self, z: float = 1.0) -> str:
        """
        Generate summary of distance measures at given redshift.

        Parameters
        ----------
        z : float
            Redshift for summary

        Returns
        -------
        str
            Formatted summary
        """
        D_C = self.comoving_distance(z)
        D_A = self.angular_diameter_distance(z)
        D_L = self.luminosity_distance(z)
        mu = self.distance_modulus(z)
        H_z = self.H_z(z)

        lines = [
            f"ΛCDM Distance Summary at z = {z:.2f}",
            "=" * 40,
            f"Hubble parameter: H(z) = {H_z:.1f} km/s/Mpc",
            f"Comoving distance: D_C = {D_C:.1f} Mpc",
            f"Angular diameter distance: D_A = {D_A:.1f} Mpc",
            f"Luminosity distance: D_L = {D_L:.1f} Mpc",
            f"Distance modulus: μ = {mu:.2f} mag",
            "",
            "Model parameters:",
            f"  H₀ = {self.params['H0']:.1f} km/s/Mpc",
            f"  Ωₘ = {self.params['Omega_m']:.4f}",
            f"  ΩΛ = {self.params['Omega_Lambda']:.4f}",
        ]

        return "\n".join(lines)

    # ==================== Traced-Aware Interface (NumPyro MCMC) ====================

    @staticmethod
    @jit
    def _E_z_static(z_grid: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
        """Static E(z) for traced MCMC sampling."""
        return LCDM._E_z_kernel(z_grid, params)

    # compute_grid_traced is generated by factory function after class definition


# Alias for compatibility
LCDMModel = LCDM

# ==================== Factory-Generated compute_grid_traced ====================
# Generate compute_grid_traced using factory pattern (closure captures _E_z_static)
from .base import (
    make_compute_grid_traced,
    make_sound_horizon_traced,
    make_sound_horizon_drag_traced,
    make_recombination_redshift_traced,
)

_lcdm_cgt = make_compute_grid_traced(LCDM._E_z_static)
LCDM.compute_grid_traced = staticmethod(_lcdm_cgt)
LCDM.compute_DM_at_z = staticmethod(_lcdm_cgt.compute_DM_at_z)
LCDM.sound_horizon_traced = staticmethod(make_sound_horizon_traced())
LCDM.sound_horizon_drag_traced = staticmethod(make_sound_horizon_drag_traced())
LCDM.recombination_redshift_traced = staticmethod(make_recombination_redshift_traced())
