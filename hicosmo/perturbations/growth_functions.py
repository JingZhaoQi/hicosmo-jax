"""
Growth Function Calculator
==========================

Specialized calculator for cosmological growth functions.
Provides high-precision calculations of linear growth factors and rates
for arbitrary cosmological models including dark energy and modified gravity.

Key features:
- Exact ODE solutions for growth functions
- Support for general w(z) dark energy models
- Modified gravity extensions
- Redshift-space distortion parameters
- Growth index calculations
"""

import jax.numpy as jnp
from jax import jit, vmap, grad
# JAX doesn't have solve_ivp and quad, use scipy
from scipy.integrate import solve_ivp, quad
from functools import partial
from typing import Union, Tuple, Dict, Optional, Callable
import numpy as np

from ..background.background import BackgroundEvolution


class GrowthCalculator:
    """
    High-precision growth function calculator.
    
    Solves the exact growth equation for arbitrary cosmological models:
    
    d²D/da² + (1/a + H'/H) dD/da - (3/2) Ω_m(a) H₀²/(a³H²) D = 0
    
    where H' = dH/da and all quantities are functions of scale factor a.
    """
    
    def __init__(self, background: BackgroundEvolution):
        """
        Initialize growth calculator.
        
        Parameters
        ----------
        background : BackgroundEvolution
            Background cosmology calculator
        """
        self.bg = background
        self.params = background.model.params
        
        # Physical parameters
        self.H0 = background.H0_value if hasattr(background, 'H0_value') else background.H0
        
        # Handle parameter extraction - compatible with dict and parameter manager
        if hasattr(self.params, 'get_value'):
            self.Omega_m_0 = self.params.get_value('Omega_m')
            self.Omega_Lambda_0 = self.params.get_value('Omega_Lambda')
        else:
            # Parameters in dictionary format
            self.Omega_m_0 = self.params.get('Omega_m', 0.3)
            self.Omega_Lambda_0 = self.params.get('Omega_Lambda', 0.7)
        
        # Pre-compile functions
        self._compile_functions()
    
    def _compile_functions(self):
        """Pre-compile JAX functions for optimal performance."""
        
        # Growth ODE solver
        self._growth_ode_jit = jit(self._solve_growth_ode)
        
        # Vectorized growth functions
        self._growth_factor_exact_vmap = jit(vmap(self._growth_factor_exact_single))
        self._growth_rate_exact_vmap = jit(vmap(self._growth_rate_exact_single))
        
        # Growth index calculation
        self._growth_index_jit = jit(self._compute_growth_index)
    
    # ==================== Exact Growth Solutions ====================
    
    @partial(jit, static_argnums=(1, 2, 3))
    def _solve_growth_ode(self, a_init: float = 1e-3, a_final: float = 1.0,
                         n_points: int = 1000, return_rate: bool = False) -> Tuple:
        """
        Solve the exact growth equation as function of scale factor.
        
        The growth equation in terms of scale factor a:
        d²D/da² + (1/a + H'/H) dD/da - (3/2) Ω_m(a) H₀²/(a³H²) D = 0
        """
        def growth_ode_system(a, y):
            """
            Growth equation as first-order ODE system.
            
            y[0] = D(a)      # Growth factor
            y[1] = dD/da     # First derivative
            """
            if a <= 1e-6:  # Avoid singularity
                return jnp.array([0.0, 0.0])
            
            D, dD_da = y
            
            # Convert to redshift for background functions
            z = (1.0 / a) - 1.0
            
            # Background quantities
            E_z = self.bg.E_z(z)
            Omega_m_a = self.Omega_m_0 * (a**(-3)) / E_z**2
            
            # Growth equation coefficients
            # H'/H term (numerical derivative of Hubble parameter)
            da = 0.001 * a
            z_plus = 1.0/max(a + da, 1e-6) - 1.0
            z_minus = 1.0/max(a - da, 1e-6) - 1.0
            
            E_plus = self.bg.E_z(z_plus)
            E_minus = self.bg.E_z(z_minus)
            
            # H'/H = d(ln H)/da = (1/H) dH/da = (1/E) dE/da * H₀/a²
            dE_da = (E_plus - E_minus) / (2 * da)
            H_prime_over_H = dE_da / E_z / a
            
            # Second derivative
            coeff1 = 1.0/a + H_prime_over_H
            coeff2 = 1.5 * Omega_m_a * (self.H0 / (a * self.H0 * E_z))**2
            
            d2D_da2 = coeff2 * D - coeff1 * dD_da
            
            return jnp.array([dD_da, d2D_da2])
        
        # Initial conditions in matter-dominated era
        # D(a) ∝ a, dD/da ∝ 1
        D_init = a_init
        dD_da_init = 1.0
        y_init = jnp.array([D_init, dD_da_init])
        
        # Solve ODE
        a_grid = jnp.linspace(a_init, a_final, n_points)
        solution = solve_ivp(
            growth_ode_system,
            (a_init, a_final),
            y_init,
            t_eval=a_grid,
            method='Radau',  # Stiff solver for better stability
            rtol=1e-8, atol=1e-10
        )
        
        D_solution = solution.y[0]
        dD_da_solution = solution.y[1]
        
        # Normalize to D(a=1) = 1
        D_normalized = D_solution / D_solution[-1]
        
        if return_rate:
            # Compute growth rate f = d ln D / d ln a = a/D * dD/da
            f_solution = a_grid * dD_da_solution / D_solution
            f_normalized = f_solution  # f is already normalized
            return a_grid, D_normalized, f_normalized
        else:
            return a_grid, D_normalized
    
    def _growth_factor_exact_single(self, z: float) -> float:
        """Exact growth factor for single redshift via ODE solution."""
        if z <= 0:
            return 1.0
        
        a = 1.0 / (1.0 + z)
        
        # Solve ODE to this scale factor
        a_grid, D_solution = self._growth_ode_jit(1e-3, a, 500)
        
        # Return final value
        return D_solution[-1]
    
    def growth_factor_exact(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Exact growth factor D(z) from ODE solution.
        
        More accurate than fitting formulas but computationally expensive.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
            
        Returns
        -------
        float or array_like
            Exact growth factor D(z)
        """
        z = jnp.asarray(z)
        if z.ndim == 0:
            return self._growth_factor_exact_single(z)
        else:
            return self._growth_factor_exact_vmap(z)
    
    def _growth_rate_exact_single(self, z: float) -> float:
        """Exact growth rate for single redshift."""
        if z <= 0:
            # Need to compute f at z=0 from derivative
            a_grid, D_solution, f_solution = self._growth_ode_jit(1e-3, 1.0, 1000, True)
            return f_solution[-1]
        
        a = 1.0 / (1.0 + z)
        a_grid, D_solution, f_solution = self._growth_ode_jit(1e-3, a, 500, True)
        
        return f_solution[-1]
    
    def growth_rate_exact(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Exact growth rate f(z) = d ln D / d ln a from ODE solution.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
            
        Returns
        -------
        float or array_like
            Exact growth rate f(z)
        """
        z = jnp.asarray(z)
        if z.ndim == 0:
            return self._growth_rate_exact_single(z)
        else:
            return self._growth_rate_exact_vmap(z)
    
    # ==================== Growth Index ====================
    
    def _compute_growth_index(self, z_max: float = 3.0) -> float:
        """
        Compute growth index γ where f ≈ Ω_m^γ.
        
        Fit f(z) to Ω_m(z)^γ over redshift range.
        """
        # Sample redshifts for fitting
        z_sample = jnp.linspace(0, z_max, 20)
        
        # Get exact growth rates and matter densities
        f_sample = self.growth_rate_exact(z_sample)
        Omega_m_sample = vmap(self.bg.Omega_m_z)(z_sample)
        
        # Fit log(f) = γ * log(Ω_m) + const
        log_f = jnp.log(f_sample)
        log_Om = jnp.log(Omega_m_sample)
        
        # Linear regression
        X = jnp.vstack([log_Om, jnp.ones(len(log_Om))]).T
        gamma, const = jnp.linalg.lstsq(X, log_f, rcond=None)[0]
        
        return gamma
    
    def growth_index(self, z_max: float = 3.0) -> float:
        """
        Growth index γ where f ≈ Ω_m^γ.
        
        For ΛCDM: γ ≈ 0.545
        For general models: fitted from exact growth rate
        
        Parameters
        ----------
        z_max : float
            Maximum redshift for fitting
            
        Returns
        -------
        float
            Growth index γ
        """
        return self._growth_index_jit(z_max)
    
    # ==================== Redshift-Space Distortions ====================
    
    def redshift_space_parameter(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Redshift-space distortion parameter β(z) = f(z)/b.
        
        Assumes linear bias b = 1 (can be generalized).
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
            
        Returns
        -------
        float or array_like
            RSD parameter β(z)
        """
        # For linear bias b=1, β = f
        return self.growth_rate_exact(z)
    
    def kaiser_factor(self, z: Union[float, jnp.ndarray],
                     mu: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Kaiser effect factor for redshift-space power spectrum.
        
        K(μ) = (1 + β μ²)²
        
        where μ is cosine of angle to line of sight.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        mu : float or array_like
            Line-of-sight cosine
            
        Returns
        -------
        float or array_like
            Kaiser factor
        """
        beta = self.redshift_space_parameter(z)
        mu = jnp.asarray(mu)
        
        return (1 + beta * mu**2)**2
    
    # ==================== Modified Gravity Extensions ====================
    
    def effective_newton_constant(self, k: Union[float, jnp.ndarray],
                                z: Union[float, jnp.ndarray],
                                model: str = 'GR') -> Union[float, jnp.ndarray]:
        """
        Effective Newton's constant for modified gravity theories.
        
        G_eff(k,z) / G_N for structure formation.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
        z : float or array_like
            Redshift(s)
        model : str
            Modified gravity model ('GR', 'f(R)', 'DGP', etc.)
            
        Returns
        -------
        float or array_like
            G_eff / G_N ratio
        """
        if model == 'GR':
            return 1.0
        elif model == 'f(R)':
            # Simplified f(R) model
            # G_eff/G = 1 + k²/(3(k² + m²))
            m_fR = 1e-3  # Compton wavelength parameter (h/Mpc)
            k = jnp.asarray(k)
            return 1.0 + k**2 / (3 * (k**2 + m_fR**2))
        elif model == 'DGP':
            # DGP model
            H_z = self.bg.H_z(z)
            r_c = 3000  # Crossover scale in Mpc/h
            return 1.0 + 1.0 / (3 * (H_z * r_c)**2)
        else:
            return 1.0  # Default to GR
    
    def modified_growth_rate(self, z: Union[float, jnp.ndarray],
                           model: str = 'GR') -> Union[float, jnp.ndarray]:
        """
        Growth rate in modified gravity theories.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        model : str
            Modified gravity model
            
        Returns
        -------
        float or array_like
            Modified growth rate
        """
        f_GR = self.growth_rate_exact(z)
        
        if model == 'GR':
            return f_GR
        elif model == 'f(R)':
            # Enhancement in f(R) gravity
            enhancement = 1.1  # Simplified
            return f_GR * enhancement
        else:
            return f_GR
    
    # ==================== Utility Methods ====================
    
    def growth_evolution_table(self, z_min: float = 0, z_max: float = 3.0,
                             n_points: int = 50) -> Dict[str, jnp.ndarray]:
        """
        Generate table of growth evolution for interpolation.
        
        Parameters
        ----------
        z_min, z_max : float
            Redshift range
        n_points : int
            Number of points
            
        Returns
        -------
        dict
            Dictionary with growth evolution data
        """
        z_array = jnp.linspace(z_min, z_max, n_points)
        
        # Compute exact growth functions
        D_array = self.growth_factor_exact(z_array)
        f_array = self.growth_rate_exact(z_array)
        
        # Additional quantities
        Omega_m_array = vmap(self.bg.Omega_m_z)(z_array)
        
        return {
            'z': z_array,
            'D': D_array,
            'f': f_array,
            'Omega_m': Omega_m_array,
            'growth_index': self.growth_index()
        }
    
    def compare_approximations(self, z: float = 1.0) -> Dict[str, float]:
        """
        Compare different growth function approximations.
        
        Parameters
        ----------
        z : float
            Redshift for comparison
            
        Returns
        -------
        dict
            Comparison of different methods
        """
        # Exact solution
        D_exact = self.growth_factor_exact(z)
        f_exact = self.growth_rate_exact(z)
        
        # Simple fitting formula
        Omega_m_z = self.bg.Omega_m_z(z)
        f_approx = Omega_m_z**0.545  # Standard approximation
        
        # Matter-dominated approximation
        a = 1.0 / (1 + z)
        D_matter = a
        
        return {
            'z': z,
            'D_exact': float(D_exact),
            'f_exact': float(f_exact),
            'f_Omega_approx': float(f_approx),
            'D_matter_dom': float(D_matter),
            'f_error_percent': float(100 * abs(f_exact - f_approx) / f_exact)
        }
    
    def growth_summary(self, z: float = 0) -> str:
        """
        Generate comprehensive growth function summary.
        
        Parameters
        ----------
        z : float
            Redshift for summary
            
        Returns
        -------
        str
            Formatted summary
        """
        D = self.growth_factor_exact(z)
        f = self.growth_rate_exact(z)
        gamma = self.growth_index()
        beta = self.redshift_space_parameter(z)
        
        Omega_m_z = self.bg.Omega_m_z(z)
        f_approx = Omega_m_z**0.545
        
        lines = [
            f"Growth Functions Summary at z = {z:.2f}",
            "=" * 40,
            f"Growth factor: D = {D:.6f}",
            f"Growth rate: f = {f:.6f}",
            f"Growth index: γ = {gamma:.4f}",
            f"RSD parameter: β = {beta:.4f}",
            f"Ω_m(z) = {Omega_m_z:.4f}",
            "",
            "Approximation Check:",
            f"f_exact = {f:.6f}",
            f"f_Ω^0.545 = {f_approx:.6f}",
            f"Error = {100*abs(f-f_approx)/f:.2f}%",
        ]
        
        return "\n".join(lines)