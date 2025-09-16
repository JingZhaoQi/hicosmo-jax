"""
CLASS Interface
===============

Professional interface with CLASS (Cosmic Linear Anisotropy Solving System).
Provides parameter conversion, result comparison, and validation tools.

Key features:
- Parameter format conversion between HIcosmo and CLASS
- Result comparison and validation  
- Performance benchmarking
- Cross-validation with CAMB

CLASS provides an independent validation of cosmological calculations
and is particularly strong for large-scale structure computations.
"""

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Union, Tuple, Dict, Optional, Any, List
import numpy as np
import warnings

from ..core.parameters_professional import CosmologyParameters
from ..background.background import BackgroundEvolution
from ..cmb.temperature_cl import TemperaturePowerSpectrum
from ..powerspectrum.linear_power import LinearPowerSpectrum


class CLASSInterface:
    """
    Interface between HIcosmo and CLASS.
    
    Provides seamless conversion between parameter formats,
    comparison of results, and validation tools for ensuring
    HIcosmo calculations match CLASS standards.
    """
    
    def __init__(self, hicosmo_components: Dict[str, Any]):
        """
        Initialize CLASS interface.
        
        Parameters
        ----------
        hicosmo_components : dict
            Dictionary containing HIcosmo calculation components
        """
        self.background = hicosmo_components.get('background')
        self.linear_power = hicosmo_components.get('linear_power')
        self.temperature_cl = hicosmo_components.get('temperature_cl')
        
        if self.background is None:
            raise ValueError("Background evolution component is required")
        
        self.params = self.background.model.params
        
        # CLASS availability check
        self.class_available = self._check_class_availability()
        
        # Comparison tolerances
        self.tolerances = {
            'background': 1e-3,
            'power_spectrum': 5e-3,
            'cmb_temperature': 1e-2
        }
    
    def _check_class_availability(self) -> bool:
        """Check if CLASS is available for comparison."""
        try:
            from classy import Class
            return True
        except ImportError:
            warnings.warn("CLASS not available. Comparison features disabled.")
            return False
    
    # ==================== Parameter Conversion ====================
    
    def hicosmo_to_class_params(self) -> Dict[str, Any]:
        """
        Convert HIcosmo parameters to CLASS format.
        
        Returns
        -------
        dict
            CLASS-formatted parameters
        """
        # Get HIcosmo parameters
        H0 = self.params.get_value('H0')
        h = self.params.get_value('h')
        Omega_m = self.params.get_value('Omega_m')
        Omega_b = self.params.get_value('Omega_b')
        Omega_k = self.params.get_value('Omega_k')
        n_s = self.params.get_value('n_s')
        ln_A_s_1e10 = self.params.get_value('ln_A_s_1e10')
        tau_reio = self.params.get_value('tau_reio')
        T_cmb = self.params.get_value('T_cmb')
        N_eff = self.params.get_value('N_eff')
        
        # Convert to CLASS format
        class_params = {
            # Hubble parameter  
            'h': h,
            
            # Density parameters
            'Omega_b': Omega_b,
            'Omega_cdm': Omega_m - Omega_b,
            'Omega_k': Omega_k,
            
            # Primordial parameters
            'n_s': n_s,
            'ln10^{10}A_s': ln_A_s_1e10,
            'k_pivot': 0.05,  # Mpc^-1
            
            # Reionization
            'tau_reio': tau_reio,
            
            # Radiation parameters
            'T_cmb': T_cmb,
            'N_ur': N_eff,
            'N_ncdm': 1,
            'm_ncdm': 0.06,  # eV
            
            # Output requests
            'output': 'tCl,pCl,lCl,mPk',
            'l_max_scalars': 2500,
            'P_k_max_h/Mpc': 10.0,
            
            # Precision parameters
            'accurate_lensing': 1,
            'gauge': 'synchronous',
            'k_per_decade_for_pk': 50,
            'k_per_decade_for_bao': 200,
            
            # Numerical precision
            'perturb_integration_stepsize': 0.01,
            'tol_background_integration': 1.e-2,
            'tol_perturb_integration': 1.e-6,
            'tol_thermo_integration': 1.e-5
        }
        
        return class_params
    
    def class_to_hicosmo_params(self, class_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert CLASS parameters to HIcosmo format.
        
        Parameters
        ----------
        class_params : dict
            CLASS parameter dictionary
            
        Returns
        -------
        dict
            HIcosmo parameter dictionary
        """
        # Extract CLASS parameters
        h = class_params.get('h', 0.674)
        Omega_b = class_params.get('Omega_b', 0.049)
        Omega_cdm = class_params.get('Omega_cdm', 0.266)
        Omega_k = class_params.get('Omega_k', 0.0)
        n_s = class_params.get('n_s', 0.965)
        ln_A_s_1e10 = class_params.get('ln10^{10}A_s', 3.044)
        tau_reio = class_params.get('tau_reio', 0.054)
        T_cmb = class_params.get('T_cmb', 2.7255)
        N_eff = class_params.get('N_ur', 3.046)
        
        # Convert to HIcosmo format
        H0 = h * 100.0
        Omega_m = Omega_b + Omega_cdm
        
        hicosmo_params = {
            'H0': H0,
            'h': h,
            'Omega_m': Omega_m,
            'Omega_b': Omega_b, 
            'Omega_k': Omega_k,
            'n_s': n_s,
            'ln_A_s_1e10': ln_A_s_1e10,
            'tau_reio': tau_reio,
            'T_cmb': T_cmb,
            'N_eff': N_eff
        }
        
        return hicosmo_params
    
    # ==================== CLASS Calculations ====================
    
    def run_class_calculation(self) -> Dict[str, Any]:
        """
        Run CLASS calculation with HIcosmo parameters.
        
        Returns
        -------
        dict
            CLASS results dictionary
        """
        if not self.class_available:
            raise RuntimeError("CLASS not available")
        
        from classy import Class
        
        # Convert parameters
        class_params = self.hicosmo_to_class_params()
        
        # Initialize CLASS
        cosmo = Class()
        cosmo.set(class_params)
        cosmo.compute()
        
        class_results = {}
        
        # Background quantities
        z_bg = np.linspace(0, 5, 100)
        class_results['background'] = {
            'z': z_bg,
            'H': np.array([cosmo.Hubble(z) * 299792.458 for z in z_bg]),  # Convert to km/s/Mpc
            'DA': np.array([cosmo.angular_distance(z) for z in z_bg]),    # Mpc
            'rs_drag': cosmo.rs_drag()  # Sound horizon at drag epoch
        }
        
        # Matter power spectrum
        k_h = np.logspace(-4, 1, 200)  # h/Mpc
        z_pk = [0.0, 0.5, 1.0, 2.0]
        pk_array = np.zeros((len(z_pk), len(k_h)))
        
        for i, z in enumerate(z_pk):
            for j, k in enumerate(k_h):
                pk_array[i, j] = cosmo.pk(k, z)
        
        class_results['matter_power'] = {
            'k_h': k_h,
            'z': z_pk, 
            'P_k': pk_array
        }
        
        # CMB power spectra
        l_max = min(2500, cosmo.l_max_scalars)
        cls = cosmo.lensed_cl(l_max)
        l_cmb = cls['ell'][2:]  # Start from l=2
        
        # Convert to μK² units
        T_cmb_muK = cosmo.T_cmb() * 1e6
        conversion_factor = T_cmb_muK**2
        
        class_results['cmb_spectra'] = {
            'ell': l_cmb,
            'TT': cls['tt'][2:] * conversion_factor,
            'EE': cls['ee'][2:] * conversion_factor,
            'BB': cls['bb'][2:] * conversion_factor,
            'TE': cls['te'][2:] * conversion_factor
        }
        
        # Lensing potential (if available)
        if 'pp' in cls:
            # Convert phi-phi to potential units
            l_lens = l_cmb
            C_l_phi_phi = cls['pp'][2:]
            class_results['lensing'] = {
                'ell': l_lens,
                'phi_phi': C_l_phi_phi
            }
        
        # Clean up
        cosmo.struct_cleanup()
        cosmo.empty()
        
        return class_results
    
    # ==================== Comparison Methods ====================
    
    def compare_background_evolution(self, z_array: jnp.ndarray = None) -> Dict[str, Any]:
        """
        Compare HIcosmo and CLASS background evolution.
        
        Parameters
        ----------
        z_array : jnp.ndarray, optional
            Redshift array for comparison
            
        Returns
        -------
        dict
            Comparison results
        """
        if z_array is None:
            z_array = jnp.linspace(0, 5, 50)
        
        # HIcosmo calculations
        hicosmo_H = self.background.H_z(z_array)
        hicosmo_DA = self.background.distances.angular_diameter_distance(z_array)
        
        comparison = {
            'z': z_array,
            'hicosmo_H': hicosmo_H,
            'hicosmo_DA': hicosmo_DA,
            'class_available': self.class_available
        }
        
        if self.class_available:
            # CLASS calculation
            class_results = self.run_class_calculation()
            
            # Interpolate CLASS results to HIcosmo grid
            class_H_interp = jnp.interp(z_array, class_results['background']['z'],
                                       class_results['background']['H'])
            class_DA_interp = jnp.interp(z_array, class_results['background']['z'],
                                        class_results['background']['DA'])
            
            comparison['class_H'] = class_H_interp
            comparison['class_DA'] = class_DA_interp
            
            # Relative differences
            comparison['H_diff_percent'] = 100 * (hicosmo_H - class_H_interp) / class_H_interp
            comparison['DA_diff_percent'] = 100 * (hicosmo_DA - class_DA_interp) / class_DA_interp
            
            # Statistics
            comparison['H_rms_error'] = float(jnp.sqrt(jnp.mean(comparison['H_diff_percent']**2)))
            comparison['DA_rms_error'] = float(jnp.sqrt(jnp.mean(comparison['DA_diff_percent']**2)))
            
            # Tolerance check
            comparison['H_within_tolerance'] = comparison['H_rms_error'] < self.tolerances['background'] * 100
            comparison['DA_within_tolerance'] = comparison['DA_rms_error'] < self.tolerances['background'] * 100
        
        return comparison
    
    def compare_matter_power_spectrum(self, 
                                    k_array: jnp.ndarray = None,
                                    z_array: jnp.ndarray = None) -> Dict[str, Any]:
        """
        Compare HIcosmo and CLASS matter power spectra.
        
        Parameters
        ----------
        k_array : jnp.ndarray, optional
            k values in h/Mpc
        z_array : jnp.ndarray, optional
            Redshift array
            
        Returns
        -------
        dict
            Comparison results
        """
        if k_array is None:
            k_array = jnp.logspace(-3, 1, 100)
        if z_array is None:
            z_array = jnp.array([0.0, 0.5, 1.0])
        
        # HIcosmo calculations
        hicosmo_pk = {}
        for z in z_array:
            hicosmo_pk[f'z_{z:.1f}'] = self.linear_power.linear_power_spectrum(k_array, z)
        
        comparison = {
            'k_h': k_array,
            'z': z_array,
            'hicosmo_pk': hicosmo_pk,
            'class_available': self.class_available
        }
        
        if self.class_available:
            # CLASS calculation
            class_results = self.run_class_calculation()
            
            # Interpolate CLASS results
            class_pk = {}
            for i, z in enumerate(z_array):
                if i < len(class_results['matter_power']['z']):
                    pk_interp = jnp.interp(k_array, class_results['matter_power']['k_h'],
                                          class_results['matter_power']['P_k'][i, :])
                    class_pk[f'z_{z:.1f}'] = pk_interp
            
            comparison['class_pk'] = class_pk
            
            # Compute differences
            pk_differences = {}
            for z_key in hicosmo_pk.keys():
                if z_key in class_pk:
                    diff = 100 * (hicosmo_pk[z_key] - class_pk[z_key]) / class_pk[z_key]
                    pk_differences[z_key] = diff
                    
                    rms_error = float(jnp.sqrt(jnp.mean(diff**2)))
                    pk_differences[f'{z_key}_rms'] = rms_error
                    pk_differences[f'{z_key}_within_tol'] = rms_error < self.tolerances['power_spectrum'] * 100
            
            comparison['pk_differences'] = pk_differences
        
        return comparison
    
    def compare_cmb_spectra(self, l_max: int = 2500) -> Dict[str, Any]:
        """
        Compare HIcosmo and CLASS CMB power spectra.
        
        Parameters
        ----------
        l_max : int
            Maximum multipole
            
        Returns
        -------
        dict
            Comparison results
        """
        if self.temperature_cl is None:
            raise ValueError("Temperature power spectrum calculator not available")
        
        # Multipole array
        l_array = jnp.arange(2, min(l_max + 1, 2501))
        
        # HIcosmo calculation
        hicosmo_TT = self.temperature_cl.temperature_power_spectrum(l_array)
        
        comparison = {
            'ell': l_array,
            'hicosmo_TT': hicosmo_TT,
            'class_available': self.class_available
        }
        
        if self.class_available:
            # CLASS calculation
            class_results = self.run_class_calculation()
            
            # Interpolate to HIcosmo l grid
            class_TT = jnp.interp(l_array, class_results['cmb_spectra']['ell'],
                                 class_results['cmb_spectra']['TT'])
            comparison['class_TT'] = class_TT
            
            # Temperature comparison
            TT_diff = 100 * (hicosmo_TT - class_TT) / class_TT
            comparison['TT_diff_percent'] = TT_diff
            comparison['TT_rms_error'] = float(jnp.sqrt(jnp.mean(TT_diff**2)))
            comparison['TT_within_tolerance'] = comparison['TT_rms_error'] < self.tolerances['cmb_temperature'] * 100
        
        return comparison
    
    # ==================== Validation Report ====================
    
    def generate_validation_report(self) -> str:
        """
        Generate comprehensive validation report.
        
        Returns
        -------
        str
            Formatted validation report
        """
        lines = [
            "HIcosmo-CLASS Validation Report",
            "=" * 35,
            f"CLASS Available: {self.class_available}",
            ""
        ]
        
        if not self.class_available:
            lines.extend([
                "CLASS not available for comparison.",
                "Install CLASS Python wrapper to enable validation:",
                "  pip install classy",
                ""
            ])
            return "\n".join(lines)
        
        # Parameter conversion
        class_params = self.hicosmo_to_class_params()
        lines.extend([
            "Parameter Conversion:",
            f"  h = {class_params['h']:.4f}",
            f"  Ωᵦ = {class_params['Omega_b']:.4f}",
            f"  Ωᶜᵈᵐ = {class_params['Omega_cdm']:.4f}",
            f"  τᵣₑᵢₒ = {class_params['tau_reio']:.4f}",
            f"  nₛ = {class_params['n_s']:.4f}",
            f"  ln(10¹⁰Aₛ) = {class_params['ln10^{10}A_s']:.3f}",
            ""
        ])
        
        # Background comparison
        bg_comparison = self.compare_background_evolution()
        lines.extend([
            "Background Evolution Comparison:",
            f"  H(z) RMS error: {bg_comparison['H_rms_error']:.3f}%",
            f"  D_A(z) RMS error: {bg_comparison['DA_rms_error']:.3f}%",
            f"  Within tolerance: {bg_comparison['H_within_tolerance'] and bg_comparison['DA_within_tolerance']}",
            ""
        ])
        
        # Power spectrum comparison
        pk_comparison = self.compare_matter_power_spectrum()
        lines.append("Matter Power Spectrum Comparison:")
        for z_key, z_val in zip(['z_0.0', 'z_0.5', 'z_1.0'], [0.0, 0.5, 1.0]):
            if f'{z_key}_rms' in pk_comparison.get('pk_differences', {}):
                rms = pk_comparison['pk_differences'][f'{z_key}_rms']
                within_tol = pk_comparison['pk_differences'][f'{z_key}_within_tol']
                lines.append(f"  P(k,z={z_val}) RMS error: {rms:.3f}% ({'✓' if within_tol else '✗'})")
        lines.append("")
        
        # CMB comparison
        if self.temperature_cl is not None:
            cmb_comparison = self.compare_cmb_spectra()
            lines.extend([
                "CMB Power Spectrum Comparison:",
                f"  C_l^TT RMS error: {cmb_comparison['TT_rms_error']:.3f}%",
                f"  Within tolerance: {cmb_comparison['TT_within_tolerance']}",
                ""
            ])
        
        # Summary
        lines.extend([
            "Validation Summary:",
            "  HIcosmo calculations are compared against CLASS",
            "  using identical cosmological parameters.",
            "  Agreement within tolerances validates the",
            "  accuracy of HIcosmo implementations.",
        ])
        
        return "\n".join(lines)
    
    # ==================== Utility Methods ====================
    
    def export_class_ini_file(self, filename: str = "hicosmo_class.ini"):
        """
        Export parameters in CLASS .ini file format.
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        class_params = self.hicosmo_to_class_params()
        
        ini_content = [
            "# CLASS parameters exported from HIcosmo",
            "# Generated automatically",
            "",
            "# Cosmological parameters",
            f"h = {class_params['h']:.6f}",
            f"Omega_b = {class_params['Omega_b']:.6f}",
            f"Omega_cdm = {class_params['Omega_cdm']:.6f}",
            f"Omega_k = {class_params['Omega_k']:.6f}",
            f"tau_reio = {class_params['tau_reio']:.4f}",
            "",
            "# Primordial parameters", 
            f"n_s = {class_params['n_s']:.4f}",
            f"ln10^{{10}}A_s = {class_params['ln10^{10}A_s']:.3f}",
            f"k_pivot = {class_params['k_pivot']:.3f}",
            "",
            "# Radiation and neutrinos",
            f"T_cmb = {class_params['T_cmb']:.4f}",
            f"N_ur = {class_params['N_ur']:.3f}",
            f"N_ncdm = {class_params['N_ncdm']}",
            f"m_ncdm = {class_params['m_ncdm']:.3f}",
            "",
            "# Output and precision",
            f"output = {class_params['output']}",
            f"l_max_scalars = {class_params['l_max_scalars']}",
            f"P_k_max_h/Mpc = {class_params['P_k_max_h/Mpc']:.1f}",
            f"gauge = {class_params['gauge']}",
            "",
            f"accurate_lensing = {class_params['accurate_lensing']}",
            f"k_per_decade_for_pk = {class_params['k_per_decade_for_pk']}",
            f"k_per_decade_for_bao = {class_params['k_per_decade_for_bao']}",
        ]
        
        with open(filename, 'w') as f:
            f.write('\n'.join(ini_content))
        
        print(f"CLASS parameter file saved as: {filename}")
    
    def set_tolerances(self, **tolerances):
        """Set comparison tolerances."""
        for key, value in tolerances.items():
            if key in self.tolerances:
                self.tolerances[key] = value
            else:
                warnings.warn(f"Unknown tolerance key: {key}")