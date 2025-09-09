"""
CAMB Interface
==============

Professional interface with CAMB (Code for Anisotropies in the Microwave Background).
Provides parameter conversion, result comparison, and validation tools.

Key features:
- Parameter format conversion between HiCosmo and CAMB
- Result comparison and validation
- Performance benchmarking
- Error analysis and diagnostics

CAMB is the gold standard for CMB power spectrum calculations.
This interface ensures HiCosmo results are validated against CAMB.
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
from ..cmb.polarization_cl import PolarizationPowerSpectrum
from ..cmb.lensing_cl import CMBLensingCalculator
from ..powerspectrum.linear_power import LinearPowerSpectrum


class CAMBInterface:
    """
    Interface between HiCosmo and CAMB.
    
    Provides seamless conversion between parameter formats,
    comparison of results, and validation tools for ensuring
    HiCosmo calculations match CAMB standards.
    """
    
    def __init__(self, hicosmo_components: Dict[str, Any]):
        """
        Initialize CAMB interface.
        
        Parameters
        ----------
        hicosmo_components : dict
            Dictionary containing HiCosmo calculation components:
            - 'background': BackgroundEvolution
            - 'linear_power': LinearPowerSpectrum  
            - 'temperature_cl': TemperaturePowerSpectrum
            - 'polarization_cl': PolarizationPowerSpectrum
            - 'lensing_cl': CMBLensingCalculator
        """
        self.background = hicosmo_components.get('background')
        self.linear_power = hicosmo_components.get('linear_power')
        self.temperature_cl = hicosmo_components.get('temperature_cl')
        self.polarization_cl = hicosmo_components.get('polarization_cl')
        self.lensing_cl = hicosmo_components.get('lensing_cl')
        
        if self.background is None:
            raise ValueError("Background evolution component is required")
        
        self.params = self.background.model.params
        
        # CAMB availability check
        self.camb_available = self._check_camb_availability()
        
        # Comparison tolerances
        self.tolerances = {
            'background': 1e-3,      # 0.1% for H(z), D_A, etc.
            'power_spectrum': 5e-3,   # 0.5% for P(k)
            'cmb_temperature': 1e-2,  # 1% for C_l^TT
            'cmb_polarization': 2e-2, # 2% for C_l^EE, C_l^TE
            'cmb_lensing': 5e-2       # 5% for C_l^φφ
        }
    
    def _check_camb_availability(self) -> bool:
        """Check if CAMB is available for comparison."""
        try:
            import camb
            return True
        except ImportError:
            warnings.warn("CAMB not available. Comparison features disabled.")
            return False
    
    # ==================== Parameter Conversion ====================
    
    def hicosmo_to_camb_params(self) -> Dict[str, Any]:
        """
        Convert HiCosmo parameters to CAMB format.
        
        Returns
        -------
        dict
            CAMB-formatted parameters
        """
        # Get HiCosmo parameter values
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
        
        # Convert to CAMB format
        camb_params = {
            # Basic cosmological parameters
            'H0': H0,
            'ombh2': Omega_b * h**2,
            'omch2': (Omega_m - Omega_b) * h**2,
            'omk': Omega_k,
            'tau': tau_reio,
            
            # Primordial parameters
            'ns': n_s,
            'As': jnp.exp(ln_A_s_1e10) * 1e-10,
            'pivot_scalar': 0.05,  # k_pivot in Mpc^-1
            
            # Other parameters
            'TCMB': T_cmb,
            'nnu': N_eff,
            'num_massive_neutrinos': 1,
            'mnu': 0.06,  # Sum of neutrino masses in eV
            
            # Accuracy parameters
            'accuracy_boost': 1.0,
            'l_accuracy_boost': 1.0,
            'l_sample_boost': 1.0
        }
        
        return camb_params
    
    def camb_to_hicosmo_params(self, camb_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert CAMB parameters to HiCosmo format.
        
        Parameters
        ----------
        camb_params : dict
            CAMB parameter dictionary
            
        Returns
        -------
        dict
            HiCosmo parameter dictionary
        """
        # Extract CAMB parameters
        H0 = camb_params.get('H0', 67.4)
        ombh2 = camb_params.get('ombh2', 0.022)
        omch2 = camb_params.get('omch2', 0.12)
        omk = camb_params.get('omk', 0.0)
        tau = camb_params.get('tau', 0.054)
        ns = camb_params.get('ns', 0.965)
        As = camb_params.get('As', 2.1e-9)
        TCMB = camb_params.get('TCMB', 2.7255)
        nnu = camb_params.get('nnu', 3.046)
        
        # Convert to HiCosmo format
        h = H0 / 100.0
        Omega_b = ombh2 / h**2
        Omega_cdm = omch2 / h**2
        Omega_m = Omega_b + Omega_cdm
        ln_A_s_1e10 = jnp.log(As * 1e10)
        
        hicosmo_params = {
            'H0': H0,
            'h': h,
            'Omega_m': Omega_m,
            'Omega_b': Omega_b,
            'Omega_k': omk,
            'n_s': ns,
            'ln_A_s_1e10': ln_A_s_1e10,
            'tau_reio': tau,
            'T_cmb': TCMB,
            'N_eff': nnu
        }
        
        return hicosmo_params
    
    # ==================== CAMB Calculations ====================
    
    def run_camb_calculation(self, 
                           lmax: int = 2500,
                           kmax: float = 10.0,
                           z_max: float = 10.0) -> Dict[str, Any]:
        """
        Run CAMB calculation with HiCosmo parameters.
        
        Parameters
        ----------
        lmax : int
            Maximum multipole for CMB spectra
        kmax : float
            Maximum k for matter power spectrum (h/Mpc)
        z_max : float
            Maximum redshift for background evolution
            
        Returns
        -------
        dict
            CAMB results dictionary
        """
        if not self.camb_available:
            raise RuntimeError("CAMB not available")
        
        import camb
        
        # Convert parameters
        camb_params_dict = self.hicosmo_to_camb_params()
        
        # Set up CAMB parameters
        pars = camb.CAMBparams()
        
        # Set cosmological parameters
        pars.set_cosmology(
            H0=camb_params_dict['H0'],
            ombh2=camb_params_dict['ombh2'],
            omch2=camb_params_dict['omch2'],
            omk=camb_params_dict['omk'],
            tau=camb_params_dict['tau'],
            TCMB=camb_params_dict['TCMB'],
            nnu=camb_params_dict['nnu'],
            num_massive_neutrinos=camb_params_dict['num_massive_neutrinos'],
            mnu=camb_params_dict['mnu']
        )
        
        # Set initial power spectrum
        pars.InitPower.set_params(
            As=camb_params_dict['As'],
            ns=camb_params_dict['ns'],
            pivot_scalar=camb_params_dict['pivot_scalar']
        )
        
        # Set accuracy
        pars.set_accuracy(
            AccuracyBoost=camb_params_dict['accuracy_boost'],
            lSampleBoost=camb_params_dict['l_sample_boost'],
            lAccuracyBoost=camb_params_dict['l_accuracy_boost']
        )
        
        # Set output options
        pars.set_for_lmax(lmax, lens_potential_accuracy=1)
        
        # Set matter power spectrum
        pars.set_matter_power(redshifts=[0., 0.5, 1.0, 2.0], kmax=kmax)
        
        # Run CAMB
        results = camb.get_results(pars)
        
        # Extract results
        camb_results = {}
        
        # CMB power spectra
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        camb_results['cmb_spectra'] = {
            'ell': powers['ell'],
            'TT': powers['total'][:, 0],  # C_l^TT
            'EE': powers['total'][:, 1],  # C_l^EE
            'BB': powers['total'][:, 2],  # C_l^BB
            'TE': powers['total'][:, 3]   # C_l^TE
        }
        
        # Lensing power spectrum
        if hasattr(powers, 'lens_potential'):
            camb_results['lensing'] = {
                'ell': powers['ell'],
                'phi_phi': powers['lens_potential'][:, 0]
            }
        
        # Matter power spectrum
        k_h, z_pk, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints=200)
        camb_results['matter_power'] = {
            'k_h': k_h,
            'z': z_pk,
            'P_k': pk
        }
        
        # Background evolution
        z_bg = np.linspace(0, z_max, 100)
        bg_results = results.get_background_time_evolution(z_bg, ['H', 'DA', 'rs_drag'])
        camb_results['background'] = {
            'z': z_bg,
            'H': bg_results[:, 0],      # H(z) in km/s/Mpc
            'DA': bg_results[:, 1],     # Angular diameter distance in Mpc
            'rs_drag': bg_results[:, 2] # Sound horizon at drag epoch in Mpc
        }
        
        return camb_results
    
    # ==================== Comparison and Validation ====================
    
    def compare_background_evolution(self, z_array: jnp.ndarray = None) -> Dict[str, Any]:
        """
        Compare HiCosmo and CAMB background evolution.
        
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
        
        # HiCosmo calculations
        hicosmo_H = self.background.H_z(z_array)
        hicosmo_DA = self.background.distances.angular_diameter_distance(z_array)
        
        comparison = {
            'z': z_array,
            'hicosmo_H': hicosmo_H,
            'hicosmo_DA': hicosmo_DA,
            'camb_available': self.camb_available
        }
        
        if self.camb_available:
            # CAMB calculation
            camb_results = self.run_camb_calculation(z_max=float(z_array.max()))
            
            # Interpolate CAMB results to HiCosmo redshift grid
            camb_H_interp = jnp.interp(z_array, camb_results['background']['z'], 
                                      camb_results['background']['H'])
            camb_DA_interp = jnp.interp(z_array, camb_results['background']['z'],
                                       camb_results['background']['DA'])
            
            # Add CAMB results
            comparison['camb_H'] = camb_H_interp
            comparison['camb_DA'] = camb_DA_interp
            
            # Compute relative differences
            comparison['H_diff_percent'] = 100 * (hicosmo_H - camb_H_interp) / camb_H_interp
            comparison['DA_diff_percent'] = 100 * (hicosmo_DA - camb_DA_interp) / camb_DA_interp
            
            # Summary statistics
            comparison['H_rms_error'] = float(jnp.sqrt(jnp.mean(comparison['H_diff_percent']**2)))
            comparison['DA_rms_error'] = float(jnp.sqrt(jnp.mean(comparison['DA_diff_percent']**2)))
            
            # Check if within tolerance
            comparison['H_within_tolerance'] = comparison['H_rms_error'] < self.tolerances['background'] * 100
            comparison['DA_within_tolerance'] = comparison['DA_rms_error'] < self.tolerances['background'] * 100
        
        return comparison
    
    def compare_matter_power_spectrum(self, 
                                    k_array: jnp.ndarray = None,
                                    z_array: jnp.ndarray = None) -> Dict[str, Any]:
        """
        Compare HiCosmo and CAMB matter power spectra.
        
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
        
        # HiCosmo calculations
        hicosmo_pk = {}
        for z in z_array:
            hicosmo_pk[f'z_{z:.1f}'] = self.linear_power.linear_power_spectrum(k_array, z)
        
        comparison = {
            'k_h': k_array,
            'z': z_array,
            'hicosmo_pk': hicosmo_pk,
            'camb_available': self.camb_available
        }
        
        if self.camb_available:
            # CAMB calculation
            camb_results = self.run_camb_calculation(kmax=float(k_array.max()))
            
            # Extract and interpolate CAMB power spectra
            camb_pk = {}
            for i, z in enumerate(z_array):
                if i < len(camb_results['matter_power']['z']):
                    # Interpolate to HiCosmo k grid
                    pk_interp = jnp.interp(k_array, camb_results['matter_power']['k_h'],
                                          camb_results['matter_power']['P_k'][i, :])
                    camb_pk[f'z_{z:.1f}'] = pk_interp
            
            comparison['camb_pk'] = camb_pk
            
            # Compute relative differences
            pk_differences = {}
            for z_key in hicosmo_pk.keys():
                if z_key in camb_pk:
                    diff = 100 * (hicosmo_pk[z_key] - camb_pk[z_key]) / camb_pk[z_key]
                    pk_differences[z_key] = diff
                    
                    # RMS error
                    rms_error = float(jnp.sqrt(jnp.mean(diff**2)))
                    pk_differences[f'{z_key}_rms'] = rms_error
                    pk_differences[f'{z_key}_within_tol'] = rms_error < self.tolerances['power_spectrum'] * 100
            
            comparison['pk_differences'] = pk_differences
        
        return comparison
    
    def compare_cmb_spectra(self, l_max: int = 2500) -> Dict[str, Any]:
        """
        Compare HiCosmo and CAMB CMB power spectra.
        
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
        
        # HiCosmo calculations
        hicosmo_TT = self.temperature_cl.temperature_power_spectrum(l_array)
        
        comparison = {
            'ell': l_array,
            'hicosmo_TT': hicosmo_TT,
            'camb_available': self.camb_available
        }
        
        # Add polarization if available
        if self.polarization_cl is not None:
            hicosmo_EE = self.polarization_cl.EE_power_spectrum(l_array)
            hicosmo_TE = self.polarization_cl.TE_cross_spectrum(l_array)
            comparison['hicosmo_EE'] = hicosmo_EE
            comparison['hicosmo_TE'] = hicosmo_TE
        
        if self.camb_available:
            # CAMB calculation
            camb_results = self.run_camb_calculation(lmax=l_max)
            
            # Interpolate to HiCosmo l grid
            camb_TT = jnp.interp(l_array, camb_results['cmb_spectra']['ell'],
                                camb_results['cmb_spectra']['TT'])
            comparison['camb_TT'] = camb_TT
            
            # Temperature comparison
            TT_diff = 100 * (hicosmo_TT - camb_TT) / camb_TT
            comparison['TT_diff_percent'] = TT_diff
            comparison['TT_rms_error'] = float(jnp.sqrt(jnp.mean(TT_diff**2)))
            comparison['TT_within_tolerance'] = comparison['TT_rms_error'] < self.tolerances['cmb_temperature'] * 100
            
            # Polarization comparison (if available)
            if self.polarization_cl is not None:
                camb_EE = jnp.interp(l_array, camb_results['cmb_spectra']['ell'],
                                    camb_results['cmb_spectra']['EE'])
                camb_TE = jnp.interp(l_array, camb_results['cmb_spectra']['ell'],
                                    camb_results['cmb_spectra']['TE'])
                
                comparison['camb_EE'] = camb_EE
                comparison['camb_TE'] = camb_TE
                
                # Polarization differences
                EE_diff = 100 * (hicosmo_EE - camb_EE) / camb_EE
                TE_diff = 100 * (hicosmo_TE - camb_TE) / jnp.abs(camb_TE)  # Abs for TE which can be negative
                
                comparison['EE_diff_percent'] = EE_diff
                comparison['TE_diff_percent'] = TE_diff
                comparison['EE_rms_error'] = float(jnp.sqrt(jnp.mean(EE_diff**2)))
                comparison['TE_rms_error'] = float(jnp.sqrt(jnp.mean(TE_diff**2)))
                comparison['EE_within_tolerance'] = comparison['EE_rms_error'] < self.tolerances['cmb_polarization'] * 100
                comparison['TE_within_tolerance'] = comparison['TE_rms_error'] < self.tolerances['cmb_polarization'] * 100
        
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
            "HiCosmo-CAMB Validation Report",
            "=" * 35,
            f"CAMB Available: {self.camb_available}",
            ""
        ]
        
        if not self.camb_available:
            lines.extend([
                "CAMB not available for comparison.",
                "Install CAMB to enable validation features:",
                "  pip install camb",
                ""
            ])
            return "\n".join(lines)
        
        # Parameter comparison
        camb_params = self.hicosmo_to_camb_params()
        lines.extend([
            "Parameter Conversion:",
            f"  H₀ = {camb_params['H0']:.2f} km/s/Mpc",
            f"  Ωᵦh² = {camb_params['ombh2']:.5f}",
            f"  Ωᶜh² = {camb_params['omch2']:.4f}",
            f"  τ = {camb_params['tau']:.4f}",
            f"  nₛ = {camb_params['ns']:.4f}",
            f"  Aₛ = {camb_params['As']:.2e}",
            ""
        ])
        
        # Background evolution comparison
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
            ])
            
            if 'EE_rms_error' in cmb_comparison:
                lines.extend([
                    f"  C_l^EE RMS error: {cmb_comparison['EE_rms_error']:.3f}%",
                    f"  C_l^TE RMS error: {cmb_comparison['TE_rms_error']:.3f}%",
                ])
            lines.append("")
        
        # Summary
        lines.extend([
            "Validation Summary:",
            "  HiCosmo calculations are compared against CAMB",
            "  using identical cosmological parameters.",
            "  Differences within expected tolerances indicate",
            "  successful validation of HiCosmo implementation.",
        ])
        
        return "\n".join(lines)
    
    # ==================== Utility Methods ====================
    
    def set_tolerances(self, **tolerances):
        """
        Set comparison tolerances.
        
        Parameters
        ----------
        **tolerances : float
            Tolerance values for different quantities
        """
        for key, value in tolerances.items():
            if key in self.tolerances:
                self.tolerances[key] = value
            else:
                warnings.warn(f"Unknown tolerance key: {key}")
    
    def export_camb_ini_file(self, filename: str = "hicosmo_params.ini"):
        """
        Export parameters in CAMB .ini file format.
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        camb_params = self.hicosmo_to_camb_params()
        
        ini_content = [
            "# CAMB parameters exported from HiCosmo",
            f"# Generated automatically",
            "",
            "# Cosmological parameters",
            f"hubble = {camb_params['H0']}",
            f"ombh2 = {camb_params['ombh2']:.6f}",
            f"omch2 = {camb_params['omch2']:.5f}",
            f"omk = {camb_params['omk']}",
            f"tau = {camb_params['tau']:.4f}",
            "",
            "# Primordial parameters",
            f"scalar_spectral_index(1) = {camb_params['ns']:.4f}",
            f"scalar_amp(1) = {camb_params['As']:.3e}",
            f"pivot_scalar = {camb_params['pivot_scalar']:.3f}",
            "",
            "# Other parameters",
            f"temp_cmb = {camb_params['TCMB']:.4f}",
            f"massless_neutrinos = {camb_params['nnu']:.3f}",
            "",
            "# Output settings",
            "get_scalar_cls = T",
            "get_tensor_cls = F", 
            "get_transfer = T",
            "l_max_scalar = 2500",
            "k_eta_max_scalar = 25000",
        ]
        
        with open(filename, 'w') as f:
            f.write('\n'.join(ini_content))
        
        print(f"CAMB parameter file saved as: {filename}")