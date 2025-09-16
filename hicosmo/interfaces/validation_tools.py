"""
Validation Tools Suite
======================

Comprehensive validation and testing tools for HIcosmo.
Provides systematic comparison, benchmarking, and quality assurance.

Key features:
- Cross-validation between CAMB and CLASS
- Numerical precision testing
- Performance benchmarking
- Regression testing framework
- Parameter space exploration
- Statistical validation methods

Ensures HIcosmo maintains accuracy and reliability across updates.
"""

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Union, Tuple, Dict, Optional, Any, List, Callable
import numpy as np
import warnings
import time
from dataclasses import dataclass

from .camb_interface import CAMBInterface
from .class_interface import CLASSInterface


@dataclass
class ValidationResult:
    """Data structure for validation results."""
    test_name: str
    passed: bool
    rms_error: float
    max_error: float
    tolerance: float
    details: Dict[str, Any]


class ValidationSuite:
    """
    Comprehensive validation suite for HIcosmo.
    
    Provides systematic testing and validation of all HIcosmo components
    against established codes and theoretical benchmarks.
    """
    
    def __init__(self, hicosmo_components: Dict[str, Any]):
        """
        Initialize validation suite.
        
        Parameters
        ----------
        hicosmo_components : dict
            Dictionary containing all HIcosmo components
        """
        self.components = hicosmo_components
        
        # Initialize interfaces
        self.camb_interface = None
        self.class_interface = None
        
        if 'background' in hicosmo_components:
            try:
                self.camb_interface = CAMBInterface(hicosmo_components)
            except Exception as e:
                warnings.warn(f"CAMB interface initialization failed: {e}")
            
            try:
                self.class_interface = CLASSInterface(hicosmo_components)
            except Exception as e:
                warnings.warn(f"CLASS interface initialization failed: {e}")
        
        # Validation results storage
        self.validation_results: List[ValidationResult] = []
        
        # Default tolerances
        self.tolerances = {
            'background_precision': 1e-3,
            'power_spectrum_precision': 5e-3,
            'cmb_precision': 1e-2,
            'cross_validation': 2e-2,
            'numerical_precision': 1e-6
        }
    
    # ==================== Core Validation Tests ====================
    
    def validate_background_evolution(self) -> List[ValidationResult]:
        """
        Validate background evolution calculations.
        
        Returns
        -------
        List[ValidationResult]
            Background validation results
        """
        results = []
        
        if 'background' not in self.components:
            return results
        
        background = self.components['background']
        
        # Test 1: Friedmann equation consistency
        result = self._test_friedmann_consistency(background)
        results.append(result)
        
        # Test 2: CAMB comparison
        if self.camb_interface and self.camb_interface.camb_available:
            result = self._test_camb_background_comparison(background)
            results.append(result)
        
        # Test 3: CLASS comparison
        if self.class_interface and self.class_interface.class_available:
            result = self._test_class_background_comparison(background)
            results.append(result)
        
        # Test 4: Numerical precision
        result = self._test_background_numerical_precision(background)
        results.append(result)
        
        return results
    
    def _test_friedmann_consistency(self, background) -> ValidationResult:
        """Test Friedmann equation consistency."""
        z_test = jnp.linspace(0, 5, 50)
        
        # Check Friedmann equation: H²(z) = H₀² E²(z)
        H_z = background.H_z(z_test)
        E_z = background.E_z(z_test)
        H0 = background.H0
        
        # Relative error
        friedmann_check = jnp.abs((H_z / H0)**2 - E_z**2) / E_z**2
        rms_error = float(jnp.sqrt(jnp.mean(friedmann_check**2)))
        max_error = float(jnp.max(friedmann_check))
        
        passed = rms_error < self.tolerances['numerical_precision']
        
        return ValidationResult(
            test_name="Friedmann Equation Consistency",
            passed=passed,
            rms_error=rms_error,
            max_error=max_error,
            tolerance=self.tolerances['numerical_precision'],
            details={'z_range': (0, 5), 'n_points': len(z_test)}
        )
    
    def _test_camb_background_comparison(self, background) -> ValidationResult:
        """Test background comparison with CAMB."""
        try:
            comparison = self.camb_interface.compare_background_evolution()
            
            # Combined RMS error for H(z) and D_A(z)
            H_rms = comparison['H_rms_error']
            DA_rms = comparison['DA_rms_error']
            combined_rms = jnp.sqrt((H_rms**2 + DA_rms**2) / 2)
            
            passed = (comparison['H_within_tolerance'] and 
                     comparison['DA_within_tolerance'])
            
            return ValidationResult(
                test_name="CAMB Background Comparison",
                passed=passed,
                rms_error=float(combined_rms),
                max_error=float(max(H_rms, DA_rms)),
                tolerance=self.tolerances['cross_validation'],
                details=comparison
            )
        except Exception as e:
            return ValidationResult(
                test_name="CAMB Background Comparison",
                passed=False,
                rms_error=float('inf'),
                max_error=float('inf'),
                tolerance=self.tolerances['cross_validation'],
                details={'error': str(e)}
            )
    
    def _test_class_background_comparison(self, background) -> ValidationResult:
        """Test background comparison with CLASS."""
        try:
            comparison = self.class_interface.compare_background_evolution()
            
            H_rms = comparison['H_rms_error']
            DA_rms = comparison['DA_rms_error']
            combined_rms = jnp.sqrt((H_rms**2 + DA_rms**2) / 2)
            
            passed = (comparison['H_within_tolerance'] and 
                     comparison['DA_within_tolerance'])
            
            return ValidationResult(
                test_name="CLASS Background Comparison",
                passed=passed,
                rms_error=float(combined_rms),
                max_error=float(max(H_rms, DA_rms)),
                tolerance=self.tolerances['cross_validation'],
                details=comparison
            )
        except Exception as e:
            return ValidationResult(
                test_name="CLASS Background Comparison", 
                passed=False,
                rms_error=float('inf'),
                max_error=float('inf'),
                tolerance=self.tolerances['cross_validation'],
                details={'error': str(e)}
            )
    
    def _test_background_numerical_precision(self, background) -> ValidationResult:
        """Test numerical precision of background calculations."""
        # Test at different redshifts with high precision
        z_test = jnp.array([0.1, 1.0, 5.0])
        
        # Compute with different integration points
        distances_low = background.distances.comoving_distance(z_test, n_points=100)
        distances_high = background.distances.comoving_distance(z_test, n_points=1000)
        
        # Relative difference
        precision_check = jnp.abs(distances_high - distances_low) / distances_high
        rms_error = float(jnp.sqrt(jnp.mean(precision_check**2)))
        max_error = float(jnp.max(precision_check))
        
        passed = rms_error < self.tolerances['numerical_precision']
        
        return ValidationResult(
            test_name="Background Numerical Precision",
            passed=passed,
            rms_error=rms_error,
            max_error=max_error,
            tolerance=self.tolerances['numerical_precision'],
            details={'test_redshifts': z_test.tolist()}
        )
    
    # ==================== Power Spectrum Validation ====================
    
    def validate_power_spectrum(self) -> List[ValidationResult]:
        """
        Validate power spectrum calculations.
        
        Returns
        -------
        List[ValidationResult]
            Power spectrum validation results
        """
        results = []
        
        if 'linear_power' not in self.components:
            return results
        
        linear_power = self.components['linear_power']
        
        # Test 1: Normalization consistency
        result = self._test_power_spectrum_normalization(linear_power)
        results.append(result)
        
        # Test 2: CAMB comparison
        if self.camb_interface and self.camb_interface.camb_available:
            result = self._test_camb_power_comparison(linear_power)
            results.append(result)
        
        # Test 3: CLASS comparison
        if self.class_interface and self.class_interface.class_available:
            result = self._test_class_power_comparison(linear_power)
            results.append(result)
        
        return results
    
    def _test_power_spectrum_normalization(self, linear_power) -> ValidationResult:
        """Test power spectrum normalization via σ8."""
        # Compute σ8 from power spectrum integral
        sigma8_computed = linear_power.sigma8(0)
        
        # Compare with target (from parameters or default)
        if hasattr(linear_power, 'sigma8_target'):
            sigma8_target = linear_power.sigma8_target
        else:
            sigma8_target = 0.811  # Default Planck value
        
        relative_error = jnp.abs(sigma8_computed - sigma8_target) / sigma8_target
        
        passed = relative_error < self.tolerances['power_spectrum_precision']
        
        return ValidationResult(
            test_name="Power Spectrum Normalization",
            passed=passed,
            rms_error=float(relative_error),
            max_error=float(relative_error),
            tolerance=self.tolerances['power_spectrum_precision'],
            details={
                'sigma8_computed': float(sigma8_computed),
                'sigma8_target': float(sigma8_target)
            }
        )
    
    def _test_camb_power_comparison(self, linear_power) -> ValidationResult:
        """Test power spectrum comparison with CAMB."""
        try:
            comparison = self.camb_interface.compare_matter_power_spectrum()
            
            if 'pk_differences' not in comparison:
                return ValidationResult(
                    test_name="CAMB Power Spectrum Comparison",
                    passed=False,
                    rms_error=float('inf'),
                    max_error=float('inf'),
                    tolerance=self.tolerances['cross_validation'],
                    details={'error': 'No comparison data available'}
                )
            
            # Average RMS error across redshifts
            rms_errors = [comparison['pk_differences'][key] 
                         for key in comparison['pk_differences'].keys() 
                         if key.endswith('_rms')]
            
            if rms_errors:
                avg_rms = float(jnp.mean(jnp.array(rms_errors)))
                max_rms = float(jnp.max(jnp.array(rms_errors)))
                
                # Check if all redshifts within tolerance
                passed = all(comparison['pk_differences'][key] 
                           for key in comparison['pk_differences'].keys()
                           if key.endswith('_within_tol'))
            else:
                avg_rms = float('inf')
                max_rms = float('inf')
                passed = False
            
            return ValidationResult(
                test_name="CAMB Power Spectrum Comparison",
                passed=passed,
                rms_error=avg_rms,
                max_error=max_rms,
                tolerance=self.tolerances['cross_validation'],
                details=comparison
            )
        except Exception as e:
            return ValidationResult(
                test_name="CAMB Power Spectrum Comparison",
                passed=False,
                rms_error=float('inf'),
                max_error=float('inf'),
                tolerance=self.tolerances['cross_validation'],
                details={'error': str(e)}
            )
    
    def _test_class_power_comparison(self, linear_power) -> ValidationResult:
        """Test power spectrum comparison with CLASS."""
        try:
            comparison = self.class_interface.compare_matter_power_spectrum()
            
            if 'pk_differences' not in comparison:
                return ValidationResult(
                    test_name="CLASS Power Spectrum Comparison",
                    passed=False,
                    rms_error=float('inf'),
                    max_error=float('inf'),
                    tolerance=self.tolerances['cross_validation'],
                    details={'error': 'No comparison data available'}
                )
            
            # Average RMS error across redshifts  
            rms_errors = [comparison['pk_differences'][key]
                         for key in comparison['pk_differences'].keys()
                         if key.endswith('_rms')]
            
            if rms_errors:
                avg_rms = float(jnp.mean(jnp.array(rms_errors)))
                max_rms = float(jnp.max(jnp.array(rms_errors)))
                
                passed = all(comparison['pk_differences'][key]
                           for key in comparison['pk_differences'].keys() 
                           if key.endswith('_within_tol'))
            else:
                avg_rms = float('inf')
                max_rms = float('inf')
                passed = False
            
            return ValidationResult(
                test_name="CLASS Power Spectrum Comparison",
                passed=passed,
                rms_error=avg_rms,
                max_error=max_rms,
                tolerance=self.tolerances['cross_validation'],
                details=comparison
            )
        except Exception as e:
            return ValidationResult(
                test_name="CLASS Power Spectrum Comparison",
                passed=False,
                rms_error=float('inf'),
                max_error=float('inf'),
                tolerance=self.tolerances['cross_validation'],
                details={'error': str(e)}
            )
    
    # ==================== CMB Validation ====================
    
    def validate_cmb_spectra(self) -> List[ValidationResult]:
        """
        Validate CMB power spectra calculations.
        
        Returns
        -------
        List[ValidationResult]
            CMB validation results
        """
        results = []
        
        if 'temperature_cl' not in self.components:
            return results
        
        temperature_cl = self.components['temperature_cl']
        
        # Test 1: Acoustic peak positions
        result = self._test_acoustic_peak_positions(temperature_cl)
        results.append(result)
        
        # Test 2: CAMB comparison
        if self.camb_interface and self.camb_interface.camb_available:
            result = self._test_camb_cmb_comparison(temperature_cl)
            results.append(result)
        
        # Test 3: CLASS comparison
        if self.class_interface and self.class_interface.class_available:
            result = self._test_class_cmb_comparison(temperature_cl)
            results.append(result)
        
        return results
    
    def _test_acoustic_peak_positions(self, temperature_cl) -> ValidationResult:
        """Test acoustic peak positions against theoretical predictions."""
        # Compute theoretical peak positions
        if hasattr(temperature_cl, 'acoustic_peak_positions'):
            predicted_peaks = temperature_cl.acoustic_peak_positions(3)
            
            # Expected peak positions (approximate)
            # These depend on sound horizon and angular diameter distance
            theta_s = temperature_cl.theta_s
            expected_peaks = jnp.array([1, 2, 3]) * jnp.pi / theta_s
            
            # Relative differences
            peak_errors = jnp.abs(predicted_peaks - expected_peaks) / expected_peaks
            rms_error = float(jnp.sqrt(jnp.mean(peak_errors**2)))
            max_error = float(jnp.max(peak_errors))
            
            passed = rms_error < self.tolerances['cmb_precision']
            
            return ValidationResult(
                test_name="Acoustic Peak Positions",
                passed=passed,
                rms_error=rms_error,
                max_error=max_error,
                tolerance=self.tolerances['cmb_precision'],
                details={
                    'predicted_peaks': predicted_peaks.tolist(),
                    'expected_peaks': expected_peaks.tolist(),
                    'sound_horizon_angle': float(theta_s)
                }
            )
        else:
            return ValidationResult(
                test_name="Acoustic Peak Positions",
                passed=False,
                rms_error=float('inf'),
                max_error=float('inf'), 
                tolerance=self.tolerances['cmb_precision'],
                details={'error': 'Peak position method not available'}
            )
    
    def _test_camb_cmb_comparison(self, temperature_cl) -> ValidationResult:
        """Test CMB comparison with CAMB."""
        try:
            comparison = self.camb_interface.compare_cmb_spectra()
            
            rms_error = comparison['TT_rms_error']
            passed = comparison['TT_within_tolerance']
            
            return ValidationResult(
                test_name="CAMB CMB Comparison",
                passed=passed,
                rms_error=rms_error,
                max_error=rms_error,  # Simplified
                tolerance=self.tolerances['cross_validation'],
                details=comparison
            )
        except Exception as e:
            return ValidationResult(
                test_name="CAMB CMB Comparison",
                passed=False,
                rms_error=float('inf'),
                max_error=float('inf'),
                tolerance=self.tolerances['cross_validation'],
                details={'error': str(e)}
            )
    
    def _test_class_cmb_comparison(self, temperature_cl) -> ValidationResult:
        """Test CMB comparison with CLASS."""
        try:
            comparison = self.class_interface.compare_cmb_spectra()
            
            rms_error = comparison['TT_rms_error']
            passed = comparison['TT_within_tolerance']
            
            return ValidationResult(
                test_name="CLASS CMB Comparison",
                passed=passed,
                rms_error=rms_error,
                max_error=rms_error,
                tolerance=self.tolerances['cross_validation'],
                details=comparison
            )
        except Exception as e:
            return ValidationResult(
                test_name="CLASS CMB Comparison",
                passed=False,
                rms_error=float('inf'),
                max_error=float('inf'),
                tolerance=self.tolerances['cross_validation'],
                details={'error': str(e)}
            )
    
    # ==================== Full Validation Suite ====================
    
    def run_full_validation(self) -> Dict[str, List[ValidationResult]]:
        """
        Run complete validation suite.
        
        Returns
        -------
        dict
            Complete validation results organized by category
        """
        print("Running HIcosmo Validation Suite...")
        print("=" * 40)
        
        all_results = {}
        
        # Background validation
        print("Validating background evolution...")
        bg_results = self.validate_background_evolution()
        all_results['background'] = bg_results
        self.validation_results.extend(bg_results)
        
        # Power spectrum validation
        print("Validating power spectrum calculations...")
        ps_results = self.validate_power_spectrum()
        all_results['power_spectrum'] = ps_results
        self.validation_results.extend(ps_results)
        
        # CMB validation
        print("Validating CMB calculations...")
        cmb_results = self.validate_cmb_spectra()
        all_results['cmb'] = cmb_results
        self.validation_results.extend(cmb_results)
        
        print("Validation complete.")
        return all_results
    
    def generate_validation_report(self, detailed: bool = True) -> str:
        """
        Generate comprehensive validation report.
        
        Parameters
        ----------
        detailed : bool
            Include detailed results for each test
            
        Returns
        -------
        str
            Formatted validation report
        """
        lines = [
            "HIcosmo Validation Report",
            "=" * 30,
            f"Total Tests: {len(self.validation_results)}",
        ]
        
        # Summary statistics
        passed_tests = sum(1 for r in self.validation_results if r.passed)
        failed_tests = len(self.validation_results) - passed_tests
        
        lines.extend([
            f"Passed: {passed_tests}",
            f"Failed: {failed_tests}",
            f"Success Rate: {100 * passed_tests / len(self.validation_results):.1f}%" 
                if self.validation_results else "No tests run",
            ""
        ])
        
        # External code availability
        camb_avail = self.camb_interface.camb_available if self.camb_interface else False
        class_avail = self.class_interface.class_available if self.class_interface else False
        
        lines.extend([
            "External Code Availability:",
            f"  CAMB: {'✓' if camb_avail else '✗'}",
            f"  CLASS: {'✓' if class_avail else '✗'}",
            ""
        ])
        
        if detailed and self.validation_results:
            lines.append("Detailed Test Results:")
            lines.append("-" * 25)
            
            for result in self.validation_results:
                status = "PASS" if result.passed else "FAIL"
                lines.extend([
                    f"{result.test_name}: {status}",
                    f"  RMS Error: {result.rms_error:.2e}",
                    f"  Max Error: {result.max_error:.2e}",
                    f"  Tolerance: {result.tolerance:.2e}",
                    ""
                ])
        
        # Recommendations
        lines.extend([
            "Recommendations:",
            "• Install CAMB and CLASS for full validation coverage" 
                if not (camb_avail and class_avail) else "• All validation tools available",
            "• Monitor failed tests for accuracy issues",
            "• Update tolerances if systematic differences persist",
        ])
        
        return "\n".join(lines)
    
    # ==================== Performance Benchmarking ====================
    
    def benchmark_performance(self, n_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark performance of HIcosmo components.
        
        Parameters
        ----------
        n_iterations : int
            Number of benchmark iterations
            
        Returns
        -------
        dict
            Performance benchmarks in seconds
        """
        benchmarks = {}
        
        # Background evolution benchmark
        if 'background' in self.components:
            bg = self.components['background']
            z_array = jnp.linspace(0, 5, 100)
            
            start_time = time.time()
            for _ in range(n_iterations):
                H_z = bg.H_z(z_array)
                D_A = bg.distances.angular_diameter_distance(z_array)
            end_time = time.time()
            
            benchmarks['background_evolution'] = (end_time - start_time) / n_iterations
        
        # Power spectrum benchmark
        if 'linear_power' in self.components:
            ps = self.components['linear_power']
            k_array = jnp.logspace(-3, 1, 100)
            
            start_time = time.time()
            for _ in range(n_iterations):
                P_k = ps.linear_power_spectrum(k_array, 0)
            end_time = time.time()
            
            benchmarks['power_spectrum'] = (end_time - start_time) / n_iterations
        
        # CMB benchmark
        if 'temperature_cl' in self.components:
            cmb = self.components['temperature_cl']
            l_array = jnp.arange(2, 501)  # Reduced range for speed
            
            start_time = time.time()
            for _ in range(n_iterations):
                C_l = cmb.temperature_power_spectrum(l_array)
            end_time = time.time()
            
            benchmarks['cmb_temperature'] = (end_time - start_time) / n_iterations
        
        return benchmarks
    
    # ==================== Utility Methods ====================
    
    def set_tolerances(self, **tolerances):
        """Set validation tolerances."""
        for key, value in tolerances.items():
            if key in self.tolerances:
                self.tolerances[key] = value
            else:
                warnings.warn(f"Unknown tolerance key: {key}")
    
    def clear_results(self):
        """Clear stored validation results."""
        self.validation_results.clear()
    
    def export_results(self, filename: str = "validation_results.json"):
        """Export validation results to JSON file."""
        import json
        
        # Convert results to serializable format
        results_dict = []
        for result in self.validation_results:
            result_dict = {
                'test_name': result.test_name,
                'passed': result.passed,
                'rms_error': float(result.rms_error),
                'max_error': float(result.max_error),
                'tolerance': float(result.tolerance),
                'details': result.details
            }
            results_dict.append(result_dict)
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Validation results exported to: {filename}")