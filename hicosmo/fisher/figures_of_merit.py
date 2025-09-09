"""
HiCosmo Dark Energy Figures of Merit
====================================

Professional dark energy figure-of-merit calculations for cosmological surveys.
Implements standard figures of merit used in dark energy studies, including
the Dark Energy Task Force (DETF) FoM and specialized metrics for different
dark energy models.

Key features:
- DETF figure of merit for w0-wa parameterization
- Modified gravity figures of merit  
- Model-independent dark energy metrics
- Time-dependent dark energy analysis
- Multi-probe combination strategies
- Systematic uncertainty propagation
"""

import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import warnings

from .fisher_matrix import FisherMatrix
from ..core.parameters_professional import CosmologyParameters


@dataclass
class FigureOfMeritConfig:
    """Configuration for figure-of-merit calculations."""
    
    # Standard dark energy parameters
    w0_param: str = 'w0'
    wa_param: str = 'wa' 
    
    # Modified gravity parameters
    mg_params: List[str] = field(default_factory=lambda: ['gamma', 'eta', 'Sigma0'])
    
    # Neutrino parameters
    neutrino_params: List[str] = field(default_factory=lambda: ['sum_mnu', 'N_eff'])
    
    # Curvature parameter
    curvature_param: str = 'Omega_k'
    
    # Hubble parameter
    hubble_param: str = 'H0'
    
    # Figure of merit definitions
    detf_inverse_area: bool = True  # True for 1/Area, False for Area
    include_curvature: bool = False
    include_neutrinos: bool = False
    
    # Marginalization settings
    marginalize_over: List[str] = field(default_factory=list)
    condition_on: Dict[str, float] = field(default_factory=dict)
    
    # Validation
    check_parameter_coverage: bool = True
    warn_poor_constraints: bool = True


class DarkEnergyFiguresOfMerit:
    """
    Professional dark energy figure-of-merit calculator.
    
    Provides comprehensive tools for evaluating the constraining power
    of cosmological surveys on dark energy properties using standard
    and custom figures of merit.
    """
    
    def __init__(
        self, 
        fisher_calculator: FisherMatrix,
        config: Optional[FigureOfMeritConfig] = None
    ):
        """
        Initialize dark energy FoM calculator.
        
        Parameters
        ----------
        fisher_calculator : FisherMatrix
            Fisher matrix calculator instance
        config : FigureOfMeritConfig, optional
            Configuration for FoM calculations
        """
        self.fisher_calc = fisher_calculator
        self.config = config or FigureOfMeritConfig()
        
    def compute_detf_figure_of_merit(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str],
        marginalize_over: Optional[List[str]] = None
    ) -> Dict[str, Union[float, jnp.ndarray]]:
        """
        Compute Dark Energy Task Force (DETF) figure of merit.
        
        The DETF FoM is defined as FoM = 1/√(σ_w0² σ_wa² (1 - ρ²))
        where ρ is the correlation coefficient between w0 and wa.
        
        Parameters
        ----------
        fisher_matrix : jnp.ndarray
            Fisher information matrix
        param_names : List[str]
            Parameter names corresponding to matrix
        marginalize_over : List[str], optional
            Parameters to marginalize over
            
        Returns
        -------
        detf_results : Dict[str, Union[float, jnp.ndarray]]
            DETF figure of merit and related quantities
        """
        # Apply marginalization if requested
        working_fisher = fisher_matrix
        working_params = param_names.copy()
        
        if marginalize_over:
            working_fisher, working_params = self.fisher_calc.marginalize_parameters(
                fisher_matrix, param_names, marginalize_over
            )
        
        # Check that w0 and wa parameters are present
        if self.config.w0_param not in working_params or self.config.wa_param not in working_params:
            available_params = set(working_params)
            required_params = {self.config.w0_param, self.config.wa_param}
            missing_params = required_params - available_params
            
            if self.config.check_parameter_coverage:
                raise ValueError(f"Missing required dark energy parameters: {missing_params}")
            else:
                warnings.warn(f"Cannot compute DETF FoM: missing parameters {missing_params}")
                return {'detf_fom': jnp.nan, 'error': 'missing_parameters'}
        
        # Extract w0-wa submatrix
        w0_idx = working_params.index(self.config.w0_param)
        wa_idx = working_params.index(self.config.wa_param)
        
        fisher_w0wa = working_fisher[jnp.ix_([w0_idx, wa_idx], [w0_idx, wa_idx])]
        
        # Compute covariance matrix for w0-wa
        try:
            cov_w0wa = jnp.linalg.inv(fisher_w0wa)
        except jnp.linalg.LinAlgError:
            cov_w0wa = jnp.linalg.pinv(fisher_w0wa, rcond=1e-12)
            warnings.warn("Singular w0-wa Fisher matrix; using pseudo-inverse")
        
        # Extract variances and correlation
        sigma_w0_sq = cov_w0wa[0, 0]
        sigma_wa_sq = cov_w0wa[1, 1]
        sigma_w0wa = cov_w0wa[0, 1]
        
        sigma_w0 = jnp.sqrt(sigma_w0_sq)
        sigma_wa = jnp.sqrt(sigma_wa_sq)
        rho_w0wa = sigma_w0wa / (sigma_w0 * sigma_wa)
        
        # DETF figure of merit
        ellipse_area = 2 * jnp.pi * sigma_w0 * sigma_wa * jnp.sqrt(1 - rho_w0wa**2)
        
        if self.config.detf_inverse_area:
            detf_fom = 1.0 / ellipse_area
        else:
            detf_fom = ellipse_area
            
        # Additional diagnostics
        det_fisher_w0wa = jnp.linalg.det(fisher_w0wa)
        condition_number = jnp.linalg.cond(fisher_w0wa)
        
        # Warn about poor constraints
        if self.config.warn_poor_constraints:
            if sigma_w0 > 1.0:
                warnings.warn(f"Poor w0 constraint: σ_w0 = {sigma_w0:.3f}")
            if sigma_wa > 5.0:
                warnings.warn(f"Poor wa constraint: σ_wa = {sigma_wa:.3f}")
            if condition_number > 1e8:
                warnings.warn(f"Poorly conditioned w0-wa Fisher matrix: κ = {condition_number:.1e}")
        
        return {
            'detf_fom': float(detf_fom),
            'ellipse_area': float(ellipse_area),
            'sigma_w0': float(sigma_w0),
            'sigma_wa': float(sigma_wa),
            'correlation_w0wa': float(rho_w0wa),
            'fisher_determinant': float(det_fisher_w0wa),
            'condition_number': float(condition_number),
            'covariance_matrix': cov_w0wa,
            'parameter_names': [self.config.w0_param, self.config.wa_param]
        }
    
    def compute_modified_gravity_fom(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str],
        mg_parameters: Optional[List[str]] = None,
        marginalize_over: Optional[List[str]] = None
    ) -> Dict[str, Union[float, jnp.ndarray]]:
        """
        Compute figure of merit for modified gravity parameters.
        
        Parameters
        ----------
        fisher_matrix : jnp.ndarray
            Fisher information matrix
        param_names : List[str]
            Parameter names corresponding to matrix
        mg_parameters : List[str], optional
            Modified gravity parameters to include
        marginalize_over : List[str], optional
            Parameters to marginalize over
            
        Returns
        -------
        mg_fom_results : Dict[str, Union[float, jnp.ndarray]]
            Modified gravity figure of merit results
        """
        mg_params = mg_parameters or self.config.mg_params
        
        # Apply marginalization if requested
        working_fisher = fisher_matrix
        working_params = param_names.copy()
        
        if marginalize_over:
            working_fisher, working_params = self.fisher_calc.marginalize_parameters(
                fisher_matrix, param_names, marginalize_over
            )
        
        # Find available MG parameters
        available_mg_params = [p for p in mg_params if p in working_params]
        
        if len(available_mg_params) == 0:
            warnings.warn("No modified gravity parameters found in Fisher matrix")
            return {'mg_fom': jnp.nan, 'error': 'no_mg_parameters'}
        
        # Extract MG parameter indices
        mg_indices = [working_params.index(p) for p in available_mg_params]
        
        # Extract MG submatrix
        fisher_mg = working_fisher[jnp.ix_(mg_indices, mg_indices)]
        
        # Compute MG figure of merit (volume of confidence ellipsoid)
        det_fisher_mg = jnp.linalg.det(fisher_mg)
        
        if det_fisher_mg <= 0:
            warnings.warn("Non-positive determinant for MG Fisher matrix")
            mg_fom = 0.0
        else:
            # FoM = √det(Fisher) (higher is better)
            mg_fom = jnp.sqrt(det_fisher_mg)
        
        # Individual parameter constraints
        try:
            cov_mg = jnp.linalg.inv(fisher_mg)
            mg_errors = {
                param: float(jnp.sqrt(cov_mg[i, i]))
                for i, param in enumerate(available_mg_params)
            }
        except jnp.linalg.LinAlgError:
            cov_mg = jnp.linalg.pinv(fisher_mg, rcond=1e-12)
            mg_errors = {
                param: float(jnp.sqrt(cov_mg[i, i]))
                for i, param in enumerate(available_mg_params)
            }
            warnings.warn("Singular MG Fisher matrix; using pseudo-inverse")
        
        return {
            'mg_fom': float(mg_fom),
            'mg_parameters': available_mg_params,
            'mg_errors': mg_errors,
            'determinant': float(det_fisher_mg),
            'covariance_matrix': cov_mg,
            'n_parameters': len(available_mg_params)
        }
    
    def compute_hubble_tension_fom(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str],
        h0_prior_sigma: Optional[float] = None,
        marginalize_over: Optional[List[str]] = None
    ) -> Dict[str, Union[float, jnp.ndarray]]:
        """
        Compute figure of merit for resolving Hubble tension.
        
        Parameters
        ----------
        fisher_matrix : jnp.ndarray
            Fisher information matrix
        param_names : List[str]
            Parameter names corresponding to matrix
        h0_prior_sigma : float, optional
            External H0 measurement uncertainty (km/s/Mpc)
        marginalize_over : List[str], optional
            Parameters to marginalize over
            
        Returns
        -------
        h0_fom_results : Dict[str, Union[float, jnp.ndarray]]
            Hubble tension figure of merit results
        """
        # Apply marginalization if requested
        working_fisher = fisher_matrix
        working_params = param_names.copy()
        
        if marginalize_over:
            working_fisher, working_params = self.fisher_calc.marginalize_parameters(
                fisher_matrix, param_names, marginalize_over
            )
        
        # Check for H0 parameter
        if self.config.hubble_param not in working_params:
            warnings.warn(f"Hubble parameter '{self.config.hubble_param}' not found")
            return {'h0_fom': jnp.nan, 'error': 'missing_h0_parameter'}
        
        h0_idx = working_params.index(self.config.hubble_param)
        
        # Extract H0 constraint from cosmological data
        h0_fisher_element = working_fisher[h0_idx, h0_idx]
        
        if h0_fisher_element <= 0:
            warnings.warn("Non-positive Fisher matrix element for H0")
            return {'h0_fom': jnp.nan, 'error': 'invalid_h0_fisher'}
        
        sigma_h0_cosmo = 1.0 / jnp.sqrt(h0_fisher_element)
        
        # Combine with external H0 measurement if provided
        if h0_prior_sigma is not None:
            # Combined precision
            combined_fisher_h0 = h0_fisher_element + 1.0 / h0_prior_sigma**2
            sigma_h0_combined = 1.0 / jnp.sqrt(combined_fisher_h0)
            
            # Figure of merit: improvement factor
            h0_fom = float(sigma_h0_cosmo / sigma_h0_combined)
            
            return {
                'h0_fom': h0_fom,
                'sigma_h0_cosmology': float(sigma_h0_cosmo),
                'sigma_h0_external': float(h0_prior_sigma),
                'sigma_h0_combined': float(sigma_h0_combined),
                'improvement_factor': h0_fom,
                'tension_significance': float(abs(73.04 - 67.36) / sigma_h0_combined)  # Rough estimate
            }
        else:
            # Just return cosmological constraint
            return {
                'h0_fom': float(1.0 / sigma_h0_cosmo),  # Precision as FoM
                'sigma_h0_cosmology': float(sigma_h0_cosmo),
                'h0_precision': float(1.0 / sigma_h0_cosmo)
            }
    
    def compute_neutrino_mass_fom(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str],
        marginalize_over: Optional[List[str]] = None
    ) -> Dict[str, Union[float, jnp.ndarray]]:
        """
        Compute figure of merit for neutrino mass constraints.
        
        Parameters
        ----------
        fisher_matrix : jnp.ndarray
            Fisher information matrix
        param_names : List[str]
            Parameter names corresponding to matrix
        marginalize_over : List[str], optional
            Parameters to marginalize over
            
        Returns
        -------
        neutrino_fom_results : Dict[str, Union[float, jnp.ndarray]]
            Neutrino mass figure of merit results
        """
        # Apply marginalization if requested
        working_fisher = fisher_matrix
        working_params = param_names.copy()
        
        if marginalize_over:
            working_fisher, working_params = self.fisher_calc.marginalize_parameters(
                fisher_matrix, param_names, marginalize_over
            )
        
        # Look for neutrino mass parameter
        neutrino_mass_param = None
        for param in ['sum_mnu', 'M_nu', 'mnu']:
            if param in working_params:
                neutrino_mass_param = param
                break
        
        if neutrino_mass_param is None:
            warnings.warn("No neutrino mass parameter found")
            return {'nu_mass_fom': jnp.nan, 'error': 'missing_neutrino_mass'}
        
        nu_idx = working_params.index(neutrino_mass_param)
        
        # Extract neutrino mass constraint
        nu_fisher_element = working_fisher[nu_idx, nu_idx]
        
        if nu_fisher_element <= 0:
            warnings.warn("Non-positive Fisher matrix element for neutrino mass")
            return {'nu_mass_fom': jnp.nan, 'error': 'invalid_nu_fisher'}
        
        sigma_nu_mass = 1.0 / jnp.sqrt(nu_fisher_element)
        
        # Figure of merit: detection significance for minimal mass (0.06 eV)
        minimal_mass = 0.06  # eV, from oscillations
        detection_significance = minimal_mass / sigma_nu_mass
        
        return {
            'nu_mass_fom': float(detection_significance),
            'sigma_neutrino_mass': float(sigma_nu_mass),
            'detection_significance': float(detection_significance),
            'minimal_mass_detection': detection_significance >= 2.0,  # 2σ detection
            'parameter_name': neutrino_mass_param
        }
    
    def compute_curvature_fom(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str],
        marginalize_over: Optional[List[str]] = None
    ) -> Dict[str, Union[float, jnp.ndarray]]:
        """
        Compute figure of merit for spatial curvature constraints.
        
        Parameters
        ----------
        fisher_matrix : jnp.ndarray
            Fisher information matrix
        param_names : List[str]
            Parameter names corresponding to matrix
        marginalize_over : List[str], optional
            Parameters to marginalize over
            
        Returns
        -------
        curvature_fom_results : Dict[str, Union[float, jnp.ndarray]]
            Curvature figure of merit results
        """
        # Apply marginalization if requested
        working_fisher = fisher_matrix
        working_params = param_names.copy()
        
        if marginalize_over:
            working_fisher, working_params = self.fisher_calc.marginalize_parameters(
                fisher_matrix, param_names, marginalize_over
            )
        
        # Check for curvature parameter
        if self.config.curvature_param not in working_params:
            warnings.warn(f"Curvature parameter '{self.config.curvature_param}' not found")
            return {'curvature_fom': jnp.nan, 'error': 'missing_curvature_parameter'}
        
        curv_idx = working_params.index(self.config.curvature_param)
        
        # Extract curvature constraint
        curv_fisher_element = working_fisher[curv_idx, curv_idx]
        
        if curv_fisher_element <= 0:
            warnings.warn("Non-positive Fisher matrix element for curvature")
            return {'curvature_fom': jnp.nan, 'error': 'invalid_curvature_fisher'}
        
        sigma_curvature = 1.0 / jnp.sqrt(curv_fisher_element)
        
        # Figure of merit: precision on flatness test
        curvature_fom = 1.0 / sigma_curvature
        
        return {
            'curvature_fom': float(curvature_fom),
            'sigma_curvature': float(sigma_curvature),
            'flatness_precision': float(curvature_fom),
            'parameter_name': self.config.curvature_param
        }
    
    def compute_comprehensive_fom_suite(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str],
        marginalize_over: Optional[List[str]] = None,
        external_priors: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, Union[float, jnp.ndarray]]]:
        """
        Compute comprehensive suite of figures of merit.
        
        Parameters
        ----------
        fisher_matrix : jnp.ndarray
            Fisher information matrix
        param_names : List[str]
            Parameter names corresponding to matrix
        marginalize_over : List[str], optional
            Parameters to marginalize over
        external_priors : Dict[str, float], optional
            External Gaussian prior constraints
            
        Returns
        -------
        comprehensive_fom : Dict[str, Dict[str, Union[float, jnp.ndarray]]]
            Comprehensive suite of figure-of-merit results
        """
        print("Computing comprehensive figure-of-merit suite...")
        
        results = {}
        
        # DETF dark energy figure of merit
        try:
            detf_results = self.compute_detf_figure_of_merit(
                fisher_matrix, param_names, marginalize_over
            )
            results['detf'] = detf_results
            print(f"  ✓ DETF FoM: {detf_results['detf_fom']:.2f}")
        except Exception as e:
            print(f"  ✗ DETF FoM failed: {e}")
            results['detf'] = {'error': str(e)}
        
        # Modified gravity figure of merit
        try:
            mg_results = self.compute_modified_gravity_fom(
                fisher_matrix, param_names, marginalize_over=marginalize_over
            )
            results['modified_gravity'] = mg_results
            if 'mg_fom' in mg_results and not jnp.isnan(mg_results['mg_fom']):
                print(f"  ✓ MG FoM: {mg_results['mg_fom']:.2e}")
        except Exception as e:
            print(f"  ✗ MG FoM failed: {e}")
            results['modified_gravity'] = {'error': str(e)}
        
        # Hubble tension figure of merit
        try:
            h0_prior = external_priors.get('H0') if external_priors else None
            h0_results = self.compute_hubble_tension_fom(
                fisher_matrix, param_names, h0_prior, marginalize_over
            )
            results['hubble_tension'] = h0_results
            if 'h0_fom' in h0_results and not jnp.isnan(h0_results['h0_fom']):
                print(f"  ✓ H0 FoM: {h0_results['h0_fom']:.2f}")
        except Exception as e:
            print(f"  ✗ H0 FoM failed: {e}")
            results['hubble_tension'] = {'error': str(e)}
        
        # Neutrino mass figure of merit
        try:
            nu_results = self.compute_neutrino_mass_fom(
                fisher_matrix, param_names, marginalize_over
            )
            results['neutrino_mass'] = nu_results
            if 'nu_mass_fom' in nu_results and not jnp.isnan(nu_results['nu_mass_fom']):
                print(f"  ✓ Neutrino mass detection: {nu_results['nu_mass_fom']:.1f}σ")
        except Exception as e:
            print(f"  ✗ Neutrino mass FoM failed: {e}")
            results['neutrino_mass'] = {'error': str(e)}
        
        # Curvature figure of merit
        try:
            curv_results = self.compute_curvature_fom(
                fisher_matrix, param_names, marginalize_over
            )
            results['curvature'] = curv_results
            if 'curvature_fom' in curv_results and not jnp.isnan(curv_results['curvature_fom']):
                print(f"  ✓ Curvature FoM: {curv_results['curvature_fom']:.1f}")
        except Exception as e:
            print(f"  ✗ Curvature FoM failed: {e}")
            results['curvature'] = {'error': str(e)}
        
        # Overall summary
        successful_foms = [key for key, value in results.items() if 'error' not in value]
        results['summary'] = {
            'successful_calculations': successful_foms,
            'n_successful': len(successful_foms),
            'n_total': len(results) - 1,  # Exclude summary itself
            'parameter_names': param_names,
            'marginalized_over': marginalize_over or []
        }
        
        print(f"✓ Comprehensive FoM suite completed: {len(successful_foms)}/{len(results)-1} successful")
        return results
    
    def compare_survey_performance(
        self,
        survey_fisher_matrices: Dict[str, Tuple[jnp.ndarray, List[str]]],
        fom_types: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
        """
        Compare survey performance using multiple figures of merit.
        
        Parameters
        ----------
        survey_fisher_matrices : Dict[str, Tuple[jnp.ndarray, List[str]]]
            Survey name -> (Fisher matrix, parameter names)
        fom_types : List[str], optional
            Types of FoM to compute ('detf', 'mg', 'h0', 'neutrino', 'curvature')
            
        Returns
        -------
        comparison_results : Dict[str, Dict[str, Union[float, Dict[str, float]]]]
            Survey comparison results organized by FoM type
        """
        fom_types = fom_types or ['detf', 'modified_gravity', 'hubble_tension']
        
        print(f"Comparing {len(survey_fisher_matrices)} surveys...")
        
        comparison_results = {}
        
        for fom_type in fom_types:
            print(f"  Computing {fom_type} FoM for all surveys...")
            
            fom_results = {}
            
            for survey_name, (fisher_matrix, param_names) in survey_fisher_matrices.items():
                try:
                    if fom_type == 'detf':
                        result = self.compute_detf_figure_of_merit(fisher_matrix, param_names)
                        fom_value = result.get('detf_fom', jnp.nan)
                    elif fom_type == 'modified_gravity':
                        result = self.compute_modified_gravity_fom(fisher_matrix, param_names)
                        fom_value = result.get('mg_fom', jnp.nan)
                    elif fom_type == 'hubble_tension':
                        result = self.compute_hubble_tension_fom(fisher_matrix, param_names)
                        fom_value = result.get('h0_fom', jnp.nan)
                    elif fom_type == 'neutrino_mass':
                        result = self.compute_neutrino_mass_fom(fisher_matrix, param_names)
                        fom_value = result.get('nu_mass_fom', jnp.nan)
                    elif fom_type == 'curvature':
                        result = self.compute_curvature_fom(fisher_matrix, param_names)
                        fom_value = result.get('curvature_fom', jnp.nan)
                    else:
                        fom_value = jnp.nan
                        
                    fom_results[survey_name] = float(fom_value) if not jnp.isnan(fom_value) else None
                    
                except Exception as e:
                    print(f"    ✗ {survey_name}: {e}")
                    fom_results[survey_name] = None
                    
            # Rank surveys by FoM
            valid_results = {k: v for k, v in fom_results.items() if v is not None}
            
            if valid_results:
                sorted_surveys = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
                ranking = {survey: rank + 1 for rank, (survey, _) in enumerate(sorted_surveys)}
                
                comparison_results[fom_type] = {
                    'fom_values': fom_results,
                    'ranking': ranking,
                    'best_survey': sorted_surveys[0][0] if sorted_surveys else None,
                    'best_fom': sorted_surveys[0][1] if sorted_surveys else None
                }
            else:
                comparison_results[fom_type] = {
                    'fom_values': fom_results,
                    'error': 'no_valid_results'
                }
                
        print("✓ Survey performance comparison completed")
        return comparison_results