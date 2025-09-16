"""
HIcosmo Parameter Forecasting
=============================

Professional cosmological parameter forecasting for future surveys and experiments.
Provides tools for survey optimization, experimental design, and constraint predictions
using Fisher matrix analysis.

Key features:
- Parameter constraint forecasting for planned surveys
- Survey optimization and figure-of-merit analysis
- Experimental design and resource allocation
- Bias and systematic uncertainty modeling
- Multi-probe synergy analysis
- Prior degradation and realistic modeling
"""

import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings

from .fisher_matrix import FisherMatrix, FisherMatrixConfig
from ..core.unified_parameters import CosmologicalParameters


@dataclass
class SurveySpecification:
    """Specification for a cosmological survey."""
    
    # Survey identification
    name: str
    survey_type: str  # 'photometric', 'spectroscopic', 'cmb', 'gravitational_waves'
    
    # Sky coverage and geometry
    sky_area: float  # deg²
    redshift_range: Tuple[float, float]
    redshift_bins: Optional[int] = None
    
    # Statistical properties  
    number_density: Optional[float] = None  # objects per deg² per unit redshift
    total_objects: Optional[int] = None
    
    # Observational uncertainties
    photometric_errors: Optional[Dict[str, float]] = None
    spectroscopic_errors: Optional[Dict[str, float]] = None
    systematic_floor: Optional[float] = None
    
    # Survey-specific parameters
    survey_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate survey specification."""
        if self.redshift_range[0] >= self.redshift_range[1]:
            raise ValueError("Invalid redshift range")
        if self.sky_area <= 0:
            raise ValueError("Sky area must be positive")


@dataclass  
class ExperimentalDesign:
    """Experimental design configuration for optimization."""
    
    # Optimization targets
    optimize_for: List[str] = field(default_factory=lambda: ['dark_energy_fom'])
    constraint_targets: Dict[str, float] = field(default_factory=dict)
    
    # Resource constraints
    total_budget: Optional[float] = None
    time_constraint: Optional[float] = None
    technology_limits: Dict[str, Any] = field(default_factory=dict)
    
    # Design variables
    variable_parameters: List[str] = field(default_factory=list)
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Systematic modeling
    include_systematics: bool = True
    systematic_budget: Optional[float] = None
    calibration_priors: Dict[str, float] = field(default_factory=dict)


class CosmologicalForecaster:
    """
    Professional cosmological parameter forecasting system.
    
    Provides comprehensive tools for predicting parameter constraints from
    future surveys, optimizing survey design, and analyzing multi-probe synergies.
    """
    
    def __init__(
        self,
        fisher_calculator: Optional[FisherMatrix] = None,
        config: Optional[FisherMatrixConfig] = None
    ):
        """
        Initialize cosmological forecaster.
        
        Parameters
        ----------
        fisher_calculator : FisherMatrix, optional
            Fisher matrix calculator instance
        config : FisherMatrixConfig, optional
            Configuration for Fisher calculations
        """
        self.fisher_calc = fisher_calculator or FisherMatrix(config)
        self.survey_cache = {}
        self.forecast_cache = {}
        
    def forecast_constraints(
        self,
        surveys: List[SurveySpecification],
        likelihood_funcs: List[Callable],
        parameters: CosmologicalParameters,
        fiducial_values: Dict[str, float],
        include_priors: Optional[Dict[str, float]] = None,
        systematic_budget: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Forecast parameter constraints for given surveys.
        
        Parameters
        ----------
        surveys : List[SurveySpecification]
            List of survey specifications
        likelihood_funcs : List[Callable]
            Corresponding likelihood functions for each survey
        parameters : CosmologicalParameters
            Parameter manager with specifications
        fiducial_values : Dict[str, float]
            Fiducial cosmological parameter values
        include_priors : Dict[str, float], optional
            Gaussian prior constraints (1σ values)
        systematic_budget : Dict[str, float], optional
            Systematic uncertainty budget per parameter
            
        Returns
        -------
        forecasts : Dict[str, Dict[str, float]]
            Nested dict: survey_name -> parameter_name -> 1σ_error
        """
        print(f"Forecasting constraints for {len(surveys)} surveys...")
        
        # Compute individual Fisher matrices
        fisher_matrices = []
        survey_names = []
        
        for survey, likelihood_func in zip(surveys, likelihood_funcs):
            print(f"  Computing Fisher matrix for {survey.name}...")
            
            # Apply systematic uncertainties to likelihood if specified
            if systematic_budget:
                likelihood_func = self._add_systematic_uncertainties(
                    likelihood_func, systematic_budget, parameters
                )
            
            fisher_matrix, param_names = self.fisher_calc.compute_fisher_matrix(
                likelihood_func, parameters, fiducial_values, survey.name
            )
            
            fisher_matrices.append((fisher_matrix, param_names))
            survey_names.append(survey.name)
            
        # Add prior constraints if specified
        if include_priors:
            prior_fisher = self._construct_prior_fisher(include_priors, param_names)
            fisher_matrices.append((prior_fisher, param_names))
            survey_names.append('priors')
            
        # Compute individual constraints
        forecasts = {}
        for (fisher_matrix, param_names), survey_name in zip(fisher_matrices, survey_names):
            errors = self.fisher_calc.compute_parameter_errors(fisher_matrix, param_names)
            forecasts[survey_name] = errors
            
        # Compute combined constraints
        if len(fisher_matrices) > 1:
            combined_fisher, param_names = self.fisher_calc.combine_fisher_matrices(
                fisher_matrices, survey_names
            )
            combined_errors = self.fisher_calc.compute_parameter_errors(
                combined_fisher, param_names
            )
            forecasts['combined'] = combined_errors
            
        print("✓ Parameter constraint forecasting completed")
        return forecasts
    
    def optimize_survey_design(
        self,
        base_survey: SurveySpecification,
        design_config: ExperimentalDesign,
        likelihood_func: Callable,
        parameters: CosmologicalParameters,
        fiducial_values: Dict[str, float],
        n_iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Optimize survey design parameters to maximize science return.
        
        Parameters
        ----------
        base_survey : SurveySpecification
            Base survey specification to optimize
        design_config : ExperimentalDesign
            Experimental design configuration
        likelihood_func : Callable
            Likelihood function for the survey
        parameters : CosmologicalParameters
            Parameter manager
        fiducial_values : Dict[str, float]
            Fiducial parameter values
        n_iterations : int, default=50
            Number of optimization iterations
            
        Returns
        -------
        optimization_result : Dict[str, Any]
            Optimization results with best parameters and performance
        """
        print("Optimizing survey design...")
        
        # Initialize optimization
        best_fom = -jnp.inf
        best_params = {}
        optimization_history = []
        
        # Simple grid search optimization (can be replaced with more sophisticated methods)
        for iteration in range(n_iterations):
            # Sample design parameters
            design_params = self._sample_design_parameters(design_config)
            
            # Create modified survey
            modified_survey = self._modify_survey(base_survey, design_params)
            
            # Compute Fisher matrix for modified survey
            fisher_matrix, param_names = self.fisher_calc.compute_fisher_matrix(
                likelihood_func, parameters, fiducial_values, f"opt_iter_{iteration}"
            )
            
            # Evaluate figure of merit
            fom = self._evaluate_figure_of_merit(
                fisher_matrix, param_names, design_config.optimize_for
            )
            
            optimization_history.append({
                'iteration': iteration,
                'design_params': design_params.copy(),
                'figure_of_merit': float(fom),
                'parameter_errors': self.fisher_calc.compute_parameter_errors(fisher_matrix, param_names)
            })
            
            # Update best result
            if fom > best_fom:
                best_fom = fom
                best_params = design_params.copy()
                
        print(f"✓ Survey optimization completed. Best FoM: {best_fom:.2f}")
        
        return {
            'best_parameters': best_params,
            'best_figure_of_merit': float(best_fom),
            'optimization_history': optimization_history,
            'n_iterations': n_iterations,
            'optimized_survey': self._modify_survey(base_survey, best_params)
        }
    
    def analyze_probe_synergies(
        self,
        surveys: List[SurveySpecification],
        likelihood_funcs: List[Callable],
        parameters: CosmologicalParameters,
        fiducial_values: Dict[str, float],
        target_parameters: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze synergies between different cosmological probes.
        
        Parameters
        ----------
        surveys : List[SurveySpecification]
            List of survey specifications
        likelihood_funcs : List[Callable]
            Corresponding likelihood functions
        parameters : CosmologicalParameters
            Parameter manager
        fiducial_values : Dict[str, float]
            Fiducial parameter values
        target_parameters : List[str], optional
            Parameters to focus synergy analysis on
            
        Returns
        -------
        synergy_analysis : Dict[str, Any]
            Comprehensive synergy analysis results
        """
        print("Analyzing multi-probe synergies...")
        
        # Compute individual Fisher matrices
        individual_fishers = []
        survey_names = [survey.name for survey in surveys]
        
        for survey, likelihood_func in zip(surveys, likelihood_funcs):
            fisher_matrix, param_names = self.fisher_calc.compute_fisher_matrix(
                likelihood_func, parameters, fiducial_values, survey.name
            )
            individual_fishers.append((fisher_matrix, param_names))
            
        # Compute combined Fisher matrix
        combined_fisher, param_names = self.fisher_calc.combine_fisher_matrices(
            individual_fishers, survey_names
        )
        
        # Focus on target parameters if specified
        if target_parameters:
            # Marginalize over non-target parameters
            marginalize_over = [p for p in param_names if p not in target_parameters]
            combined_fisher, param_names = self.fisher_calc.marginalize_parameters(
                combined_fisher, param_names, marginalize_over
            )
            
            # Also marginalize individual Fisher matrices
            marginalized_individuals = []
            for fisher_matrix, full_param_names in individual_fishers:
                marg_fisher, marg_params = self.fisher_calc.marginalize_parameters(
                    fisher_matrix, full_param_names, marginalize_over
                )
                marginalized_individuals.append((marg_fisher, marg_params))
            individual_fishers = marginalized_individuals
        
        # Calculate synergy metrics
        synergy_metrics = self._calculate_synergy_metrics(
            individual_fishers, combined_fisher, param_names, survey_names
        )
        
        # Analyze parameter correlations
        correlation_analysis = self._analyze_parameter_correlations(
            individual_fishers, combined_fisher, param_names, survey_names
        )
        
        print("✓ Multi-probe synergy analysis completed")
        
        return {
            'survey_names': survey_names,
            'target_parameters': target_parameters or param_names,
            'synergy_metrics': synergy_metrics,
            'correlation_analysis': correlation_analysis,
            'individual_constraints': {
                name: self.fisher_calc.compute_parameter_errors(fisher, params)
                for (fisher, params), name in zip(individual_fishers, survey_names)
            },
            'combined_constraints': self.fisher_calc.compute_parameter_errors(
                combined_fisher, param_names
            )
        }
    
    def forecast_bias_impact(
        self,
        survey: SurveySpecification,
        likelihood_func: Callable,
        parameters: CosmologicalParameters,
        fiducial_values: Dict[str, float],
        bias_scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Forecast impact of systematic biases on parameter constraints.
        
        Parameters
        ----------
        survey : SurveySpecification
            Survey specification
        likelihood_func : Callable
            Survey likelihood function
        parameters : CosmologicalParameters
            Parameter manager
        fiducial_values : Dict[str, float]
            Fiducial parameter values
        bias_scenarios : Dict[str, Dict[str, float]]
            Bias scenarios: scenario_name -> {param: bias_level}
            
        Returns
        -------
        bias_impact : Dict[str, Any]
            Analysis of bias impact on parameter estimation
        """
        print(f"Analyzing bias impact for survey: {survey.name}")
        
        # Compute unbiased Fisher matrix
        unbiased_fisher, param_names = self.fisher_calc.compute_fisher_matrix(
            likelihood_func, parameters, fiducial_values, f"{survey.name}_unbiased"
        )
        unbiased_errors = self.fisher_calc.compute_parameter_errors(unbiased_fisher, param_names)
        
        bias_results = {}
        
        for scenario_name, bias_levels in bias_scenarios.items():
            print(f"  Analyzing bias scenario: {scenario_name}")
            
            # Create biased fiducial values
            biased_fiducial = fiducial_values.copy()
            for param, bias in bias_levels.items():
                if param in biased_fiducial:
                    biased_fiducial[param] += bias
                    
            # Compute biased Fisher matrix
            biased_fisher, _ = self.fisher_calc.compute_fisher_matrix(
                likelihood_func, parameters, biased_fiducial, f"{survey.name}_{scenario_name}"
            )
            biased_errors = self.fisher_calc.compute_parameter_errors(biased_fisher, param_names)
            
            # Calculate bias impact metrics
            bias_results[scenario_name] = {
                'bias_levels': bias_levels,
                'biased_errors': biased_errors,
                'error_degradation': {
                    param: biased_errors[param] / unbiased_errors[param]
                    for param in param_names if param in unbiased_errors
                },
                'parameter_shifts': {
                    param: bias_levels.get(param, 0.0) / unbiased_errors[param]
                    for param in param_names if param in unbiased_errors
                }
            }
            
        print("✓ Bias impact analysis completed")
        
        return {
            'survey_name': survey.name,
            'unbiased_constraints': unbiased_errors,
            'bias_scenarios': bias_results,
            'summary': self._summarize_bias_impact(bias_results, unbiased_errors)
        }
    
    def _add_systematic_uncertainties(
        self,
        likelihood_func: Callable,
        systematic_budget: Dict[str, float],
        parameters: CosmologicalParameters
    ) -> Callable:
        """Add systematic uncertainties to likelihood function."""
        
        def systematic_likelihood(param_dict: Dict[str, float]) -> float:
            # Original likelihood
            log_like = likelihood_func(param_dict)
            
            # Add systematic penalty
            systematic_penalty = 0.0
            for param_name, sys_sigma in systematic_budget.items():
                if param_name in param_dict:
                    # Simple Gaussian systematic prior
                    systematic_penalty -= 0.5 * (param_dict[param_name] / sys_sigma) ** 2
                    
            return log_like + systematic_penalty
            
        return systematic_likelihood
    
    def _construct_prior_fisher(
        self,
        priors: Dict[str, float],
        param_names: List[str]
    ) -> jnp.ndarray:
        """Construct Fisher matrix from Gaussian priors."""
        n_params = len(param_names)
        prior_fisher = jnp.zeros((n_params, n_params))
        
        for i, param_name in enumerate(param_names):
            if param_name in priors:
                prior_sigma = priors[param_name]
                prior_fisher = prior_fisher.at[i, i].set(1.0 / prior_sigma**2)
                
        return prior_fisher
    
    def _sample_design_parameters(self, design_config: ExperimentalDesign) -> Dict[str, float]:
        """Sample design parameters for optimization."""
        sampled_params = {}
        
        for param_name in design_config.variable_parameters:
            if param_name in design_config.parameter_ranges:
                param_min, param_max = design_config.parameter_ranges[param_name]
                # Simple uniform sampling (can be made more sophisticated)
                sampled_params[param_name] = np.random.uniform(param_min, param_max)
                
        return sampled_params
    
    def _modify_survey(
        self,
        base_survey: SurveySpecification,
        design_params: Dict[str, float]
    ) -> SurveySpecification:
        """Create modified survey with new design parameters."""
        modified = SurveySpecification(
            name=f"{base_survey.name}_modified",
            survey_type=base_survey.survey_type,
            sky_area=design_params.get('sky_area', base_survey.sky_area),
            redshift_range=base_survey.redshift_range,
            redshift_bins=base_survey.redshift_bins,
            number_density=design_params.get('number_density', base_survey.number_density),
            total_objects=base_survey.total_objects,
            photometric_errors=base_survey.photometric_errors,
            spectroscopic_errors=base_survey.spectroscopic_errors,
            systematic_floor=design_params.get('systematic_floor', base_survey.systematic_floor),
            survey_params=base_survey.survey_params.copy()
        )
        
        # Update survey parameters with design parameters
        for key, value in design_params.items():
            if key not in ['sky_area', 'number_density', 'systematic_floor']:
                modified.survey_params[key] = value
                
        return modified
    
    def _evaluate_figure_of_merit(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str],
        optimize_for: List[str]
    ) -> float:
        """Evaluate figure of merit for optimization."""
        if 'dark_energy_fom' in optimize_for:
            # Dark energy figure of merit (area of w0-wa ellipse)
            if 'w0' in param_names and 'wa' in param_names:
                w0_idx = param_names.index('w0')
                wa_idx = param_names.index('wa')
                
                # Extract 2x2 submatrix for w0, wa
                sub_fisher = fisher_matrix[jnp.ix_([w0_idx, wa_idx], [w0_idx, wa_idx])]
                
                # FoM = 1 / sqrt(det(Cov)) = sqrt(det(Fisher))
                det_fisher = jnp.linalg.det(sub_fisher)
                if det_fisher > 0:
                    return float(jnp.sqrt(det_fisher))
                    
        # Default: determinant of full Fisher matrix
        det_fisher = jnp.linalg.det(fisher_matrix)
        if det_fisher > 0:
            return float(jnp.sqrt(det_fisher))
        else:
            return 0.0
    
    def _calculate_synergy_metrics(
        self,
        individual_fishers: List[Tuple[jnp.ndarray, List[str]]],
        combined_fisher: jnp.ndarray,
        param_names: List[str],
        survey_names: List[str]
    ) -> Dict[str, Any]:
        """Calculate synergy metrics between probes."""
        
        # Individual errors
        individual_errors = {}
        for (fisher, params), name in zip(individual_fishers, survey_names):
            errors = self.fisher_calc.compute_parameter_errors(fisher, params)
            individual_errors[name] = errors
            
        # Combined errors  
        combined_errors = self.fisher_calc.compute_parameter_errors(combined_fisher, param_names)
        
        # Synergy factor: improvement from combination vs quadrature sum
        synergy_factors = {}
        for param in param_names:
            # Quadrature sum of individual errors
            individual_variances = []
            for name in survey_names:
                if param in individual_errors[name]:
                    individual_variances.append(individual_errors[name][param]**2)
                    
            if individual_variances:
                quadrature_error = jnp.sqrt(1.0 / jnp.sum(1.0 / jnp.array(individual_variances)))
                synergy_factors[param] = float(quadrature_error / combined_errors[param])
                
        return {
            'synergy_factors': synergy_factors,
            'individual_errors': individual_errors,
            'combined_errors': combined_errors
        }
    
    def _analyze_parameter_correlations(
        self,
        individual_fishers: List[Tuple[jnp.ndarray, List[str]]],
        combined_fisher: jnp.ndarray,
        param_names: List[str],
        survey_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze parameter correlations between probes."""
        
        # Combined covariance matrix
        combined_cov = jnp.linalg.pinv(combined_fisher)
        
        # Correlation matrix
        combined_corr = self._covariance_to_correlation(combined_cov)
        
        # Individual correlation matrices
        individual_correlations = {}
        for (fisher, params), name in zip(individual_fishers, survey_names):
            cov = jnp.linalg.pinv(fisher)
            corr = self._covariance_to_correlation(cov)
            individual_correlations[name] = {
                'correlation_matrix': corr,
                'parameter_names': params
            }
            
        return {
            'combined_correlation_matrix': combined_corr,
            'individual_correlations': individual_correlations,
            'parameter_names': param_names
        }
    
    def _covariance_to_correlation(self, covariance: jnp.ndarray) -> jnp.ndarray:
        """Convert covariance matrix to correlation matrix."""
        std_devs = jnp.sqrt(jnp.diag(covariance))
        correlation = covariance / jnp.outer(std_devs, std_devs)
        return correlation
    
    def _summarize_bias_impact(
        self,
        bias_results: Dict[str, Dict[str, Any]],
        unbiased_errors: Dict[str, float]
    ) -> Dict[str, Any]:
        """Summarize bias impact analysis."""
        
        max_degradation = {}
        critical_biases = {}
        
        for param in unbiased_errors.keys():
            max_deg = 0.0
            worst_scenario = None
            
            for scenario_name, results in bias_results.items():
                if param in results['error_degradation']:
                    degradation = results['error_degradation'][param]
                    if degradation > max_deg:
                        max_deg = degradation
                        worst_scenario = scenario_name
                        
            max_degradation[param] = max_deg
            critical_biases[param] = worst_scenario
            
        return {
            'max_error_degradation': max_degradation,
            'critical_bias_scenarios': critical_biases,
            'overall_robustness': float(np.mean(list(max_degradation.values())))
        }