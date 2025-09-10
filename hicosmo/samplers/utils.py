#!/usr/bin/env python3
"""
MCMC Utilities and Helper Functions.

This module provides utility functions for MCMC parameter analysis,
function introspection, and automatic parameter-likelihood mapping.
"""

import inspect
from typing import Dict, Any, Optional, Set, List, Tuple, Callable, Union
from dataclasses import dataclass
import warnings
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .config import ParameterConfig, AutoParameter


# =============================================================================
# Function Introspection Tools
# =============================================================================

@dataclass
class ParameterSignature:
    """
    Signature analysis result for a function parameter.
    
    Attributes
    ----------
    name : str
        Parameter name.
    required : bool
        Whether parameter is required (no default value).
    has_default : bool
        Whether parameter has a default value.
    default_value : Any
        Default value if available.
    annotation : Any
        Type annotation if available.
    is_data : bool
        Whether this is identified as a data parameter.
    """
    name: str
    required: bool
    has_default: bool
    default_value: Any = None
    annotation: Any = None
    is_data: bool = False


@dataclass
class FunctionSignature:
    """
    Complete function signature analysis result.
    
    Attributes
    ----------
    function : Callable
        Original function.
    parameters : Dict[str, ParameterSignature]
        All parameter signatures.
    model_parameters : List[str]
        Model parameter names (excluding data).
    data_parameters : List[str]  
        Data parameter names.
    total_params : int
        Total number of parameters.
    required_model_params : List[str]
        Required model parameters (no defaults).
    optional_model_params : List[str]
        Optional model parameters (have defaults).
    """
    function: Callable
    parameters: Dict[str, ParameterSignature]
    model_parameters: List[str]
    data_parameters: List[str] 
    total_params: int
    required_model_params: List[str]
    optional_model_params: List[str]


@dataclass
class CompatibilityCheck:
    """
    Function compatibility analysis result.
    
    Attributes
    ----------
    compatible : bool
        Whether function is compatible with available parameters/data.
    missing_model_params : List[str]
        Required model parameters that are missing.
    extra_model_params : List[str]
        Available parameters not used by function.
    missing_data_params : List[str]
        Required data parameters that are missing.
    function_signature : FunctionSignature
        Complete function signature analysis.
    warnings : List[str]
        Warning messages about compatibility issues.
    """
    compatible: bool
    missing_model_params: List[str]
    extra_model_params: List[str]  
    missing_data_params: List[str]
    function_signature: FunctionSignature
    warnings: List[str]


class FunctionInspector:
    """
    Intelligent function introspection for automatic parameter mapping.
    
    This class analyzes likelihood functions to automatically determine:
    1. What parameters the function expects
    2. Which parameters are data vs. model parameters
    3. Parameter types and constraints
    4. Function signature compatibility
    """
    
    def __init__(
        self,
        data_param_names: Optional[Set[str]] = None,
        exclude_params: Optional[Set[str]] = None
    ):
        """
        Initialize function inspector.
        
        Parameters
        ----------
        data_param_names : Set[str], optional
            Known data parameter names to exclude from model parameters.
        exclude_params : Set[str], optional
            Parameter names to exclude from analysis.
        """
        # Common data parameter names
        self.default_data_names = {
            'data', 'x_data', 'y_data', 'obs_data', 
            'x', 'y', 'y_obs', 'y_err', 'yerr', 'sigma', 'cov',
            'sn_data', 'cmb_data', 'bao_data', 'observations',
            'dataset', 'measurements', 'errors', 'uncertainties'
        }
        
        # Combine with user-provided data names
        if data_param_names:
            self.data_param_names = self.default_data_names.union(data_param_names)
        else:
            self.data_param_names = self.default_data_names
        
        # Common parameters to exclude
        self.default_exclude = {
            'self', 'cls', 'args', 'kwargs', 'return', 'returns'
        }
        
        if exclude_params:
            self.exclude_params = self.default_exclude.union(exclude_params)
        else:
            self.exclude_params = self.default_exclude
    
    def analyze_function(self, func: Callable) -> FunctionSignature:
        """
        Analyze a function's signature to extract parameter information.
        
        Parameters
        ----------
        func : Callable
            Function to analyze.
            
        Returns
        -------
        FunctionSignature
            Complete signature analysis.
        """
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot inspect function {func}: {e}")
        
        parameters = {}
        model_parameters = []
        data_parameters = []
        
        for param_name, param in sig.parameters.items():
            # Skip excluded parameters
            if param_name in self.exclude_params:
                continue
                
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            
            # Create parameter signature
            param_sig = ParameterSignature(
                name=param_name,
                required=(param.default == param.empty),
                has_default=(param.default != param.empty),
                default_value=param.default if param.default != param.empty else None,
                annotation=param.annotation if param.annotation != param.empty else None,
                is_data=(param_name in self.data_param_names)
            )
            
            parameters[param_name] = param_sig
            
            # Categorize parameters
            if param_sig.is_data:
                data_parameters.append(param_name)
            else:
                model_parameters.append(param_name)
        
        return FunctionSignature(
            function=func,
            parameters=parameters,
            model_parameters=model_parameters,
            data_parameters=data_parameters,
            total_params=len(parameters),
            required_model_params=[p for p in model_parameters 
                                 if parameters[p].required],
            optional_model_params=[p for p in model_parameters 
                                 if not parameters[p].required]
        )
    
    def check_compatibility(
        self, 
        func: Callable, 
        available_params: Set[str],
        available_data: Optional[Dict[str, Any]] = None
    ) -> CompatibilityCheck:
        """
        Check if a function is compatible with available parameters and data.
        
        Parameters
        ----------
        func : Callable
            Function to check.
        available_params : Set[str]
            Available model parameter names.
        available_data : Dict[str, Any], optional
            Available data arguments.
            
        Returns
        -------
        CompatibilityCheck
            Detailed compatibility analysis.
        """
        sig = self.analyze_function(func)
        
        # Check model parameters
        missing_params = set(sig.required_model_params) - available_params
        extra_params = available_params - set(sig.model_parameters)
        
        # Check data parameters
        missing_data = []
        if available_data:
            available_data_names = set(available_data.keys())
            missing_data = [p for p in sig.data_parameters 
                          if p not in available_data_names and sig.parameters[p].required]
        
        is_compatible = (len(missing_params) == 0 and len(missing_data) == 0)
        
        return CompatibilityCheck(
            compatible=is_compatible,
            missing_model_params=list(missing_params),
            extra_model_params=list(extra_params),
            missing_data_params=missing_data,
            function_signature=sig,
            warnings=self._generate_warnings(sig, missing_params, extra_params, missing_data)
        )
    
    def _generate_warnings(
        self, 
        sig: FunctionSignature,
        missing_params: Set[str],
        extra_params: Set[str], 
        missing_data: List[str]
    ) -> List[str]:
        """Generate warning messages for compatibility issues."""
        warnings = []
        
        if missing_params:
            warnings.append(f"Missing required parameters: {list(missing_params)}")
        
        if missing_data:
            warnings.append(f"Missing required data parameters: {missing_data}")
        
        if extra_params and len(extra_params) > 3:
            warnings.append(f"Many unused parameters provided: {len(extra_params)} extra")
        
        if sig.total_params == 0:
            warnings.append("Function has no parameters - may not be a likelihood function")
        
        if len(sig.model_parameters) == 0:
            warnings.append("Function has no model parameters - may not be suitable for MCMC")
            
        return warnings


def analyze_likelihood_function(
    func: Callable,
    available_params: Optional[Set[str]] = None,
    available_data: Optional[Dict[str, Any]] = None
) -> Tuple[FunctionSignature, Optional[CompatibilityCheck]]:
    """
    Convenience function to analyze a likelihood function.
    
    Parameters
    ----------
    func : Callable
        Likelihood function to analyze.
    available_params : Set[str], optional
        Available parameter names.
    available_data : Dict[str, Any], optional
        Available data.
        
    Returns
    -------
    signature : FunctionSignature
        Function signature analysis.
    compatibility : CompatibilityCheck, optional
        Compatibility check if parameters/data provided.
    """
    inspector = FunctionInspector()
    signature = inspector.analyze_function(func)
    
    compatibility = None
    if available_params is not None:
        compatibility = inspector.check_compatibility(func, available_params, available_data)
    
    return signature, compatibility


# =============================================================================
# Automatic Parameter Mapping
# =============================================================================

@dataclass
class MappingResult:
    """
    Result of automatic parameter mapping.
    
    Attributes
    ----------
    parameter_mapping : Dict[str, str]
        Mapping from function parameters to config parameters.
    data_arguments : Dict[str, Any]
        Data arguments to pass to function.
    missing_parameters : List[str]
        Required parameters that couldn't be mapped.
    unused_parameters : List[str]
        Config parameters that aren't used.
    warnings : List[str]
        Warning messages about the mapping.
    """
    parameter_mapping: Dict[str, str]
    data_arguments: Dict[str, Any]
    missing_parameters: List[str]
    unused_parameters: List[str]
    warnings: List[str]


class ParameterMapper:
    """
    Intelligent automatic parameter mapping system.
    
    This class bridges the gap between user parameter configurations and
    likelihood function signatures, providing automatic argument mapping
    and validation.
    """
    
    def __init__(
        self, 
        param_config: ParameterConfig,
        strict_mode: bool = False,
        enable_fuzzy_matching: bool = True
    ):
        """
        Initialize automatic mapper.
        
        Parameters
        ----------
        param_config : ParameterConfig
            Parameter configuration to map from.
        strict_mode : bool
            If True, raise errors for missing parameters instead of warnings.
        enable_fuzzy_matching : bool
            Whether to enable fuzzy parameter name matching.
        """
        self.param_config = param_config
        self.strict_mode = strict_mode
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.inspector = FunctionInspector()
        
        # Cache for function signatures
        self._signature_cache = {}
    
    def map_to_function(
        self, 
        likelihood_func: Callable,
        **data_kwargs
    ) -> MappingResult:
        """
        Create automatic mapping between parameters and function arguments.
        
        Parameters
        ----------
        likelihood_func : Callable
            Likelihood function to map to.
        **data_kwargs
            Data arguments to pass to the function.
            
        Returns
        -------
        MappingResult
            Complete mapping result with diagnostics.
        """
        # Analyze function signature (with caching)
        func_id = id(likelihood_func)
        if func_id not in self._signature_cache:
            signature, _ = analyze_likelihood_function(likelihood_func)
            self._signature_cache[func_id] = signature
        else:
            signature = self._signature_cache[func_id]
        
        # Get available parameters
        available_params = set(self.param_config.get_parameter_names())
        
        # Check compatibility
        compatibility = self.inspector.check_compatibility(
            likelihood_func, available_params, data_kwargs
        )
        
        # Create basic mapping
        parameter_mapping = {}
        
        # Direct matches first
        for func_param in signature.model_parameters:
            if func_param in available_params:
                parameter_mapping[func_param] = func_param
        
        # Fuzzy matching for unmapped parameters
        if self.enable_fuzzy_matching:
            unmapped = set(signature.model_parameters) - set(parameter_mapping.keys())
            if unmapped:
                fuzzy_mapping = self._suggest_fuzzy_mapping(
                    likelihood_func, unmapped, available_params
                )
                
                # Add fuzzy matches
                for func_param, config_param in fuzzy_mapping.items():
                    if config_param not in parameter_mapping.values():
                        parameter_mapping[func_param] = config_param
        
        # Identify missing and unused parameters
        missing_parameters = [
            p for p in signature.required_model_params 
            if p not in parameter_mapping
        ]
        
        used_params = set(parameter_mapping.values())
        unused_parameters = list(available_params - used_params)
        
        # Generate warnings
        warnings_list = []
        
        if missing_parameters:
            msg = f"Missing required parameters: {missing_parameters}"
            warnings_list.append(msg)
            if self.strict_mode:
                raise ValueError(msg)
        
        if unused_parameters:
            warnings_list.append(f"Unused parameters: {unused_parameters}")
        
        # Add compatibility warnings
        warnings_list.extend(compatibility.warnings)
        
        return MappingResult(
            parameter_mapping=parameter_mapping,
            data_arguments=data_kwargs,
            missing_parameters=missing_parameters,
            unused_parameters=unused_parameters,
            warnings=warnings_list
        )
    
    def _suggest_fuzzy_mapping(
        self, 
        likelihood_func: Callable,
        unmapped_params: Set[str],
        available_params: Set[str]
    ) -> Dict[str, str]:
        """Suggest fuzzy parameter mappings."""
        mapping = {}
        unused = available_params.copy()
        
        for func_param in unmapped_params:
            best_match = self._find_fuzzy_match(func_param, unused)
            if best_match:
                mapping[func_param] = best_match
                unused.remove(best_match)
        
        return mapping
    
    def _find_fuzzy_match(self, target: str, candidates: Set[str]) -> Optional[str]:
        """Find best fuzzy match using simple heuristics."""
        target_lower = target.lower()
        
        # Direct substring matching
        for candidate in candidates:
            if target_lower in candidate.lower() or candidate.lower() in target_lower:
                return candidate
        
        # Common parameter name variations
        variations = {
            'h0': ['hubble', 'h_0', 'h0'],
            'omega_m': ['om', 'omegam', 'omega_matter', 'omm'],
            'omega_b': ['ob', 'omegab', 'omega_baryon', 'omb'], 
            'omega_de': ['ode', 'omega_lambda', 'ol', 'w'],
            'ns': ['n_s', 'spectral_index'],
            'sigma8': ['s8', 'sigma_8'],
            'tau': ['tau_reio', 'optical_depth']
        }
        
        for canonical, aliases in variations.items():
            if target_lower == canonical or target_lower in aliases:
                for candidate in candidates:
                    if (candidate.lower() == canonical or 
                        candidate.lower() in aliases or
                        any(alias in candidate.lower() for alias in aliases)):
                        return candidate
        
        return None
    
    def create_numpyro_model(
        self, 
        likelihood_func: Callable,
        **data_kwargs
    ) -> Callable:
        """
        Create NumPyro model from likelihood function and parameter config.
        
        Parameters
        ----------
        likelihood_func : Callable
            Likelihood function that returns log-likelihood value.
        **data_kwargs
            Data to pass to likelihood function.
            
        Returns
        -------
        Callable
            NumPyro model function ready for MCMC sampling.
        """
        mapping_result = self.map_to_function(likelihood_func, **data_kwargs)
        
        # Print warnings
        for warning in mapping_result.warnings:
            warnings.warn(f"ParameterMapper: {warning}")
        
        # Get parameter configurations
        param_mapping = mapping_result.parameter_mapping
        data_args = mapping_result.data_arguments
        
        def numpyro_model(**model_kwargs):
            """Auto-generated NumPyro model."""
            # Sample all mapped parameters
            sampled_params = {}
            
            for func_param, config_param in param_mapping.items():
                param_config = self.param_config.parameters[config_param]
                
                # Sample from the configured prior
                sampled_value = numpyro.sample(
                    config_param,  # Use config name for sampling
                    param_config.to_numpyro_dist()
                )
                sampled_params[func_param] = sampled_value
            
            # Handle derived parameters if any
            for derived_name in self.param_config.derived:
                # Placeholder for future derived parameter support
                pass
            
            # Combine data arguments: prefer model_kwargs, fallback to stored data_args
            combined_data = dict(data_args)
            combined_data.update(model_kwargs)
            
            # Call likelihood function
            try:
                log_likelihood = likelihood_func(**sampled_params, **combined_data)
                
                # Use numpyro.factor to incorporate the likelihood
                numpyro.factor("likelihood", log_likelihood)
                
            except Exception as e:
                raise RuntimeError(
                    f"Error calling likelihood function with parameters "
                    f"{list(sampled_params.keys())}: {e}"
                )
        
        return numpyro_model


# =============================================================================
# Convenience Functions
# =============================================================================

def create_auto_model(
    param_config: ParameterConfig,
    likelihood_func: Callable,
    **data_kwargs
) -> Callable:
    """
    Convenience function to create NumPyro model with automatic mapping.
    
    Parameters
    ----------
    param_config : ParameterConfig
        Parameter configuration.
    likelihood_func : Callable
        Likelihood function.
    **data_kwargs
        Data arguments.
        
    Returns
    -------
    Callable
        NumPyro model ready for MCMC.
    """
    mapper = ParameterMapper(param_config)
    return mapper.create_numpyro_model(likelihood_func, **data_kwargs)


def validate_likelihood_compatibility(
    param_config: ParameterConfig,
    likelihood_func: Callable,
    **data_kwargs
) -> MappingResult:
    """
    Validate compatibility between parameter config and likelihood function.
    
    Parameters
    ----------
    param_config : ParameterConfig
        Parameter configuration.
    likelihood_func : Callable
        Likelihood function.
    **data_kwargs
        Data arguments.
        
    Returns
    -------
    MappingResult
        Detailed mapping analysis.
    """
    mapper = ParameterMapper(param_config, strict_mode=False)
    return mapper.map_to_function(likelihood_func, **data_kwargs)