"""
HiCosmo Fisher Matrix Calculator
===============================

Professional Fisher matrix calculations for cosmological parameter estimation.
Computes parameter forecasts and error predictions using automatic differentiation
via JAX for high-performance numerical derivatives.

Key features:
- Automatic differentiation of likelihood functions
- Multi-probe Fisher matrix combination  
- Parameter marginalization and conditioning
- Numerical stability with advanced regularization
- Professional validation and diagnostics
"""

import jax.numpy as jnp
from jax import grad, hessian, vmap, jit
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
import warnings

from ..core.unified_parameters import CosmologicalParameters


@dataclass
class FisherMatrixConfig:
    """Configuration for Fisher matrix calculations."""
    
    # Numerical differentiation parameters
    step_size: float = 1e-6
    step_method: str = 'adaptive'  # 'fixed', 'adaptive', 'optimal'
    max_condition_number: float = 1e12
    
    # Regularization parameters
    regularization: bool = True
    reg_lambda: float = 1e-10
    svd_threshold: float = 1e-12
    
    # Validation parameters
    check_symmetry: bool = True
    symmetry_tolerance: float = 1e-10
    check_positive_definite: bool = True
    
    # Performance parameters
    vectorize_derivatives: bool = True
    cache_derivatives: bool = True
    parallel_probes: bool = True


class FisherMatrix:
    """
    Professional Fisher matrix calculator for cosmological parameter estimation.
    
    Computes Fisher information matrices from likelihood functions using automatic
    differentiation, with advanced numerical methods for stability and accuracy.
    """
    
    def __init__(self, config: Optional[FisherMatrixConfig] = None):
        """
        Initialize Fisher matrix calculator.
        
        Parameters
        ----------
        config : FisherMatrixConfig, optional
            Configuration for Fisher matrix calculations
        """
        self.config = config or FisherMatrixConfig()
        self._derivative_cache = {}
        
    def compute_fisher_matrix(
        self, 
        likelihood_func: Callable,
        parameters: CosmologicalParameters,
        fiducial_values: Dict[str, float],
        probe_name: str = 'combined'
    ) -> Tuple[jnp.ndarray, List[str]]:
        """
        Compute Fisher information matrix for given likelihood function.
        
        Parameters
        ----------
        likelihood_func : Callable
            Log-likelihood function taking parameter dict as input
        parameters : CosmologicalParameters
            Parameter manager with parameter specifications
        fiducial_values : Dict[str, float]
            Fiducial parameter values for derivatives
        probe_name : str, default='combined'
            Name identifier for this probe
            
        Returns
        -------
        fisher_matrix : jnp.ndarray
            Fisher information matrix (N_params x N_params)
        param_names : List[str]
            Ordered list of parameter names corresponding to matrix indices
        """
        # Get free parameters for Fisher matrix
        free_params = parameters.get_free_parameters()
        param_names = [p.name for p in free_params]
        n_params = len(param_names)
        
        if n_params == 0:
            raise ValueError("No free parameters found for Fisher matrix calculation")
            
        # Validate fiducial values
        self._validate_fiducial_values(fiducial_values, param_names)
        
        # Create parameter vector at fiducial values
        fiducial_vector = jnp.array([fiducial_values[name] for name in param_names])
        
        # Compute second derivatives (Fisher matrix elements)
        print(f"Computing Fisher matrix for {n_params} parameters...")
        fisher_matrix = self._compute_hessian_matrix(
            likelihood_func, param_names, fiducial_vector, parameters
        )
        
        # Apply negative sign (Fisher = -d²ln L/dθᵢdθⱼ)
        fisher_matrix = -fisher_matrix
        
        # Validation and regularization
        fisher_matrix = self._validate_and_regularize_fisher(fisher_matrix, param_names)
        
        # Cache result (convert JAX array to tuple for hashing)
        cache_key = (probe_name, tuple(param_names), tuple(float(x) for x in fiducial_vector))
        self._derivative_cache[cache_key] = fisher_matrix
        
        print(f"✓ Fisher matrix computed successfully for probe '{probe_name}'")
        return fisher_matrix, param_names
    
    def combine_fisher_matrices(
        self,
        fisher_list: List[Tuple[jnp.ndarray, List[str]]],
        probe_names: List[str]
    ) -> Tuple[jnp.ndarray, List[str]]:
        """
        Combine multiple Fisher matrices from different probes.
        
        Parameters
        ----------
        fisher_list : List[Tuple[jnp.ndarray, List[str]]]
            List of (fisher_matrix, param_names) tuples
        probe_names : List[str]
            Names of probes corresponding to Fisher matrices
            
        Returns
        -------
        combined_fisher : jnp.ndarray
            Combined Fisher information matrix
        param_names : List[str]
            Parameter names for combined matrix
        """
        if len(fisher_list) == 0:
            raise ValueError("No Fisher matrices provided for combination")
            
        if len(fisher_list) == 1:
            return fisher_list[0]
            
        print(f"Combining {len(fisher_list)} Fisher matrices from probes: {probe_names}")
        
        # Get union of all parameters
        all_param_names = []
        for _, param_names in fisher_list:
            all_param_names.extend(param_names)
        unique_params = list(dict.fromkeys(all_param_names))  # Preserve order
        n_params = len(unique_params)
        
        # Initialize combined Fisher matrix
        combined_fisher = jnp.zeros((n_params, n_params))
        
        # Add each Fisher matrix to the combined matrix
        for (fisher_matrix, param_names), probe_name in zip(fisher_list, probe_names):
            # Create index mapping
            indices = [unique_params.index(name) for name in param_names]
            
            # Add to combined matrix using advanced indexing
            combined_fisher = combined_fisher.at[jnp.ix_(indices, indices)].add(fisher_matrix)
            
        print("✓ Fisher matrices combined successfully")
        return combined_fisher, unique_params
    
    def marginalize_parameters(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str],
        marginalize_over: List[str]
    ) -> Tuple[jnp.ndarray, List[str]]:
        """
        Marginalize Fisher matrix over specified parameters.
        
        Parameters
        ----------
        fisher_matrix : jnp.ndarray
            Input Fisher information matrix
        param_names : List[str]
            Parameter names corresponding to matrix
        marginalize_over : List[str]
            Parameters to marginalize over
            
        Returns
        -------
        marginalized_fisher : jnp.ndarray
            Marginalized Fisher matrix
        remaining_params : List[str]
            Remaining parameter names after marginalization
        """
        if not marginalize_over:
            return fisher_matrix, param_names
            
        # Find indices of parameters to keep
        keep_indices = []
        remaining_params = []
        
        for i, name in enumerate(param_names):
            if name not in marginalize_over:
                keep_indices.append(i)
                remaining_params.append(name)
                
        if len(keep_indices) == 0:
            raise ValueError("Cannot marginalize over all parameters")
            
        # Extract submatrix
        keep_indices = jnp.array(keep_indices)
        marginalized_fisher = fisher_matrix[jnp.ix_(keep_indices, keep_indices)]
        
        print(f"✓ Marginalized over {len(marginalize_over)} parameters")
        return marginalized_fisher, remaining_params
    
    def condition_on_parameters(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str],
        condition_on: Dict[str, float]
    ) -> Tuple[jnp.ndarray, List[str]]:
        """
        Condition Fisher matrix on fixed parameter values (Gaussian prior).
        
        Parameters
        ----------
        fisher_matrix : jnp.ndarray
            Input Fisher information matrix
        param_names : List[str]
            Parameter names corresponding to matrix
        condition_on : Dict[str, float]
            Parameters and their fixed values (infinite prior precision)
            
        Returns
        -------
        conditioned_fisher : jnp.ndarray
            Conditioned Fisher matrix
        remaining_params : List[str]
            Remaining free parameter names
        """
        if not condition_on:
            return fisher_matrix, param_names
            
        # Find indices of parameters to condition on
        condition_indices = []
        remaining_params = []
        remaining_indices = []
        
        for i, name in enumerate(param_names):
            if name in condition_on:
                condition_indices.append(i)
            else:
                remaining_indices.append(i)
                remaining_params.append(name)
                
        if len(remaining_indices) == 0:
            raise ValueError("Cannot condition on all parameters")
            
        condition_indices = jnp.array(condition_indices)
        remaining_indices = jnp.array(remaining_indices)
        
        # Schur complement for conditioning
        F_AA = fisher_matrix[jnp.ix_(remaining_indices, remaining_indices)]
        F_AB = fisher_matrix[jnp.ix_(remaining_indices, condition_indices)]  
        F_BB = fisher_matrix[jnp.ix_(condition_indices, condition_indices)]
        
        # Conditioned Fisher: F_A|B = F_AA - F_AB F_BB^-1 F_AB^T
        F_BB_inv = jnp.linalg.pinv(F_BB, rcond=self.config.svd_threshold)
        conditioned_fisher = F_AA - F_AB @ F_BB_inv @ F_AB.T
        
        print(f"✓ Conditioned on {len(condition_on)} parameters")
        return conditioned_fisher, remaining_params
    
    def compute_parameter_errors(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str],
        marginalize_over: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute 1σ parameter uncertainties from Fisher matrix.
        
        Parameters
        ----------
        fisher_matrix : jnp.ndarray
            Fisher information matrix
        param_names : List[str]
            Parameter names corresponding to matrix
        marginalize_over : List[str], optional
            Parameters to marginalize over before computing errors
            
        Returns
        -------
        errors : Dict[str, float]
            1σ parameter uncertainties
        """
        # Marginalize if requested
        if marginalize_over:
            fisher_matrix, param_names = self.marginalize_parameters(
                fisher_matrix, param_names, marginalize_over
            )
        
        # Compute covariance matrix (inverse of Fisher matrix)
        try:
            covariance = jnp.linalg.inv(fisher_matrix)
        except jnp.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            covariance = jnp.linalg.pinv(fisher_matrix, rcond=self.config.svd_threshold)
            warnings.warn("Fisher matrix is singular; using pseudo-inverse")
            
        # Extract 1σ errors from diagonal
        errors = {}
        for i, name in enumerate(param_names):
            variance = covariance[i, i]
            if variance < 0:
                warnings.warn(f"Negative variance for parameter {name}: {variance}")
                errors[name] = jnp.nan
            else:
                errors[name] = float(jnp.sqrt(variance))
                
        return errors
    
    def _compute_hessian_matrix(
        self,
        likelihood_func: Callable,
        param_names: List[str],
        fiducial_vector: jnp.ndarray,
        parameters: CosmologicalParameters
    ) -> jnp.ndarray:
        """Compute Hessian matrix using JAX automatic differentiation."""
        
        # Create wrapper function for vector input
        def likelihood_wrapper(param_vector: jnp.ndarray) -> float:
            param_dict = dict(zip(param_names, param_vector))
            # Update parameter manager and compute derived parameters
            updated_params = parameters.update_and_compute_derived(param_dict)
            return likelihood_func(updated_params)
            
        # Use JAX hessian for automatic differentiation
        hessian_func = jit(hessian(likelihood_wrapper))
        hessian_matrix = hessian_func(fiducial_vector)
        
        return hessian_matrix
    
    def _validate_fiducial_values(
        self, 
        fiducial_values: Dict[str, float], 
        param_names: List[str]
    ) -> None:
        """Validate that all required fiducial values are provided."""
        missing_params = set(param_names) - set(fiducial_values.keys())
        if missing_params:
            raise ValueError(f"Missing fiducial values for parameters: {missing_params}")
            
        for name, value in fiducial_values.items():
            if not jnp.isfinite(value):
                raise ValueError(f"Non-finite fiducial value for parameter {name}: {value}")
    
    def _validate_and_regularize_fisher(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str]
    ) -> jnp.ndarray:
        """Validate and regularize Fisher matrix for numerical stability."""
        
        # Check symmetry
        if self.config.check_symmetry:
            asymmetry = jnp.max(jnp.abs(fisher_matrix - fisher_matrix.T))
            if asymmetry > self.config.symmetry_tolerance:
                warnings.warn(f"Fisher matrix asymmetry: {asymmetry}")
                # Symmetrize
                fisher_matrix = 0.5 * (fisher_matrix + fisher_matrix.T)
        
        # Check condition number
        try:
            cond_num = jnp.linalg.cond(fisher_matrix)
            if cond_num > self.config.max_condition_number:
                warnings.warn(f"Fisher matrix poorly conditioned: {cond_num}")
        except jnp.linalg.LinAlgError:
            warnings.warn("Could not compute Fisher matrix condition number")
            
        # Regularization for numerical stability
        if self.config.regularization:
            # Add small diagonal regularization
            n_params = fisher_matrix.shape[0]
            reg_matrix = self.config.reg_lambda * jnp.eye(n_params)
            fisher_matrix = fisher_matrix + reg_matrix
            
        # Check positive definiteness
        if self.config.check_positive_definite:
            eigenvals = jnp.linalg.eigvals(fisher_matrix)
            min_eigenval = jnp.min(eigenvals)
            if min_eigenval <= 0:
                warnings.warn(f"Fisher matrix not positive definite: min eigenvalue = {min_eigenval}")
                
        return fisher_matrix
    
    def get_fisher_summary(
        self,
        fisher_matrix: jnp.ndarray,
        param_names: List[str]
    ) -> Dict[str, Union[float, int]]:
        """
        Get summary statistics for Fisher matrix.
        
        Returns
        -------
        summary : Dict[str, Union[float, int]]
            Summary statistics including condition number, determinant, etc.
        """
        n_params = len(param_names)
        
        summary = {
            'n_parameters': n_params,
            'matrix_shape': fisher_matrix.shape,
            'determinant': float(jnp.linalg.det(fisher_matrix)),
            'trace': float(jnp.trace(fisher_matrix)),
            'frobenius_norm': float(jnp.linalg.norm(fisher_matrix, 'fro')),
        }
        
        try:
            summary['condition_number'] = float(jnp.linalg.cond(fisher_matrix))
        except jnp.linalg.LinAlgError:
            summary['condition_number'] = jnp.inf
            
        # Eigenvalue analysis  
        eigenvals = jnp.linalg.eigvals(fisher_matrix)
        summary['max_eigenvalue'] = float(jnp.max(eigenvals))
        summary['min_eigenvalue'] = float(jnp.min(eigenvals))
        summary['eigenvalue_ratio'] = float(jnp.max(eigenvals) / jnp.max(jnp.abs(eigenvals)))
        
        return summary