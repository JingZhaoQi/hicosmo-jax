"""
Parameter manager for handling cosmological and nuisance parameters.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp
import yaml
from pathlib import Path
from copy import deepcopy


class ParameterManager:
    """
    Manages cosmological and nuisance parameters for MCMC sampling.
    
    Features:
    - Flexible parameter definition (free/fixed)
    - Prior distribution specification
    - Parameter transformations
    - Nuisance parameter handling
    - YAML configuration support
    """
    
    def __init__(self):
        """Initialize parameter manager."""
        self.params = {}
        self.param_order = []
        self._likelihood_nuisance = {}
    
    def add_param(self, 
                  name: str,
                  value: Optional[float] = None,
                  prior: Optional[Dict[str, Any]] = None,
                  fixed: bool = False,
                  latex: Optional[str] = None,
                  ref: Optional[float] = None,
                  proposal: Optional[float] = None,
                  transform: Optional[str] = None) -> None:
        """
        Add a parameter to the manager.
        
        Args:
            name: Parameter name
            value: Fixed value (if fixed=True)
            prior: Prior distribution specification
            fixed: Whether parameter is fixed
            latex: LaTeX representation for plotting
            ref: Reference value for initialization
            proposal: Proposal width for MCMC
            transform: Parameter transformation ('log', 'logit', etc.)
        """
        param_info = {
            'name': name,
            'fixed': fixed,
            'latex': latex or name
        }
        
        if fixed:
            if value is None:
                raise ValueError(f"Fixed parameter {name} must have a value")
            param_info['value'] = value
        else:
            if prior is None:
                raise ValueError(f"Free parameter {name} must have a prior")
            param_info['prior'] = prior
            
            if ref is not None:
                param_info['ref'] = ref
            elif 'min' in prior and 'max' in prior:
                param_info['ref'] = (prior['min'] + prior['max']) / 2
            elif 'loc' in prior:
                param_info['ref'] = prior['loc']
            
            if proposal is not None:
                param_info['proposal'] = proposal
            
            if transform is not None:
                param_info['transform'] = transform
        
        self.params[name] = param_info
        if name not in self.param_order:
            self.param_order.append(name)
    
    def add_cosmological_params(self, model: str = 'LCDM') -> None:
        """
        Add standard cosmological parameters for a given model.
        
        Args:
            model: Cosmological model name
        """
        if model == 'LCDM':
            # Hubble constant
            self.add_param(
                'H0',
                prior={'min': 50, 'max': 100, 'dist': 'uniform'},
                ref=70.0,
                proposal=2.0,
                latex=r'H_0'
            )
            
            # Matter density
            self.add_param(
                'Omega_m',
                prior={'min': 0.1, 'max': 0.5, 'dist': 'uniform'},
                ref=0.3,
                proposal=0.02,
                latex=r'\Omega_m'
            )
            
            # Baryon density (optional)
            self.add_param(
                'Omega_b',
                prior={'min': 0.03, 'max': 0.07, 'dist': 'uniform'},
                ref=0.05,
                proposal=0.005,
                latex=r'\Omega_b'
            )
            
        elif model == 'wCDM':
            # Add LCDM parameters
            self.add_cosmological_params('LCDM')
            
            # Dark energy equation of state
            self.add_param(
                'w',
                prior={'min': -2.0, 'max': 0.0, 'dist': 'uniform'},
                ref=-1.0,
                proposal=0.1,
                latex=r'w'
            )
            
        elif model == 'w0waCDM':
            # Add LCDM parameters
            self.add_cosmological_params('LCDM')
            
            # CPL parameterization
            self.add_param(
                'w0',
                prior={'min': -2.0, 'max': 0.0, 'dist': 'uniform'},
                ref=-1.0,
                proposal=0.1,
                latex=r'w_0'
            )
            
            self.add_param(
                'wa',
                prior={'min': -2.0, 'max': 2.0, 'dist': 'uniform'},
                ref=0.0,
                proposal=0.3,
                latex=r'w_a'
            )
        else:
            raise ValueError(f"Unknown cosmological model: {model}")
    
    def add_likelihood_nuisance(self, 
                               likelihood_name: str,
                               nuisance_params: Dict[str, Dict[str, Any]]) -> None:
        """
        Add nuisance parameters from a likelihood.
        
        Args:
            likelihood_name: Name of the likelihood
            nuisance_params: Dictionary of nuisance parameters
        """
        for param_name, param_info in nuisance_params.items():
            full_name = f"{likelihood_name}_{param_name}"
            
            # Add to parameter list
            self.add_param(
                full_name,
                prior=param_info.get('prior'),
                fixed=param_info.get('fixed', False),
                value=param_info.get('value'),
                latex=param_info.get('latex', param_name),
                ref=param_info.get('ref'),
                proposal=param_info.get('proposal')
            )
            
            # Track which likelihood owns this parameter
            if likelihood_name not in self._likelihood_nuisance:
                self._likelihood_nuisance[likelihood_name] = []
            self._likelihood_nuisance[likelihood_name].append(full_name)
    
    def get_free_params(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all free (non-fixed) parameters.
        
        Returns:
            Dictionary of free parameters
        """
        return {name: info for name, info in self.params.items() 
                if not info['fixed']}
    
    def get_fixed_params(self) -> Dict[str, float]:
        """
        Get all fixed parameters.
        
        Returns:
            Dictionary of fixed parameter values
        """
        return {name: info['value'] for name, info in self.params.items() 
                if info['fixed']}
    
    def get_numpyro_prior(self, param_name: str) -> dist.Distribution:
        """
        Get NumPyro distribution for a parameter's prior.
        
        Args:
            param_name: Parameter name
            
        Returns:
            NumPyro distribution object
        """
        if param_name not in self.params:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        param_info = self.params[param_name]
        if param_info['fixed']:
            raise ValueError(f"Parameter {param_name} is fixed")
        
        prior = param_info['prior']
        dist_type = prior.get('dist', 'uniform')
        
        if dist_type == 'uniform':
            return dist.Uniform(prior['min'], prior['max'])
        
        elif dist_type == 'normal' or dist_type == 'gaussian':
            return dist.Normal(prior['loc'], prior['scale'])
        
        elif dist_type == 'truncated_normal':
            base_dist = dist.Normal(prior['loc'], prior['scale'])
            return dist.TruncatedNormal(
                loc=prior['loc'],
                scale=prior['scale'],
                low=prior.get('min', -np.inf),
                high=prior.get('max', np.inf)
            )
        
        elif dist_type == 'lognormal':
            return dist.LogNormal(prior['loc'], prior['scale'])
        
        elif dist_type == 'beta':
            return dist.Beta(prior['a'], prior['b'])
        
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    
    def transform_param(self, name: str, value: float, inverse: bool = False) -> float:
        """
        Apply parameter transformation.
        
        Args:
            name: Parameter name
            value: Parameter value
            inverse: Whether to apply inverse transformation
            
        Returns:
            Transformed value
        """
        if name not in self.params:
            return value
        
        transform = self.params[name].get('transform')
        if transform is None:
            return value
        
        if transform == 'log':
            if inverse:
                return jnp.exp(value)
            else:
                return jnp.log(value)
        
        elif transform == 'logit':
            prior = self.params[name]['prior']
            min_val = prior.get('min', 0)
            max_val = prior.get('max', 1)
            
            if inverse:
                # From logit space to original space
                exp_val = jnp.exp(value)
                p = exp_val / (1 + exp_val)
                return min_val + p * (max_val - min_val)
            else:
                # From original space to logit space
                p = (value - min_val) / (max_val - min_val)
                return jnp.log(p / (1 - p))
        
        else:
            raise ValueError(f"Unknown transformation: {transform}")
    
    def get_param_bounds(self, param_name: str) -> Tuple[float, float]:
        """
        Get parameter bounds from prior.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Tuple of (min, max) bounds
        """
        if param_name not in self.params:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        param_info = self.params[param_name]
        if param_info['fixed']:
            value = param_info['value']
            return (value, value)
        
        prior = param_info['prior']
        
        # Get bounds based on distribution type
        dist_type = prior.get('dist', 'uniform')
        
        if dist_type in ['uniform', 'truncated_normal']:
            return (prior.get('min', -np.inf), prior.get('max', np.inf))
        
        elif dist_type in ['normal', 'gaussian']:
            # Use 5-sigma bounds for normal distribution
            loc = prior['loc']
            scale = prior['scale']
            return (loc - 5*scale, loc + 5*scale)
        
        elif dist_type == 'lognormal':
            return (0, np.inf)
        
        elif dist_type == 'beta':
            return (0, 1)
        
        else:
            return (-np.inf, np.inf)
    
    def to_yaml(self, filename: str) -> None:
        """
        Save parameter configuration to YAML file.
        
        Args:
            filename: Output YAML file path
        """
        config = {
            'parameters': deepcopy(self.params),
            'order': self.param_order,
            'likelihood_nuisance': self._likelihood_nuisance
        }
        
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, filename: str) -> 'ParameterManager':
        """
        Load parameter configuration from YAML file.
        
        Args:
            filename: Input YAML file path
            
        Returns:
            ParameterManager instance
        """
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        
        manager = cls()
        manager.params = config['parameters']
        manager.param_order = config.get('order', list(manager.params.keys()))
        manager._likelihood_nuisance = config.get('likelihood_nuisance', {})
        
        return manager
    
    def get_summary_table(self) -> str:
        """
        Get a summary table of all parameters.
        
        Returns:
            Formatted parameter table string
        """
        lines = []
        lines.append("Parameter Summary:")
        lines.append("-" * 60)
        lines.append(f"{'Name':<15} {'Type':<10} {'Prior/Value':<30}")
        lines.append("-" * 60)
        
        for name in self.param_order:
            info = self.params[name]
            
            if info['fixed']:
                param_type = "Fixed"
                prior_str = f"{info['value']:.4f}"
            else:
                param_type = "Free"
                prior = info['prior']
                dist_type = prior.get('dist', 'uniform')
                
                if dist_type == 'uniform':
                    prior_str = f"U({prior['min']:.2f}, {prior['max']:.2f})"
                elif dist_type in ['normal', 'gaussian']:
                    prior_str = f"N({prior['loc']:.2f}, {prior['scale']:.2f})"
                else:
                    prior_str = dist_type
            
            lines.append(f"{name:<15} {param_type:<10} {prior_str:<30}")
        
        lines.append("-" * 60)
        lines.append(f"Total parameters: {len(self.params)}")
        lines.append(f"Free parameters: {len(self.get_free_params())}")
        lines.append(f"Fixed parameters: {len(self.get_fixed_params())}")
        
        return "\n".join(lines)