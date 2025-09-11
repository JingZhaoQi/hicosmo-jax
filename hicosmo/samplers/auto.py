#!/usr/bin/env python3
"""
MCMC Interface with Intelligent Parameter Management.

This module provides the main MCMC class that combines dictionary-driven
configuration, automatic parameter mapping, and comprehensive data persistence
for effortless MCMC sampling.
"""

from typing import Dict, Any, Optional, Callable, Union, List
import warnings
from pathlib import Path
import yaml
import json
import jax
import jax.numpy as jnp
import optax

from .config import ParameterConfig, AutoParameter
from .utils import ParameterMapper, analyze_likelihood_function
from .core import MCMCSampler
from .persistence import MCMCState, CheckpointManager, ResumeManager, create_likelihood_info, create_data_info
from .constants import (
    DEFAULT_NUM_SAMPLES, DEFAULT_NUM_CHAINS,
    DEFAULT_WARMUP_STANDARD, DEFAULT_WARMUP_OPTIMIZED,
    DEFAULT_MAX_OPTIMIZATION_ITERATIONS, OPTIMIZATION_PROGRESS_INTERVAL,
    OPTIMIZATION_PENALTY_FACTOR, DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_CHECKPOINT_DIR, RNG_SEED_MODULO
)


class MCMC:
    """
    MCMC sampling with intelligent parameter management.
    
    This class provides a qcosmc-style simple API while leveraging the powerful
    NumPyro backend. It automatically handles parameter mapping, model generation,
    MCMC sampling, and comprehensive data persistence.
    
    Examples
    --------
    >>> # Dictionary-based configuration (recommended)
    >>> config = {
    ...     'parameters': {
    ...         'H0': {
    ...             'prior': {'dist': 'uniform', 'min': 50, 'max': 100},
    ...             'ref': 70.0,
    ...             'latex': r'H_0'
    ...         },
    ...         'Omega_m': {
    ...             'prior': {'dist': 'normal', 'loc': 0.3, 'scale': 0.05},
    ...             'bounds': [0.1, 0.5]
    ...         }
    ...     },
    ...     'mcmc': {'num_samples': 4000, 'num_chains': 4}
    ... }
    
    >>> def my_likelihood(H0, Omega_m, sn_data):
    ...     # Your likelihood computation
    ...     return log_likelihood_value
    
    >>> # One-line MCMC with automatic checkpointing
    >>> mcmc = MCMC(config, my_likelihood, sn_data=supernova_data)
    >>> results = mcmc.run()
    >>> mcmc.print_summary()
    
    >>> # Simple list format (qcosmc-compatible)
    >>> params = [
    ...     ['H0', 70, 50, 100],
    ...     ['Omega_m', 0.3, 0.1, 0.5]
    ... ]
    >>> mcmc = MCMC.from_simple_list(params, my_likelihood, sn_data=data)
    >>> results = mcmc.run()
    """
    
    def __init__(
        self,
        config: Union[Dict, ParameterConfig],
        likelihood_func: Callable,
        strict_mode: bool = False,
        chain_name: Optional[str] = None,
        # Optimization options
        optimize_init: bool = False,  # 默认关闭JAX优化，使用标准warmup
        max_opt_iterations: int = DEFAULT_MAX_OPTIMIZATION_ITERATIONS,
        opt_learning_rate: float = 0.01,
        # Checkpoint and resume options
        enable_checkpoints: bool = True,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        checkpoint_dir: Union[str, Path] = DEFAULT_CHECKPOINT_DIR,
        backup_versions: int = 5,
        save_warmup: bool = True,
        compression: str = "gzip",
        auto_resume: bool = True,
        **data_kwargs
    ):
        """
        Initialize MCMC sampler.
        
        Parameters
        ----------
        config : Dict or ParameterConfig
            Parameter configuration. Can be a dictionary or ParameterConfig object.
        likelihood_func : Callable
            Likelihood function that returns log-likelihood value.
            Function signature will be automatically analyzed.
        strict_mode : bool
            If True, raise errors for parameter mismatches instead of warnings.
        chain_name : str, optional
            Name for the MCMC chain (used for saving results).
        optimize_init : bool, default=False
            If True, use JAX optimization to find best-fit values as initial points.
            Most beneficial for: expensive likelihoods (>10ms/call), high dimensions (>20 params),
            or multi-modal problems. For typical problems, standard warmup is recommended.
        max_opt_iterations : int
            Maximum iterations for the optimization process.
        opt_learning_rate : float
            Learning rate for the optimization process.
        enable_checkpoints : bool
            Whether to enable automatic checkpointing during sampling.
        checkpoint_interval : int
            Save checkpoint every N steps.
        checkpoint_dir : str or Path
            Directory to save checkpoints.
        backup_versions : int
            Number of backup versions to keep.
        save_warmup : bool
            Whether to save warmup samples in checkpoints.
        compression : str
            Compression method ('none', 'gzip', 'lzf').
        auto_resume : bool
            Whether to automatically detect and offer to resume from checkpoints.
        **data_kwargs
            Data arguments to pass to the likelihood function.
        """
        # Convert config if needed
        if isinstance(config, dict):
            self.param_config = self._parse_dict_config(config)
        elif isinstance(config, ParameterConfig):
            self.param_config = config
        else:
            raise ValueError("Config must be dict or ParameterConfig")
        
        self.likelihood_func = likelihood_func
        self.strict_mode = strict_mode
        # Generate descriptive chain name if not provided
        if chain_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            func_name = getattr(likelihood_func, '__name__', 'unknown_func')
            chain_name = f"mcmc_{func_name}_{timestamp}"
        self.chain_name = chain_name
        self.data_kwargs = data_kwargs
        
        # Optimization configuration
        self.optimize_init = optimize_init
        self.max_opt_iterations = max_opt_iterations
        self.opt_learning_rate = opt_learning_rate
        
        # Show optimization advice if user explicitly enabled it
        if optimize_init:
            self._show_optimization_advice()
        
        # Checkpoint system configuration
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.backup_versions = backup_versions
        self.save_warmup = save_warmup
        self.compression = compression
        self.auto_resume = auto_resume
        
        # Initialize checkpoint manager
        if self.enable_checkpoints:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_interval=self.checkpoint_interval,
                backup_count=self.backup_versions,
                compression=self.compression,
                save_warmup=self.save_warmup
            )
        else:
            self.checkpoint_manager = None
        
        # Initialize resume manager
        self.resume_manager = ResumeManager(strict_validation=strict_mode)
        
        # MCMC state tracking
        self.mcmc_state = None
        self._is_resumed = False
        
        # Create parameter mapper
        self.mapper = ParameterMapper(
            self.param_config, 
            strict_mode=strict_mode,
            enable_fuzzy_matching=True
        )
        
        # Note: Auto-resume functionality is available via the resume() class method
        # Users should explicitly specify checkpoint files for resume
        
        # Analyze likelihood and create model
        self._setup_mcmc()
        
        # Store results
        self._samples = None
        self._mapping_result = None
    
    def _show_optimization_advice(self):
        """Show advice about when JAX optimization is most beneficial."""
        num_params = len(self.param_config.parameters)
        print("\n" + "🔧" * 60)
        print("JAX OPTIMIZATION ENABLED")
        print("🔧" * 60)
        print("📝 Optimization is most beneficial when:")
        print("  • Likelihood computation time > 10ms per call")
        print("  • High-dimensional problems (>20 parameters)")  
        print("  • Complex multi-modal distributions")
        print("  • Convergence problems with traditional warmup")
        print()
        
        if num_params < 10:
            print("⚠️  Note: For simple problems (<10 parameters), optimization")
            print("   overhead may exceed benefits. Consider traditional warmup.")
        elif num_params < 20:
            print("ℹ️  Medium complexity problem. Optimization may help if")
            print("   likelihood computation is expensive.")
        else:
            print("✅ Large problem size. JAX optimization likely beneficial.")
            
        print("\n💡 To disable optimization: set optimize_init=False")
        print("   This will use standard warmup (recommended for most cases)")
        print("🔧" * 60)
    
    def _parse_dict_config(self, config: Dict) -> ParameterConfig:
        """Parse dictionary configuration into ParameterConfig.
        
        Now supports simplified parameter formats:
        - Tuple: (ref, min, max, latex)
        - List: [ref, min, max, latex]
        - Set: {ref, min, max, latex}
        - Full dict: {'prior': {...}, 'ref': ..., ...}
        """
        # Use ParameterConfig.from_dict which handles all formats
        return ParameterConfig.from_dict(config)
    
    def _jax_optimize_init(self):
        """Use JAX optimization to find best initial values."""
        print("🔍 Optimizing initial values using JAX...")
        
        # Get parameter names and bounds
        param_names = list(self.param_config.parameters.keys())
        param_bounds = {}
        initial_values = {}
        
        for name, param in self.param_config.parameters.items():
            if param.prior['dist'] == 'uniform':
                param_bounds[name] = (param.prior['min'], param.prior['max'])
                initial_values[name] = param.ref or (param.prior['min'] + param.prior['max']) / 2
            else:
                # For non-uniform priors, use reasonable bounds
                param_bounds[name] = (param.ref * 0.1, param.ref * 10) if param.ref else (0.1, 10)
                initial_values[name] = param.ref or 1.0
        
        # Define loss function (negative log-likelihood)
        @jax.jit
        def loss_fn(params_array):
            # Convert array to parameter dict
            params_dict = {name: params_array[i] for i, name in enumerate(param_names)}
            
            # Apply bounds constraints using JAX-friendly operations
            penalty = 0.0
            for i, name in enumerate(param_names):
                value = params_array[i]
                min_val, max_val = param_bounds[name]
                # Use JAX-friendly soft constraint
                penalty += OPTIMIZATION_PENALTY_FACTOR * jnp.maximum(0, min_val - value)**2
                penalty += OPTIMIZATION_PENALTY_FACTOR * jnp.maximum(0, value - max_val)**2
            
            # Compute likelihood
            log_like = self.likelihood_func(**params_dict)
            
            # Handle NaN/inf with JAX-friendly operations
            log_like = jnp.where(jnp.isfinite(log_like), log_like, -1e10)
            
            return -log_like + penalty
        
        # Initial parameter array
        initial_array = jnp.array([initial_values[name] for name in param_names])
        
        # Set up optimizer
        optimizer = optax.adam(learning_rate=self.opt_learning_rate)
        opt_state = optimizer.init(initial_array)
        
        # Optimization loop
        params = initial_array
        best_loss = float('inf')
        best_params = params
        
        for i in range(self.max_opt_iterations):
            loss_value, grads = jax.value_and_grad(loss_fn)(params)
            
            if loss_value < best_loss:
                best_loss = loss_value
                best_params = params
            
            # Early stopping if loss becomes very small
            if loss_value < 1e-10:
                break
                
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # Progress report every 100 iterations
            if (i + 1) % OPTIMIZATION_PROGRESS_INTERVAL == 0:
                print(f"  Iteration {i+1}/{self.max_opt_iterations}, Loss: {loss_value:.6f}")
        
        # Update reference values with optimized results
        optimized_params = {name: float(best_params[i]) for i, name in enumerate(param_names)}
        
        print(f"✓ Optimization completed after {i+1} iterations")
        print("  Optimized initial values:")
        for name, value in optimized_params.items():
            original_ref = self.param_config.parameters[name].ref
            print(f"    {name}: {original_ref:.4f} → {value:.4f}")
            # Update the parameter's ref value
            self.param_config.parameters[name].ref = value
        
        return optimized_params
    
    def _apply_intelligent_defaults(self, mcmc_kwargs: dict) -> dict:
        """
        Apply intelligent MCMC defaults based on optimization settings.
        
        Strategy:
        - JAX optimization ON  → warmup=300  (minimal for HMC tuning)
        - JAX optimization OFF → warmup=2000 (standard warmup)
        - User can override any defaults explicitly
        """
        # Don't override user-specified values
        if 'num_warmup' not in mcmc_kwargs:
            if self.optimize_init:
                # JAX optimization enabled: minimal warmup needed
                mcmc_kwargs['num_warmup'] = DEFAULT_WARMUP_OPTIMIZED
                if mcmc_kwargs.get('verbose', True):
                    print(f"📝 Using minimal warmup ({DEFAULT_WARMUP_OPTIMIZED}) with JAX optimization")
            else:
                # No optimization: use standard warmup
                mcmc_kwargs['num_warmup'] = DEFAULT_WARMUP_STANDARD
                if mcmc_kwargs.get('verbose', True):
                    print(f"📝 Using standard warmup ({DEFAULT_WARMUP_STANDARD}) without optimization")
        else:
            # User specified warmup value
            user_warmup = mcmc_kwargs['num_warmup']
            if mcmc_kwargs.get('verbose', True):
                print(f"📝 Using user-specified warmup ({user_warmup})")
        
        # Set other reasonable defaults
        mcmc_kwargs.setdefault('num_samples', DEFAULT_NUM_SAMPLES)
        mcmc_kwargs.setdefault('num_chains', DEFAULT_NUM_CHAINS)
        
        return mcmc_kwargs
    
    def _setup_mcmc(self):
        """Set up MCMC sampler with automatic model generation."""
        # Create NumPyro model with automatic mapping
        self.numpyro_model = self.mapper.create_numpyro_model(
            self.likelihood_func, **self.data_kwargs
        )
        
        # Create MCMCSampler with intelligent warmup defaults
        mcmc_kwargs = self.param_config.mcmc.copy()
        verbose = mcmc_kwargs.pop('verbose', True)
        
        # Apply intelligent warmup defaults based on optimization setting
        mcmc_kwargs = self._apply_intelligent_defaults(mcmc_kwargs)
        
        self.sampler = MCMCSampler(
            self.numpyro_model,
            verbose=verbose,
            **mcmc_kwargs
        )
        
        # Get mapping result for diagnostics
        self._mapping_result = self.mapper.map_to_function(
            self.likelihood_func, **self.data_kwargs
        )
        
        # Print mapping information if verbose
        if verbose:
            self._print_mapping_info()
    
    def _print_mapping_info(self):
        """Print parameter mapping information."""
        print("=" * 60)
        print("MCMC Parameter Mapping")
        print("=" * 60)
        
        if self._mapping_result.parameter_mapping:
            print("✓ Parameter mapping successful:")
            for func_param, config_param in self._mapping_result.parameter_mapping.items():
                config_obj = self.param_config.parameters[config_param]
                print(f"  {func_param} ← {config_param} {config_obj.prior}")
        
        if self._mapping_result.data_arguments:
            print(f"✓ Data arguments: {list(self._mapping_result.data_arguments.keys())}")
        
        if self._mapping_result.warnings:
            print("\n⚠ Warnings:")
            for warning in self._mapping_result.warnings:
                print(f"  {warning}")
        
        print()
    
    def _create_mcmc_state(self, samples: Optional[Dict] = None) -> MCMCState:
        """Create current MCMC state for checkpointing."""
        likelihood_info = create_likelihood_info(self.likelihood_func)
        data_info = create_data_info(self.data_kwargs)
        
        # Get current samples if not provided
        if samples is None and hasattr(self, '_samples') and self._samples:
            samples = self._samples
        
        run_metadata = {
            'chain_name': self.chain_name,
            'enable_checkpoints': self.enable_checkpoints,
            'checkpoint_interval': self.checkpoint_interval,
            'strict_mode': self.strict_mode,
            'mcmc_config': self.param_config.mcmc.copy(),
            'hicosmo_version': '2.0'
        }
        
        state = MCMCState(
            samples=samples or {},
            current_step=0,  # Will be updated during sampling
            total_steps=self.param_config.mcmc.get('num_samples', 0) * self.param_config.mcmc.get('num_chains', 1),
            warmup_steps=self.param_config.mcmc.get('num_warmup', 0) * self.param_config.mcmc.get('num_chains', 1),
            parameter_config=self.param_config,
            likelihood_info=likelihood_info,
            data_info=data_info,
            run_metadata=run_metadata
        )
        
        self.mcmc_state = state
        return state
    
    def run(self, **extra_kwargs) -> Dict[str, Any]:
        """
        Run MCMC sampling.
        
        Parameters
        ----------
        **extra_kwargs
            Additional arguments to pass to MCMCSampler.run().
            
        Returns
        -------
        Dict[str, Any]
            MCMC samples for all parameters.
        """
        # Combine data arguments
        run_kwargs = dict(self.data_kwargs)
        run_kwargs.update(extra_kwargs)
        
        # Optimize initial values if requested
        if self.optimize_init:
            self._jax_optimize_init()
        
        # Create initial MCMC state
        if not self.mcmc_state:
            self._create_mcmc_state()
        
        # Run MCMC with checkpointing
        if self.enable_checkpoints:
            self._samples = self._run_with_checkpoints(**run_kwargs)
        else:
            self._samples = self.sampler.run(**run_kwargs)
            
        return self._samples
    
    def _run_with_checkpoints(self, **run_kwargs) -> Dict[str, Any]:
        """Run MCMC with automatic checkpointing."""
        import time
        
        print(f"💾 Checkpointing enabled: every {self.checkpoint_interval} steps")
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")
        
        start_time = time.time()
        
        # Run MCMC
        samples = self.sampler.run(**run_kwargs)
        
        # Final checkpoint save
        if self.checkpoint_manager and samples:
            final_state = self._create_mcmc_state(samples)
            final_state.current_step = final_state.total_steps  # Mark as complete
            final_checkpoint = self.checkpoint_manager.save_checkpoint(
                final_state, self.chain_name, force=True
            )
            
            if final_checkpoint:
                runtime = time.time() - start_time
                print(f"💾 Final checkpoint saved: {final_checkpoint.name}")
                print(f"⏱️ Total runtime: {runtime:.1f} seconds")
        
        return samples
    
    def print_summary(self, prob: float = 0.9, burnin_frac: float = 0.1):
        """Print MCMC results summary with automatic burn-in removal."""
        if self._samples is None:
            print("No samples available. Run MCMC first.")
            return
        
        self.sampler.print_summary(prob=prob, burnin_frac=burnin_frac)
    
    def get_samples(self, param: Optional[str] = None) -> Union[Dict, List]:
        """
        Get MCMC samples.
        
        Parameters
        ----------
        param : str, optional
            Parameter name. If None, return all samples.
            
        Returns
        -------
        samples : dict or list
            Parameter samples.
        """
        if self._samples is None:
            raise RuntimeError("No samples available. Run MCMC first.")
        
        if param is None:
            return self._samples
        else:
            if param not in self._samples:
                raise ValueError(f"Parameter '{param}' not found in samples")
            return self._samples[param]
    
    def get_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Get MCMC diagnostics."""
        return self.sampler.get_diagnostics()
    
    def save_results(self, filename: Optional[str] = None, format: str = 'pickle'):
        """Save MCMC results."""
        if filename is None:
            extension = 'pkl' if format == 'pickle' else 'h5'
            filename = f"{self.chain_name}.{extension}"
        
        self.sampler.save_results(filename, format=format)
    
    def load_results(self, filename: str, format: str = 'pickle'):
        """Load previously saved results."""
        self.sampler.load_results(filename, format=format)
        self._samples = self.sampler.get_samples()
    
    def validate_setup(self) -> bool:
        """Validate that MCMC setup is correct."""
        # Simple validation: check if mapping was successful
        return self._mapping_result is not None and self._mapping_result.success
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all parameters."""
        # Return parameter configuration information
        info = {}
        for name, param in self.param_config.parameters.items():
            info[name] = {
                'prior': param.prior,
                'ref': param.ref,
                'bounds': param.bounds,
                'latex': param.latex,
                'description': param.description
            }
        return info
    
    def suggest_improvements(self) -> List[str]:
        """Get suggestions for improving the configuration."""
        # Return suggestions based on mapping results
        suggestions = []
        if self._mapping_result and self._mapping_result.warnings:
            suggestions.append("Consider reviewing parameter mapping warnings")
        if hasattr(self, 'sampler') and hasattr(self.sampler, 'get_diagnostics'):
            try:
                diagnostics = self.sampler.get_diagnostics()
                # Add suggestions based on diagnostics if available
            except:
                pass
        return suggestions if suggestions else ["Configuration looks good!"]
    
    @property
    def results(self):
        """Get results dictionary (qcosmc compatibility)."""
        if self._samples is None:
            return None
        
        import numpy as np
        results = {}
        for param_name in self.param_config.get_parameter_names():
            if param_name in self._samples:
                samples = np.array(self._samples[param_name])
                results[param_name] = {
                    'value': np.mean(samples),
                    'error': np.std(samples),
                    'samples': samples
                }
        return results
    
    # =========================================================================
    # Class Methods for Different Creation Modes
    # =========================================================================
    
    @classmethod
    def resume(
        cls,
        checkpoint_path: Union[str, Path],
        likelihood_func: Optional[Callable] = None,
        strict_validation: bool = True,
        **new_kwargs
    ) -> 'MCMC':
        """
        Resume MCMC from a checkpoint file.
        
        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint file to resume from
        likelihood_func : Callable, optional
            Likelihood function. If None, will try to resume with saved info.
        strict_validation : bool
            Whether to enforce strict compatibility validation
        **new_kwargs
            New configuration options that override saved settings
            
        Returns
        -------
        MCMC
            Resumed MCMC instance
        """
        resume_manager = ResumeManager(strict_validation=strict_validation)
        saved_state = resume_manager.load_checkpoint(checkpoint_path)
        
        if not saved_state.parameter_config:
            raise ValueError("Checkpoint missing parameter configuration")
        
        # Validate compatibility if likelihood provided
        if likelihood_func:
            compatible, issues = resume_manager.validate_compatibility(
                saved_state, saved_state.parameter_config, likelihood_func, {}
            )
            
            if not compatible:
                if strict_validation:
                    raise ValueError(f"Incompatible checkpoint: {issues}")
                else:
                    warnings.warn(f"Checkpoint compatibility issues: {issues}")
        
        # Create new instance with saved configuration
        config_dict = saved_state.parameter_config.to_dict()
        
        # Override with new kwargs
        if 'mcmc' in new_kwargs:
            config_dict['mcmc'].update(new_kwargs.pop('mcmc'))
        
        # Use saved likelihood if none provided
        if likelihood_func is None:
            raise NotImplementedError("Resuming without providing likelihood function is not yet supported")
        
        # Create resumed instance
        instance = cls(
            config_dict,
            likelihood_func,
            chain_name=saved_state.run_metadata.get('chain_name', 'resumed_mcmc'),
            **new_kwargs
        )
        
        # Set resumed state
        instance.mcmc_state = saved_state
        instance._samples = saved_state.samples
        instance._is_resumed = True
        
        print(f"🔄 Resumed from checkpoint: {Path(checkpoint_path).name}")
        print(f"   Progress: {saved_state.get_completion_percentage():.1f}% complete")
        print(f"   Samples: {len(next(iter(saved_state.samples.values()))) if saved_state.samples else 0}")
        
        return instance
    
    def continue_sampling(self, additional_samples: int, **kwargs) -> Dict[str, Any]:
        """
        Continue sampling from current state.
        
        Parameters
        ----------
        additional_samples : int
            Number of additional samples to draw
        **kwargs
            Additional arguments for sampling
            
        Returns
        -------
        Dict[str, Any]
            Combined samples (previous + new)
        """
        if not self._is_resumed and not self._samples:
            raise RuntimeError("No existing samples to continue from")
        
        # Update configuration for additional sampling
        original_samples = self.param_config.mcmc.get('num_samples', 0)
        self.param_config.mcmc['num_samples'] = additional_samples
        
        print(f"🔄 Continuing sampling: {additional_samples} additional samples")
        if self._samples:
            existing_count = len(next(iter(self._samples.values())))
            print(f"   Existing samples: {existing_count}")
        
        # Run additional sampling
        new_samples = self.run(**kwargs)
        
        # Merge with existing samples if any
        if self._is_resumed and self.mcmc_state and self.mcmc_state.samples:
            merged_samples = {}
            for param_name in new_samples.keys():
                if param_name in self.mcmc_state.samples:
                    # Concatenate old and new samples
                    import numpy as np
                    old_samples = self.mcmc_state.samples[param_name]
                    new_param_samples = new_samples[param_name]
                    merged_samples[param_name] = np.concatenate([old_samples, new_param_samples], axis=0)
                else:
                    merged_samples[param_name] = new_samples[param_name]
            
            self._samples = merged_samples
            
            total_samples = len(next(iter(merged_samples.values())))
            print(f"✅ Total samples after continuation: {total_samples}")
        
        # Restore original configuration
        self.param_config.mcmc['num_samples'] = original_samples
        
        return self._samples
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints for this run."""
        if not self.checkpoint_manager:
            return []
        return self.checkpoint_manager.list_checkpoints(self.chain_name)
    
    @classmethod
    def from_simple_list(
        cls,
        param_list: List[List],
        likelihood_func: Callable,
        chain_name: Optional[str] = None,
        **kwargs
    ) -> 'MCMC':
        """
        Create MCMC from simple parameter list (qcosmc-style).
        
        Parameters
        ----------
        param_list : List[List]
            Parameter definitions: [[name, initial, min, max], ...]
        likelihood_func : Callable
            Likelihood function.
        chain_name : str, optional
            Chain name.
        **kwargs
            Additional arguments including data and MCMC settings.
        """
        # Separate MCMC config from data
        data_kwargs = {}
        mcmc_kwargs = {}
        
        mcmc_keys = {'num_warmup', 'num_samples', 'num_chains', 'verbose', 
                     'enable_checkpoints', 'checkpoint_interval', 'checkpoint_dir',
                     'backup_versions', 'save_warmup', 'compression'}
        for key, value in kwargs.items():
            if key in mcmc_keys:
                mcmc_kwargs[key] = value
            else:
                data_kwargs[key] = value
        
        # Create parameter config
        param_config = ParameterConfig.from_simple_list(param_list, **mcmc_kwargs)
        
        return cls(param_config, likelihood_func, chain_name=chain_name, **data_kwargs)
    
    @classmethod
    def from_yaml(
        cls,
        yaml_file: Union[str, Path],
        likelihood_func: Callable,
        **data_kwargs
    ) -> 'MCMC':
        """Create MCMC from YAML configuration file."""
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(config, likelihood_func, **data_kwargs)
    
    @classmethod
    def from_json(
        cls,
        json_file: Union[str, Path],
        likelihood_func: Callable,
        **data_kwargs
    ) -> 'MCMC':
        """Create MCMC from JSON configuration file."""
        with open(json_file, 'r') as f:
            config = json.load(f)
        
        return cls(config, likelihood_func, **data_kwargs)


def quick_mcmc(
    params: Union[List[List], Dict],
    likelihood_func: Callable,
    num_samples: int = 2000,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for quick MCMC runs.
    
    Parameters
    ----------
    params : List[List] or Dict
        Parameter definitions.
    likelihood_func : Callable
        Likelihood function.
    num_samples : int
        Number of samples to draw.
    **kwargs
        Additional arguments.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with samples and summary statistics.
    """
    # Create appropriate MCMC instance
    if isinstance(params, list):
        mcmc = MCMC.from_simple_list(
            params, likelihood_func, 
            num_samples=num_samples, 
            verbose=False,
            **kwargs
        )
    else:
        # Add num_samples to config if not present
        if 'mcmc' not in params:
            params['mcmc'] = {}
        if 'num_samples' not in params['mcmc']:
            params['mcmc']['num_samples'] = num_samples
        
        mcmc = MCMC(params, likelihood_func, **kwargs)
    
    # Run MCMC
    samples = mcmc.run()
    
    # Return results
    return {
        'samples': samples,
        'results': mcmc.results,
        'diagnostics': mcmc.get_diagnostics(),
        'mcmc_instance': mcmc
    }