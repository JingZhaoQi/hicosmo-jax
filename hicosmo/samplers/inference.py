#!/usr/bin/env python3
"""
MCMC interface for cosmological parameter estimation.

This module provides the MCMC class for dictionary-driven configuration with
automatic parameter mapping, comprehensive data persistence, and checkpoint
management. Ideal for complex analyses requiring fine-grained control.

Leverages NumPyro's NUTS sampler for efficient Bayesian inference.
"""

from typing import Dict, Any, Optional, Callable, Union, List, Type
import copy
import os
import sys
import warnings
import inspect
from pathlib import Path
import yaml
import json
import jax
import jax.numpy as jnp
import optax
import numpyro
from numpyro.infer import SA, NUTS, HMC, MCMC as NumPyro_MCMC

from .config import ParameterConfig
from .utils import MappingResult, create_direct_mapping
from .persistence import (
    MCMCState,
    CheckpointManager,
    ResumeManager,
    create_likelihood_info,
    create_data_info,
)
from ..parameters import ParameterRegistry, build_parameter_error_message
from .constants import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_NUM_CHAINS,
    DEFAULT_WARMUP_STANDARD,
    DEFAULT_WARMUP_OPTIMIZED,
    DEFAULT_MAX_OPTIMIZATION_ITERATIONS,
    OPTIMIZATION_PROGRESS_INTERVAL,
    OPTIMIZATION_PENALTY_FACTOR,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
    DEFAULT_CHECKPOINT_DIR,
    RNG_SEED_MODULO,
    get_chains_dir,
)
from .init import Config

# Multi-sampler backend support
from .base import (
    SamplerBackend,
    SamplerConfig,
    SamplerResults,
    InitializationError,
    ConvergenceError,
)
from .derived import compute_derived_parameters
from .numpyro_backend import NumPyroSampler
from .emcee_backend import EmceeSampler
from ..utils.logging import get_logger
from ..likelihoods.base import Likelihood

logger = get_logger(__name__)
_JAX_ARRAY_TYPE = getattr(jax, "Array", ())


def _is_jax_value(value: Any) -> bool:
    """Return True for JAX arrays or tracers (avoids float() on tracers)."""
    return isinstance(value, _JAX_ARRAY_TYPE) or isinstance(value, jax.core.Tracer)


class MCMC:
    """
    MCMC sampling with intelligent parameter management and multi-sampler support.

    This class provides a qcosmc-style simple API while supporting multiple MCMC backends
    (NumPyro NUTS, emcee, etc.). It automatically handles parameter mapping, model generation,
    MCMC sampling, and comprehensive data persistence.

    Supported samplers:
    - 'numpyro' (default): NumPyro NUTS (fast, gradient-based, JAX-compatible)
    - 'emcee': Ensemble sampler (robust, no gradients needed, handles NaN/Inf gracefully)

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
    ...     'mcmc': {'num_samples': 4000, 'num_chains': 4}  # 4000 total samples
    ... }

    # Note: num_samples represents TOTAL samples across all chains.
    # With 4 chains, this will be 1000 samples per chain (4000/4=1000).

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

    # Available sampler backends (class-level registry)
    AVAILABLE_SAMPLERS = {
        "numpyro": NumPyroSampler,
        "emcee": EmceeSampler,
    }

    def __init__(
        self,
        config: Union[Dict, ParameterConfig, "ParameterRegistry"],
        likelihood_func: Callable,
        strict_mode: bool = False,
        chain_name: Optional[str] = None,
        # Automatic nuisance parameter collection (NEW)
        likelihoods: Optional[List] = None,  # Auto-collect nuisance parameters
        # Sampler selection (NEW - multi-sampler support)
        sampler: str = "numpyro",  # 'numpyro' (default) or 'emcee'
        # Chain parallelization method
        chain_method: str = "auto",  # 'auto', 'vectorized', 'sequential', or 'parallel'
        # Optimization options
        optimize_init: bool = False,  # Default: OFF. Standard warmup recommended for most cases
        max_opt_iterations: int = DEFAULT_MAX_OPTIMIZATION_ITERATIONS,
        opt_learning_rate: float = 0.01,
        # Checkpoint and resume options
        enable_checkpoints: bool = True,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        checkpoint_interval_seconds: Optional[
            float
        ] = DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
        checkpoint_dir: Union[str, Path] = DEFAULT_CHECKPOINT_DIR,
        backup_versions: int = 5,
        save_warmup: bool = True,
        compression: str = "gzip",
        auto_resume: bool = True,
        **data_kwargs,
    ):
        """
        Initialize MCMC sampler.

        Parameters
        ----------
        config : Dict, ParameterConfig, or ParameterRegistry
            Parameter configuration. Can be a dictionary, ParameterConfig, or ParameterRegistry.
        likelihood_func : Callable
            Likelihood function that returns log-likelihood value.
            Function signature will be automatically analyzed.
        strict_mode : bool
            If True, raise errors for parameter mismatches instead of warnings.
        chain_name : str, optional
            Name for the MCMC chain (used for saving results).
        likelihoods : List[Likelihood], optional
            List of likelihood objects to automatically extract nuisance parameters.
            This enables automatic nuisance parameter discovery without manual specification.
            Each likelihood should have a `nuisance_parameters` property returning Parameter objects.
        chain_method : str, default='auto'
            How to run multiple chains:
            - 'auto': Choose 'parallel' when enough devices are available, else 'vectorized'
            - 'vectorized': Run chains in parallel using vmap (recommended for CPU multi-core)
            - 'sequential': Run chains one after another (single-threaded)
            - 'parallel': Run chains on multiple devices using pmap (for multi-GPU)
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
            Save checkpoint every N steps (TOTAL samples across all chains).
        checkpoint_interval_seconds : float or None
            Save checkpoint every N seconds (time-based). If set, this takes
            precedence over step-based saving.
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
        elif isinstance(config, ParameterRegistry):
            # Convert ParameterRegistry to ParameterConfig
            self.param_config = self._convert_registry_to_config(config)
        else:
            raise ValueError(
                "Config must be dict, ParameterConfig, or ParameterRegistry"
            )

        # Store likelihoods for derived parameter computation
        self._likelihoods = likelihoods or []

        # Auto-collect nuisance parameters from likelihoods
        if likelihoods:
            self._collect_nuisance_from_likelihoods(likelihoods)

        self.likelihood_func = likelihood_func
        self.strict_mode = strict_mode
        # Generate descriptive chain name if not provided
        if chain_name is None:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            func_name = getattr(likelihood_func, "__name__", "unknown_func")
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
        if checkpoint_interval_seconds is not None and checkpoint_interval_seconds <= 0:
            checkpoint_interval_seconds = None
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        if self.checkpoint_interval_seconds is not None:
            logger.debug(
                "Using time-based checkpointing; step-based interval will be ignored."
            )
        # Use global chains directory if default, otherwise use user-specified
        if checkpoint_dir == DEFAULT_CHECKPOINT_DIR:
            self.checkpoint_dir = get_chains_dir()
        else:
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
                save_warmup=self.save_warmup,
            )
        else:
            self.checkpoint_manager = None

        # Initialize resume manager
        self.resume_manager = ResumeManager(strict_validation=strict_mode)

        # MCMC state tracking
        self.mcmc_state = None
        self._is_resumed = False
        self._resume_total_steps = None
        self._resume_current_step = None
        self._resume_backend_state = None
        self._mcmc_counts_are_per_chain = False

        # Sampler selection and validation (NEW - multi-sampler support)
        self.sampler_name = sampler.lower()
        if self.sampler_name not in self.AVAILABLE_SAMPLERS:
            raise ValueError(
                f"Unknown sampler: '{sampler}'. "
                f"Available samplers: {list(self.AVAILABLE_SAMPLERS.keys())}"
            )

        # Chain parallelization method
        self.chain_method = chain_method

        # Note: Simplified architecture uses direct parameter mapping
        # No more ParameterMapper - parameters must match exactly by name

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
        logger.debug("\n" + "🔧" * 60)
        logger.debug("JAX OPTIMIZATION ENABLED")
        logger.debug("🔧" * 60)
        logger.debug("📝 Optimization is most beneficial when:")
        logger.debug("  • Likelihood computation time > 10ms per call")
        logger.debug("  • High-dimensional problems (>20 parameters)")
        logger.debug("  • Complex multi-modal distributions")
        logger.debug("  • Convergence problems with traditional warmup")
        logger.debug("")

        if num_params < 10:
            logger.debug("Note: For simple problems (<10 parameters), optimization")
            logger.debug("overhead may exceed benefits. Consider traditional warmup.")
        elif num_params < 20:
            logger.debug("Medium complexity problem. Optimization may help if")
            logger.debug("likelihood computation is expensive.")
        else:
            logger.debug("Large problem size. JAX optimization likely beneficial.")

        logger.debug("\n💡 To disable optimization: set optimize_init=False")
        logger.debug("This will use standard warmup (recommended for most cases)")
        logger.debug("🔧" * 60)

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

    def _convert_registry_to_config(
        self, registry: "ParameterRegistry"
    ) -> ParameterConfig:
        """Convert ParameterRegistry to ParameterConfig.

        Parameters
        ----------
        registry : ParameterRegistry
            Parameter registry containing Parameter objects.

        Returns
        -------
        ParameterConfig
            Equivalent ParameterConfig for MCMC sampling.
        """
        from ..parameters import Parameter

        all_params = registry.get_all()
        if not any(param.is_free() for param in all_params.values()):
            raise ValueError(
                "ParameterRegistry has no free parameters. "
                "Use registry.set_free(['param1', 'param2']) to mark parameters for sampling."
            )

        return ParameterConfig(parameters=copy.deepcopy(all_params))

    def _collect_nuisance_from_likelihoods(self, likelihoods: List) -> None:
        """Collect nuisance parameters from likelihood objects.

        Parameters
        ----------
        likelihoods : List[Likelihood]
            List of likelihood objects with `nuisance_parameters` property.

        Notes
        -----
        This method automatically extracts nuisance parameters from likelihoods
        and adds them to the parameter configuration. Parameters already present
        in the config are skipped to avoid conflicts.
        """
        from ..parameters import Parameter

        added_params = []

        for lik in likelihoods:
            if not hasattr(lik, "nuisance_parameters"):
                continue

            # Handle both method and property for nuisance_parameters
            nuisance_attr = lik.nuisance_parameters
            nuisance_params = (
                nuisance_attr() if callable(nuisance_attr) else nuisance_attr
            )

            if not nuisance_params:
                continue

            # Support legacy dict-style nuisance parameters
            if isinstance(nuisance_params, dict):
                converted = []
                for name, spec in nuisance_params.items():
                    try:
                        converted.append(Parameter.from_simple_config(name, spec))
                    except Exception as exc:
                        warnings.warn(
                            f"Skipping nuisance parameter '{name}' (invalid spec): {exc}"
                        )
                nuisance_params = converted

            for param in nuisance_params:
                if not isinstance(param, Parameter):
                    warnings.warn(
                        f"Skipping non-Parameter object in {lik.__class__.__name__}.nuisance_parameters"
                    )
                    continue

                # Skip if already in config
                if param.name in self.param_config.parameters:
                    continue

                # Add nuisance parameter to config
                try:
                    self.param_config.add_parameter(param)
                    added_params.append((param.name, lik.__class__.__name__))
                except ValueError:
                    # Parameter already exists, skip
                    pass

        if added_params:
            param_list = ", ".join(
                [f"{name} (from {src})" for name, src in added_params]
            )
            logger.debug(f"Auto-collected nuisance parameters: {param_list}")

    def _collect_nuisance_from_likelihood_object(
        self, likelihood_obj, verbose: bool = True
    ) -> None:
        """
        Collect nuisance parameters from a likelihood object or CombinedLikelihood.

        Parameters
        ----------
        likelihood_obj : callable
            A likelihood object (SN_likelihood, BAO_likelihood, CombinedLikelihood, etc.)
        verbose : bool
            Whether to print info about collected parameters.

        Notes
        -----
        This method handles:
        - Single likelihood objects with `nuisance_parameters` property
        - CombinedLikelihood objects (iterates over all contained likelihoods)
        - Callable methods like `nuisance_parameters()` or properties
        """
        from ..parameters import Parameter

        # Collect all likelihood objects to check
        likelihoods_to_check = []

        # Check if it's a CombinedLikelihood (has 'likelihoods' or '_likelihoods' attribute)
        if hasattr(likelihood_obj, "_likelihoods"):
            likelihoods_to_check.extend(likelihood_obj._likelihoods)
        elif hasattr(likelihood_obj, "likelihoods"):
            liks = likelihood_obj.likelihoods
            if callable(liks):
                liks = liks()
            likelihoods_to_check.extend(liks)
        else:
            # Single likelihood object
            likelihoods_to_check.append(likelihood_obj)

        added_params = []

        for lik in likelihoods_to_check:
            # Get nuisance parameters (could be property or method)
            nuisance_params = None

            if hasattr(lik, "nuisance_parameters"):
                nuisance = lik.nuisance_parameters
                # Could be a property (returns list) or method (needs to be called)
                if callable(nuisance):
                    nuisance_params = nuisance()
                else:
                    nuisance_params = nuisance

            if not nuisance_params:
                continue

            # Support legacy dict-style nuisance parameters
            if isinstance(nuisance_params, dict):
                converted = []
                for name, spec in nuisance_params.items():
                    try:
                        converted.append(Parameter.from_simple_config(name, spec))
                    except Exception as exc:
                        warnings.warn(
                            f"Skipping nuisance parameter '{name}' (invalid spec): {exc}"
                        )
                nuisance_params = converted

            for param in nuisance_params:
                if not isinstance(param, Parameter):
                    if verbose:
                        warnings.warn(
                            f"Skipping non-Parameter object in {lik.__class__.__name__}.nuisance_parameters"
                        )
                    continue

                # Skip if already in config
                if param.name in self.param_config.parameters:
                    continue

                # Add nuisance parameter to config
                try:
                    self.param_config.add_parameter(param)
                    added_params.append((param.name, lik.__class__.__name__))
                except (ValueError, AttributeError):
                    # Parameter already exists or add_parameter not available, skip
                    pass

        if added_params and verbose:
            param_list = ", ".join(
                [f"{name} (from {src})" for name, src in added_params]
            )
            logger.debug(f"Auto-collected nuisance parameters: {param_list}")

    def _collect_likelihood_metadata(self) -> tuple:
        """
        Collect normalization constant and num_data from likelihood objects.

        This is needed for correct AIC/BIC calculation:
            χ² = -2 * (log_likelihood - normalization_constant)
            AIC = χ²_min + 2k
            BIC = χ²_min + k * ln(N)

        Returns
        -------
        tuple
            (normalization_constant, num_data) - either may be None
        """
        norm_const = 0.0
        num_data = 0
        has_norm = False
        has_num_data = False

        # Collect from registered likelihoods
        likelihoods_to_check = list(self._likelihoods) if self._likelihoods else []

        # Also check if likelihood_func is a likelihood object
        if hasattr(self.likelihood_func, "get_normalization_constant"):
            if self.likelihood_func not in likelihoods_to_check:
                likelihoods_to_check.append(self.likelihood_func)

        for lik in likelihoods_to_check:
            # Handle CombinedLikelihood - get its constituent likelihoods
            if hasattr(lik, "_likelihoods"):
                for sub_lik in lik._likelihoods:
                    if hasattr(sub_lik, "get_normalization_constant"):
                        try:
                            norm_const += float(sub_lik.get_normalization_constant())
                            has_norm = True
                        except Exception:
                            pass
                    if hasattr(sub_lik, "get_num_data"):
                        try:
                            nd = sub_lik.get_num_data()
                            if nd is not None:
                                num_data += int(nd)
                                has_num_data = True
                        except Exception:
                            pass
            else:
                # Single likelihood
                if hasattr(lik, "get_normalization_constant"):
                    try:
                        norm_const += float(lik.get_normalization_constant())
                        has_norm = True
                    except Exception:
                        pass
                if hasattr(lik, "get_num_data"):
                    try:
                        nd = lik.get_num_data()
                        if nd is not None:
                            num_data += int(nd)
                            has_num_data = True
                    except Exception:
                        pass

        return (norm_const if has_norm else None, num_data if has_num_data else None)

    def _jax_optimize_init(self):
        """Use JAX optimization to find best initial values."""
        logger.debug("Optimizing initial values using JAX...")

        # Get parameter names and bounds
        param_names = list(self.param_config.parameters.keys())
        param_bounds = {}
        initial_values = {}

        for name, param in self.param_config.parameters.items():
            if param.prior["dist"] == "uniform":
                param_bounds[name] = (param.prior["min"], param.prior["max"])
                initial_values[name] = (
                    param.ref or (param.prior["min"] + param.prior["max"]) / 2
                )
            else:
                # For non-uniform priors, use reasonable bounds
                param_bounds[name] = (
                    (param.ref * 0.1, param.ref * 10) if param.ref else (0.1, 10)
                )
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
                penalty += (
                    OPTIMIZATION_PENALTY_FACTOR * jnp.maximum(0, min_val - value) ** 2
                )
                penalty += (
                    OPTIMIZATION_PENALTY_FACTOR * jnp.maximum(0, value - max_val) ** 2
                )

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
        best_loss = float("inf")
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
                logger.debug(
                    f"Iteration {i+1}/{self.max_opt_iterations}, Loss: {loss_value:.6f}"
                )

        # Update reference values with optimized results
        optimized_params = {
            name: float(best_params[i]) for i, name in enumerate(param_names)
        }

        logger.debug(f"Optimization completed after {i+1} iterations")
        logger.debug("Optimized initial values:")
        for name, value in optimized_params.items():
            original_ref = self.param_config.parameters[name].ref
            logger.debug(f"  {name}: {original_ref:.4f} → {value:.4f}")
            # Update the parameter's ref value
            self.param_config.parameters[name].ref = value

        return optimized_params

    def _apply_intelligent_defaults(self, mcmc_kwargs: dict) -> dict:
        """
        Apply intelligent MCMC defaults based on system configuration.

        Smart Defaults Strategy:
        1. num_chains: Auto-detect based on available CPU cores
        2. num_samples: interpreted as TOTAL across chains (split per chain)
        3. num_warmup: interpreted as PER CHAIN (adaptation cannot be
           amortized across chains); default max(200, min(1000, per-chain))
        4. optimize_init: False by default (standard warmup preferred)
        5. checkpointing: Enabled by default

        User can override any defaults explicitly.
        """
        verbose = mcmc_kwargs.get("verbose", True)
        if self._mcmc_counts_are_per_chain:
            return mcmc_kwargs

        # 1. Smart default for num_samples
        mcmc_kwargs.setdefault("num_samples", DEFAULT_NUM_SAMPLES)
        total_samples = mcmc_kwargs["num_samples"]

        # 2. Smart default for num_chains: match CPU cores (but reasonable limits)
        default_warmup = False

        if "num_chains" not in mcmc_kwargs:
            config_state = getattr(Config, "_config", {})
            preferred_chains = config_state.get("preferred_num_chains")
            if preferred_chains:
                optimal_chains = max(1, int(preferred_chains))
            else:
                configured = config_state.get("actual_cores")
                if configured:
                    optimal_chains = max(1, min(int(configured), 8))
                else:
                    available_threads = int(
                        os.environ.get("JAX_NUM_THREADS", os.cpu_count() or 1)
                    )
                    optimal_chains = max(1, min(available_threads, 8))

            mcmc_kwargs["num_chains"] = optimal_chains
            if verbose:
                if preferred_chains:
                    logger.debug(
                        f"Auto-detected num_chains: {optimal_chains} (based on device configuration)"
                    )
                else:
                    logger.debug(
                        f"Auto-detected num_chains: {optimal_chains} (based on configured CPU threads)"
                    )
        else:
            if verbose:
                logger.debug(
                    f"Using user-specified num_chains: {mcmc_kwargs['num_chains']}"
                )

        num_chains = mcmc_kwargs["num_chains"]

        # 🔧 num_samples is TOTAL across chains; num_warmup is PER CHAIN.
        # Warmup is each chain's independent mass-matrix/step-size adaptation
        # and cannot be amortized across chains: dividing it by num_chains
        # (the previous behavior) left 4 chains with 100 steps each, far below
        # what NUTS window adaptation needs, and degraded tau for d>=3.
        total_samples = mcmc_kwargs["num_samples"]
        num_chains = mcmc_kwargs["num_chains"]

        samples_per_chain = (total_samples + num_chains - 1) // num_chains
        mcmc_kwargs["num_samples"] = samples_per_chain

        # 3. Smart default for num_warmup: per-chain, aligned with NumPyro's
        # adaptation needs (capped by the per-chain sample count for tiny runs)
        if "num_warmup" not in mcmc_kwargs:
            warmup_per_chain = max(200, min(1000, samples_per_chain))
            mcmc_kwargs["num_warmup"] = warmup_per_chain
            if verbose:
                logger.debug(f"Auto-calculated warmup: {warmup_per_chain} per chain")
        else:
            warmup_per_chain = mcmc_kwargs["num_warmup"]
            if verbose:
                logger.debug(
                    f"Using user-specified warmup: {warmup_per_chain} per chain"
                )

        if verbose and num_chains > 1:
            logger.debug(
                f"Sample distribution: {total_samples} total → "
                f"{samples_per_chain}/chain × {num_chains}; "
                f"warmup {warmup_per_chain}/chain"
            )

        self._mcmc_counts_are_per_chain = True
        return mcmc_kwargs

    def _setup_mcmc(self):
        """Set up MCMC sampler with automatic model generation and multi-sampler support."""
        # Get verbose setting
        mcmc_kwargs = self.param_config.mcmc.copy()
        verbose = mcmc_kwargs.pop("verbose", True)

        # NEW: Multi-sampler backend selection
        if self.sampler_name in ["numpyro", "emcee"]:
            # Use new unified backend architecture
            self._setup_unified_backend(verbose)
        else:
            # Legacy path (should not reach here due to validation in __init__)
            raise ValueError(f"Unsupported sampler: {self.sampler_name}")

    def _split_sampled_and_fixed_parameters(self) -> tuple[List[str], Dict[str, Any]]:
        """Return sampled parameter names plus fixed values for likelihood calls."""
        sampled_param_names = [
            name
            for name, param in self.param_config.parameters.items()
            if param.is_free()
        ]
        fixed_values = {
            name: param.value
            for name, param in self.param_config.parameters.items()
            if not param.is_free()
        }
        if not sampled_param_names:
            raise ValueError(
                "No free parameters configured for MCMC. "
                "Mark at least one parameter as free before sampling."
            )
        return sampled_param_names, fixed_values

    def _setup_unified_backend(self, verbose: bool = True):
        """Set up unified sampler backend (NumPyro or emcee)."""
        # Explicit likelihood protocol: use base class rather than heuristics
        is_likelihood_object = isinstance(self.likelihood_func, Likelihood)

        sampled_param_names, fixed_values = self._split_sampled_and_fixed_parameters()

        # For likelihood objects, we bypass the normal parameter mapping
        # and directly use the parameter names from config
        if is_likelihood_object:
            # Auto-collect nuisance parameters from likelihood object(s)
            self._collect_nuisance_from_likelihood_object(self.likelihood_func, verbose)

            sampled_param_names, fixed_values = (
                self._split_sampled_and_fixed_parameters()
            )

            # Create log probability function that directly calls the likelihood object
            def log_probability(params: Dict[str, float]) -> float:
                """
                Log probability function for likelihood objects.

                Directly passes parameters from sampler to likelihood.__call__(**params).

                Exceptions must propagate: this runs during JAX tracing, so any
                exception is a structural bug (shape/type/missing data), not a
                numerical failure. Swallowing it and returning a constant would
                make NUTS silently sample the prior. Numerical NaN/Inf at runtime
                is handled by the backend's isfinite guard instead.
                """
                # Build kwargs from sampler params
                call_kwargs = dict(fixed_values)
                call_kwargs.update(
                    {
                        name: params[name]
                        for name in sampled_param_names
                        if name in params
                    }
                )

                # Call likelihood object directly
                log_L = self.likelihood_func(**call_kwargs)

                # NumPyro expects JAX-friendly outputs; keep tracers untouched
                if _is_jax_value(log_L):
                    return log_L

                return float(log_L)

            # Create a simple mapping result for compatibility
            mapping_result = MappingResult(
                parameter_mapping={name: name for name in sampled_param_names},
                data_arguments={},
                missing_parameters=[],
                unused_parameters=[],
                warnings=[],
                success=True,
            )
        else:
            # Standard function: use direct parameter mapping (simplified)
            mapping_result = create_direct_mapping(
                sampled_param_names, self.data_kwargs
            )

            # Capture data_kwargs in closure
            data_args = dict(self.data_kwargs) if self.data_kwargs else {}

            # Create log probability function for backend
            def log_probability(params: Dict[str, float]) -> float:
                """
                Unified log probability function for sampler backends.

                Exceptions must propagate (see likelihood-object variant above):
                trace-time exceptions are structural bugs, and masking them with
                a constant makes NUTS silently sample the prior.
                """
                # Build kwargs from sampler params (direct mapping)
                call_kwargs = dict(fixed_values)
                call_kwargs.update(
                    {
                        name: params[name]
                        for name in sampled_param_names
                        if name in params
                    }
                )

                # Add data arguments
                call_kwargs.update(data_args)

                # Call likelihood function
                log_L = self.likelihood_func(**call_kwargs)

                # NumPyro expects JAX-friendly outputs; keep tracers untouched
                if _is_jax_value(log_L):
                    return log_L

                return float(log_L)

        # Extract parameter configuration for backend
        # Convert ParameterConfig to backend-compatible format
        parameters_dict = {}
        for name in sampled_param_names:
            param = self.param_config.parameters[name]
            parameters_dict[name] = {
                "prior": param.prior,
                "ref": param.value,  # Parameter uses 'value', not 'ref'
                "latex": param.latex_label,  # Parameter uses 'latex_label', not 'latex'
            }

        # Backend-specific configuration (NumPyroSampler reads init/dense_mass options from here)
        backend_config = self.param_config.mcmc.copy()

        # Instantiate selected backend
        SamplerClass = self.AVAILABLE_SAMPLERS[self.sampler_name]
        self.backend = SamplerClass(
            log_probability=log_probability,
            parameters=parameters_dict,
            config=backend_config,
        )

        # For compatibility with existing code, create a wrapper that mimics legacy sampler interface
        self._create_backend_wrapper(verbose)

        # Store mapping result for diagnostics / introspection
        self._mapping_result = mapping_result

        # Print mapping information if verbose
        if verbose:
            self._print_mapping_info()
            logger.debug(f"Using {self.sampler_name} sampler backend")

    def _create_backend_wrapper(self, verbose: bool = True):
        """Create a wrapper to make new backend compatible with existing MCMC class interface."""
        # Capture chain_method from outer MCMC instance
        outer_chain_method = self.chain_method

        # This wrapper provides backward compatibility
        class BackendWrapper:
            def __init__(wrapper_self, backend, param_config, verbose):
                wrapper_self.backend = backend
                wrapper_self.param_config = param_config
                wrapper_self.verbose = verbose
                wrapper_self.samples = None
                wrapper_self.results = None

            def _build_sampler_config(wrapper_self, **kwargs) -> SamplerConfig:
                """Build SamplerConfig with consistent chain_method handling."""
                mcmc_config = wrapper_self.param_config.mcmc.copy()
                mcmc_config.update(kwargs)

                requested_chain_method = outer_chain_method
                if requested_chain_method not in {
                    "auto",
                    "vectorized",
                    "sequential",
                    "parallel",
                }:
                    raise ValueError(
                        "chain_method must be one of: 'auto', 'vectorized', 'sequential', 'parallel'"
                    )

                device_count = jax.local_device_count()
                num_chains = int(mcmc_config.get("num_chains", 4))
                if requested_chain_method == "auto":
                    if (
                        wrapper_self.backend.name == "numpyro"
                        and device_count > 1
                        and num_chains <= device_count
                    ):
                        chain_method = "parallel"
                    else:
                        chain_method = "vectorized"
                else:
                    chain_method = requested_chain_method

                if chain_method == "parallel" and device_count < num_chains:
                    if wrapper_self.verbose:
                        logger.warning(
                            f"⚠️  chain_method='parallel' requested but only {device_count} JAX "
                            f"device(s) available for {num_chains} chain(s); falling back to "
                            "chain_method='vectorized'."
                        )
                    chain_method = "vectorized"

                # Honor an explicit progress_bar setting; otherwise follow
                # verbose but auto-disable on non-TTY output (NumPyro's
                # per-step host callback costs 5-20% and floods logs).
                default_progress = wrapper_self.verbose and sys.stdout.isatty()
                return SamplerConfig(
                    num_samples=mcmc_config.get("num_samples", 1000),
                    num_chains=mcmc_config.get("num_chains", 4),
                    num_warmup=mcmc_config.get("num_warmup", 500),
                    seed=mcmc_config.get("seed", 42),
                    progress_bar=mcmc_config.get("progress_bar", default_progress),
                    chain_method=chain_method,
                    max_tree_depth=mcmc_config.get("max_tree_depth", 10),
                    n_walkers=mcmc_config.get("n_walkers", None),
                    moves=mcmc_config.get("moves", None),
                )

            def run(wrapper_self, **kwargs) -> Dict[str, Any]:
                """Run sampling using new backend."""
                sampler_config = wrapper_self._build_sampler_config(**kwargs)
                wrapper_self.results = wrapper_self.backend.run(sampler_config)
                wrapper_self.samples = wrapper_self.results.samples
                return wrapper_self.samples

            def run_chunked(
                wrapper_self,
                *,
                checkpoint_callback=None,
                checkpoint_interval_seconds: Optional[float] = None,
                checkpoint_interval_steps: Optional[int] = None,
                **kwargs,
            ) -> Dict[str, Any]:
                """Run sampling in chunks with time-based checkpoints (NumPyro only)."""
                sampler_config = wrapper_self._build_sampler_config(**kwargs)
                if not hasattr(wrapper_self.backend, "run_chunked"):
                    # Fallback to single run if backend doesn't support chunking
                    return wrapper_self.run(**kwargs)

                wrapper_self.results = wrapper_self.backend.run_chunked(
                    sampler_config,
                    checkpoint_callback=checkpoint_callback,
                    checkpoint_interval_seconds=checkpoint_interval_seconds,
                    checkpoint_interval_steps=checkpoint_interval_steps,
                )
                wrapper_self.samples = wrapper_self.results.samples
                return wrapper_self.samples

            def get_diagnostics(wrapper_self) -> Dict[str, Any]:
                """Get diagnostics from backend."""
                if wrapper_self.results:
                    return wrapper_self.results.diagnostics
                return {}

            def print_summary(
                wrapper_self, prob: float = 0.9, burnin_frac: float = 0.1
            ):
                """
                Print MCMC results summary with parameter estimates.

                Parameters
                ----------
                prob : float
                    Probability for credible interval (default 0.9 = 90% CI)
                burnin_frac : float
                    Fraction of samples to discard as burn-in (default 0.1 = 10%)
                """
                import numpy as np

                if wrapper_self.samples is None:
                    logger.info("No samples available.")
                    return

                logger.info("\n" + "=" * 70)
                logger.info("MCMC Results Summary")
                logger.info("=" * 70)

                # Parameter estimates table
                logger.info(
                    f"\n{'Parameter':<15} {'Mean':>12} {'Std':>12} {f'{int(prob*100)}% CI':>20}"
                )
                logger.info("-" * 70)

                alpha = (1 - prob) / 2
                for param_name, param_samples in wrapper_self.samples.items():
                    samples_arr = np.array(param_samples).flatten()

                    # Apply burn-in removal
                    n_burnin = int(len(samples_arr) * burnin_frac)
                    samples_arr = samples_arr[n_burnin:]

                    mean = np.mean(samples_arr)
                    std = np.std(samples_arr)
                    lo = np.percentile(samples_arr, alpha * 100)
                    hi = np.percentile(samples_arr, (1 - alpha) * 100)

                    logger.info(
                        f"{param_name:<15} {mean:>12.4f} {std:>12.4f} [{lo:>8.4f}, {hi:>8.4f}]"
                    )

                logger.info("-" * 70)

                # Sampling info
                logger.info(f"\nSampler: {wrapper_self.results.sampler_name}")
                logger.info(f"Elapsed time: {wrapper_self.results.elapsed_time:.2f}s")
                n_samples = MCMC._count_total_samples(wrapper_self.samples)
                logger.info(f"Total samples: {n_samples}")

                # Diagnostics (if available)
                if wrapper_self.results.diagnostics:
                    r_hats = {
                        k: v
                        for k, v in wrapper_self.results.diagnostics.items()
                        if isinstance(v, (int, float)) and "r_hat" in k.lower()
                    }
                    if r_hats:
                        max_rhat = max(r_hats.values())
                        converged = max_rhat < 1.1
                        status = "converged" if converged else "NOT converged"
                        logger.info(
                            f"Convergence: {status} (max R-hat = {max_rhat:.3f})"
                        )

                logger.info("=" * 70)

        # Set self.sampler to the wrapper
        self.sampler = BackendWrapper(self.backend, self.param_config, verbose)

    def _print_mapping_info(self):
        """Print parameter mapping information."""
        logger.debug("=" * 60)
        logger.debug("MCMC Parameter Mapping")
        logger.debug("=" * 60)

        if self._mapping_result.parameter_mapping:
            logger.debug("Parameter mapping successful:")
            for (
                func_param,
                config_param,
            ) in self._mapping_result.parameter_mapping.items():
                config_obj = self.param_config.parameters[config_param]
                logger.debug(f"  {func_param} ← {config_param} {config_obj.prior}")

        if self._mapping_result.data_arguments:
            logger.debug(
                f"Data arguments: {list(self._mapping_result.data_arguments.keys())}"
            )

        if self._mapping_result.warnings:
            logger.warning("Warnings:")
            for warning in self._mapping_result.warnings:
                logger.warning(f"  {warning}")

    @staticmethod
    def _count_total_samples(samples: Optional[Dict[str, Any]]) -> int:
        """Return total samples across all chains."""
        if not samples:
            return 0
        import numpy as np

        arr = np.asarray(next(iter(samples.values())))
        if arr.ndim == 1:
            return int(arr.shape[0])
        if arr.ndim >= 2:
            return int(arr.shape[0] * arr.shape[1])
        return int(arr.size)

    def _create_mcmc_state(self, samples: Optional[Dict] = None) -> MCMCState:
        """Create current MCMC state for checkpointing."""
        likelihood_info = create_likelihood_info(self.likelihood_func)
        data_info = create_data_info(self.data_kwargs)

        # Add normalization constant and num_data from likelihood objects
        # This is needed for correct AIC/BIC calculation
        norm_const, num_data = self._collect_likelihood_metadata()
        if norm_const is not None:
            data_info["normalization_constant"] = float(norm_const)
        if num_data is not None:
            data_info["num_data"] = int(num_data)

        # Get current samples if not provided
        if samples is None and hasattr(self, "_samples") and self._samples:
            samples = self._samples

        # Extract key MCMC settings for report display
        mcmc_cfg = self.param_config.mcmc
        run_metadata = {
            "chain_name": self.chain_name,
            "sampler_name": mcmc_cfg.get("sampler", "numpyro"),
            "num_chains": mcmc_cfg.get("num_chains", 4),
            "num_warmup": mcmc_cfg.get("num_warmup", 500),
            "num_samples": mcmc_cfg.get("num_samples", 1000),
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "strict_mode": self.strict_mode,
            "mcmc_config": mcmc_cfg.copy(),
            "hicosmo_version": "2.0",
        }

        backend_state = self._collect_backend_state()

        total_steps = self.param_config.mcmc.get(
            "num_samples", 0
        ) * self.param_config.mcmc.get("num_chains", 1)
        warmup_steps = self.param_config.mcmc.get(
            "num_warmup", 0
        ) * self.param_config.mcmc.get("num_chains", 1)
        if self._is_resumed and self.mcmc_state:
            try:
                total_steps = int(self.mcmc_state.current_step or 0) + int(total_steps)
                warmup_steps = int(self.mcmc_state.warmup_steps or 0)
            except Exception:
                pass

        state = MCMCState(
            samples=samples or {},
            current_step=0,  # Will be updated during sampling
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            parameter_config=self.param_config,
            likelihood_info=likelihood_info,
            data_info=data_info,
            mcmc_state=backend_state,
            run_metadata=run_metadata,
        )

        self.mcmc_state = state
        return state

    def _collect_backend_state(self) -> Optional[Dict[str, Any]]:
        """Collect backend-specific state for true checkpoint/resume."""
        if not hasattr(self, "backend"):
            return None

        try:
            backend_name = getattr(self.backend, "name", None)
            if backend_name == "numpyro" and getattr(self.backend, "mcmc", None):
                last_state = self.backend.mcmc.last_state
                rng_key = getattr(last_state, "rng_key", None)
                return {
                    "backend": "numpyro",
                    "state": last_state,
                    "rng_key": rng_key,
                }
            if (
                backend_name == "emcee"
                and getattr(self.backend, "last_state", None) is not None
            ):
                return {
                    "backend": "emcee",
                    "state": self.backend.last_state,
                }
        except Exception:
            return None

        return None

    def run(self, **extra_kwargs) -> Dict[str, Any]:
        """
        Run MCMC sampling.

        Parameters
        ----------
        **extra_kwargs
            Additional arguments to pass to the sampler run method.
            Supports: num_samples, num_warmup, num_chains (override config)

        Returns
        -------
        Dict[str, Any]
            MCMC samples for all parameters.
        """
        # Allow overriding MCMC config via kwargs
        mcmc_overrides = {}
        for key in ["num_samples", "num_warmup", "num_chains"]:
            if key in extra_kwargs:
                mcmc_overrides[key] = extra_kwargs.pop(key)

        # Update param_config.mcmc with overrides
        self.param_config.mcmc.update(mcmc_overrides)
        if mcmc_overrides:
            self._mcmc_counts_are_per_chain = False

        # If resuming and num_samples not explicitly set, finish remaining samples
        if self._is_resumed and "num_samples" not in mcmc_overrides:
            try:
                remaining = int(
                    max(
                        0,
                        (self.mcmc_state.total_steps or 0)
                        - (self.mcmc_state.current_step or 0),
                    )
                )
            except Exception:
                remaining = 0

            if remaining > 0:
                self.param_config.mcmc["num_samples"] = remaining
                self._mcmc_counts_are_per_chain = False
                logger.debug(f"Resuming: remaining total samples = {remaining}")
            else:
                logger.warning(
                    "Resuming: no remaining samples detected; run() will be a no-op."
                )
                self.param_config.mcmc["num_samples"] = 0
                self._mcmc_counts_are_per_chain = False

        # Apply intelligent defaults (convert total to per-chain)
        self.param_config.mcmc = self._apply_intelligent_defaults(
            self.param_config.mcmc
        )

        # Combine data arguments
        run_kwargs = dict(self.data_kwargs)
        run_kwargs.update(extra_kwargs)

        # Optimize initial values if requested
        if self.optimize_init:
            self._jax_optimize_init()

        # Create initial MCMC state
        if not self.mcmc_state:
            self._create_mcmc_state()

        # Record start time
        import time
        from datetime import datetime

        start_time = time.time()
        start_datetime = datetime.now()

        # Run MCMC with checkpointing
        if self.enable_checkpoints:
            self._samples = self._run_with_checkpoints(**run_kwargs)
        else:
            self._samples = self.sampler.run(**run_kwargs)

        # Compute derived parameters from likelihoods
        self._samples = self._compute_derived_parameters(self._samples)

        # Record end time
        end_time = time.time()
        end_datetime = datetime.now()
        runtime = end_time - start_time

        # Auto-save results to mcmc_chains/ directory in h5 format
        if self.chain_name:
            save_path = get_chains_dir() / f"{self.chain_name}.h5"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_results(str(save_path), format="hdf5", _skip_log=True)

            # Display detailed timing information
            logger.info(f"MCMC completed:")
            logger.info(f"  Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"  End time:   {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"  Duration:   {runtime:.2f}s")
            logger.info(f"  Saved to:   {save_path}")

        return self._samples

    def _run_with_checkpoints(self, **run_kwargs) -> Dict[str, Any]:
        """Run MCMC with automatic checkpointing."""
        import time

        start_time = time.time()

        def _save_checkpoint(samples: Dict[str, Any], current_step: int) -> None:
            if not self.checkpoint_manager:
                return
            state = self._create_mcmc_state(samples)
            state.current_step = int(current_step)
            self.checkpoint_manager.save_checkpoint(state, self.chain_name, force=True)

        resume_state = None
        resume_rng_key = None
        resume_samples = None
        resume_step = 0
        if self._is_resumed and self.mcmc_state:
            resume_samples = self.mcmc_state.samples or None
            resume_step = int(self.mcmc_state.current_step or 0)
            backend_state = self.mcmc_state.mcmc_state
            if isinstance(backend_state, dict):
                if backend_state.get("backend") == "numpyro":
                    resume_state = backend_state.get("state")
                    resume_rng_key = backend_state.get("rng_key")
                elif backend_state.get("backend") == "emcee":
                    resume_state = backend_state.get("state")

        # Run MCMC with chunking when supported
        if self.checkpoint_interval_seconds or self.checkpoint_interval > 0:
            samples = self.sampler.run_chunked(
                checkpoint_callback=_save_checkpoint,
                checkpoint_interval_seconds=self.checkpoint_interval_seconds,
                checkpoint_interval_steps=(
                    self.checkpoint_interval
                    if not self.checkpoint_interval_seconds
                    else None
                ),
                initial_state=resume_state,
                initial_rng_key=resume_rng_key,
                initial_samples=resume_samples,
                initial_step=resume_step,
                **run_kwargs,
            )
        else:
            samples = self.sampler.run(**run_kwargs)

        # If resuming and backend did not append previous samples, merge manually
        if resume_samples:
            try:
                prev_count = self._count_total_samples(resume_samples)
                new_count = self._count_total_samples(samples)
            except Exception:
                prev_count = 0
                new_count = 0

            if new_count < prev_count:
                merged_samples = {}
                for name, values in samples.items():
                    if name in resume_samples:
                        import numpy as np

                        merged_samples[name] = np.concatenate(
                            [resume_samples[name], values], axis=0
                        )
                    else:
                        merged_samples[name] = values
                samples = merged_samples

        # Note: Final results are saved in run() method as {chain_name}.h5
        # No need to save checkpoint here since run() will save the complete results

        return samples

    def _compute_derived_parameters(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        """Compute non-conflicting derived parameters from likelihood objects."""
        return compute_derived_parameters(samples, self._likelihoods)

    def print_summary(self, prob: float = 0.9, burnin_frac: float = 0.1):
        """Print MCMC results summary with automatic burn-in removal."""
        if self._samples is None:
            logger.warning("No samples available. Run MCMC first.")
            return

        self.sampler.print_summary(prob=prob, burnin_frac=burnin_frac)

    def summary(self, prob: float = 0.9, burnin_frac: float = 0.1):
        """Alias for :meth:`print_summary` (matches InferenceRunner)."""
        self.print_summary(prob=prob, burnin_frac=burnin_frac)

    def corner_plot(
        self,
        filename: Union[str, Path],
        params: Optional[List[str]] = None,
        **kwargs,
    ):
        """Generate a corner plot from current samples.

        Mirrors ``InferenceRunner.corner_plot`` so the object returned by the
        top-level ``hicosmo()`` API closes the sample-to-plot loop.
        """
        if self._samples is None:
            raise RuntimeError("No samples available. Run MCMC first.")

        from ..visualization import Plotter

        plotter = Plotter(
            self.get_samples(),
            labels=self.labels,
            ranges=self.ranges,
        )
        return plotter.corner(params=params, filename=filename, **kwargs)

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

    def save_results(
        self,
        filename: Optional[str] = None,
        format: str = "pickle",
        _skip_log: bool = False,
    ):
        """
        Save MCMC results with parameter metadata (labels, ranges).

        This saves:
        - samples: The MCMC samples for each parameter
        - labels: LaTeX labels for each parameter
        - ranges: Prior bounds for each parameter
        - diagnostics: Convergence diagnostics
        - config: MCMC configuration

        Parameters
        ----------
        filename : str, optional
            Output filename. If None, uses chain_name.
        format : str
            Output format: 'pickle' (default), 'hdf5', or 'npz'
        _skip_log : bool
            Internal flag to skip logging (used by run() to avoid duplicate logs)
        """
        import pickle
        import numpy as np
        from pathlib import Path
        from datetime import datetime

        if self._samples is None:
            raise RuntimeError("No samples available. Run MCMC first.")

        if filename is None:
            extension = {"pickle": "pkl", "hdf5": "h5", "npz": "npz"}.get(format, "pkl")
            filename = f"{self.chain_name}.{extension}"

        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Extract labels and ranges from ParameterConfig
        labels = self.param_config.get_labels()
        ranges = self.param_config.get_bounds()

        # Add labels for derived parameters (not in config but in samples)
        from ..parameters import DEFAULT_LATEX_LABELS
        from .. import __version__ as hicosmo_version

        for param_name in self._samples.keys():
            if param_name not in labels:
                labels[param_name] = DEFAULT_LATEX_LABELS.get(param_name, param_name)

        # Get diagnostics if available
        try:
            diagnostics = self.sampler.get_diagnostics()
        except:
            diagnostics = {}

        # Prepare metadata
        # Build parameter metadata (priors/bounds/labels)
        param_info = {}
        for name, param in self.param_config.parameters.items():
            param_info[name] = {
                "prior": param.prior,
                "bounds": param.bounds,
                "latex": param.latex_label,
                "description": param.description,
                "free": param.free,
            }

        metadata = {
            "labels": labels,
            "ranges": ranges,
            "chain_name": self.chain_name,
            "timestamp": datetime.now().isoformat(),
            "param_info": param_info,
            "param_names": list(self._samples.keys()),
            "hicosmo_version": hicosmo_version,
            "config": {
                "num_samples": self.param_config.mcmc.get("num_samples"),
                "num_chains": self.param_config.mcmc.get("num_chains"),
                "num_warmup": self.param_config.mcmc.get("num_warmup"),
            },
        }

        if format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "samples": {k: np.array(v) for k, v in self._samples.items()},
                        "metadata": metadata,
                        "diagnostics": diagnostics,
                    },
                    f,
                )

        elif format == "hdf5":
            import h5py

            with h5py.File(filepath, "w") as f:
                # Save samples
                samples_group = f.create_group("samples")
                for param_name, param_samples in self._samples.items():
                    samples_group.create_dataset(
                        param_name, data=np.array(param_samples)
                    )

                # Save metadata as attributes
                meta_group = f.create_group("metadata")
                meta_group.attrs["chain_name"] = self.chain_name
                meta_group.attrs["timestamp"] = metadata["timestamp"]
                meta_group.attrs["creation_time"] = metadata["timestamp"]
                meta_group.attrs["hicosmo_version"] = metadata["hicosmo_version"]
                meta_group.attrs["param_names"] = json.dumps(metadata["param_names"])
                meta_group.attrs["param_info"] = json.dumps(metadata["param_info"])

                # Save labels
                labels_group = meta_group.create_group("labels")
                for name, label in labels.items():
                    labels_group.attrs[name] = label

                # Save ranges
                ranges_group = meta_group.create_group("ranges")
                for name, (lo, hi) in ranges.items():
                    ranges_group.create_dataset(name, data=[lo, hi])

                # Save run_metadata for report()
                mcmc_cfg = self.param_config.mcmc
                run_metadata = {
                    "chain_name": self.chain_name,
                    "sampler_name": mcmc_cfg.get("sampler", "numpyro"),
                    "num_chains": mcmc_cfg.get("num_chains", 4),
                    "num_warmup": mcmc_cfg.get("num_warmup", 500),
                    "num_samples": mcmc_cfg.get("num_samples", 1000),
                }
                meta_group.create_dataset("run_metadata", data=json.dumps(run_metadata))

                # Save likelihood_info if available
                try:
                    likelihood_info = create_likelihood_info(self.likelihood_func)
                    meta_group.create_dataset(
                        "likelihood_info", data=json.dumps(likelihood_info)
                    )
                except:
                    pass

                # Save parameter_config for nuisance params display
                param_cfg_dict = self.param_config.to_dict()
                meta_group.create_dataset(
                    "parameter_config", data=json.dumps(param_cfg_dict)
                )

                # Save data_info with normalization_constant and num_data
                # This is needed for correct AIC/BIC/chi2 calculation
                # Must save to f['data']['data_info'] to match loading code in information_criteria.py
                norm_const, num_data = self._collect_likelihood_metadata()
                data_info = {
                    "normalization_constant": (
                        float(norm_const) if norm_const is not None else None
                    ),
                    "num_data": int(num_data) if num_data is not None else None,
                }
                data_group = f.create_group("data") if "data" not in f else f["data"]
                data_group.create_dataset("data_info", data=json.dumps(data_info))

                # Save progress info
                progress_group = f.create_group("progress")
                total_samples = sum(len(v) for v in self._samples.values()) // len(
                    self._samples
                )
                progress_group.attrs["total_steps"] = total_samples

        elif format == "npz":
            # Save as compressed numpy archive
            save_dict = {f"samples_{k}": np.array(v) for k, v in self._samples.items()}
            # Store metadata as special keys
            save_dict["_param_names"] = np.array(list(self._samples.keys()))
            save_dict["_labels"] = np.array(
                [labels.get(k, k) for k in self._samples.keys()]
            )
            save_dict["_metadata"] = np.array(json.dumps(metadata))
            # Store ranges as array (param_name -> [lo, hi])
            range_data = []
            for k in self._samples.keys():
                bounds = ranges.get(k, (-np.inf, np.inf))
                range_data.append([bounds[0], bounds[1]])
            save_dict["_ranges"] = np.array(range_data)
            np.savez_compressed(filepath, **save_dict)
        else:
            raise ValueError(
                f"Unknown format: {format}. Use 'pickle', 'hdf5', or 'npz'"
            )

        if not _skip_log:
            logger.info(f"Results saved to {filepath}")

    def load_results(self, filename: str, format: str = "pickle"):
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
                "prior": param.prior,
                "ref": param.ref,
                "bounds": param.bounds,
                "latex": param.latex,
                "description": param.description,
            }
        return info

    def suggest_improvements(self) -> List[str]:
        """Get suggestions for improving the configuration."""
        # Return suggestions based on mapping results
        suggestions = []
        if self._mapping_result and self._mapping_result.warnings:
            suggestions.append("Consider reviewing parameter mapping warnings")
        if hasattr(self, "sampler") and hasattr(self.sampler, "get_diagnostics"):
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
                    "value": np.mean(samples),
                    "error": np.std(samples),
                    "samples": samples,
                }
        return results

    @property
    def labels(self) -> Dict[str, str]:
        """Get LaTeX labels for all parameters (including nuisance parameters).

        Returns dictionary mapping parameter names to their LaTeX labels.
        Labels are automatically filled during Parameter creation.

        Example
        -------
        >>> mcmc.run()
        >>> plotter = Plotter(mcmc.samples, labels=mcmc.labels)
        >>> plotter.corner(['H0', 'Omega_m'])
        """
        return self.param_config.get_labels()

    @property
    def ranges(self) -> Dict[str, tuple]:
        """Get prior bounds for all free parameters.

        Returns dictionary mapping parameter names to (min, max) tuples.

        Example
        -------
        >>> mcmc.run()
        >>> plotter = Plotter(mcmc.samples, labels=mcmc.labels, ranges=mcmc.ranges)
        """
        return self.param_config.get_bounds()

    # =========================================================================
    # Class Methods for Different Creation Modes
    # =========================================================================

    @classmethod
    def resume(
        cls,
        checkpoint_path: Union[str, Path],
        likelihood_func: Optional[Callable] = None,
        strict_validation: bool = True,
        **new_kwargs,
    ) -> "MCMC":
        """
        Resume MCMC from a checkpoint file.

        Notes
        -----
        If the checkpoint contains backend internal state (NumPyro/NUTS),
        this performs a true continuation from the last state. Otherwise,
        new samples will be appended after the existing samples.

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
        if "mcmc" in new_kwargs:
            config_dict["mcmc"].update(new_kwargs.pop("mcmc"))

        # Use saved likelihood if none provided
        if likelihood_func is None:
            raise NotImplementedError(
                "Resuming without providing likelihood function is not yet supported"
            )

        # Restore checkpoint settings from saved metadata when not explicitly overridden
        saved_meta = saved_state.run_metadata or {}
        if "enable_checkpoints" not in new_kwargs:
            new_kwargs["enable_checkpoints"] = saved_meta.get(
                "enable_checkpoints", True
            )
        if "checkpoint_interval" not in new_kwargs:
            new_kwargs["checkpoint_interval"] = saved_meta.get(
                "checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL
            )
        if "checkpoint_interval_seconds" not in new_kwargs:
            new_kwargs["checkpoint_interval_seconds"] = saved_meta.get(
                "checkpoint_interval_seconds", DEFAULT_CHECKPOINT_INTERVAL_SECONDS
            )

        # Create resumed instance
        instance = cls(
            config_dict,
            likelihood_func,
            chain_name=saved_state.run_metadata.get("chain_name", "resumed_mcmc"),
            **new_kwargs,
        )

        # Set resumed state
        instance.mcmc_state = saved_state
        instance._samples = saved_state.samples
        instance._is_resumed = True
        instance._resume_total_steps = saved_state.total_steps
        instance._resume_current_step = saved_state.current_step
        instance._resume_backend_state = saved_state.mcmc_state

        logger.debug(f"Resumed from checkpoint: {Path(checkpoint_path).name}")
        logger.debug(
            f"Progress: {saved_state.get_completion_percentage():.1f}% complete"
        )
        logger.debug(f"Samples: {instance._count_total_samples(saved_state.samples)}")

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
        original_samples = self.param_config.mcmc.get("num_samples", 0)
        self.param_config.mcmc["num_samples"] = additional_samples

        logger.debug(f"Continuing sampling: {additional_samples} additional samples")
        if self._samples:
            existing_count = self._count_total_samples(self._samples)
            logger.debug(f"Existing samples: {existing_count}")

        # Run additional sampling
        new_samples = self.run(**kwargs)

        # Merge with existing samples if any (avoid double-merge if resume already appended)
        if self._is_resumed and self.mcmc_state and self.mcmc_state.samples:
            try:
                existing_count = self._count_total_samples(self.mcmc_state.samples)
                new_count = self._count_total_samples(new_samples)
            except Exception:
                existing_count = 0
                new_count = 0

            if new_count >= existing_count and existing_count > 0:
                # Assume new_samples already include previous samples
                self._samples = new_samples
                logger.debug(f"Total samples after continuation: {new_count}")
            else:
                merged_samples = {}
                for param_name in new_samples.keys():
                    if param_name in self.mcmc_state.samples:
                        import numpy as np

                        old_samples = self.mcmc_state.samples[param_name]
                        new_param_samples = new_samples[param_name]
                        merged_samples[param_name] = np.concatenate(
                            [old_samples, new_param_samples], axis=0
                        )
                    else:
                        merged_samples[param_name] = new_samples[param_name]

                self._samples = merged_samples

                total_samples = self._count_total_samples(merged_samples)
                logger.debug(f"Total samples after continuation: {total_samples}")

        # Restore original configuration
        self.param_config.mcmc["num_samples"] = original_samples

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
        **kwargs,
    ) -> "MCMC":
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

        mcmc_keys = {
            "num_warmup",
            "num_samples",
            "num_chains",
            "verbose",
            "enable_checkpoints",
            "checkpoint_interval",
            "checkpoint_interval_seconds",
            "checkpoint_dir",
            "backup_versions",
            "save_warmup",
            "compression",
        }
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
        cls, yaml_file: Union[str, Path], likelihood_func: Callable, **data_kwargs
    ) -> "MCMC":
        """Create MCMC from YAML configuration file."""
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)

        return cls(config, likelihood_func, **data_kwargs)


# Backward compatibility: legacy Inference class name
Inference = MCMC
