"""
NumPyro NUTS Sampler Backend
=============================

NumPyro implementation of the SamplerBackend interface using the No-U-Turn Sampler (NUTS).

Key Features:
- Hamiltonian Monte Carlo with automatic step size tuning
- Multiple initialization strategies (uniform, median, custom values)
- Enhanced error handling for JAX tracing issues
- Comprehensive diagnostics (R-hat, ESS, divergences)

Best suited for:
- Smooth, differentiable posterior distributions
- Medium to high dimensional parameter spaces (< 100 dimensions)
- Scenarios requiring fast convergence

Limitations:
- Requires likelihood functions compatible with JAX tracing
- May fail with non-smooth or multi-modal distributions
- Needs pure JAX arrays (no mixing with NumPy in hot paths)

Author: HIcosmo Development Team
License: MIT
"""
import time
import inspect
from typing import Dict, Any, Optional, Callable
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_uniform, init_to_value, init_to_median

from .base import (
    SamplerBackend, SamplerConfig, SamplerResults,
    InitializationError, ConvergenceError, LikelihoodError
)


class NumPyroSampler(SamplerBackend):
    """
    NumPyro NUTS (No-U-Turn Sampler) backend.

    Uses Hamiltonian Monte Carlo for efficient sampling of smooth posterior distributions.

    Suitable for:
    - Smooth, differentiable likelihood functions
    - Medium-dimensional parameter spaces (10-100 parameters)
    - Fast convergence requirements

    Not recommended for:
    - Likelihood functions incompatible with JAX tracing
    - Multi-modal distributions
    - Highly correlated parameters without reparameterization

    Example:
        >>> sampler = NumPyroSampler(log_prob, params, config)
        >>> sampler_config = SamplerConfig(num_samples=1000, num_chains=4, num_warmup=500)
        >>> results = sampler.run(sampler_config)
        >>> print(f"R-hat: {results.diagnostics['rhat']}")
    """

    INIT_STRATEGIES = {
        'uniform': init_to_uniform,
        'median': init_to_median,
        'value': init_to_value,
    }

    def __init__(self, log_probability, parameters, config):
        super().__init__(log_probability, parameters, config)
        self.mcmc = None
        self.numpyro_model = None

    def _build_numpyro_model(self):
        """
        Build NumPyro model function from parameter configuration.

        Converts HIcosmo's parameter configuration format to NumPyro's
        probabilistic modeling syntax.

        Returns:
            Callable: NumPyro model function
        """
        def model():
            # Sample all parameters from their priors
            sampled_params = {}
            for param_name, param_config in self.parameters.items():
                prior_config = param_config.get('prior', {})
                prior_dist = prior_config.get('dist', 'uniform')
                dist_type = str(prior_dist).lower()
                dist_aliases = {
                    'log_normal': 'lognormal',
                    'truncated_normal': 'truncnorm',
                    'half_normal': 'halfnormal',
                    'half_cauchy': 'halfcauchy',
                }
                dist_type = dist_aliases.get(dist_type, dist_type)

                # Convert to NumPyro distribution
                if dist_type == 'uniform':
                    min_val = prior_config['min']
                    max_val = prior_config['max']
                    sampled_params[param_name] = numpyro.sample(
                        param_name,
                        dist.Uniform(min_val, max_val)
                    )
                elif dist_type == 'normal':
                    mu = prior_config['loc']
                    sigma = prior_config['scale']
                    sampled_params[param_name] = numpyro.sample(
                        param_name,
                        dist.Normal(mu, sigma)
                    )
                elif dist_type == 'truncnorm':
                    low = prior_config.get('low', prior_config.get('min'))
                    high = prior_config.get('high', prior_config.get('max'))
                    sampled_params[param_name] = numpyro.sample(
                        param_name,
                        dist.TruncatedNormal(
                            loc=prior_config['loc'],
                            scale=prior_config['scale'],
                            low=low,
                            high=high
                        )
                    )
                elif dist_type == 'lognormal':
                    sampled_params[param_name] = numpyro.sample(
                        param_name,
                        dist.LogNormal(prior_config['loc'], prior_config['scale'])
                    )
                elif dist_type == 'beta':
                    alpha = prior_config.get('alpha', prior_config.get('a'))
                    beta = prior_config.get('beta', prior_config.get('b'))
                    sampled_params[param_name] = numpyro.sample(
                        param_name,
                        dist.Beta(alpha, beta)
                    )
                elif dist_type == 'gamma':
                    concentration = prior_config.get('concentration', prior_config.get('shape'))
                    rate = prior_config.get('rate')
                    if rate is None and 'scale' in prior_config:
                        rate = 1.0 / prior_config['scale']
                    sampled_params[param_name] = numpyro.sample(
                        param_name,
                        dist.Gamma(concentration, rate)
                    )
                elif dist_type == 'halfnormal':
                    sampled_params[param_name] = numpyro.sample(
                        param_name,
                        dist.HalfNormal(prior_config['scale'])
                    )
                elif dist_type == 'halfcauchy':
                    sampled_params[param_name] = numpyro.sample(
                        param_name,
                        dist.HalfCauchy(prior_config['scale'])
                    )
                elif dist_type == 'exponential':
                    sampled_params[param_name] = numpyro.sample(
                        param_name,
                        dist.Exponential(prior_config['rate'])
                    )
                else:
                    raise ValueError(
                        f"Unsupported prior distribution: {prior_dist}. "
                        "Supported: 'uniform', 'normal', 'truncnorm', 'lognormal', "
                        "'beta', 'gamma', 'halfnormal', 'halfcauchy', 'exponential'"
                    )

            # Compute log likelihood
            log_L = jnp.asarray(self.log_probability(sampled_params))

            # Handle NaN/Inf gracefully
            log_L = jnp.where(jnp.isfinite(log_L), log_L, -1e10)

            # Save log_likelihood as deterministic site for information criteria
            numpyro.deterministic('log_likelihood', log_L)

            # Add to model's log probability
            numpyro.factor('log_likelihood_factor', log_L)

        return model

    def run(self, sampler_config: SamplerConfig) -> SamplerResults:
        """
        Run NumPyro MCMC sampling.

        Args:
            sampler_config: Sampler configuration object

        Returns:
            SamplerResults: Sampling results with diagnostics

        Raises:
            InitializationError: If sampler fails to find valid initial parameters
            LikelihoodError: If likelihood computation fails
            ConvergenceError: If sampling fails for other reasons
        """
        self.validate_config()

        # Build NumPyro model
        self.numpyro_model = self._build_numpyro_model()

        # Select initialization strategy
        init_strategy_name = self.config.get('init_strategy', 'uniform')
        init_strategy = self.INIT_STRATEGIES.get(init_strategy_name, init_to_uniform)

        # Handle custom initial values
        if init_strategy_name == 'value':
            init_values = self.config.get('init_values', {})
            if not init_values:
                raise ValueError(
                    "init_strategy='value' requires 'init_values' in config"
                )
            # Convert to JAX arrays
            init_values = {k: jnp.array(v) for k, v in init_values.items()}
            init_strategy = init_to_value(values=init_values)

        # Configure NUTS kernel
        dense_mass = self.config.get("dense_mass")
        if dense_mass is None:
            # Dense mass matrices are usually beneficial for low-dimensional,
            # correlated posteriors (common in cosmology). Keep diagonal mass for
            # higher dimensions to avoid O(D^2) overhead.
            dense_mass = len(self.parameters) <= int(self.config.get("dense_mass_max_dim", 20))

        target_accept_prob = self.config.get("target_accept_prob")
        nuts_kwargs = {
            'max_tree_depth': sampler_config.max_tree_depth or 10,
            'init_strategy': init_strategy,
            'dense_mass': dense_mass,
        }
        if target_accept_prob is not None:
            nuts_kwargs["target_accept_prob"] = target_accept_prob

        kernel = NUTS(self.numpyro_model, **nuts_kwargs)

        # Optional performance knobs (see NumPyro MCMC docs)
        mcmc_kwargs = {
            'num_warmup': sampler_config.num_warmup,
            'num_samples': sampler_config.num_samples,
            'num_chains': sampler_config.num_chains,
            'progress_bar': sampler_config.progress_bar,
            'chain_method': sampler_config.chain_method,
        }
        for key in ('jit_model_args', 'thinning'):
            if key in self.config:
                mcmc_kwargs[key] = self.config[key]

        # Filter kwargs for older NumPyro versions
        mcmc_sig = inspect.signature(MCMC)
        mcmc_kwargs = {k: v for k, v in mcmc_kwargs.items() if k in mcmc_sig.parameters}

        # Configure MCMC
        # chain_method options:
        # - 'vectorized': Run chains in parallel using vmap (recommended for CPU multi-core)
        # - 'sequential': Run chains one after another (single-threaded)
        # - 'parallel': Run chains on multiple devices using pmap (for multi-GPU)
        mcmc = MCMC(
            kernel,
            **mcmc_kwargs,
        )

        # Run sampling
        rng_key = random.PRNGKey(sampler_config.seed)

        try:
            start_time = time.time()
            extra_fields = self.config.get(
                "extra_fields",
                ("num_steps", "accept_prob", "mean_accept_prob", "diverging"),
            )
            run_kwargs = {'extra_fields': extra_fields}
            if 'collect_warmup' in self.config:
                run_kwargs['collect_warmup'] = self.config['collect_warmup']
            run_sig = inspect.signature(mcmc.run)
            run_kwargs = {k: v for k, v in run_kwargs.items() if k in run_sig.parameters}
            mcmc.run(rng_key, **run_kwargs)
            elapsed_time = time.time() - start_time

        except Exception as e:
            error_str = str(e)

            # Enhanced error handling with actionable suggestions
            if "Cannot find valid initial parameters" in error_str:
                raise InitializationError(
                    "NumPyro failed to find valid initial parameters.\n\n"
                    "This typically means:\n"
                    "  1. Likelihood function returns NaN/Inf for most parameter values\n"
                    "  2. JAX tracing is incompatible with likelihood function\n"
                    "  3. Prior ranges are too wide or unrealistic\n\n"
                    "Recommended solutions:\n"
                    "  1. Switch to emcee sampler: config['sampler'] = 'emcee'\n"
                    "  2. Check likelihood function returns valid values\n"
                    "  3. Narrow prior ranges based on physical constraints\n"
                    "  4. Provide better initial values via config['init_values']\n\n"
                    f"Original error: {e}"
                )
            elif "NaN" in error_str or "Inf" in error_str:
                raise LikelihoodError(
                    "Likelihood computation produced NaN/Inf values.\n\n"
                    "Common causes:\n"
                    "  1. Log of zero or negative numbers: log(0) = -inf\n"
                    "  2. Division by zero\n"
                    "  3. Numerical overflow in exponentials\n"
                    "  4. Invalid parameter values in calculations\n\n"
                    "Debugging steps:\n"
                    "  1. Check for log operations: use jnp.log(jnp.maximum(x, 1e-100))\n"
                    "  2. Check for divisions: use jnp.maximum(denominator, 1e-100)\n"
                    "  3. Add parameter bounds checking in likelihood function\n"
                    "  4. Use jnp.where() for conditional logic\n\n"
                    f"Original error: {error_str}"
                )
            elif "Tracer" in error_str or "ConcretizationTypeError" in error_str:
                raise LikelihoodError(
                    "JAX tracing error detected.\n\n"
                    "This means your likelihood function is incompatible with JAX tracing.\n\n"
                    "Common issues:\n"
                    "  1. Using float() or int() conversions on JAX arrays\n"
                    "  2. Using if-else on JAX arrays (use jnp.where() instead)\n"
                    "  3. Mixing NumPy and JAX operations\n"
                    "  4. Dynamic array shapes\n\n"
                    "Solutions:\n"
                    "  1. Return JAX arrays directly (don't convert to Python types)\n"
                    "  2. Replace if-else with jnp.where(condition, true_val, false_val)\n"
                    "  3. Use jnp instead of np throughout\n"
                    "  4. Or switch to emcee sampler (no tracing required)\n\n"
                    f"Original error: {error_str}"
                )
            else:
                raise ConvergenceError(
                    f"NumPyro sampling failed:\n{error_str}\n\n"
                    f"If this persists, consider switching to emcee sampler."
                )

        # Extract results
        samples = mcmc.get_samples()

        # Compute diagnostics
        diagnostics = self.get_diagnostics()

        # Build results object
        results = SamplerResults(
            samples=samples,
            diagnostics=diagnostics,
            elapsed_time=elapsed_time,
            sampler_name='numpyro',
            metadata={
                'num_warmup': sampler_config.num_warmup,
                'num_samples': sampler_config.num_samples,
                'num_chains': sampler_config.num_chains,
                'max_tree_depth': nuts_kwargs['max_tree_depth'],
                'init_strategy': init_strategy_name,
            }
        )

        self.results = results
        self.mcmc = mcmc

        return results

    def run_chunked(
        self,
        sampler_config: SamplerConfig,
        *,
        checkpoint_callback: Optional[Callable[[Dict[str, np.ndarray], int], None]] = None,
        checkpoint_interval_seconds: Optional[float] = None,
        checkpoint_interval_steps: Optional[int] = None,
        initial_state: Optional[Any] = None,
        initial_rng_key: Optional[Any] = None,
        initial_samples: Optional[Dict[str, Any]] = None,
        initial_step: int = 0,
    ) -> SamplerResults:
        """
        Run NumPyro MCMC sampling in chunks to enable time-based checkpoints.

        This uses NumPyro's post_warmup_state to continue sampling without
        re-running warmup. Samples are concatenated across chunks.
        """
        self.validate_config()

        # Build NumPyro model
        self.numpyro_model = self._build_numpyro_model()

        # Select initialization strategy
        init_strategy_name = self.config.get('init_strategy', 'uniform')
        init_strategy = self.INIT_STRATEGIES.get(init_strategy_name, init_to_uniform)

        if init_strategy_name == 'value':
            init_values = self.config.get('init_values', {})
            if not init_values:
                raise ValueError(
                    "init_strategy='value' requires 'init_values' in config"
                )
            init_values = {k: jnp.array(v) for k, v in init_values.items()}
            init_strategy = init_to_value(values=init_values)

        dense_mass = self.config.get("dense_mass")
        if dense_mass is None:
            dense_mass = len(self.parameters) <= int(self.config.get("dense_mass_max_dim", 20))

        target_accept_prob = self.config.get("target_accept_prob")
        nuts_kwargs = {
            'max_tree_depth': sampler_config.max_tree_depth or 10,
            'init_strategy': init_strategy,
            'dense_mass': dense_mass,
        }
        if target_accept_prob is not None:
            nuts_kwargs["target_accept_prob"] = target_accept_prob

        kernel = NUTS(self.numpyro_model, **nuts_kwargs)

        # Determine target sample count (per chain) and chunk size
        total_per_chain = int(sampler_config.num_samples)
        if total_per_chain <= 0:
            return SamplerResults(
                samples={},
                diagnostics={},
                elapsed_time=0.0,
                sampler_name='numpyro',
                metadata={'num_samples': 0, 'num_chains': sampler_config.num_chains},
            )

        num_chains = int(sampler_config.num_chains)
        total_target = total_per_chain * num_chains

        default_chunk = max(10, min(100, total_per_chain))
        chunk_size = int(self.config.get("checkpoint_chunk_size", default_chunk))
        chunk_size = max(1, min(chunk_size, total_per_chain))

        # Configure MCMC for the first chunk (with warmup)
        mcmc_kwargs = {
            'num_warmup': sampler_config.num_warmup,
            'num_samples': chunk_size,
            'num_chains': num_chains,
            'progress_bar': sampler_config.progress_bar,
            'chain_method': sampler_config.chain_method,
        }
        for key in ('jit_model_args', 'thinning'):
            if key in self.config:
                mcmc_kwargs[key] = self.config[key]
        mcmc_sig = inspect.signature(MCMC)
        mcmc_kwargs = {k: v for k, v in mcmc_kwargs.items() if k in mcmc_sig.parameters}
        mcmc = MCMC(kernel, **mcmc_kwargs)

        rng_key = random.PRNGKey(sampler_config.seed)
        if initial_rng_key is not None:
            rng_key = initial_rng_key
        extra_fields = self.config.get(
            "extra_fields",
            ("num_steps", "accept_prob", "mean_accept_prob", "diverging"),
        )

        self.mcmc = mcmc

        samples_accum: Dict[str, np.ndarray] = {}
        total_collected = 0
        if initial_samples:
            for name, values in initial_samples.items():
                samples_accum[name] = np.array(values)
            try:
                total_collected = int(initial_step or len(next(iter(samples_accum.values()))))
            except Exception:
                total_collected = int(initial_step or 0)
        last_checkpoint_time = time.time()
        last_checkpoint_step = int(total_collected)
        start_time = last_checkpoint_time

        if initial_state is not None:
            mcmc.post_warmup_state = initial_state
            mcmc.num_warmup = 0

        overall_target_total = int(total_collected + total_target)
        while total_collected < overall_target_total:
            chunk_start = time.time()
            run_kwargs = {'extra_fields': extra_fields}
            if mcmc.num_warmup > 0 and 'collect_warmup' in self.config:
                run_kwargs['collect_warmup'] = self.config['collect_warmup']
            run_sig = inspect.signature(mcmc.run)
            run_kwargs = {k: v for k, v in run_kwargs.items() if k in run_sig.parameters}
            mcmc.run(rng_key, **run_kwargs)
            chunk_elapsed = max(time.time() - chunk_start, 1e-6)

            new_samples = mcmc.get_samples()
            for name, values in new_samples.items():
                values_np = np.array(values)
                if name not in samples_accum:
                    samples_accum[name] = values_np
                else:
                    samples_accum[name] = np.concatenate([samples_accum[name], values_np], axis=0)

            total_collected = len(next(iter(samples_accum.values())))

            # Prepare for next chunk
            rng_key = mcmc.last_state.rng_key
            mcmc.post_warmup_state = mcmc.last_state
            mcmc.num_warmup = 0

            # Save checkpoint if time interval reached
            if checkpoint_callback and checkpoint_interval_seconds:
                now = time.time()
                if (now - last_checkpoint_time) >= checkpoint_interval_seconds:
                    checkpoint_callback(samples_accum, total_collected)
                    last_checkpoint_time = now
                    last_checkpoint_step = total_collected

            # Save checkpoint if step interval reached
            if checkpoint_callback and checkpoint_interval_steps:
                if (total_collected - last_checkpoint_step) >= checkpoint_interval_steps:
                    checkpoint_callback(samples_accum, total_collected)
                    last_checkpoint_step = total_collected
                    last_checkpoint_time = time.time()

            remaining_total = overall_target_total - total_collected
            if remaining_total <= 0:
                break

            remaining_per_chain = max(1, remaining_total // num_chains)

            if checkpoint_interval_seconds:
                # Estimate chunk size based on observed throughput
                chunk_total = len(next(iter(new_samples.values())))
                samples_per_sec = max(1.0, chunk_total / chunk_elapsed)
                chunk_target_total = max(
                    1, int(samples_per_sec * checkpoint_interval_seconds * 0.9)
                )
                target_per_chain = max(1, chunk_target_total // num_chains)
                chunk_size = min(remaining_per_chain, max(1, target_per_chain))
            elif checkpoint_interval_steps:
                target_per_chain = max(1, int(checkpoint_interval_steps) // num_chains)
                chunk_size = min(remaining_per_chain, max(1, target_per_chain))
            else:
                chunk_size = min(remaining_per_chain, chunk_size)

            mcmc.num_samples = int(chunk_size)

        elapsed_time = time.time() - start_time
        samples = {k: jnp.array(v) for k, v in samples_accum.items()}
        diagnostics = self._diagnostics_from_samples(samples)

        results = SamplerResults(
            samples=samples,
            diagnostics=diagnostics,
            elapsed_time=elapsed_time,
            sampler_name='numpyro',
            metadata={
                'num_warmup': sampler_config.num_warmup,
                'num_samples': sampler_config.num_samples,
                'num_chains': sampler_config.num_chains,
                'max_tree_depth': nuts_kwargs['max_tree_depth'],
                'init_strategy': init_strategy_name,
                'chunked': True,
            }
        )

        self.results = results
        self.mcmc = mcmc

        return results

    def _diagnostics_from_samples(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        """Compute diagnostics from aggregated samples using NumPyro's summary."""
        try:
            from numpyro.diagnostics import summary
            # summary returns: mean, std, median, 5%, 95%, n_eff, r_hat
            stats = summary(samples, prob=0.9, group_by_chain=False)
            return {
                'rhat': {k: float(v['r_hat']) for k, v in stats.items()},
                'ess': {k: float(v['n_eff']) for k, v in stats.items()},
                'summary': stats,  # Full summary for advanced users
            }
        except Exception:
            return {}

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Extract NumPyro diagnostic information using numpyro.diagnostics.summary.

        Returns:
            Dictionary containing:
                - 'rhat': Split R-hat convergence statistic per parameter
                - 'ess': Effective sample size per parameter
                - 'summary': Full NumPyro summary table (mean, std, median, HPDI, n_eff, r_hat)
                - 'num_divergences': Number of divergent transitions
                - 'mean_accept_prob': Mean acceptance probability

        Note:
            Returns empty dict if MCMC hasn't been run yet.
        """
        if self.mcmc is None:
            return {}

        diagnostics = {}

        # Use NumPyro's summary for all standard diagnostics
        try:
            from numpyro.diagnostics import summary
            samples = self.mcmc.get_samples()
            # summary computes: mean, std, median, HPDI, n_eff (ESS), r_hat (split R-hat)
            stats = summary(samples, prob=0.9, group_by_chain=False)
            diagnostics['rhat'] = {k: float(v['r_hat']) for k, v in stats.items()}
            diagnostics['ess'] = {k: float(v['n_eff']) for k, v in stats.items()}
            diagnostics['summary'] = stats  # Full summary for advanced access
        except Exception:
            pass

        # Divergences (from extra_fields, not in summary)
        try:
            extra_fields = self.mcmc.get_extra_fields()
            if 'diverging' in extra_fields:
                divergences = extra_fields['diverging']
                n_divergences = int(jnp.sum(divergences))
                diagnostics['num_divergences'] = n_divergences
                if n_divergences > 0:
                    diagnostics['divergence_rate'] = float(n_divergences / divergences.size)
        except Exception:
            pass

        # Mean acceptance probability (from extra_fields)
        try:
            extra_fields = self.mcmc.get_extra_fields()
            if 'mean_accept_prob' in extra_fields:
                diagnostics['mean_accept_prob'] = float(jnp.mean(extra_fields['mean_accept_prob']))
        except Exception:
            pass

        return diagnostics
