"""
emcee Ensemble Sampler Backend
===============================

emcee implementation using the affine-invariant ensemble sampler algorithm.

Key Features:
- No gradient information required (derivative-free)
- Robust handling of NaN/Inf in likelihood function
- Effective for multi-modal distributions
- Low requirements for likelihood function (only needs to return a float)

Best suited for:
- When NumPyro initialization fails (JAX tracing incompatibility)
- Multi-modal posterior distributions
- Likelihood functions that are not JAX-compatible
- Exploratory analysis when distribution shape is unknown

Limitations:
- Slower convergence compared to HMC methods like NUTS
- Requires more samples for equivalent effective sample size
- Needs sufficient walkers (typically 2× parameter dimension)

Reference:
    Foreman-Mackey et al. (2013) "emcee: The MCMC Hammer"
    https://arxiv.org/abs/1202.3665

Author: HIcosmo Development Team
License: MIT
"""

import time
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Optional
import emcee

from .base import (
    SamplerBackend,
    SamplerConfig,
    SamplerResults,
    InitializationError,
    ConvergenceError,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EmceeSampler(SamplerBackend):
    """
    emcee Ensemble Sampler backend.

    Uses the affine-invariant ensemble sampler algorithm for MCMC sampling.
    Particularly useful when NumPyro fails due to JAX tracing issues.

    Advantages:
    - No gradient computation required
    - Handles NaN/Inf gracefully (automatically assigns low probability)
    - Works with any callable log probability function
    - Good for multi-modal distributions

    Disadvantages:
    - Slower convergence than gradient-based methods
    - Requires more walkers (memory overhead)
    - May need longer chains for good ESS

    Example:
        >>> sampler = EmceeSampler(log_prob, params, config)
        >>> sampler_config = SamplerConfig(
        ...     num_samples=2000,
        ...     num_chains=4,
        ...     num_warmup=1000,
        ...     n_walkers=32
        ... )
        >>> results = sampler.run(sampler_config)
        >>> print(f"Acceptance rate: {results.diagnostics['acceptance_rate']['mean']}")
    """

    def __init__(self, log_probability, parameters, config):
        super().__init__(log_probability, parameters, config)
        self.sampler = None
        self.param_names = list(parameters.keys())
        self.param_bounds = self._extract_bounds()
        self.last_state = None

    def _extract_bounds(self) -> Dict[str, tuple]:
        """
        Extract parameter bounds from configuration.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        bounds = {}
        for name, config in self.parameters.items():
            prior = config.get("prior", {})
            dist_type = str(prior.get("dist", "uniform")).lower()
            if dist_type in {
                "log_normal",
                "truncated_normal",
                "half_normal",
                "half_cauchy",
            }:
                dist_type = {
                    "log_normal": "lognormal",
                    "truncated_normal": "truncnorm",
                    "half_normal": "halfnormal",
                    "half_cauchy": "halfcauchy",
                }[dist_type]

            if dist_type == "uniform":
                bounds[name] = (prior["min"], prior["max"])
            elif dist_type == "normal":
                # Use ±5σ as bounds for normal priors
                mu = prior["loc"]
                sigma = prior["scale"]
                bounds[name] = (mu - 5 * sigma, mu + 5 * sigma)
            elif dist_type == "truncnorm":
                low = prior.get("low", prior.get("min", -1e10))
                high = prior.get("high", prior.get("max", 1e10))
                bounds[name] = (low, high)
            elif dist_type == "lognormal":
                mu = prior["loc"]
                sigma = prior["scale"]
                # Conservative bounds in log space
                bounds[name] = (np.exp(mu - 5 * sigma), np.exp(mu + 5 * sigma))
            elif dist_type == "halfnormal":
                scale = prior["scale"]
                bounds[name] = (0.0, 5 * scale)
            elif dist_type == "halfcauchy":
                scale = prior["scale"]
                bounds[name] = (0.0, 25 * scale)
            elif dist_type == "exponential":
                rate = prior["rate"]
                bounds[name] = (0.0, 10.0 / rate)
            else:
                # Conservative wide bounds for unknown distributions
                bounds[name] = (-1e10, 1e10)
        return bounds

    def _log_prior(self, theta: np.ndarray) -> float:
        """
        Compute log prior probability.

        Args:
            theta: Parameter array in order of self.param_names

        Returns:
            float: Log prior probability (0 for uniform, -inf for out of bounds)
        """
        for i, param_name in enumerate(self.param_names):
            value = theta[i]
            min_val, max_val = self.param_bounds[param_name]

            # Check bounds
            if not (min_val <= value <= max_val):
                return -np.inf

        return 0.0  # Uniform prior log probability = 0 within bounds

    def _log_likelihood_wrapper(self, theta: np.ndarray) -> float:
        """
        Wrap likelihood function for emcee compatibility.

        Handles:
        - Parameter dictionary construction
        - JAX array to Python float conversion
        - NaN/Inf handling (returns -1e10)
        - Exception handling (returns -1e10)

        Args:
            theta: Parameter array

        Returns:
            float: Log likelihood (or -1e10 if computation fails)
        """
        # Build parameter dictionary
        params_dict = {
            self.param_names[i]: float(theta[i]) for i in range(len(self.param_names))
        }

        try:
            log_L = self.log_probability(params_dict)

            # Convert JAX array to Python float
            if hasattr(log_L, "item"):
                log_L = log_L.item()
            else:
                log_L = float(log_L)

            # Handle NaN/Inf gracefully
            if not np.isfinite(log_L):
                return -1e10

            return log_L

        except Exception:
            # Any exception returns very low probability
            # This allows emcee to continue sampling
            return -1e10

    def _log_probability_emcee(self, theta: np.ndarray) -> float:
        """
        Total log probability function for emcee.

        Computes: log_posterior = log_prior + log_likelihood

        Args:
            theta: Parameter array

        Returns:
            float: Log posterior probability
        """
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        ll = self._log_likelihood_wrapper(theta)

        return lp + ll

    def _initialize_walkers(self, n_walkers: int, seed: int = 42) -> np.ndarray:
        """
        Initialize walker positions in parameter space.

        Strategy: Uniform sampling within prior bounds with 1% margin.

        Args:
            n_walkers: Number of walkers to initialize
            seed: Random seed for reproducibility

        Returns:
            ndarray: Initial positions with shape (n_walkers, n_dim)
        """
        np.random.seed(seed)
        n_dim = len(self.param_names)

        pos = []
        for _ in range(n_walkers):
            walker_pos = []
            for param_name in self.param_names:
                min_val, max_val = self.param_bounds[param_name]
                # Sample within bounds with 1% margin to avoid boundary issues
                margin = 0.01 * (max_val - min_val)
                value = np.random.uniform(min_val + margin, max_val - margin)
                walker_pos.append(value)
            pos.append(walker_pos)

        return np.array(pos)

    def run(self, sampler_config: SamplerConfig) -> SamplerResults:
        """
        Run emcee MCMC sampling.

        Args:
            sampler_config: Sampler configuration object

        Returns:
            SamplerResults: Sampling results with diagnostics

        Raises:
            ConvergenceError: If sampling fails catastrophically
        """
        self.validate_config()

        n_dim = len(self.param_names)

        # Determine number of walkers
        # emcee recommendation: at least 2 × ndim
        n_walkers = sampler_config.n_walkers
        if n_walkers is None:
            n_walkers = max(32, 2 * n_dim)

        # Initialize walker positions
        pos = self._initialize_walkers(n_walkers, sampler_config.seed)

        # Create emcee sampler
        self.sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            self._log_probability_emcee,
            moves=sampler_config.moves,
        )

        # Run MCMC
        try:
            start_time = time.time()

            # Burn-in phase
            if sampler_config.progress_bar:
                logger.info(f"[emcee] Burn-in: {sampler_config.num_warmup} steps...")

            state = self.sampler.run_mcmc(
                pos, sampler_config.num_warmup, progress=sampler_config.progress_bar
            )
            self.sampler.reset()

            # Sampling phase
            if sampler_config.progress_bar:
                logger.info(f"[emcee] Sampling: {sampler_config.num_samples} steps...")

            state = self.sampler.run_mcmc(
                state, sampler_config.num_samples, progress=sampler_config.progress_bar
            )

            elapsed_time = time.time() - start_time

        except Exception as e:
            raise ConvergenceError(
                f"emcee sampling failed:\n{e}\n\n"
                f"This is unusual for emcee, which is typically very robust.\n"
                f"Check that your likelihood function is callable and returns a number."
            )

        # Persist last state for resume
        self.last_state = state

        # Extract samples
        chain = self.sampler.get_chain()  # (nsteps, nwalkers, ndim)
        flat_samples = self.sampler.get_chain(flat=True)  # (nsteps*nwalkers, ndim)

        # Convert to HIcosmo format: {param_name: array(num_chains, num_samples)}
        # We select num_chains walkers to represent independent chains
        samples = {}
        samples_per_walker = chain.shape[0]  # nsteps

        num_chains = sampler_config.num_chains
        # Select evenly spaced walkers as "chains"
        selected_walkers = np.linspace(0, n_walkers - 1, num_chains, dtype=int)

        for i, param_name in enumerate(self.param_names):
            # Extract samples from selected walkers
            chain_samples = chain[:, selected_walkers, i].T  # (num_chains, nsteps)
            samples[param_name] = jnp.array(chain_samples)

        # Compute diagnostics
        diagnostics = self.get_diagnostics()

        # Build results object
        results = SamplerResults(
            samples=samples,
            diagnostics=diagnostics,
            elapsed_time=elapsed_time,
            sampler_name="emcee",
            metadata={
                "num_warmup": sampler_config.num_warmup,
                "num_samples": sampler_config.num_samples,
                "n_walkers": n_walkers,
                "n_dim": n_dim,
            },
        )

        self.results = results
        return results

    def run_chunked(
        self,
        sampler_config: SamplerConfig,
        *,
        checkpoint_callback=None,
        checkpoint_interval_seconds: Optional[float] = None,
        checkpoint_interval_steps: Optional[int] = None,
        initial_state: Optional[Any] = None,
        initial_samples: Optional[Dict[str, Any]] = None,
        initial_step: int = 0,
        **_kwargs,
    ) -> SamplerResults:
        """Run emcee sampling in chunks to enable checkpointing."""
        self.validate_config()

        n_dim = len(self.param_names)
        n_walkers = sampler_config.n_walkers
        if initial_state is not None and hasattr(initial_state, "coords"):
            n_walkers = int(initial_state.coords.shape[0])
        if n_walkers is None:
            n_walkers = max(32, 2 * n_dim)

        # Initialize walkers
        if initial_state is None:
            pos = self._initialize_walkers(n_walkers, sampler_config.seed)
        else:
            pos = initial_state

        # Create sampler
        self.sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            self._log_probability_emcee,
            moves=sampler_config.moves,
        )

        # Warmup (skip if resuming with state)
        if initial_state is None:
            if sampler_config.progress_bar:
                logger.info(f"[emcee] Burn-in: {sampler_config.num_warmup} steps...")
            state = self.sampler.run_mcmc(
                pos, sampler_config.num_warmup, progress=sampler_config.progress_bar
            )
            self.sampler.reset()
        else:
            state = pos

        # Accumulate samples
        samples_accum: Dict[str, np.ndarray] = {}
        if initial_samples:
            for name, values in initial_samples.items():
                samples_accum[name] = np.array(values)

        def _count_total(samples_dict: Dict[str, Any]) -> int:
            if not samples_dict:
                return 0
            arr = np.asarray(next(iter(samples_dict.values())))
            if arr.ndim == 1:
                return int(arr.shape[0])
            return int(arr.shape[0] * arr.shape[1])

        total_collected = max(int(initial_step or 0), _count_total(samples_accum))
        num_chains = int(sampler_config.num_chains)
        selected_walkers = np.linspace(0, n_walkers - 1, num_chains, dtype=int)

        target_total = total_collected + (sampler_config.num_samples * num_chains)
        default_chunk = max(10, min(100, sampler_config.num_samples))
        chunk_size = int(self.config.get("checkpoint_chunk_size", default_chunk))
        chunk_size = max(1, min(chunk_size, sampler_config.num_samples))

        last_checkpoint_time = time.time()
        last_checkpoint_step = int(total_collected)
        start_time = last_checkpoint_time

        while total_collected < target_total:
            chunk_start = time.time()
            state = self.sampler.run_mcmc(
                state, chunk_size, progress=sampler_config.progress_bar
            )
            chunk_elapsed = max(time.time() - chunk_start, 1e-6)

            chain = self.sampler.get_chain()  # (nsteps, nwalkers, ndim)
            for i, param_name in enumerate(self.param_names):
                chunk_samples = chain[:, selected_walkers, i].T  # (num_chains, nsteps)
                if param_name not in samples_accum:
                    samples_accum[param_name] = chunk_samples
                else:
                    samples_accum[param_name] = np.concatenate(
                        [samples_accum[param_name], chunk_samples], axis=1
                    )

            total_collected = _count_total(samples_accum)

            # Save checkpoints
            if checkpoint_callback and checkpoint_interval_seconds:
                now = time.time()
                if (now - last_checkpoint_time) >= checkpoint_interval_seconds:
                    checkpoint_callback(samples_accum, total_collected)
                    last_checkpoint_time = now
                    last_checkpoint_step = total_collected

            if checkpoint_callback and checkpoint_interval_steps:
                if (
                    total_collected - last_checkpoint_step
                ) >= checkpoint_interval_steps:
                    checkpoint_callback(samples_accum, total_collected)
                    last_checkpoint_step = total_collected
                    last_checkpoint_time = time.time()

            remaining_total = target_total - total_collected
            if remaining_total <= 0:
                break

            remaining_per_chain = max(1, remaining_total // num_chains)

            if checkpoint_interval_seconds:
                chunk_total = num_chains * chunk_size
                samples_per_sec = max(1.0, chunk_total / chunk_elapsed)
                target_total_interval = max(
                    1, int(samples_per_sec * checkpoint_interval_seconds * 0.9)
                )
                target_per_chain = max(1, target_total_interval // num_chains)
                chunk_size = min(remaining_per_chain, max(1, target_per_chain))
            elif checkpoint_interval_steps:
                target_per_chain = max(1, int(checkpoint_interval_steps) // num_chains)
                chunk_size = min(remaining_per_chain, max(1, target_per_chain))
            else:
                chunk_size = min(remaining_per_chain, chunk_size)

            if remaining_total > 0:
                self.sampler.reset()

        elapsed_time = time.time() - start_time

        # Persist last state for resume
        self.last_state = state

        # Convert to HIcosmo format
        samples = {name: jnp.array(values) for name, values in samples_accum.items()}

        diagnostics = self.get_diagnostics()

        results = SamplerResults(
            samples=samples,
            diagnostics=diagnostics,
            elapsed_time=elapsed_time,
            sampler_name="emcee",
            metadata={
                "num_warmup": sampler_config.num_warmup,
                "num_samples": sampler_config.num_samples,
                "n_walkers": n_walkers,
                "n_dim": n_dim,
                "chunked": True,
            },
        )

        self.results = results
        return results

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Extract emcee diagnostic information.

        Returns:
            Dictionary containing:
                - 'acceptance_rate': Mean, std, min, max acceptance fractions
                - 'autocorr_time': Autocorrelation time per parameter (if available)

        Note:
            Returns empty dict if sampler hasn't been run yet.
        """
        if self.sampler is None:
            return {}

        diagnostics = {}

        # Acceptance fraction
        acceptance_fraction = self.sampler.acceptance_fraction
        diagnostics["acceptance_rate"] = {
            "mean": float(np.mean(acceptance_fraction)),
            "std": float(np.std(acceptance_fraction)),
            "min": float(np.min(acceptance_fraction)),
            "max": float(np.max(acceptance_fraction)),
        }

        # Autocorrelation time (may fail if chain is too short)
        try:
            autocorr_time = self.sampler.get_autocorr_time(quiet=True)
            diagnostics["autocorr_time"] = {
                self.param_names[i]: float(autocorr_time[i])
                for i in range(len(self.param_names))
            }
        except Exception:
            # Silently skip if autocorrelation computation fails
            # (common for short chains)
            pass

        return diagnostics
