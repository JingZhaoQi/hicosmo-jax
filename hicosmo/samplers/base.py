"""
HIcosmo Samplers - Unified Sampler Interface
=============================================

Abstract base class defining the interface all MCMC samplers must implement.

Design Principles:
- Strategy Pattern: Different samplers can be used interchangeably
- Minimal Interface: Only define essential methods to avoid over-abstraction
- Type Safety: Use typing to ensure interface consistency

Author: HIcosmo Development Team
License: MIT
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Tuple
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class SamplerConfig:
    """
    Base sampler configuration class.

    Contains shared configuration parameters for all samplers.

    Attributes:
        num_samples: Number of samples per chain
        num_chains: Number of independent chains
        num_warmup: Number of warmup/burn-in steps
        seed: Random seed for reproducibility
        progress_bar: Whether to show progress bar during sampling
        chain_method: How to run multiple chains ('vectorized', 'sequential', 'parallel')
            - 'vectorized': Run chains in parallel using vmap (recommended for CPU)
            - 'sequential': Run chains one after another (single-threaded)
            - 'parallel': Run chains on multiple devices using pmap (for multi-GPU)
        max_tree_depth: NUTS-specific parameter (optional)
        n_walkers: emcee-specific parameter (optional)
        moves: emcee-specific parameter (optional)
    """
    num_samples: int
    num_chains: int
    num_warmup: int
    seed: int = 42
    progress_bar: bool = True
    chain_method: str = 'vectorized'  # 'vectorized', 'sequential', 'parallel'

    # Optional sampler-specific parameters
    max_tree_depth: Optional[int] = None
    n_walkers: Optional[int] = None
    moves: Optional[Any] = None


@dataclass
class SamplerResults:
    """
    Unified sampler results format.

    All samplers must return results in this standardized format.

    Attributes:
        samples: Posterior samples as {param_name: array(num_chains, num_samples)}
        diagnostics: Diagnostic information (R-hat, ESS, acceptance rate, etc.)
        elapsed_time: Total sampling time in seconds
        sampler_name: Name of the sampler used ('numpyro', 'emcee', etc.)
        metadata: Additional sampler-specific metadata
    """
    samples: Dict[str, jnp.ndarray]
    diagnostics: Dict[str, Any]
    elapsed_time: float
    sampler_name: str
    metadata: Dict[str, Any]


class SamplerBackend(ABC):
    """
    Abstract base class for MCMC sampler backends.

    All sampler implementations (NumPyro, emcee, Zeus, etc.) must inherit from
    this class and implement the abstract methods.

    Design Philosophy:
    - Keep base class minimal: Only define must-have interface methods
    - Specialize via inheritance: Each sampler has its own config and behavior
    - Avoid conditionals: Don't use if statements to distinguish sampler types

    Example:
        class NumPyroSampler(SamplerBackend):
            def run(self, config):
                # NumPyro-specific implementation
                ...

            def get_diagnostics(self):
                # Extract NumPyro diagnostics
                ...
    """

    def __init__(
        self,
        log_probability: Callable,
        parameters: Dict[str, Dict[str, Any]],
        config: Dict[str, Any]
    ):
        """
        Initialize sampler backend.

        Args:
            log_probability: Log probability function with signature f(params_dict) -> float
            parameters: Parameter configuration dictionary with format:
                {
                    'param_name': {
                        'prior': {'dist': 'uniform', 'min': 0, 'max': 1},
                        'ref': 0.5,  # Reference value (optional)
                        ...
                    }
                }
            config: Sampler-specific configuration dictionary
        """
        self.log_probability = log_probability
        self.parameters = parameters
        self.config = config
        self.results: Optional[SamplerResults] = None

    @abstractmethod
    def run(self, sampler_config: SamplerConfig) -> SamplerResults:
        """
        Run MCMC sampling.

        Args:
            sampler_config: Sampler configuration object

        Returns:
            SamplerResults: Standardized sampling results

        Raises:
            MCMCError: If sampling fails
        """
        pass

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get MCMC diagnostic information.

        Returns:
            Dictionary containing:
                - 'rhat': Gelman-Rubin R-hat statistic
                - 'ess': Effective sample size
                - 'acceptance_rate': Acceptance rate
                - Sampler-specific diagnostics
        """
        pass

    def validate_config(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            bool: True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.parameters:
            raise ValueError("Parameters dictionary cannot be empty")

        if not callable(self.log_probability):
            raise TypeError("log_probability must be callable")

        return True

    @property
    def name(self) -> str:
        """
        Return sampler name.

        Returns:
            str: Sampler name (e.g., 'numpyro', 'emcee')
        """
        return self.__class__.__name__.replace('Sampler', '').lower()


# Custom exception classes for better error handling

class MCMCError(Exception):
    """Base class for MCMC sampling errors."""
    pass


class InitializationError(MCMCError):
    """Raised when sampler initialization fails."""
    pass


class ConvergenceError(MCMCError):
    """Raised when sampling fails to converge."""
    pass


class LikelihoodError(MCMCError):
    """Raised when likelihood computation fails."""
    pass
