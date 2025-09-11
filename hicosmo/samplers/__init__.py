"""
HiCosmo MCMC Sampling Module

Provides flexible Bayesian inference tools built on NumPyro.
Features both low-level NumPyro wrappers and high-level automatic systems.
"""

from .core import MCMCSampler, DiagnosticsTools, run_mcmc
from .inference import MCMC, quick_mcmc
from .config import ParameterConfig, AutoParameter
from .utils import ParameterMapper, FunctionInspector, create_auto_model, validate_likelihood_compatibility, analyze_likelihood_function
from .persistence import MCMCState, CheckpointManager, ResumeManager, create_likelihood_info, create_data_info
from .multicore import setup_multicore_execution, get_optimal_chain_count, check_multicore_status, print_multicore_info
from .init import Config, init_hicosmo

__all__ = [
    # Core MCMC (Low-level)
    'MCMCSampler', 'DiagnosticsTools', 'run_mcmc',
    # MCMC System (High-level)
    'MCMC', 'quick_mcmc',
    # Parameter Management
    'ParameterConfig', 'AutoParameter', 
    # Advanced Tools
    'ParameterMapper', 'FunctionInspector',
    'create_auto_model', 'validate_likelihood_compatibility', 'analyze_likelihood_function',
    # Checkpoint System
    'MCMCState', 'CheckpointManager', 'ResumeManager',
    'create_likelihood_info', 'create_data_info',
    # Multi-core Optimization
    'setup_multicore_execution', 'get_optimal_chain_count', 'check_multicore_status', 'print_multicore_info',
    # Elegant Initialization
    'Config', 'init_hicosmo'
]