"""
HiCosmo MCMC Sampling Module

Provides flexible Bayesian inference tools built on NumPyro.
Features both low-level NumPyro wrappers and high-level automatic systems.
"""

from .core import MCMCSampler, DiagnosticsTools, run_mcmc
from .auto import AutoMCMC, quick_mcmc
from .config import ParameterConfig, AutoParameter
from .utils import ParameterMapper, FunctionInspector, create_auto_model, validate_likelihood_compatibility, analyze_likelihood_function
from .persistence import MCMCState, CheckpointManager, ResumeManager, create_likelihood_info, create_data_info

__all__ = [
    # Core MCMC (Low-level)
    'MCMCSampler', 'DiagnosticsTools', 'run_mcmc',
    # Auto MCMC System (High-level)
    'AutoMCMC', 'quick_mcmc',
    # Parameter Management
    'ParameterConfig', 'AutoParameter', 
    # Advanced Tools
    'ParameterMapper', 'FunctionInspector',
    'create_auto_model', 'validate_likelihood_compatibility', 'analyze_likelihood_function',
    # Checkpoint System
    'MCMCState', 'CheckpointManager', 'ResumeManager',
    'create_likelihood_info', 'create_data_info'
]