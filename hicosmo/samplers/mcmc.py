"""
MCMC sampler wrapper using NumPyro's NUTS sampler.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from numpyro.diagnostics import summary, effective_sample_size, gelman_rubin
import arviz as az
from pathlib import Path
import pickle
import json
from datetime import datetime
from rich.console import Console
from rich.progress import track
from rich.table import Table


class MCMCSampler:
    """
    MCMC sampler for cosmological parameter estimation using NumPyro.
    
    Features:
    - NUTS sampler with automatic step size adaptation
    - Multiple chains for convergence assessment
    - Smart initialization strategies
    - Real-time diagnostics and monitoring
    - Checkpoint/resume capability
    - Automatic convergence checking
    """
    
    def __init__(self, 
                 model_fn: Callable,
                 param_manager: 'ParameterManager',
                 num_warmup: int = 1000,
                 num_samples: int = 2000,
                 num_chains: int = 4,
                 target_accept: float = 0.8,
                 max_tree_depth: int = 10,
                 checkpoint_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize MCMC sampler.
        
        Args:
            model_fn: NumPyro model function
            param_manager: Parameter manager instance
            num_warmup: Number of warmup steps
            num_samples: Number of samples per chain
            num_chains: Number of parallel chains
            target_accept: Target acceptance probability
            max_tree_depth: Maximum tree depth for NUTS
            checkpoint_dir: Directory for saving checkpoints
            verbose: Whether to show progress and diagnostics
        """
        self.model_fn = model_fn
        self.param_manager = param_manager
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.target_accept = target_accept
        self.max_tree_depth = max_tree_depth
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.verbose = verbose
        
        self.console = Console() if verbose else None
        self.mcmc = None
        self.samples = None
        
        # Setup checkpoint directory
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, 
            rng_key: Optional[jax.random.PRNGKey] = None,
            init_strategy: str = 'median',
            resume: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Run MCMC sampling.
        
        Args:
            rng_key: JAX random key
            init_strategy: Initialization strategy ('median', 'uniform', 'prior')
            resume: Whether to resume from checkpoint
            
        Returns:
            Dictionary of samples
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        # Resume from checkpoint if requested
        if resume and self.checkpoint_dir:
            checkpoint_file = self.checkpoint_dir / 'mcmc_checkpoint.pkl'
            if checkpoint_file.exists():
                self._resume_from_checkpoint(checkpoint_file)
                if self.verbose:
                    self.console.print("[green]Resumed from checkpoint[/green]")
        
        # Initialize sampler if not resuming
        if self.mcmc is None:
            self._initialize_sampler(rng_key, init_strategy)
        
        # Run sampling
        if self.verbose:
            self.console.print(f"[cyan]Running MCMC with {self.num_chains} chains[/cyan]")
            self.console.print(f"Warmup: {self.num_warmup}, Samples: {self.num_samples}")
        
        self.mcmc.run(rng_key)
        self.samples = self.mcmc.get_samples()
        
        # Save checkpoint
        if self.checkpoint_dir:
            self._save_checkpoint()
        
        # Print diagnostics
        if self.verbose:
            self._print_diagnostics()
        
        # Check convergence
        if not self._check_convergence():
            if self.verbose:
                self.console.print("[yellow]Warning: Chains may not have converged![/yellow]")
        
        return self.samples
    
    def _initialize_sampler(self, rng_key: jax.random.PRNGKey, init_strategy: str):
        """Initialize MCMC sampler with specified strategy."""
        
        # Get initialization function
        if init_strategy == 'median':
            init_fn = init_to_median(num_samples=100)
        elif init_strategy == 'uniform':
            init_fn = self._init_to_uniform()
        elif init_strategy == 'prior':
            init_fn = None  # Use prior samples
        else:
            raise ValueError(f"Unknown initialization strategy: {init_strategy}")
        
        # Create NUTS kernel
        kernel = NUTS(
            self.model_fn,
            target_accept_prob=self.target_accept,
            max_tree_depth=self.max_tree_depth,
            init_strategy=init_fn
        )
        
        # Create MCMC object
        self.mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=self.verbose,
            jit_model_args=True
        )
    
    def _init_to_uniform(self):
        """Initialize parameters uniformly within prior bounds."""
        def init_fn(site=None):
            if site is None:
                return {}
            
            values = {}
            for param_name, param_info in self.param_manager.get_free_params().items():
                if 'prior' in param_info:
                    prior = param_info['prior']
                    if 'min' in prior and 'max' in prior:
                        values[param_name] = (prior['min'] + prior['max']) / 2
                    elif 'loc' in prior:
                        values[param_name] = prior['loc']
                    else:
                        values[param_name] = 0.0
                else:
                    values[param_name] = param_info.get('ref', 0.0)
            
            return values
        
        return init_fn
    
    def _check_convergence(self, r_hat_threshold: float = 1.01) -> bool:
        """
        Check MCMC convergence using R-hat statistic.
        
        Args:
            r_hat_threshold: Threshold for R-hat (should be < 1.01 for convergence)
            
        Returns:
            True if converged, False otherwise
        """
        if self.samples is None:
            return False
        
        # Calculate R-hat for each parameter
        r_hats = {}
        for param_name in self.samples.keys():
            param_samples = self.samples[param_name]
            if param_samples.ndim == 1:
                param_samples = param_samples.reshape(self.num_chains, -1)
            r_hat = gelman_rubin(param_samples)
            r_hats[param_name] = r_hat
        
        # Check if all R-hats are below threshold
        converged = all(r_hat < r_hat_threshold for r_hat in r_hats.values())
        
        return converged
    
    def _print_diagnostics(self):
        """Print MCMC diagnostics."""
        if self.samples is None or not self.verbose:
            return
        
        # Create diagnostics table
        table = Table(title="MCMC Diagnostics")
        table.add_column("Parameter", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("R-hat", justify="right")
        table.add_column("ESS", justify="right")
        
        for param_name in sorted(self.samples.keys()):
            param_samples = self.samples[param_name]
            
            # Reshape if needed
            if param_samples.ndim == 1:
                param_samples = param_samples.reshape(self.num_chains, -1)
            
            # Calculate statistics
            mean_val = float(jnp.mean(param_samples))
            std_val = float(jnp.std(param_samples))
            r_hat = float(gelman_rubin(param_samples))
            ess = float(effective_sample_size(param_samples))
            
            # Color code R-hat
            r_hat_str = f"{r_hat:.3f}"
            if r_hat > 1.01:
                r_hat_str = f"[red]{r_hat_str}[/red]"
            elif r_hat > 1.005:
                r_hat_str = f"[yellow]{r_hat_str}[/yellow]"
            else:
                r_hat_str = f"[green]{r_hat_str}[/green]"
            
            table.add_row(
                param_name,
                f"{mean_val:.4f}",
                f"{std_val:.4f}",
                r_hat_str,
                f"{ess:.0f}"
            )
        
        self.console.print(table)
    
    def _save_checkpoint(self):
        """Save MCMC state to checkpoint file."""
        if not self.checkpoint_dir:
            return
        
        checkpoint = {
            'samples': self.samples,
            'num_warmup': self.num_warmup,
            'num_samples': self.num_samples,
            'num_chains': self.num_chains,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = self.checkpoint_dir / 'mcmc_checkpoint.pkl'
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save samples in text format for compatibility
        self._save_chains_txt()
    
    def _resume_from_checkpoint(self, checkpoint_file: Path):
        """Resume MCMC from checkpoint."""
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.samples = checkpoint['samples']
        if self.verbose:
            self.console.print(f"Loaded checkpoint from {checkpoint['timestamp']}")
    
    def _save_chains_txt(self):
        """Save chains in text format for GetDist compatibility."""
        if not self.checkpoint_dir or self.samples is None:
            return
        
        # Flatten samples
        flat_samples = {}
        param_names = []
        for param_name, param_samples in self.samples.items():
            flat_samples[param_name] = param_samples.flatten()
            param_names.append(param_name)
        
        # Create chains array
        n_samples_total = len(flat_samples[param_names[0]])
        chains_array = np.zeros((n_samples_total, len(param_names) + 2))
        
        # Add weight and -log(likelihood) columns (placeholders)
        chains_array[:, 0] = 1.0  # weights
        chains_array[:, 1] = 0.0  # -log(likelihood)
        
        # Add parameter samples
        for i, param_name in enumerate(param_names):
            chains_array[:, i + 2] = flat_samples[param_name]
        
        # Save chains
        chains_file = self.checkpoint_dir / 'chains.txt'
        np.savetxt(chains_file, chains_array)
        
        # Save parameter names
        params_file = self.checkpoint_dir / 'chains.paramnames'
        with open(params_file, 'w') as f:
            for param_name in param_names:
                f.write(f"{param_name}\t{param_name}\n")
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all parameters.
        
        Returns:
            Dictionary with parameter statistics
        """
        if self.samples is None:
            raise RuntimeError("No samples available. Run MCMC first.")
        
        summary_dict = {}
        for param_name, param_samples in self.samples.items():
            flat_samples = param_samples.flatten()
            
            summary_dict[param_name] = {
                'mean': float(jnp.mean(flat_samples)),
                'std': float(jnp.std(flat_samples)),
                'median': float(jnp.median(flat_samples)),
                'q16': float(jnp.percentile(flat_samples, 16)),
                'q84': float(jnp.percentile(flat_samples, 84)),
                'q025': float(jnp.percentile(flat_samples, 2.5)),
                'q975': float(jnp.percentile(flat_samples, 97.5))
            }
        
        return summary_dict
    
    def to_arviz(self) -> az.InferenceData:
        """
        Convert samples to ArviZ InferenceData object.
        
        Returns:
            ArviZ InferenceData object
        """
        if self.samples is None:
            raise RuntimeError("No samples available. Run MCMC first.")
        
        # Reshape samples for ArviZ
        reshaped_samples = {}
        for param_name, param_samples in self.samples.items():
            if param_samples.ndim == 1:
                param_samples = param_samples.reshape(self.num_chains, -1)
            reshaped_samples[param_name] = param_samples
        
        return az.from_dict(posterior=reshaped_samples)