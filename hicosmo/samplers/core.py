"""
Lightweight MCMC wrapper around NumPyro for flexible Bayesian inference.

This module provides a thin wrapper around NumPyro's MCMC implementation,
adding convenience features while maintaining full flexibility for users
to define their own models and parameter management.

Key Features:
- Direct use of NumPyro's mature MCMC implementations (NUTS, HMC, etc.)
- Enhanced diagnostics and visualization
- Checkpoint save/restore functionality
- Beautiful progress display with rich
- No assumptions about parameter management or model structure
"""

from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, HMC, SA
from numpyro.diagnostics import summary, effective_sample_size, gelman_rubin, hpdi
import arviz as az
from pathlib import Path
import pickle
import json
import h5py
import time
from datetime import datetime
from .constants import (
    DEFAULT_NUM_SAMPLES, DEFAULT_NUM_CHAINS,
    DEFAULT_WARMUP_STANDARD, RNG_SEED_MODULO
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
import warnings

import jax
import jax.numpy as jnp
from jax import jit, vmap


class MCMCSampler:
    """
    A lightweight wrapper around NumPyro's MCMC with enhanced utilities.
    
    This class provides a thin abstraction layer over NumPyro's MCMC,
    offering convenience features while maintaining full flexibility.
    Users can define any NumPyro-compatible model without restrictions.
    
    Example
    -------
    >>> import numpyro.distributions as dist
    >>> 
    >>> def my_model(data):
    >>>     # User defines their own parameters and priors
    >>>     theta = numpyro.sample("theta", dist.Normal(0, 1))
    >>>     sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    >>>     # User defines their own likelihood
    >>>     numpyro.sample("obs", dist.Normal(theta, sigma), obs=data)
    >>> 
    >>> sampler = MCMCSampler(my_model)
    >>> sampler.run(data=my_data)
    >>> samples = sampler.get_samples()
    """
    
    def __init__(self,
                 model_fn: Callable,
                 kernel: Optional[object] = None,
                 num_warmup: int = DEFAULT_WARMUP_STANDARD,  # Default value, intelligently adjusted by MCMC
                 num_samples: int = DEFAULT_NUM_SAMPLES,
                 num_chains: int = 4,
                 chain_method: Optional[str] = None,  # None means auto-detect
                 progress_bar: bool = True,
                 verbose: bool = True,
                 **mcmc_kwargs):
        """
        Initialize the MCMC sampler.
        
        Parameters
        ----------
        model_fn : Callable
            A NumPyro model function. Users have complete freedom
            in how they define this model.
        kernel : object, optional
            A NumPyro kernel (NUTS, HMC, SA, etc.). If None, uses NUTS
            with default settings.
        num_warmup : int
            Number of warmup/burn-in steps
        num_samples : int
            Number of samples to draw per chain
        num_chains : int
            Number of MCMC chains to run
        chain_method : str, optional
            How to run chains: 'parallel', 'sequential', or 'vectorized'.
            If None, automatically detects based on device availability.
        progress_bar : bool
            Whether to show progress bar
        verbose : bool
            Whether to print diagnostic information
        **mcmc_kwargs
            Additional keyword arguments passed to NumPyro's MCMC class
        """
        self.model_fn = model_fn
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.progress_bar = progress_bar
        self.verbose = verbose
        
        # Intelligently detect chain method
        self.chain_method = self._detect_chain_method(chain_method)
        
        # Initialize console for pretty output
        self.console = Console() if verbose else None
        
        # Setup kernel (default to NUTS if not provided)
        if kernel is None:
            kernel = NUTS(model_fn)
        
        # Create NumPyro MCMC object
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=self.chain_method,  # Use detected method
            progress_bar=False,  # We'll use our own progress display
            **mcmc_kwargs
        )
        
        # Storage for results
        self._samples = None
        self._run_time = None
        
    def _detect_chain_method(self, chain_method: Optional[str]) -> str:
        """
        Intelligently detect chain execution method
        
        Parameters
        ----------
        chain_method : str, optional
            User-specified chain method, auto-detect if None
            
        Returns
        -------
        str
            Actual chain method to use
        """
        # If user explicitly specified a method, use it
        if chain_method is not None:
            return chain_method
        
        try:
            device_count = jax.local_device_count()
            
            # Determine method based on device count and chain count
            if device_count >= self.num_chains and self.num_chains > 1:
                return 'parallel'
            elif self.num_chains > 1:
                return 'sequential'  # Multiple chains but insufficient devices, use sequential
            else:
                return 'sequential'  # Single chain case
                
        except ImportError:
            # JAX not available, default to sequential
            return 'sequential'
        
    def run(self, rng_key = None,
            *args, **kwargs):
        """
        Run MCMC sampling.
        
        This method directly calls NumPyro's MCMC.run() with enhanced
        progress display and timing.
        
        Parameters
        ----------
        rng_key : optional
            Random key for JAX. If None, generates one from current time.
        *args, **kwargs
            Arguments passed directly to the model function

        Returns
        -------
        Dict
            Dictionary of samples for each parameter
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(int(time.time() * 1000) % RNG_SEED_MODULO)
        
        # Print configuration if verbose
        if self.verbose:
            self._print_config()
        
        # Track timing with detailed information
        from datetime import datetime
        start_time = time.time()
        start_datetime = datetime.now()
        
        if self.verbose:
            self.console.print(f"🚀 [bold]MCMC Started:[/bold] {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run MCMC with progress display
        if self.progress_bar and self.verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console
            ) as progress:
                # Create progress task
                total_steps = self.num_warmup + self.num_samples
                task = progress.add_task(
                    f"Running MCMC ({self.num_chains} chains)", 
                    total=total_steps
                )
                
                # Run MCMC (NumPyro handles everything internally)
                self.mcmc.run(rng_key, *args, **kwargs)
                
                # Update progress to complete
                progress.update(task, completed=total_steps)
        else:
            # Run without progress display
            self.mcmc.run(rng_key, *args, **kwargs)
        
        # Record run time with detailed information
        end_time = time.time()
        end_datetime = datetime.now()
        self._run_time = end_time - start_time
        
        # Store timing information
        self._start_time = start_datetime
        self._end_time = end_datetime
        
        # Get samples - force evaluation to avoid JAX lazy evaluation issues
        if self.verbose:
            print("📊 Retrieving samples...")

        # Force JAX to complete all computations
        if hasattr(self.mcmc, '_states'):
            # Block until ready to ensure computation is complete
            _ = jax.device_get(self.mcmc._states)

        # Now get samples normally
        self._samples = self.mcmc.get_samples()

        # Convert to numpy arrays to avoid further JAX issues
        self._samples = {k: np.array(v) for k, v in self._samples.items()}

        if self.verbose:
            n_samples = len(next(iter(self._samples.values())))
            print(f"✓ Retrieved {n_samples} samples")

        # Print summary if verbose
        if self.verbose:
            self._print_summary()
        
        return self._samples
    
    def get_samples(self, group_by_chain: bool = False):
        """
        Get MCMC samples.
        
        Parameters
        ----------
        group_by_chain : bool
            If True, returns samples grouped by chain.
            Shape will be (num_chains, num_samples, ...)
            If False, returns flattened samples.
            Shape will be (num_chains * num_samples, ...)
            
        Returns
        -------
        Dict
            Dictionary of samples for each parameter
        """
        if self._samples is None:
            self._samples = self.mcmc.get_samples(group_by_chain=group_by_chain)
        return self._samples
    
    def print_summary(self, prob: float = 0.9, burnin_frac: float = 0.1) -> None:
        """
        Print a beautiful summary of MCMC results.
        
        Parameters
        ----------
        prob : float
            Probability for credible intervals
        burnin_frac : float
            Fraction of samples to discard as burn-in (default: 0.1 = 10%)
        """
        if self._samples is None:
            raise RuntimeError("No samples available. Run MCMC first.")
        
        # Get samples grouped by chain for summary statistics
        samples_grouped = self.mcmc.get_samples(group_by_chain=True)
        
        # Apply burn-in removal
        if burnin_frac > 0:
            samples_with_burnin = {}
            for param_name, samples in samples_grouped.items():
                # samples shape: (num_chains, num_samples, ...)
                num_samples = samples.shape[1]
                burnin_samples = int(burnin_frac * num_samples)
                samples_with_burnin[param_name] = samples[:, burnin_samples:]
            samples_grouped = samples_with_burnin
        
        # Use NumPyro's summary function
        summary_dict = summary(samples_grouped, prob=prob)
        
        # Create beautiful table with rich
        table = Table(title="MCMC Summary Statistics")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column(f"{int(prob*100)}% CI", justify="center")
        table.add_column("n_eff", justify="right")
        table.add_column("r_hat", justify="right")
        
        for param_name in self._samples.keys():
            stats = summary_dict[param_name]
            mean = stats['mean']
            std = stats['std']
            # Get credible interval - NumPyro uses different naming convention
            alpha = (1 - prob) / 2
            ci_low_key = f'{alpha*100:.1f}%'
            ci_high_key = f'{(1-alpha)*100:.1f}%'
            ci_low = stats[ci_low_key]
            ci_high = stats[ci_high_key]
            n_eff = stats['n_eff']
            r_hat = stats['r_hat']
            
            # Color code r_hat
            if r_hat < 1.01:
                r_hat_str = f"[green]{r_hat:.3f}[/green]"
            elif r_hat < 1.05:
                r_hat_str = f"[yellow]{r_hat:.3f}[/yellow]"
            else:
                r_hat_str = f"[red]{r_hat:.3f}[/red]"
            
            table.add_row(
                param_name,
                f"{mean:.4f}",
                f"{std:.4f}",
                f"[{ci_low:.4f}, {ci_high:.4f}]",
                f"{n_eff:.0f}",
                r_hat_str
            )
        
        if self.console:
            self.console.print(table)
        else:
            print(table)
    
    def get_diagnostics(self, burnin_frac: float = 0.1) -> Dict[str, Any]:
        """
        Get comprehensive MCMC diagnostics.
        
        Parameters
        ----------
        burnin_frac : float
            Fraction of samples to discard as burn-in (default: 0.1 = 10%)
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing various diagnostic metrics
        """
        if self._samples is None:
            raise RuntimeError("No samples available. Run MCMC first.")
        
        # Get samples grouped by chain for diagnostics
        samples_by_chain = self.mcmc.get_samples(group_by_chain=True)
        
        # Apply burn-in removal
        if burnin_frac > 0:
            samples_with_burnin = {}
            for param_name, samples in samples_by_chain.items():
                # samples shape: (num_chains, num_samples, ...)
                num_samples = samples.shape[1]
                burnin_samples = int(burnin_frac * num_samples)
                samples_with_burnin[param_name] = samples[:, burnin_samples:]
            samples_by_chain = samples_with_burnin
        
        diagnostics = {}
        
        # For each parameter, compute diagnostics
        for param_name, param_samples in samples_by_chain.items():
            try:
                # Gelman-Rubin statistic (requires at least 2 chains)
                r_hat = float(gelman_rubin(param_samples))
            except (AssertionError, ValueError) as e:
                # Single chain or insufficient data for R-hat
                if self.num_chains == 1:
                    r_hat = 1.0  # Perfect convergence assumed for single chain
                else:
                    r_hat = float('nan')  # Mark as unavailable
            
            # Effective sample size
            try:
                ess = float(effective_sample_size(param_samples))
            except (AssertionError, ValueError):
                ess = float('nan')
            
            # Effective samples per second
            ess_per_sec = ess / self._run_time if (self._run_time and not np.isnan(ess)) else 0
            
            diagnostics[param_name] = {
                'r_hat': r_hat,
                'ess': ess,
                'ess_per_sec': ess_per_sec,
                'mean': float(jnp.mean(param_samples)),
                'std': float(jnp.std(param_samples)),
                'min': float(jnp.min(param_samples)),
                'max': float(jnp.max(param_samples))
            }
        
        # Add overall diagnostics
        diagnostics['_overall'] = {
            'total_time': self._run_time,
            'num_chains': self.num_chains,
            'num_samples': self.num_samples,
            'num_warmup': self.num_warmup
        }
        
        return diagnostics
    
    def save_results(self, filepath: Union[str, Path], format: str = 'pickle') -> None:
        """
        Save MCMC results to file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save file
        format : str
            Format to save in: 'pickle', 'hdf5', or 'arviz'
        """
        if self._samples is None:
            raise RuntimeError("No samples available. Run MCMC first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'samples': self._samples,
                    'diagnostics': self.get_diagnostics(),
                    'config': {
                        'num_warmup': self.num_warmup,
                        'num_samples': self.num_samples,
                        'num_chains': self.num_chains
                    },
                    'timestamp': datetime.now().isoformat()
                }, f)
                
        elif format == 'hdf5':
            with h5py.File(filepath, 'w') as f:
                # Save samples
                samples_group = f.create_group('samples')
                for param_name, param_samples in self._samples.items():
                    samples_group.create_dataset(param_name, data=np.array(param_samples))
                
                # Save config
                config_group = f.create_group('config')
                config_group.attrs['num_warmup'] = self.num_warmup
                config_group.attrs['num_samples'] = self.num_samples
                config_group.attrs['num_chains'] = self.num_chains
                config_group.attrs['timestamp'] = datetime.now().isoformat()
                
        elif format == 'arviz':
            # Convert to ArviZ InferenceData
            inference_data = self.to_arviz()
            inference_data.to_netcdf(filepath)
            
        else:
            raise ValueError(f"Unknown format: {format}")
        
        if self.verbose:
            self.console.print(f"[green]✓ Results saved to {filepath}[/green]")
    
    def load_results(self, filepath: Union[str, Path], format: str = 'pickle') -> None:
        """
        Load MCMC results from file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to load file
        format : str
            Format to load from: 'pickle', 'hdf5', or 'arviz'
        """
        filepath = Path(filepath)
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self._samples = data['samples']
                self._run_time = data.get('diagnostics', {}).get('_overall', {}).get('total_time')
                
        elif format == 'hdf5':
            with h5py.File(filepath, 'r') as f:
                self._samples = {}
                for param_name in f['samples'].keys():
                    self._samples[param_name] = jnp.array(f['samples'][param_name][:])
                    
        elif format == 'arviz':
            inference_data = az.from_netcdf(filepath)
            # Convert from ArviZ format
            self._samples = {}
            for var_name in inference_data.posterior.data_vars:
                values = inference_data.posterior[var_name].values
                # Flatten chains and draws
                self._samples[var_name] = jnp.array(values.reshape(-1, *values.shape[2:]))
                
        else:
            raise ValueError(f"Unknown format: {format}")
        
        if self.verbose:
            self.console.print(f"[green]✓ Results loaded from {filepath}[/green]")
    
    def to_arviz(self) -> az.InferenceData:
        """
        Convert samples to ArviZ InferenceData for analysis.
        
        Returns
        -------
        arviz.InferenceData
            ArviZ data structure for further analysis
        """
        if self._samples is None:
            raise RuntimeError("No samples available. Run MCMC first.")
        
        # Get samples grouped by chain
        samples_by_chain = self.mcmc.get_samples(group_by_chain=True)
        
        # Convert to ArviZ format
        return az.from_dict(posterior=samples_by_chain)
    
    def _print_config(self) -> None:
        """Print MCMC configuration."""
        # Get device information - prefer configured cores over JAX device count
        try:
            # First try to get configured CPU cores from Config
            from .init import Config
            if hasattr(Config, '_config') and 'actual_cores' in Config._config:
                configured_cores = Config._config['actual_cores']
                device_info = f"{configured_cores} CPU cores (thread pool)"
            else:
                # Try JAX_NUM_THREADS environment variable
                import os
                jax_threads = os.environ.get('JAX_NUM_THREADS')
                if jax_threads:
                    device_info = f"{jax_threads} CPU cores (from JAX_NUM_THREADS)"
                else:
                    # Fallback to system CPU count
                    cpu_count = os.cpu_count() or 1
                    device_info = f"{cpu_count} CPU cores (system detected)"
        except:
            device_info = "Unknown"
            
        config_panel = Panel.fit(
            f"[cyan]Chains:[/cyan] {self.num_chains}\n"
            f"[cyan]Warmup:[/cyan] {self.num_warmup}\n"
            f"[cyan]Samples:[/cyan] {self.num_samples}\n"
            f"[cyan]Method:[/cyan] {self.chain_method}\n"
            f"[cyan]CPU Cores:[/cyan] {device_info}",
            title="MCMC Configuration",
            border_style="blue"
        )
        self.console.print(config_panel)
    
    def _print_summary(self) -> None:
        """Print post-run summary."""
        # Print completion with timing details
        self.console.print(f"\n[green]✓ MCMC completed in {self._run_time:.2f} seconds[/green]")
        
        # Print detailed timing information
        if hasattr(self, '_start_time') and hasattr(self, '_end_time'):
            self.console.print(f"🕐 [bold]Started:[/bold] {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.console.print(f"🕐 [bold]Finished:[/bold] {self._end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.console.print(f"⏱️  [bold]Duration:[/bold] {self._run_time:.3f} seconds")
        
        # Quick convergence check
        diagnostics = self.get_diagnostics()
        converged = all(
            d['r_hat'] < 1.01 
            for param, d in diagnostics.items() 
            if param != '_overall'
        )
        
        if converged:
            self.console.print("[green]✓ All parameters converged (R̂ < 1.01)[/green]")
        else:
            unconverged = [
                param for param, d in diagnostics.items()
                if param != '_overall' and d['r_hat'] >= 1.01
            ]
            self.console.print(f"[yellow]⚠ Parameters not converged: {unconverged}[/yellow]")


class DiagnosticsTools:
    """
    Enhanced diagnostic tools for MCMC analysis.
    
    This class provides JAX-optimized diagnostic computations
    and beautiful visualizations for MCMC results.
    """
    
    @staticmethod
    @jit
    def autocorrelation(x, max_lag: Optional[int] = None):
        """
        Compute autocorrelation function using JAX.
        
        Parameters
        ----------
        x : jnp.ndarray
            1D array of samples
        max_lag : int, optional
            Maximum lag to compute
            
        Returns
        -------
        jnp.ndarray
            Autocorrelation values for each lag
        """
        x = x - jnp.mean(x)
        c0 = jnp.dot(x, x) / len(x)
        
        if max_lag is None:
            max_lag = len(x) // 4
        
        acf = jnp.zeros(max_lag)
        for k in range(max_lag):
            if k == 0:
                acf = acf.at[k].set(1.0)
            else:
                ck = jnp.dot(x[:-k], x[k:]) / len(x)
                acf = acf.at[k].set(ck / c0)
        
        return acf
    
    @staticmethod
    @jit
    def integrated_autocorrelation_time(x: jnp.ndarray) -> float:
        """
        Estimate integrated autocorrelation time using JAX.
        
        Parameters
        ----------
        x : jnp.ndarray
            1D array of samples
            
        Returns
        -------
        float
            Integrated autocorrelation time
        """
        acf = DiagnosticsTools.autocorrelation(x)
        
        # Find first negative autocorrelation
        first_negative = jnp.where(acf < 0)[0]
        if len(first_negative) > 0:
            cutoff = first_negative[0]
        else:
            cutoff = len(acf)
        
        # Integrate up to cutoff
        tau = 1 + 2 * jnp.sum(acf[1:cutoff])
        return tau
    
    @staticmethod
    def plot_trace(samples: Dict[str, jnp.ndarray], 
                   params: Optional[List[str]] = None) -> None:
        """
        Create trace plots for MCMC chains.
        
        Parameters
        ----------
        samples : Dict[str, jnp.ndarray]
            Dictionary of samples
        params : List[str], optional
            Parameters to plot. If None, plots all.
        """
        import matplotlib.pyplot as plt
        
        if params is None:
            params = list(samples.keys())
        
        n_params = len(params)
        fig, axes = plt.subplots(n_params, 2, figsize=(12, 3*n_params))
        
        if n_params == 1:
            axes = axes.reshape(1, -1)
        
        for i, param in enumerate(params):
            param_samples = samples[param]
            
            # Trace plot
            axes[i, 0].plot(param_samples, alpha=0.7)
            axes[i, 0].set_ylabel(param)
            axes[i, 0].set_xlabel('Iteration')
            axes[i, 0].set_title(f'{param} Trace')
            
            # Histogram
            axes[i, 1].hist(param_samples, bins=50, alpha=0.7, density=True)
            axes[i, 1].set_xlabel(param)
            axes[i, 1].set_ylabel('Density')
            axes[i, 1].set_title(f'{param} Distribution')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_corner(samples: Dict[str, jnp.ndarray], 
                    params: Optional[List[str]] = None,
                    **corner_kwargs) -> None:
        """
        Create corner plot for MCMC samples.
        
        Parameters
        ----------
        samples : Dict[str, jnp.ndarray]
            Dictionary of samples
        params : List[str], optional
            Parameters to plot. If None, plots all.
        **corner_kwargs
            Additional arguments for corner.corner()
        """
        try:
            import corner
        except ImportError:
            raise ImportError("Please install corner: pip install corner")
        
        if params is None:
            params = list(samples.keys())
        
        # Prepare data for corner
        data = np.column_stack([np.array(samples[p]).flatten() for p in params])
        
        # Create corner plot
        fig = corner.corner(
            data,
            labels=params,
            show_titles=True,
            title_fmt='.3f',
            quantiles=[0.16, 0.5, 0.84],
            **corner_kwargs
        )
        
        return fig


# Convenience function for quick MCMC runs
def run_mcmc(model_fn: Callable,
             num_samples: int = DEFAULT_NUM_SAMPLES,
             num_chains: int = 4,
             show_summary: bool = True,
             **kwargs) -> Dict[str, jnp.ndarray]:
    """
    Convenience function for quick MCMC runs.
    
    Parameters
    ----------
    model_fn : Callable
        NumPyro model function
    num_samples : int
        Number of samples per chain
    num_chains : int
        Number of chains
    show_summary : bool
        Whether to print summary
    **kwargs
        Additional arguments passed to model function
        
    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary of samples
    
    Example
    -------
    >>> def model(data):
    >>>     mu = numpyro.sample("mu", dist.Normal(0, 10))
    >>>     numpyro.sample("obs", dist.Normal(mu, 1), obs=data)
    >>> 
    >>> samples = run_mcmc(model, data=my_data)
    """
    sampler = MCMCSampler(model_fn, num_samples=num_samples, num_chains=num_chains)
    samples = sampler.run(**kwargs)
    
    if show_summary:
        sampler.print_summary()
    
    return samples
