#!/usr/bin/env python3
"""
Basic HiCosmo Analysis Example
=============================

Demonstrates the complete workflow of cosmological parameter estimation:
1. Setting up cosmological model and parameters
2. Running MCMC analysis 
3. Creating professional visualizations
4. Analyzing results

This example uses synthetic data for demonstration purposes.
"""

import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# HiCosmo imports
from hicosmo.models import LCDM
from hicosmo.core.parameters import create_standard_cosmology_parameters
from hicosmo.samplers.mcmc import MCMCSampler
from hicosmo.visualization import quick_corner, quick_trace, HiCosmoPlotter

# Set random seed for reproducibility
key = random.PRNGKey(42)

def create_mock_sne_data(n_points: int = 100) -> tuple:
    """Create mock Type Ia supernova data for demonstration."""
    key_local = random.PRNGKey(123)
    
    # Redshift range typical for SNe Ia
    z_min, z_max = 0.01, 2.0
    z = jnp.logspace(jnp.log10(z_min), jnp.log10(z_max), n_points)
    
    # True cosmological parameters (what we'll try to recover)
    true_params = {
        'H0': 70.0,
        'Omega_m': 0.3,
        'Omega_Lambda': 0.7,
        'Omega_b': 0.05,
        'n_s': 0.965,
        'tau_reio': 0.055
    }
    
    # Create LCDM model and compute true distance moduli
    model = LCDM()
    mu_true = model.distance_modulus(z, true_params)
    
    # Add realistic noise (σ_μ ≈ 0.15 mag)
    sigma_mu = 0.15
    key_noise = random.split(key_local, 1)[0]
    noise = random.normal(key_noise, shape=mu_true.shape) * sigma_mu
    mu_obs = mu_true + noise
    mu_err = jnp.full_like(mu_obs, sigma_mu)
    
    return z, mu_obs, mu_err, true_params

def simple_sne_likelihood(params_dict: dict, z_data: jnp.ndarray, 
                         mu_data: jnp.ndarray, mu_err: jnp.ndarray) -> float:
    """Simple chi-squared likelihood for SNe Ia data."""
    model = LCDM()
    
    # Compute model predictions
    mu_model = model.distance_modulus(z_data, params_dict)
    
    # Chi-squared
    chi2 = jnp.sum(((mu_data - mu_model) / mu_err) ** 2)
    
    return -0.5 * chi2

def main():
    """Run complete analysis workflow."""
    
    print("🌌 HiCosmo Basic Analysis Example")
    print("=" * 50)
    
    # 1. Create mock data
    print("\n1. Creating mock SNe Ia data...")
    z_data, mu_data, mu_err, true_params = create_mock_sne_data(50)
    print(f"   Generated {len(z_data)} data points")
    print(f"   Redshift range: {z_data.min():.3f} - {z_data.max():.3f}")
    
    # 2. Set up parameters
    print("\n2. Setting up parameter space...")
    param_manager = create_standard_cosmology_parameters()
    
    # Focus on key cosmological parameters for this example
    param_manager.fix_parameter('Omega_b', 0.05)
    param_manager.fix_parameter('n_s', 0.965)
    param_manager.fix_parameter('tau_reio', 0.055)
    
    print(f"   Free parameters: {param_manager.free_parameter_names}")
    print(f"   Fixed parameters: {param_manager.fixed_parameter_names}")
    
    # 3. Define likelihood function
    def log_likelihood(values: jnp.ndarray) -> float:
        """Likelihood function for MCMC."""
        params_dict = param_manager.get_all_parameters_dict(values)
        return simple_sne_likelihood(params_dict, z_data, mu_data, mu_err)
    
    # 4. Run MCMC analysis
    print("\n3. Running MCMC analysis...")
    print("   This may take a moment...")
    
    sampler = MCMCSampler(
        model_fn=log_likelihood,
        param_manager=param_manager,
        num_warmup=500,
        num_samples=1000,
        num_chains=2
    )
    
    # Run sampling
    samples = sampler.run(key)
    
    # Get summary statistics
    summary = sampler.get_summary()
    print("\n   MCMC Summary:")
    print(f"   - Chains: {summary['num_chains']}")
    print(f"   - Samples per chain: {summary['num_samples']}")
    print(f"   - Effective sample size (min): {summary['ess_bulk'].min():.0f}")
    print(f"   - R-hat (max): {summary['rhat'].max():.4f}")
    
    # 4. Create visualizations
    print("\n4. Creating visualizations...")
    
    # Corner plot
    labels = param_manager.get_labels()
    truths = [true_params[name] for name in param_manager.free_parameter_names]
    
    fig_corner = quick_corner(
        samples['samples'], 
        labels=labels,
        save_path='corner_plot.png'
    )
    print("   ✓ Corner plot saved as 'corner_plot.png'")
    
    # Trace plot
    fig_trace = quick_trace(
        samples['samples'],
        labels=labels,
        save_path='trace_plot.png'
    )
    print("   ✓ Trace plot saved as 'trace_plot.png'")
    
    # Hubble diagram
    plotter = HiCosmoPlotter()
    
    # Get best-fit parameters
    flat_samples = samples['samples'].reshape(-1, samples['samples'].shape[-1])
    best_idx = jnp.argmax(samples['log_prob'].flatten())
    best_params = param_manager.get_all_parameters_dict(flat_samples[best_idx])
    
    # Create theory curve
    z_theory = jnp.logspace(-2, jnp.log10(2.5), 100)
    model = LCDM()
    mu_theory = model.distance_modulus(z_theory, best_params)
    
    fig_hubble = plotter.hubble_diagram(
        z_obs=z_data,
        mu_obs=mu_data,
        mu_err=mu_err,
        z_theory=z_theory,
        mu_theory=mu_theory,
        model_name='Best-fit ΛCDM',
        save_path='hubble_diagram.png'
    )
    print("   ✓ Hubble diagram saved as 'hubble_diagram.png'")
    
    # 5. Compare results with true values
    print("\n5. Parameter estimation results:")
    print("   Parameter    True      Best-fit    68% C.I.")
    print("   " + "-" * 45)
    
    for i, name in enumerate(param_manager.free_parameter_names):
        true_val = true_params[name]
        samples_param = flat_samples[:, i]
        
        median = jnp.percentile(samples_param, 50)
        lower = jnp.percentile(samples_param, 16)
        upper = jnp.percentile(samples_param, 84)
        
        print(f"   {name:<12} {true_val:>8.3f}  {median:>8.3f}   [{lower:.3f}, {upper:.3f}]")
    
    print("\n🎉 Analysis complete! Check the generated plots.")
    print("\nFiles created:")
    print("   - corner_plot.png: Parameter posterior distributions")
    print("   - trace_plot.png: MCMC chain convergence")  
    print("   - hubble_diagram.png: Model fit to data")

if __name__ == "__main__":
    main()