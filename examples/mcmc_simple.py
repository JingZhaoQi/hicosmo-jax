#!/usr/bin/env python3
"""
Simple MCMC Example using HiCosmo's lightweight NumPyro wrapper.

This example demonstrates the basic usage of MCMCSampler with a simple
Gaussian model, showing how users have complete freedom in defining
their models and priors.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

# Add HiCosmo to path if needed
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers import MCMCSampler, DiagnosticsTools, run_mcmc


def example_1_basic_gaussian():
    """
    Example 1: Basic Gaussian inference.
    
    Infer the mean and standard deviation of a Gaussian distribution
    from observed data.
    """
    print("=" * 60)
    print("Example 1: Basic Gaussian Inference")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    true_mean = 5.0
    true_std = 2.0
    n_data = 100
    observed_data = np.random.normal(true_mean, true_std, n_data)
    
    print(f"True parameters: mean={true_mean}, std={true_std}")
    print(f"Number of observations: {n_data}")
    
    # Define the NumPyro model
    def gaussian_model(data=None):
        """
        Simple Gaussian model with unknown mean and standard deviation.
        
        Users have complete freedom in how they define their models.
        """
        # Define priors (user's choice)
        mean = numpyro.sample("mean", dist.Normal(0, 10))
        std = numpyro.sample("std", dist.HalfNormal(5))
        
        # Define likelihood
        with numpyro.plate("data", len(data) if data is not None else 1):
            numpyro.sample("obs", dist.Normal(mean, std), obs=data)
    
    # Create sampler
    sampler = MCMCSampler(
        gaussian_model,
        num_warmup=1000,
        num_samples=2000,
        num_chains=4
    )
    
    # Run MCMC
    samples = sampler.run(data=observed_data)
    
    # Print summary
    print("\nMCMC Results:")
    sampler.print_summary()
    
    # Get diagnostics
    diagnostics = sampler.get_diagnostics()
    print(f"\nConvergence diagnostics:")
    print(f"  Mean R-hat: {diagnostics['mean']['r_hat']:.3f}")
    print(f"  Std R-hat: {diagnostics['std']['r_hat']:.3f}")
    
    return samples


def example_2_linear_regression():
    """
    Example 2: Bayesian linear regression.
    
    Infer slope and intercept from noisy observations.
    """
    print("\n" + "=" * 60)
    print("Example 2: Bayesian Linear Regression")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(123)
    n_data = 50
    true_slope = 2.5
    true_intercept = 1.0
    true_sigma = 0.5
    
    x = np.linspace(0, 10, n_data)
    y_true = true_slope * x + true_intercept
    y_obs = y_true + np.random.normal(0, true_sigma, n_data)
    
    print(f"True parameters: slope={true_slope}, intercept={true_intercept}, sigma={true_sigma}")
    
    # Define the model
    def linear_model(x, y=None):
        """Bayesian linear regression model."""
        # Priors
        slope = numpyro.sample("slope", dist.Normal(0, 10))
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1))
        
        # Linear model
        y_pred = slope * x + intercept
        
        # Likelihood
        numpyro.sample("y", dist.Normal(y_pred, sigma), obs=y)
    
    # Use the convenience function for quick runs
    samples = run_mcmc(
        linear_model,
        num_samples=2000,
        num_chains=4,
        x=x,
        y=y_obs
    )
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Data and fit
    axes[0].scatter(x, y_obs, alpha=0.5, label='Observations')
    axes[0].plot(x, y_true, 'r--', label='True', linewidth=2)
    
    # Plot posterior samples
    slope_samples = np.array(samples['slope'])
    intercept_samples = np.array(samples['intercept'])
    
    # Random posterior draws
    for i in np.random.choice(len(slope_samples), 100, replace=False):
        y_sample = slope_samples[i] * x + intercept_samples[i]
        axes[0].plot(x, y_sample, 'b-', alpha=0.01)
    
    # Mean prediction
    y_mean = np.mean(slope_samples) * x + np.mean(intercept_samples)
    axes[0].plot(x, y_mean, 'b-', label='Posterior mean', linewidth=2)
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].set_title('Linear Regression Fit')
    
    # Parameter distributions
    axes[1].hist(slope_samples, bins=30, alpha=0.5, label='Slope', density=True)
    axes[1].axvline(true_slope, color='red', linestyle='--', label='True slope')
    axes[1].hist(intercept_samples, bins=30, alpha=0.5, label='Intercept', density=True)
    axes[1].axvline(true_intercept, color='blue', linestyle='--', label='True intercept')
    axes[1].set_xlabel('Parameter value')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].set_title('Parameter Posteriors')
    
    plt.tight_layout()
    plt.show()
    
    return samples


def example_3_custom_kernel():
    """
    Example 3: Using different MCMC kernels.
    
    Demonstrate how to use HMC instead of NUTS.
    """
    print("\n" + "=" * 60)
    print("Example 3: Custom MCMC Kernel (HMC)")
    print("=" * 60)
    
    # Simple model
    def simple_model():
        x = numpyro.sample("x", dist.Normal(0, 1))
        y = numpyro.sample("y", dist.Normal(x, 1))
        return y
    
    # Use HMC kernel instead of default NUTS
    from numpyro.infer import HMC
    
    hmc_kernel = HMC(simple_model, step_size=0.1, num_steps=10)
    
    sampler = MCMCSampler(
        simple_model,
        kernel=hmc_kernel,  # Custom kernel
        num_warmup=500,
        num_samples=1000,
        num_chains=2,
        verbose=True
    )
    
    samples = sampler.run()
    sampler.print_summary()
    
    return samples


def example_4_save_and_load():
    """
    Example 4: Saving and loading MCMC results.
    
    Demonstrate checkpoint functionality.
    """
    print("\n" + "=" * 60)
    print("Example 4: Save and Load Results")
    print("=" * 60)
    
    # Simple model
    def model():
        return numpyro.sample("theta", dist.Beta(2, 5))
    
    # Run MCMC
    sampler = MCMCSampler(model, num_samples=1000)
    samples = sampler.run()
    
    # Save in different formats
    sampler.save_results("mcmc_results.pkl", format='pickle')
    sampler.save_results("mcmc_results.h5", format='hdf5')
    print("✓ Results saved")
    
    # Load results
    new_sampler = MCMCSampler(model)  # Create new sampler
    new_sampler.load_results("mcmc_results.pkl", format='pickle')
    print("✓ Results loaded")
    
    # Verify loaded samples
    loaded_samples = new_sampler.get_samples()
    print(f"Loaded {len(loaded_samples['theta'])} samples")
    
    # Clean up
    import os
    os.remove("mcmc_results.pkl")
    os.remove("mcmc_results.h5")
    
    return loaded_samples


def example_5_diagnostics():
    """
    Example 5: Using diagnostic tools.
    
    Demonstrate enhanced diagnostics and visualization.
    """
    print("\n" + "=" * 60)
    print("Example 5: Enhanced Diagnostics")
    print("=" * 60)
    
    # Model with potential convergence issues
    def challenging_model():
        # Highly correlated parameters
        x = numpyro.sample("x", dist.Normal(0, 1))
        y = numpyro.sample("y", dist.Normal(0.9 * x, 0.1))
        z = numpyro.sample("z", dist.Normal(x + y, 0.1))
        return z
    
    # Run with fewer warmup steps (may not converge well)
    sampler = MCMCSampler(
        challenging_model,
        num_warmup=200,  # Intentionally low
        num_samples=500,
        num_chains=4
    )
    
    samples = sampler.run()
    
    # Get comprehensive diagnostics
    diagnostics = sampler.get_diagnostics()
    
    print("\nDetailed diagnostics:")
    for param, diag in diagnostics.items():
        if param != '_overall':
            print(f"\n{param}:")
            print(f"  R-hat: {diag['r_hat']:.3f}")
            print(f"  ESS: {diag['ess']:.0f}")
            print(f"  ESS/sec: {diag['ess_per_sec']:.1f}")
    
    # Plot diagnostics
    DiagnosticsTools.plot_trace(samples, params=['x', 'y', 'z'])
    
    # Corner plot (if corner is installed)
    try:
        DiagnosticsTools.plot_corner(samples)
    except ImportError:
        print("Corner plot skipped (install corner for this feature)")
    
    return samples


def main():
    """Run all examples."""
    
    print("HiCosmo MCMC Examples")
    print("=" * 60)
    print("Demonstrating the flexible, lightweight MCMC wrapper")
    print("built on NumPyro.\n")
    
    # Run examples
    samples1 = example_1_basic_gaussian()
    samples2 = example_2_linear_regression()
    samples3 = example_3_custom_kernel()
    samples4 = example_4_save_and_load()
    samples5 = example_5_diagnostics()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    
    print("\nKey takeaways:")
    print("1. Users have complete freedom in defining models")
    print("2. No assumptions about parameter management")
    print("3. Direct access to NumPyro's powerful features")
    print("4. Enhanced diagnostics and visualization")
    print("5. Convenient save/load functionality")


if __name__ == "__main__":
    main()