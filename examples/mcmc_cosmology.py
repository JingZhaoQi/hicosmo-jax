#!/usr/bin/env python3
"""
Cosmological MCMC Example using HiCosmo.

This example demonstrates how to use the MCMC sampler for cosmological
parameter inference with Type Ia supernovae data. It shows the flexibility
of combining HiCosmo's fast cosmology calculations with NumPyro's
powerful MCMC capabilities.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

# Add HiCosmo to path if needed
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers import MCMCSampler, DiagnosticsTools
from hicosmo.models import LCDM
from hicosmo.core import CosmologicalParameters


def generate_mock_sn_data(n_sn: int = 40, seed: int = 42) -> Dict:
    """
    Generate mock Type Ia supernova data.
    
    Parameters
    ----------
    n_sn : int
        Number of supernovae
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    Dict
        Dictionary with 'z' (redshifts), 'mu_obs' (distance moduli),
        and 'mu_err' (uncertainties)
    """
    np.random.seed(seed)
    
    # Redshift distribution
    z = np.linspace(0.01, 1.5, n_sn)
    
    # True cosmological parameters
    true_params = CosmologicalParameters(H0=70.0, Omega_m=0.3)
    true_model = LCDM(**true_params.to_dict())
    
    # Calculate true distance moduli
    d_L = true_model.luminosity_distance(z)
    mu_true = 5 * jnp.log10(d_L) + 25  # Distance modulus
    
    # Add observational uncertainties
    mu_err = 0.15 * np.ones(n_sn)  # Typical SN uncertainty
    mu_obs = mu_true + np.random.normal(0, mu_err)
    
    return {
        'z': jnp.array(z),
        'mu_obs': jnp.array(mu_obs),
        'mu_err': jnp.array(mu_err),
        'true_H0': 70.0,
        'true_Om': 0.3
    }


def cosmology_model_basic(sn_data: Dict):
    """
    Basic cosmological model for MCMC.
    
    This demonstrates a simple ΛCDM model with H0 and Omega_m as
    free parameters, using HiCosmo's fast distance calculations.
    """
    # Extract data
    z = sn_data['z']
    mu_obs = sn_data['mu_obs']
    mu_err = sn_data['mu_err']
    
    # Define priors (user's choice)
    H0 = numpyro.sample("H0", dist.Uniform(50, 90))
    Omega_m = numpyro.sample("Omega_m", dist.Uniform(0.1, 0.5))
    
    # Additional nuisance parameter for intrinsic scatter
    sigma_int = numpyro.sample("sigma_int", dist.HalfNormal(0.1))
    
    # Create cosmological model using sampled parameters
    # Note: In actual MCMC, we need to be careful about JAX tracing
    params = CosmologicalParameters(H0=H0, Omega_m=Omega_m)
    model = LCDM(**params.to_dict())
    
    # Calculate theoretical distance moduli
    d_L = model.luminosity_distance(z)
    mu_theory = 5 * jnp.log10(d_L) + 25
    
    # Total uncertainty
    sigma_total = jnp.sqrt(mu_err**2 + sigma_int**2)
    
    # Likelihood
    with numpyro.plate("sn_data", len(z)):
        numpyro.sample("mu", dist.Normal(mu_theory, sigma_total), obs=mu_obs)


def cosmology_model_advanced(sn_data: Dict, include_Ob: bool = False):
    """
    Advanced cosmological model with optional parameters.
    
    This demonstrates how users can build more complex models
    with additional parameters and hierarchical structure.
    """
    z = sn_data['z']
    mu_obs = sn_data['mu_obs']
    mu_err = sn_data['mu_err']
    
    # Cosmological parameters with informative priors
    H0 = numpyro.sample("H0", dist.Normal(70, 5))  # Prior from local measurements
    Omega_m = numpyro.sample("Omega_m", dist.Normal(0.3, 0.05))  # Prior from CMB
    
    if include_Ob:
        # Include baryon density with physical prior
        Omega_b = numpyro.sample("Omega_b", dist.Normal(0.048, 0.005))
    else:
        Omega_b = 0.048  # Fixed value
    
    # Systematic uncertainties
    M_abs = numpyro.sample("M_abs", dist.Normal(-19.3, 0.1))  # Absolute magnitude
    alpha = numpyro.sample("alpha", dist.Normal(0.14, 0.02))  # Stretch correction
    beta = numpyro.sample("beta", dist.Normal(3.1, 0.2))  # Color correction
    
    # Intrinsic scatter
    sigma_int = numpyro.sample("sigma_int", dist.HalfNormal(0.1))
    
    # Calculate distances
    params = CosmologicalParameters(H0=H0, Omega_m=Omega_m, Omega_b=Omega_b)
    model = LCDM(**params.to_dict())
    d_L = model.luminosity_distance(z)
    
    # Distance modulus with systematics
    mu_theory = 5 * jnp.log10(d_L) + 25 + M_abs
    # In real analysis, would also apply stretch and color corrections
    
    # Total uncertainty
    sigma_total = jnp.sqrt(mu_err**2 + sigma_int**2)
    
    # Likelihood
    with numpyro.plate("sn_data", len(z)):
        numpyro.sample("mu", dist.Normal(mu_theory, sigma_total), obs=mu_obs)


def run_cosmology_mcmc():
    """
    Run cosmological parameter inference with MCMC.
    """
    print("=" * 70)
    print("Cosmological Parameter Inference with Type Ia Supernovae")
    print("=" * 70)
    
    # Generate mock data
    print("\nGenerating mock supernova data...")
    sn_data = generate_mock_sn_data(n_sn=40)
    print(f"  Number of SNe: {len(sn_data['z'])}")
    print(f"  Redshift range: {sn_data['z'][0]:.2f} - {sn_data['z'][-1]:.2f}")
    print(f"  True parameters: H0={sn_data['true_H0']}, Ωm={sn_data['true_Om']}")
    
    # Setup MCMC sampler
    print("\nSetting up MCMC sampler...")
    sampler = MCMCSampler(
        cosmology_model_basic,
        num_warmup=2000,
        num_samples=4000,
        num_chains=4,
        verbose=True
    )
    
    # Run MCMC
    print("\nRunning MCMC...")
    samples = sampler.run(sn_data=sn_data)
    
    # Print results
    print("\n" + "=" * 70)
    print("MCMC Results")
    print("=" * 70)
    sampler.print_summary(prob=0.68)  # 68% credible intervals
    
    # Check convergence
    diagnostics = sampler.get_diagnostics()
    all_converged = all(
        d['r_hat'] < 1.01 
        for param, d in diagnostics.items() 
        if param != '_overall'
    )
    
    if all_converged:
        print("\n✓ All parameters converged successfully!")
    else:
        print("\n⚠ Some parameters may not have converged fully")
    
    # Extract samples
    H0_samples = np.array(samples['H0'])
    Om_samples = np.array(samples['Omega_m'])
    
    # Calculate statistics
    H0_mean = np.mean(H0_samples)
    H0_std = np.std(H0_samples)
    Om_mean = np.mean(Om_samples)
    Om_std = np.std(Om_samples)
    
    print(f"\nParameter constraints (mean ± std):")
    print(f"  H0 = {H0_mean:.2f} ± {H0_std:.2f} km/s/Mpc")
    print(f"  Ωm = {Om_mean:.3f} ± {Om_std:.3f}")
    
    print(f"\nTrue values:")
    print(f"  H0 = {sn_data['true_H0']} km/s/Mpc")
    print(f"  Ωm = {sn_data['true_Om']}")
    
    return samples, sn_data


def plot_cosmology_results(samples: Dict, sn_data: Dict):
    """
    Create visualization of cosmological MCMC results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract samples
    H0_samples = np.array(samples['H0'])
    Om_samples = np.array(samples['Omega_m'])
    
    # 1. H0 posterior
    axes[0, 0].hist(H0_samples, bins=50, density=True, alpha=0.7, color='blue')
    axes[0, 0].axvline(sn_data['true_H0'], color='red', linestyle='--', 
                       label=f'True: {sn_data["true_H0"]}', linewidth=2)
    axes[0, 0].axvline(np.mean(H0_samples), color='black', linestyle='-',
                       label=f'Mean: {np.mean(H0_samples):.1f}', linewidth=2)
    axes[0, 0].set_xlabel('H₀ [km/s/Mpc]')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title('Hubble Constant Posterior')
    axes[0, 0].legend()
    
    # 2. Omega_m posterior
    axes[0, 1].hist(Om_samples, bins=50, density=True, alpha=0.7, color='green')
    axes[0, 1].axvline(sn_data['true_Om'], color='red', linestyle='--',
                       label=f'True: {sn_data["true_Om"]}', linewidth=2)
    axes[0, 1].axvline(np.mean(Om_samples), color='black', linestyle='-',
                       label=f'Mean: {np.mean(Om_samples):.3f}', linewidth=2)
    axes[0, 1].set_xlabel('Ωₘ')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].set_title('Matter Density Posterior')
    axes[0, 1].legend()
    
    # 3. Joint posterior (2D)
    axes[1, 0].hexbin(H0_samples, Om_samples, gridsize=30, cmap='Blues')
    axes[1, 0].scatter(sn_data['true_H0'], sn_data['true_Om'], 
                      color='red', s=100, marker='*', label='True value')
    axes[1, 0].set_xlabel('H₀ [km/s/Mpc]')
    axes[1, 0].set_ylabel('Ωₘ')
    axes[1, 0].set_title('Joint Posterior Distribution')
    axes[1, 0].legend()
    
    # 4. Hubble diagram
    z = np.array(sn_data['z'])
    mu_obs = np.array(sn_data['mu_obs'])
    mu_err = np.array(sn_data['mu_err'])
    
    # Plot data
    axes[1, 1].errorbar(z, mu_obs, yerr=mu_err, fmt='o', alpha=0.5, 
                        label='Mock SNe data')
    
    # Plot posterior predictions
    z_theory = np.linspace(0.01, 1.5, 100)
    
    # Random posterior samples
    n_samples_plot = min(100, len(H0_samples))
    indices = np.random.choice(len(H0_samples), n_samples_plot, replace=False)
    
    for i in indices:
        params = CosmologicalParameters(H0=H0_samples[i], Omega_m=Om_samples[i])
        model = LCDM(**params.to_dict())
        d_L = model.luminosity_distance(z_theory)
        mu_theory = 5 * np.log10(d_L) + 25
        axes[1, 1].plot(z_theory, mu_theory, 'b-', alpha=0.02)
    
    # Mean prediction
    params_mean = CosmologicalParameters(
        H0=np.mean(H0_samples), 
        Omega_m=np.mean(Om_samples)
    )
    model_mean = LCDM(**params_mean.to_dict())
    d_L_mean = model_mean.luminosity_distance(z_theory)
    mu_mean = 5 * np.log10(d_L_mean) + 25
    axes[1, 1].plot(z_theory, mu_mean, 'b-', linewidth=2, 
                   label='Posterior mean')
    
    # True model
    params_true = CosmologicalParameters(
        H0=sn_data['true_H0'], 
        Omega_m=sn_data['true_Om']
    )
    model_true = LCDM(**params_true.to_dict())
    d_L_true = model_true.luminosity_distance(z_theory)
    mu_true = 5 * np.log10(d_L_true) + 25
    axes[1, 1].plot(z_theory, mu_true, 'r--', linewidth=2, 
                   label='True model')
    
    axes[1, 1].set_xlabel('Redshift z')
    axes[1, 1].set_ylabel('Distance Modulus μ')
    axes[1, 1].set_title('Hubble Diagram')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Cosmological Parameter Inference Results', fontsize=14)
    plt.tight_layout()
    plt.show()


def example_advanced_model():
    """
    Example with the advanced cosmological model.
    """
    print("\n" + "=" * 70)
    print("Advanced Cosmological Model with Systematics")
    print("=" * 70)
    
    # Generate data
    sn_data = generate_mock_sn_data(n_sn=30)
    
    # Run with advanced model
    sampler = MCMCSampler(
        lambda data: cosmology_model_advanced(data, include_Ob=True),
        num_warmup=1500,
        num_samples=3000,
        num_chains=4,
        verbose=True
    )
    
    samples = sampler.run(sn_data)
    sampler.print_summary(prob=0.95)  # 95% credible intervals
    
    # Save results for further analysis
    sampler.save_results("cosmology_mcmc_results.pkl")
    print("\nResults saved to cosmology_mcmc_results.pkl")
    
    # Convert to ArviZ for advanced diagnostics
    inference_data = sampler.to_arviz()
    print(f"\nArviZ InferenceData created with {len(inference_data.posterior.data_vars)} parameters")
    
    # Clean up
    import os
    if os.path.exists("cosmology_mcmc_results.pkl"):
        os.remove("cosmology_mcmc_results.pkl")
    
    return samples


def main():
    """Run cosmological MCMC examples."""
    
    print("HiCosmo Cosmological MCMC Example")
    print("=" * 70)
    print("Demonstrating cosmological parameter inference using")
    print("HiCosmo's fast calculations with NumPyro's MCMC.\n")
    
    # Run basic cosmology MCMC
    samples, sn_data = run_cosmology_mcmc()
    
    # Visualize results
    plot_cosmology_results(samples, sn_data)
    
    # Run advanced example
    advanced_samples = example_advanced_model()
    
    print("\n" + "=" * 70)
    print("Cosmological MCMC examples completed!")
    print("=" * 70)
    
    print("\nKey features demonstrated:")
    print("1. Flexible model definition with NumPyro")
    print("2. Integration with HiCosmo's fast cosmology calculations")
    print("3. Comprehensive diagnostics and convergence checks")
    print("4. Beautiful visualization of results")
    print("5. Support for complex hierarchical models")


if __name__ == "__main__":
    main()