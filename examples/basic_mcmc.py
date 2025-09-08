"""
Basic MCMC analysis example using HiCosmo.

This example demonstrates:
1. Setting up a ΛCDM cosmological model
2. Creating mock supernova data
3. Configuring parameters and priors
4. Running MCMC sampling with NumPyro
5. Analyzing and visualizing results
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
import corner
import numpy as np

# Import HiCosmo components
from hicosmo.models.lcdm import LCDM
from hicosmo.parameters.manager import ParameterManager
from hicosmo.likelihoods.base import GaussianLikelihood


class MockSNeLikelihood(GaussianLikelihood):
    """
    Mock Type Ia supernova likelihood for demonstration.
    """
    
    def __init__(self, n_sne=40, z_max=1.5, **kwargs):
        super().__init__(**kwargs)
        self.n_sne = n_sne
        self.z_max = z_max
        
    def _load_data(self):
        """Generate mock SNe data."""
        # Generate redshifts
        self.z_data = jnp.linspace(0.01, self.z_max, self.n_sne)
        
        # True parameters for mock data
        true_params = {'H0': 70.0, 'Omega_m': 0.3}
        
        # Generate mock distance moduli
        true_dL = LCDM.luminosity_distance(self.z_data, true_params, LCDM)
        true_mu = 5 * jnp.log10(true_dL * 1e6 / 10)  # Convert to pc
        
        # Add noise
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=(self.n_sne,)) * 0.15
        self.data_vec = true_mu + noise
        
        # Create diagonal covariance matrix
        self.cov = jnp.eye(self.n_sne) * 0.15**2
        
    def theory(self, **params):
        """Compute theoretical distance moduli."""
        # Extract cosmological parameters
        cosmo_params = {
            'H0': params['H0'],
            'Omega_m': params['Omega_m']
        }
        
        # Calculate luminosity distances
        dL = LCDM.luminosity_distance(self.z_data, cosmo_params, LCDM)
        
        # Convert to distance modulus
        mu_theory = 5 * jnp.log10(dL * 1e6 / 10)
        
        # Add absolute magnitude nuisance parameter if present
        if 'M' in params:
            mu_theory = mu_theory + params['M']
        
        return mu_theory
    
    def get_requirements(self):
        """No external requirements for this simple example."""
        return {}


def build_numpyro_model(likelihood, param_manager):
    """
    Build NumPyro model for MCMC sampling.
    
    Args:
        likelihood: Likelihood object
        param_manager: Parameter manager
        
    Returns:
        NumPyro model function
    """
    def model():
        # Sample parameters according to their priors
        params = {}
        
        for param_name, param_info in param_manager.get_free_params().items():
            prior_dist = param_manager.get_numpyro_prior(param_name)
            params[param_name] = numpyro.sample(param_name, prior_dist)
        
        # Add fixed parameters
        params.update(param_manager.get_fixed_params())
        
        # Compute log-likelihood
        logp = likelihood.logp(**params)
        
        # Observe the likelihood
        numpyro.factor("logp", logp)
    
    return model


def run_analysis():
    """Run the complete analysis pipeline."""
    
    print("HiCosmo Basic MCMC Example")
    print("=" * 50)
    
    # Step 1: Create mock SNe likelihood
    print("\n1. Creating mock supernova data...")
    likelihood = MockSNeLikelihood(n_sne=40, z_max=1.5)
    likelihood.initialize()
    print(f"   Generated {likelihood.n_sne} mock SNe up to z={likelihood.z_max}")
    
    # Step 2: Setup parameter manager
    print("\n2. Configuring parameters...")
    param_manager = ParameterManager()
    
    # Add cosmological parameters
    param_manager.add_param(
        'H0',
        prior={'min': 60, 'max': 80, 'dist': 'uniform'},
        ref=70.0,
        latex=r'$H_0$'
    )
    
    param_manager.add_param(
        'Omega_m',
        prior={'min': 0.2, 'max': 0.4, 'dist': 'uniform'},
        ref=0.3,
        latex=r'$\Omega_m$'
    )
    
    # Add nuisance parameter for absolute magnitude
    param_manager.add_param(
        'M',
        prior={'min': -20, 'max': -18, 'dist': 'uniform'},
        ref=-19.3,
        latex=r'$M$'
    )
    
    print(param_manager.get_summary_table())
    
    # Step 3: Build NumPyro model
    print("\n3. Building NumPyro model...")
    model = build_numpyro_model(likelihood, param_manager)
    
    # Step 4: Run MCMC
    print("\n4. Running MCMC sampling...")
    print("   Chains: 4")
    print("   Warmup: 1000")
    print("   Samples: 2000 per chain")
    
    # Initialize NUTS sampler
    nuts_kernel = NUTS(model, target_accept_prob=0.8)
    
    # Run MCMC
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=1000,
        num_samples=2000,
        num_chains=4,
        progress_bar=True,
        jit_model_args=True
    )
    
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key)
    
    # Step 5: Get samples and print summary
    print("\n5. Analyzing results...")
    samples = mcmc.get_samples()
    
    # Print summary statistics
    print("\nParameter Summary:")
    print("-" * 50)
    for param_name in param_manager.get_free_params().keys():
        param_samples = samples[param_name]
        mean = float(jnp.mean(param_samples))
        std = float(jnp.std(param_samples))
        median = float(jnp.median(param_samples))
        print(f"{param_name:10s}: {mean:.3f} ± {std:.3f} (median: {median:.3f})")
    
    # Check convergence
    from numpyro.diagnostics import gelman_rubin
    print("\nConvergence Diagnostics (R-hat):")
    print("-" * 50)
    for param_name in param_manager.get_free_params().keys():
        param_samples = samples[param_name].reshape(4, -1)  # Reshape to (chains, samples)
        r_hat = float(gelman_rubin(param_samples))
        status = "✓" if r_hat < 1.01 else "✗"
        print(f"{param_name:10s}: {r_hat:.4f} {status}")
    
    # Step 6: Create corner plot
    print("\n6. Creating corner plot...")
    
    # Prepare samples for corner plot
    flat_samples = np.column_stack([
        samples[param_name].flatten() 
        for param_name in ['H0', 'Omega_m', 'M']
    ])
    
    labels = [r'$H_0$', r'$\Omega_m$', r'$M$']
    truths = [70.0, 0.3, 0.0]  # True values used to generate mock data
    
    fig = corner.corner(
        flat_samples,
        labels=labels,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    
    plt.suptitle("HiCosmo MCMC Results", fontsize=16, y=1.02)
    plt.savefig("mcmc_corner.png", dpi=150, bbox_inches='tight')
    print("   Corner plot saved as 'mcmc_corner.png'")
    
    # Step 7: Plot chains
    print("\n7. Creating chain plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    for i, param_name in enumerate(['H0', 'Omega_m', 'M']):
        param_samples = samples[param_name].reshape(4, -1)
        
        for chain in range(4):
            axes[i].plot(param_samples[chain], alpha=0.7, linewidth=0.5)
        
        axes[i].set_ylabel(labels[i])
        axes[i].axhline(truths[i], color='red', linestyle='--', alpha=0.5)
        
        if i == 2:
            axes[i].set_xlabel('Sample')
    
    plt.suptitle("MCMC Chains", fontsize=14)
    plt.tight_layout()
    plt.savefig("mcmc_chains.png", dpi=150, bbox_inches='tight')
    print("   Chain plot saved as 'mcmc_chains.png'")
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    
    return samples, param_manager


if __name__ == "__main__":
    # Set NumPyro settings
    numpyro.set_platform("cpu")  # Use "gpu" if available
    numpyro.set_host_device_count(4)  # Number of CPU cores to use
    
    # Run the analysis
    samples, param_manager = run_analysis()
    
    # Additional analysis can be added here
    # For example, computing derived parameters, model comparison, etc.