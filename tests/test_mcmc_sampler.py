#!/usr/bin/env python3
"""
Test suite for HiCosmo MCMC sampler.

Tests the lightweight NumPyro wrapper functionality.
"""

import pytest
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np
from pathlib import Path
import tempfile
import os

# Add HiCosmo to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers import MCMCSampler, DiagnosticsTools, run_mcmc


class TestMCMCSampler:
    """Test MCMCSampler class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Simple test model
        def test_model():
            return numpyro.sample("x", dist.Normal(0, 1))
        
        self.model = test_model
        self.sampler = MCMCSampler(
            self.model,
            num_warmup=100,
            num_samples=200,
            num_chains=2,
            verbose=False
        )
    
    def test_initialization(self):
        """Test sampler initialization."""
        assert self.sampler.num_warmup == 100
        assert self.sampler.num_samples == 200
        assert self.sampler.num_chains == 2
        assert self.sampler.model_fn == self.model
        assert self.sampler.mcmc is not None
    
    def test_run_basic(self):
        """Test basic MCMC run."""
        samples = self.sampler.run()
        
        assert samples is not None
        assert 'x' in samples
        assert len(samples['x']) == self.sampler.num_chains * self.sampler.num_samples
        
        # Check samples are reasonable
        x_samples = np.array(samples['x'])
        assert np.abs(np.mean(x_samples)) < 0.5  # Should be close to 0
        assert 0.5 < np.std(x_samples) < 1.5  # Should be close to 1
    
    def test_run_with_data(self):
        """Test MCMC with data."""
        def model_with_data(data):
            mu = numpyro.sample("mu", dist.Normal(0, 10))
            numpyro.sample("obs", dist.Normal(mu, 1), obs=data)
        
        data = jnp.array([1.0, 2.0, 1.5, 1.8])
        sampler = MCMCSampler(model_with_data, num_warmup=200, num_samples=400, verbose=False)
        samples = sampler.run(data=data)
        
        assert 'mu' in samples
        mu_mean = np.mean(np.array(samples['mu']))
        assert 1.0 < mu_mean < 2.0  # Should be close to data mean
    
    def test_custom_kernel(self):
        """Test with custom kernel."""
        from numpyro.infer import HMC
        
        kernel = HMC(self.model, step_size=0.1, num_steps=5)
        sampler = MCMCSampler(
            self.model,
            kernel=kernel,
            num_warmup=100,
            num_samples=200,
            verbose=False
        )
        
        samples = sampler.run()
        assert 'x' in samples
    
    def test_get_samples(self):
        """Test getting samples with different options."""
        self.sampler.run()
        
        # Flattened samples
        samples_flat = self.sampler.get_samples(group_by_chain=False)
        assert samples_flat['x'].shape == (400,)  # 2 chains * 200 samples
        
        # Grouped by chain
        samples_grouped = self.sampler.mcmc.get_samples(group_by_chain=True)
        assert samples_grouped['x'].shape == (2, 200)  # 2 chains, 200 samples each
    
    def test_diagnostics(self):
        """Test diagnostic computation."""
        self.sampler.run()
        diagnostics = self.sampler.get_diagnostics()
        
        assert 'x' in diagnostics
        assert '_overall' in diagnostics
        
        x_diag = diagnostics['x']
        assert 'r_hat' in x_diag
        assert 'ess' in x_diag
        assert 'ess_per_sec' in x_diag
        assert 'mean' in x_diag
        assert 'std' in x_diag
        
        # R-hat should be close to 1 for converged chains
        assert 0.99 < x_diag['r_hat'] < 1.1
        
        # ESS should be reasonable
        assert x_diag['ess'] > 50
    
    def test_save_load_pickle(self):
        """Test saving and loading results in pickle format."""
        self.sampler.run()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name
        
        try:
            # Save
            self.sampler.save_results(filepath, format='pickle')
            assert Path(filepath).exists()
            
            # Load in new sampler
            new_sampler = MCMCSampler(self.model, verbose=False)
            new_sampler.load_results(filepath, format='pickle')
            
            # Check loaded samples
            original_samples = self.sampler.get_samples()
            loaded_samples = new_sampler.get_samples()
            
            assert 'x' in loaded_samples
            np.testing.assert_array_almost_equal(
                original_samples['x'], loaded_samples['x']
            )
        finally:
            if Path(filepath).exists():
                os.remove(filepath)
    
    def test_save_load_hdf5(self):
        """Test saving and loading results in HDF5 format."""
        self.sampler.run()
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name
        
        try:
            # Save
            self.sampler.save_results(filepath, format='hdf5')
            assert Path(filepath).exists()
            
            # Load
            new_sampler = MCMCSampler(self.model, verbose=False)
            new_sampler.load_results(filepath, format='hdf5')
            
            # Check
            loaded_samples = new_sampler.get_samples()
            assert 'x' in loaded_samples
            assert len(loaded_samples['x']) == 400
        finally:
            if Path(filepath).exists():
                os.remove(filepath)
    
    def test_to_arviz(self):
        """Test ArviZ conversion."""
        self.sampler.run()
        inference_data = self.sampler.to_arviz()
        
        assert inference_data is not None
        assert 'posterior' in inference_data.groups()
        assert 'x' in inference_data.posterior.data_vars
    
    def test_convergence_check(self):
        """Test convergence checking with poorly converged chains."""
        # Model that's hard to sample (highly correlated)
        def difficult_model():
            x = numpyro.sample("x", dist.Normal(0, 1))
            y = numpyro.sample("y", dist.Normal(0.99 * x, 0.01))
            return y
        
        # Run with very few warmup steps
        sampler = MCMCSampler(
            difficult_model,
            num_warmup=10,  # Too few!
            num_samples=50,
            num_chains=2,
            verbose=False
        )
        sampler.run()
        
        diagnostics = sampler.get_diagnostics()
        # With so few warmup steps, R-hat might be poor
        # Just check that diagnostics are computed
        assert 'x' in diagnostics
        assert 'y' in diagnostics


class TestDiagnosticsTools:
    """Test DiagnosticsTools class."""
    
    def test_autocorrelation(self):
        """Test autocorrelation computation."""
        # Generate AR(1) process
        n = 1000
        rho = 0.8
        x = np.zeros(n)
        x[0] = np.random.normal()
        for i in range(1, n):
            x[i] = rho * x[i-1] + np.random.normal(0, np.sqrt(1 - rho**2))
        
        x_jax = jnp.array(x)
        acf = DiagnosticsTools.autocorrelation(x_jax, max_lag=10)
        
        assert len(acf) == 10
        assert acf[0] == 1.0  # ACF at lag 0 is always 1
        assert acf[1] < acf[0]  # Should decay
        
        # For AR(1), theoretical ACF is rho^k
        assert 0.7 < acf[1] < 0.9  # Should be close to rho
    
    def test_integrated_autocorrelation_time(self):
        """Test integrated autocorrelation time."""
        # Independent samples
        x_independent = jnp.array(np.random.normal(0, 1, 1000))
        tau_independent = DiagnosticsTools.integrated_autocorrelation_time(x_independent)
        assert tau_independent < 5  # Should be small for independent samples
        
        # Correlated samples
        n = 1000
        x_correlated = np.zeros(n)
        x_correlated[0] = np.random.normal()
        for i in range(1, n):
            x_correlated[i] = 0.95 * x_correlated[i-1] + np.random.normal(0, 0.1)
        
        tau_correlated = DiagnosticsTools.integrated_autocorrelation_time(jnp.array(x_correlated))
        assert tau_correlated > tau_independent  # Should be larger for correlated samples


class TestConvenienceFunction:
    """Test the run_mcmc convenience function."""
    
    def test_run_mcmc_basic(self):
        """Test basic usage of run_mcmc."""
        def model():
            return numpyro.sample("theta", dist.Beta(2, 5))
        
        samples = run_mcmc(
            model,
            num_samples=500,
            num_chains=2,
            show_summary=False
        )
        
        assert 'theta' in samples
        theta_samples = np.array(samples['theta'])
        
        # Beta(2, 5) has mean = 2/(2+5) = 0.286
        assert 0.2 < np.mean(theta_samples) < 0.4
    
    def test_run_mcmc_with_data(self):
        """Test run_mcmc with data."""
        def regression_model(x, y=None):
            slope = numpyro.sample("slope", dist.Normal(0, 10))
            intercept = numpyro.sample("intercept", dist.Normal(0, 10))
            sigma = numpyro.sample("sigma", dist.HalfNormal(1))
            
            y_pred = slope * x + intercept
            numpyro.sample("y", dist.Normal(y_pred, sigma), obs=y)
        
        # Generate data
        x = jnp.linspace(0, 1, 20)
        y = 2 * x + 1 + np.random.normal(0, 0.1, 20)
        
        samples = run_mcmc(
            regression_model,
            num_samples=1000,
            x=x,
            y=y,
            show_summary=False
        )
        
        assert 'slope' in samples
        assert 'intercept' in samples
        
        # Check recovered parameters are reasonable
        slope_mean = np.mean(np.array(samples['slope']))
        intercept_mean = np.mean(np.array(samples['intercept']))
        
        assert 1.5 < slope_mean < 2.5  # True slope is 2
        assert 0.8 < intercept_mean < 1.2  # True intercept is 1


def test_imports():
    """Test that all imports work correctly."""
    from hicosmo.samplers import MCMCSampler, DiagnosticsTools, run_mcmc
    assert MCMCSampler is not None
    assert DiagnosticsTools is not None
    assert run_mcmc is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])