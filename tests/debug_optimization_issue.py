#!/usr/bin/env python3
"""
Debug the optimization initialization issue.

Investigate why JAX optimization leads to poor MCMC convergence.
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys
import time
import numpyro

# Add HiCosmo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers import AutoMCMC

# Enable multi-core sampling
numpyro.set_host_device_count(4)


def debug_simple_case():
    """Debug the simple case to understand the issue."""
    print("🔍 DEBUGGING: Why does optimization fail?")
    print("=" * 60)
    
    # Simple test case
    np.random.seed(42)
    x = np.linspace(0, 5, 30)
    true_params = [2.5, 1.2, 0.8]  # [a, b, c]
    y_true = true_params[0] * x**2 + true_params[1] * x + true_params[2]
    y_err = 0.1 * np.ones_like(x)
    y_obs = y_true + np.random.normal(0, y_err)
    
    def likelihood(a, b, c):
        y_pred = a * x**2 + b * x + c
        chi2 = jnp.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    config = {
        'parameters': {
            'a': (1.0, 0, 5),  # Start far from true value (2.5)
            'b': (0.5, -2, 4), # Start far from true value (1.2)  
            'c': (0.1, -1, 2)  # Start far from true value (0.8)
        }
    }
    
    # Test 1: Traditional approach
    print("\n1. Traditional MCMC (with warmup):")
    config1 = config.copy()
    config1['mcmc'] = {'num_warmup': 1000, 'num_samples': 1000}
    
    mcmc1 = AutoMCMC(config1, likelihood, 
                    optimize_init=False, 
                    chain_name="debug_traditional")
    start = time.time()
    samples1 = mcmc1.run()
    time1 = time.time() - start
    diag1 = mcmc1.sampler.get_diagnostics(burnin_frac=0)
    
    print(f"  Time: {time1:.2f}s")
    print(f"  Convergence: {[d.get('r_hat', 999) < 1.1 for d in diag1.values() if isinstance(d, dict) and 'r_hat' in d]}")
    print(f"  Results: a={np.mean(samples1['a']):.3f}, b={np.mean(samples1['b']):.3f}, c={np.mean(samples1['c']):.3f}")
    
    # Test 2: Check what happens with MORE warmup
    print("\n2. Traditional MCMC (with MORE warmup):")
    config2 = config.copy()
    config2['mcmc'] = {'num_warmup': 3000, 'num_samples': 1000}  # More warmup
    
    mcmc2 = AutoMCMC(config2, likelihood, 
                    optimize_init=False, 
                    chain_name="debug_more_warmup")
    start = time.time()
    samples2 = mcmc2.run()
    time2 = time.time() - start
    diag2 = mcmc2.sampler.get_diagnostics(burnin_frac=0)
    
    print(f"  Time: {time2:.2f}s")
    print(f"  Convergence: {[d.get('r_hat', 999) < 1.1 for d in diag2.values() if isinstance(d, dict) and 'r_hat' in d]}")
    print(f"  Results: a={np.mean(samples2['a']):.3f}, b={np.mean(samples2['b']):.3f}, c={np.mean(samples2['c']):.3f}")
    
    # Test 3: Optimized but with warmup too
    print("\n3. Optimized MCMC (with some warmup):")
    config3 = config.copy()
    config3['mcmc'] = {'num_warmup': 500, 'num_samples': 1000}  # Some warmup even after optimization
    
    mcmc3 = AutoMCMC(config3, likelihood, 
                    optimize_init=True,
                    max_opt_iterations=200,  # Shorter optimization
                    chain_name="debug_opt_warmup")
    start = time.time()
    samples3 = mcmc3.run()
    time3 = time.time() - start
    diag3 = mcmc3.sampler.get_diagnostics(burnin_frac=0)
    
    print(f"  Time: {time3:.2f}s")
    print(f"  Convergence: {[d.get('r_hat', 999) < 1.1 for d in diag3.values() if isinstance(d, dict) and 'r_hat' in d]}")
    print(f"  Results: a={np.mean(samples3['a']):.3f}, b={np.mean(samples3['b']):.3f}, c={np.mean(samples3['c']):.3f}")
    
    # Test 4: What if we increase number of samples?
    print("\n4. Optimized MCMC (no warmup, more samples):")
    config4 = config.copy()
    config4['mcmc'] = {'num_warmup': 0, 'num_samples': 3000}  # More samples instead of warmup
    
    mcmc4 = AutoMCMC(config4, likelihood, 
                    optimize_init=True,
                    max_opt_iterations=200,
                    chain_name="debug_opt_more_samples")
    start = time.time()
    samples4 = mcmc4.run()
    time4 = time.time() - start
    diag4 = mcmc4.sampler.get_diagnostics(burnin_frac=0.2)  # Remove more burn-in
    
    print(f"  Time: {time4:.2f}s")
    print(f"  Convergence: {[d.get('r_hat', 999) < 1.1 for d in diag4.values() if isinstance(d, dict) and 'r_hat' in d]}")
    print(f"  Results: a={np.mean(samples4['a']):.3f}, b={np.mean(samples4['b']):.3f}, c={np.mean(samples4['c']):.3f}")
    
    # Test 5: Check if optimization actually finds good values
    print("\n5. Check optimization quality:")
    
    # Test the likelihood at true values
    true_likelihood = likelihood(true_params[0], true_params[1], true_params[2])
    print(f"  True likelihood: {true_likelihood:.2f}")
    
    # Test likelihood at starting values  
    start_likelihood = likelihood(1.0, 0.5, 0.1)
    print(f"  Start likelihood: {start_likelihood:.2f}")
    
    # Get optimized values from mcmc3 or mcmc4
    opt_param_a = mcmc3.param_config.parameters['a'].ref
    opt_param_b = mcmc3.param_config.parameters['b'].ref  
    opt_param_c = mcmc3.param_config.parameters['c'].ref
    opt_likelihood = likelihood(opt_param_a, opt_param_b, opt_param_c)
    print(f"  Optimized likelihood: {opt_likelihood:.2f} (a={opt_param_a:.3f}, b={opt_param_b:.3f}, c={opt_param_c:.3f})")
    
    print(f"\n📋 Summary:")
    print(f"  Traditional (1000 warmup): {time1:.1f}s, converged: {all(d.get('r_hat', 999) < 1.1 for d in diag1.values() if isinstance(d, dict) and 'r_hat' in d)}")
    print(f"  Traditional (3000 warmup): {time2:.1f}s, converged: {all(d.get('r_hat', 999) < 1.1 for d in diag2.values() if isinstance(d, dict) and 'r_hat' in d)}")
    print(f"  Optimized + 500 warmup:    {time3:.1f}s, converged: {all(d.get('r_hat', 999) < 1.1 for d in diag3.values() if isinstance(d, dict) and 'r_hat' in d)}")
    print(f"  Optimized + 3000 samples:  {time4:.1f}s, converged: {all(d.get('r_hat', 999) < 1.1 for d in diag4.values() if isinstance(d, dict) and 'r_hat' in d)}")


if __name__ == "__main__":
    debug_simple_case()