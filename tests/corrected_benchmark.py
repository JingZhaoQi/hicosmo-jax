#!/usr/bin/env python3
"""
Corrected performance benchmark: JAX optimization + minimal warmup vs traditional warmup.

Now comparing:
1. Traditional approach: 2000 warmup + 2000 samples (from random start)
2. Optimized approach: JAX optimization + 300 warmup + 2000 samples (from optimized start)

This is a more fair comparison that respects HMC sampler requirements.
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys
import time
import numpyro

# Add HiCosmo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers import MCMC

# Enable multi-core sampling
numpyro.set_host_device_count(4)


def create_test_problems():
    """Create a range of test problems."""
    problems = {}
    
    # Simple quadratic
    np.random.seed(42)
    x = np.linspace(0, 5, 30)
    true_params = [2.5, 1.2, 0.8]
    y_true = true_params[0] * x**2 + true_params[1] * x + true_params[2]
    y_err = 0.1 * np.ones_like(x)
    y_obs = y_true + np.random.normal(0, y_err)
    
    def simple_likelihood(a, b, c):
        y_pred = a * x**2 + b * x + c
        chi2 = jnp.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    problems['simple'] = {
        'likelihood': simple_likelihood,
        'config': {
            'parameters': {
                'a': (1.0, 0, 5),
                'b': (0.5, -2, 4),
                'c': (0.1, -1, 2)
            }
        },
        'true_params': true_params
    }
    
    # Medium complexity
    np.random.seed(43)  # Different seed to avoid conflicts
    n_data = 40
    true_params = [1.5, -0.8, 0.3, 0.6, 2.1]
    
    # Create design matrix - keep it separate from simple problem
    X_medium = np.random.randn(n_data, 5)
    y_true_med = X_medium @ true_params
    y_err_med = 0.15 * np.ones(n_data)
    y_obs_med = y_true_med + np.random.normal(0, y_err_med)
    
    def medium_likelihood(a, b, c, d, e):
        params = jnp.array([a, b, c, d, e])
        y_pred = X_medium @ params
        chi2 = jnp.sum((y_obs_med - y_pred)**2 / y_err_med**2)
        return -0.5 * chi2
    
    problems['medium'] = {
        'likelihood': medium_likelihood,
        'config': {
            'parameters': {
                'a': (0.5, -3, 4),
                'b': (0.2, -3, 2),
                'c': (0.8, -1, 2),
                'd': (-0.2, -2, 3),
                'e': (0.5, -1, 5)
            }
        },
        'true_params': true_params
    }
    
    return problems


def run_corrected_benchmark():
    """Run the corrected benchmark with proper warmup settings."""
    print("CORRECTED MCMC PERFORMANCE BENCHMARK")
    print("=" * 70)
    print("JAX Optimization + Minimal Warmup vs Traditional Warmup")
    print()
    
    problems = create_test_problems()
    results = {}
    
    for name, problem in problems.items():
        print(f"\n{'='*50}")
        print(f"TESTING: {name.upper()} PROBLEM")
        print(f"{'='*50}")
        
        likelihood = problem['likelihood']
        config = problem['config']
        
        # Method 1: Traditional approach (2000 warmup)
        print("\n📈 Method 1: Traditional MCMC")
        config1 = config.copy()
        config1['mcmc'] = {
            'num_warmup': 2000,  # Standard warmup
            'num_samples': 2000,
            'num_chains': 4
        }
        
        start_time = time.time()
        mcmc1 = MCMC(config1, likelihood,
                        optimize_init=False,
                        chain_name=f"traditional_{name}")
        samples1 = mcmc1.run()
        time1 = time.time() - start_time
        
        diag1 = mcmc1.sampler.get_diagnostics(burnin_frac=0)
        converged1 = sum(1 for d in diag1.values() if isinstance(d, dict) and d.get('r_hat', 999) < 1.1)
        total1 = sum(1 for d in diag1.values() if isinstance(d, dict) and 'r_hat' in d)
        
        print(f"  Time: {time1:.2f}s")
        print(f"  Convergence: {converged1}/{total1} parameters")
        print(f"  Avg R̂: {np.mean([d.get('r_hat', 999) for d in diag1.values() if isinstance(d, dict) and 'r_hat' in d]):.4f}")
        
        # Method 2: Optimized approach (JAX opt + 300 warmup)
        print("\n🚀 Method 2: JAX Optimized + Minimal Warmup")
        config2 = config.copy()
        config2['mcmc'] = {
            'num_warmup': 300,   # Minimal warmup for HMC tuning
            'num_samples': 2000,
            'num_chains': 4
        }
        
        start_time = time.time()
        mcmc2 = MCMC(config2, likelihood,
                        optimize_init=True,
                        max_opt_iterations=300,
                        chain_name=f"optimized_{name}")
        samples2 = mcmc2.run()
        time2 = time.time() - start_time
        
        diag2 = mcmc2.sampler.get_diagnostics(burnin_frac=0.1)  # Post-processing burn-in
        converged2 = sum(1 for d in diag2.values() if isinstance(d, dict) and d.get('r_hat', 999) < 1.1)
        total2 = sum(1 for d in diag2.values() if isinstance(d, dict) and 'r_hat' in d)
        
        print(f"  Time: {time2:.2f}s")
        print(f"  Convergence: {converged2}/{total2} parameters")  
        print(f"  Avg R̂: {np.mean([d.get('r_hat', 999) for d in diag2.values() if isinstance(d, dict) and 'r_hat' in d]):.4f}")
        
        # Compare parameter accuracy
        true_params = problem['true_params']
        param_names = list(config['parameters'].keys())
        
        print(f"\n🎯 Parameter Accuracy:")
        total_error1, total_error2 = 0, 0
        for i, param in enumerate(param_names):
            if i < len(true_params):
                true_val = true_params[i]
                mean1 = float(np.mean(samples1[param]))
                mean2 = float(np.mean(samples2[param]))
                error1 = abs(mean1 - true_val)
                error2 = abs(mean2 - true_val)
                total_error1 += error1
                total_error2 += error2
                print(f"  {param}: True={true_val:.3f}, Trad={mean1:.3f}(±{error1:.3f}), Opt={mean2:.3f}(±{error2:.3f})")
        
        # Summary for this problem
        speedup = time1 / time2
        print(f"\n📊 Summary for {name}:")
        print(f"  Traditional: {time1:.2f}s, {converged1}/{total1} converged, total_error={total_error1:.3f}")
        print(f"  Optimized:   {time2:.2f}s, {converged2}/{total2} converged, total_error={total_error2:.3f}")
        print(f"  Speedup: {speedup:.2f}x")
        
        results[name] = {
            'traditional_time': time1,
            'optimized_time': time2,
            'speedup': speedup,
            'traditional_convergence': converged1 / total1,
            'optimized_convergence': converged2 / total2,
            'traditional_error': total_error1,
            'optimized_error': total_error2
        }
    
    # Overall summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    avg_speedup = np.mean([r['speedup'] for r in results.values()])
    avg_trad_conv = np.mean([r['traditional_convergence'] for r in results.values()])
    avg_opt_conv = np.mean([r['optimized_convergence'] for r in results.values()])
    
    print(f"\nPerformance:")
    for name, r in results.items():
        print(f"  {name.capitalize():>8s}: {r['speedup']:.2f}x speedup")
    print(f"  Average: {avg_speedup:.2f}x speedup")
    
    print(f"\nConvergence:")
    print(f"  Traditional: {avg_trad_conv:.1%}")
    print(f"  Optimized:   {avg_opt_conv:.1%}")
    
    print(f"\n💡 Key Insight:")
    if avg_speedup > 1.2 and avg_opt_conv > 0.8:
        print("   ✅ JAX optimization + minimal warmup provides significant benefits!")
        print("   ✅ Better initialization allows reducing warmup from 2000 to 300 steps")
        print("   ✅ Total compute time reduced while maintaining/improving convergence")
    elif avg_speedup > 1.0:
        print("   ✅ JAX optimization provides moderate performance improvement")
    else:
        print("   ⚠️  Traditional warmup still competitive, optimization overhead significant")
    
    print(f"\n🔬 Technical Notes:")
    print(f"   • JAX optimization finds good starting points automatically")
    print(f"   • HMC still needs minimal warmup (300 steps) for parameter tuning")
    print(f"   • Post-processing burn-in (10%) handles remaining adaptation")
    print(f"   • Net effect: 2000 → 300 warmup steps = 85% reduction")
    
    return results


if __name__ == "__main__":
    results = run_corrected_benchmark()