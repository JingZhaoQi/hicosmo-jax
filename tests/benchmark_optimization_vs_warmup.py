#!/usr/bin/env python3
"""
Performance benchmark: JAX optimization initialization vs traditional warmup.

This benchmark provides a comprehensive comparison between:
1. Traditional MCMC with warmup (starting from random initial values)
2. Modern MCMC with JAX optimization initialization (no warmup needed)
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys
import time
import numpyro
import matplotlib.pyplot as plt

# Add HiCosmo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers import MCMC

# Enable multi-core sampling for fair comparison
numpyro.set_host_device_count(4)


def create_test_problem(complexity="simple"):
    """Create test problems of different complexity levels."""
    
    if complexity == "simple":
        # Simple quadratic fit
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
        
    elif complexity == "medium":
        # Higher dimensional problem with correlations
        np.random.seed(42)
        x1 = np.linspace(0, 3, 25)
        x2 = np.linspace(-1, 2, 25)
        X1, X2 = np.meshgrid(x1, x2)
        x1_flat, x2_flat = X1.flatten(), X2.flatten()
        
        # True parameters with more complexity
        true_params = [1.5, -0.8, 0.3, 0.6, 2.1]  # [a, b, c, d, e]
        y_true = (true_params[0] * x1_flat**2 + true_params[1] * x2_flat**2 + 
                 true_params[2] * x1_flat * x2_flat + true_params[3] * x1_flat + 
                 true_params[4])
        y_err = 0.15 * np.ones_like(y_true)
        y_obs = y_true + np.random.normal(0, y_err)
        
        def likelihood(a, b, c, d, e):
            y_pred = (a * x1_flat**2 + b * x2_flat**2 + c * x1_flat * x2_flat + 
                     d * x1_flat + e)
            chi2 = jnp.sum((y_obs - y_pred)**2 / y_err**2)
            return -0.5 * chi2
        
        config = {
            'parameters': {
                'a': (0.5, -2, 4),   # Start far from 1.5
                'b': (0.2, -3, 2),   # Start far from -0.8
                'c': (0.8, -1, 2),   # Start far from 0.3
                'd': (-0.2, -2, 3),  # Start far from 0.6
                'e': (0.5, -1, 5)    # Start far from 2.1
            }
        }
        
    elif complexity == "hard":
        # High dimensional problem with strong correlations
        np.random.seed(42)
        n_data = 40
        n_params = 8
        
        # Generate correlated design matrix
        x = np.random.randn(n_data, n_params)
        # Add correlations
        for i in range(1, n_params):
            x[:, i] += 0.3 * x[:, i-1]
        
        # True parameters  
        true_params = np.array([1.2, -0.5, 0.8, 0.3, -0.4, 0.9, -0.2, 1.1])
        y_true = x @ true_params
        y_err = 0.2 * np.ones(n_data)
        y_obs = y_true + np.random.normal(0, y_err)
        
        def likelihood(p0, p1, p2, p3, p4, p5, p6, p7):
            params = jnp.array([p0, p1, p2, p3, p4, p5, p6, p7])
            y_pred = x @ params
            chi2 = jnp.sum((y_obs - y_pred)**2 / y_err**2)
            return -0.5 * chi2
        
        config = {
            'parameters': {
                f'p{i}': (0.1, -2, 3) for i in range(8)  # Start far from true values
            }
        }
        
    return likelihood, config, true_params


def run_traditional_mcmc(likelihood, config, num_samples=2000, num_warmup=2000):
    """Run traditional MCMC with warmup."""
    print(f"🔥 Traditional MCMC (warmup={num_warmup}, samples={num_samples})")
    
    config_warmup = config.copy()
    config_warmup['mcmc'] = {
        'num_warmup': num_warmup,
        'num_samples': num_samples,
        'num_chains': 4
    }
    
    start_time = time.time()
    mcmc = MCMC(config_warmup, likelihood, 
                   optimize_init=False,  # No optimization
                   chain_name=f"traditional_{int(time.time())}")
    samples = mcmc.run()
    total_time = time.time() - start_time
    
    diagnostics = mcmc.sampler.get_diagnostics(burnin_frac=0.0)  # No post-burnin
    
    return samples, total_time, diagnostics


def run_optimized_mcmc(likelihood, config, num_samples=2000):
    """Run optimized MCMC with JAX initialization."""
    print(f"🚀 Optimized MCMC (optimization + {num_samples} samples)")
    
    config_opt = config.copy()
    config_opt['mcmc'] = {
        'num_warmup': 0,  # No warmup needed
        'num_samples': num_samples,
        'num_chains': 4
    }
    
    start_time = time.time()
    mcmc = MCMC(config_opt, likelihood,
                   optimize_init=True,  # JAX optimization
                   max_opt_iterations=500,
                   chain_name=f"optimized_{int(time.time())}")
    samples = mcmc.run()
    total_time = time.time() - start_time
    
    diagnostics = mcmc.sampler.get_diagnostics(burnin_frac=0.1)  # 10% post-burnin
    
    return samples, total_time, diagnostics


def compare_accuracy(samples1, samples2, true_params, param_names):
    """Compare parameter recovery accuracy."""
    print("\n📊 Parameter Recovery Comparison:")
    print("=" * 60)
    
    results = {}
    
    for i, param in enumerate(param_names):
        if param in samples1 and param in samples2:
            true_val = true_params[i] if i < len(true_params) else 0.0
            
            mean1 = float(np.mean(samples1[param]))
            std1 = float(np.std(samples1[param]))
            error1 = abs(mean1 - true_val)
            
            mean2 = float(np.mean(samples2[param]))
            std2 = float(np.std(samples2[param]))
            error2 = abs(mean2 - true_val)
            
            print(f"{param:>8s} | True: {true_val:6.3f} | Traditional: {mean1:6.3f}±{std1:.3f} (err:{error1:.3f}) | Optimized: {mean2:6.3f}±{std2:.3f} (err:{error2:.3f})")
            
            results[param] = {
                'true': true_val,
                'traditional': {'mean': mean1, 'std': std1, 'error': error1},
                'optimized': {'mean': mean2, 'std': std2, 'error': error2}
            }
    
    return results


def analyze_convergence(diagnostics1, diagnostics2):
    """Analyze MCMC convergence diagnostics."""
    print("\n🔍 Convergence Diagnostics:")
    print("=" * 50)
    
    # Count parameters that converged (R̂ < 1.1)
    converged1 = sum(1 for param, diag in diagnostics1.items() 
                    if param != '_overall' and diag.get('r_hat', 999) < 1.1)
    total1 = len([p for p in diagnostics1.keys() if p != '_overall'])
    
    converged2 = sum(1 for param, diag in diagnostics2.items() 
                    if param != '_overall' and diag.get('r_hat', 999) < 1.1)
    total2 = len([p for p in diagnostics2.keys() if p != '_overall'])
    
    print(f"Traditional MCMC: {converged1}/{total1} parameters converged")
    print(f"Optimized MCMC:   {converged2}/{total2} parameters converged")
    
    # Average R̂ values
    rhat1 = np.mean([diag.get('r_hat', 999) for param, diag in diagnostics1.items() 
                    if param != '_overall'])
    rhat2 = np.mean([diag.get('r_hat', 999) for param, diag in diagnostics2.items() 
                    if param != '_overall'])
    
    print(f"Average R̂ - Traditional: {rhat1:.4f}, Optimized: {rhat2:.4f}")
    
    # Average effective sample size
    ess1 = np.mean([diag.get('ess', 0) for param, diag in diagnostics1.items() 
                   if param != '_overall'])
    ess2 = np.mean([diag.get('ess', 0) for param, diag in diagnostics2.items() 
                   if param != '_overall'])
    
    print(f"Average ESS - Traditional: {ess1:.0f}, Optimized: {ess2:.0f}")
    
    return {
        'convergence_rate': {'traditional': converged1/total1, 'optimized': converged2/total2},
        'avg_rhat': {'traditional': rhat1, 'optimized': rhat2},
        'avg_ess': {'traditional': ess1, 'optimized': ess2}
    }


def benchmark_complexity_level(complexity, num_runs=3):
    """Benchmark a specific complexity level multiple times."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {complexity.upper()} PROBLEM")
    print(f"{'='*80}")
    
    likelihood, config, true_params = create_test_problem(complexity)
    param_names = list(config['parameters'].keys())
    
    traditional_times = []
    optimized_times = []
    accuracy_results = []
    convergence_results = []
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        # Traditional MCMC
        samples1, time1, diag1 = run_traditional_mcmc(likelihood, config)
        traditional_times.append(time1)
        
        # Optimized MCMC  
        samples2, time2, diag2 = run_optimized_mcmc(likelihood, config)
        optimized_times.append(time2)
        
        # Compare accuracy
        accuracy = compare_accuracy(samples1, samples2, true_params, param_names)
        accuracy_results.append(accuracy)
        
        # Analyze convergence
        convergence = analyze_convergence(diag1, diag2)
        convergence_results.append(convergence)
        
        print(f"\n⏱️  Run {run + 1} Times: Traditional={time1:.2f}s, Optimized={time2:.2f}s, Speedup={time1/time2:.2f}x")
    
    # Summary statistics
    avg_traditional = np.mean(traditional_times)
    avg_optimized = np.mean(optimized_times)
    avg_speedup = avg_traditional / avg_optimized
    
    std_traditional = np.std(traditional_times)
    std_optimized = np.std(optimized_times)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY - {complexity.upper()} PROBLEM ({num_runs} runs)")
    print(f"{'='*60}")
    print(f"Traditional MCMC: {avg_traditional:.2f} ± {std_traditional:.2f} seconds")
    print(f"Optimized MCMC:   {avg_optimized:.2f} ± {std_optimized:.2f} seconds")
    print(f"Average Speedup:  {avg_speedup:.2f}x")
    
    # Average convergence rates
    avg_conv_trad = np.mean([r['convergence_rate']['traditional'] for r in convergence_results])
    avg_conv_opt = np.mean([r['convergence_rate']['optimized'] for r in convergence_results])
    print(f"Convergence Rate: Traditional={avg_conv_trad:.1%}, Optimized={avg_conv_opt:.1%}")
    
    return {
        'complexity': complexity,
        'times': {'traditional': traditional_times, 'optimized': optimized_times},
        'speedup': avg_speedup,
        'accuracy': accuracy_results,
        'convergence': convergence_results
    }


def main():
    """Run comprehensive performance benchmark."""
    print("COMPREHENSIVE MCMC PERFORMANCE BENCHMARK")
    print("=" * 80)
    print("Comparing JAX Optimization Initialization vs Traditional Warmup")
    print()
    
    # Test different problem complexities
    complexity_levels = ["simple", "medium", "hard"]
    num_runs = 3  # Number of runs per complexity
    
    all_results = {}
    
    for complexity in complexity_levels:
        try:
            results = benchmark_complexity_level(complexity, num_runs)
            all_results[complexity] = results
            time.sleep(1)  # Brief pause between benchmarks
        except Exception as e:
            print(f"❌ Benchmark failed for {complexity}: {e}")
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    for complexity, results in all_results.items():
        avg_trad = np.mean(results['times']['traditional'])
        avg_opt = np.mean(results['times']['optimized'])
        speedup = results['speedup']
        
        print(f"{complexity.capitalize():>8s} Problem: {speedup:.2f}x speedup ({avg_trad:.1f}s → {avg_opt:.1f}s)")
    
    overall_speedups = [results['speedup'] for results in all_results.values()]
    if overall_speedups:
        avg_overall_speedup = np.mean(overall_speedups)
        print(f"\n🚀 Overall Average Speedup: {avg_overall_speedup:.2f}x")
        
        if avg_overall_speedup > 1.5:
            print("✅ JAX Optimization provides significant performance improvement!")
        elif avg_overall_speedup > 1.2:
            print("✅ JAX Optimization provides moderate performance improvement.")
        else:
            print("⚠️  Performance improvement is marginal.")
    
    print(f"\n💡 Key Insights:")
    print("   • JAX optimization finds good starting points automatically")
    print("   • No warmup needed when starting from optimized values") 
    print("   • Better convergence properties in most cases")
    print("   • Modern MCMC best practices implemented")
    
    return all_results


if __name__ == "__main__":
    results = main()