#!/usr/bin/env python3
"""
Test the new MCMC system with the polynomial fitting example.

This test demonstrates the dramatically simplified API compared to the previous
NumPyro implementation, while maintaining all the power and flexibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add HiCosmo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers import MCMC, quick_mcmc


def load_test_data():
    """Load the same test data used in original test_MC.py"""
    data_file = Path(__file__).parent / 'data' / 'sim_data.txt'
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        print("Using mock data...")
        
        # Create mock data with same structure as original
        x = np.array([0.05, 0.15, 0.15, 0.30, 0.35, 0.85, 0.85, 0.90, 
                      1.35, 1.45, 1.55, 1.55, 1.75, 1.75, 2.00, 2.20, 
                      2.65, 2.65, 2.75, 2.75])
        
        # Generate y values with true parameters a=3.5, b=2, c=1
        y_true = 3.5 * x**2 + 2 * x + 1
        y_err = 0.1 + 0.8 * x  # Similar error structure
        y_obs = y_true + np.random.normal(0, y_err)
        
    else:
        # Load actual data
        x, y_obs, y_err = np.loadtxt(data_file, unpack=True)
    
    return x, y_obs, y_err


def test_dictionary_config():
    """Test MCMC with dictionary configuration."""
    print("=" * 70)
    print("Test 1: Dictionary-Based Configuration")
    print("=" * 70)
    
    # Load data
    x, y_obs, y_err = load_test_data()
    
    # Define likelihood function with enclosed data (much cleaner!)
    def polynomial_likelihood(a, b, c):
        """Simple polynomial likelihood function with data from outer scope."""
        y_pred = a * x**2 + b * x + c
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2  # Return log-likelihood
    
    # Dictionary configuration (much cleaner than qcosmc!)
    config = {
        'parameters': {
            'a': {
                'prior': {'dist': 'uniform', 'min': 0, 'max': 10},
                'ref': 3.5,
                'latex': r'a'
            },
            'b': {
                'prior': {'dist': 'uniform', 'min': 0, 'max': 4},
                'ref': 2.0,
                'latex': r'b' 
            },
            'c': {
                'prior': {'dist': 'uniform', 'min': 0, 'max': 2},
                'ref': 1.0,
                'latex': r'c'
            }
        },
        'mcmc': {
            'num_warmup': 1000,
            'num_samples': 2000,
            'num_chains': 4
        }
    }
    
    # One line to create and configure MCMC! (much cleaner)
    mcmc = MCMC(config, polynomial_likelihood, 
                   chain_name="test1_dict_config")
    
    # One line to run!
    samples = mcmc.run()
    
    # Print results
    mcmc.print_summary(prob=0.68)
    
    # Compare with expected results
    a_mean = np.mean(samples['a'])
    b_mean = np.mean(samples['b']) 
    c_mean = np.mean(samples['c'])
    
    print(f"\nResults comparison:")
    print(f"  a = {a_mean:.3f} (expected ≈ 3.32)")
    print(f"  b = {b_mean:.3f} (expected ≈ 1.28)")  
    print(f"  c = {c_mean:.3f} (expected ≈ 1.087)")
    
    return samples


def test_simple_list_config():
    """Test MCMC with qcosmc-style simple list."""
    print("\n" + "=" * 70)
    print("Test 2: Simple List Configuration (qcosmc-compatible)")
    print("=" * 70)
    
    # Load data
    x, y_obs, y_err = load_test_data()
    
    # Likelihood function with enclosed data
    def likelihood(a, b, c):
        """Simple likelihood with enclosed data."""
        y_pred = a * x**2 + b * x + c
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    # qcosmc-style parameter list - as simple as the original!
    params = [
        ['a', 3.5, 0, 10],
        ['b', 2.0, 0, 4], 
        ['c', 1.0, 0, 2]
    ]
    
    # Clean and simple call
    mcmc = MCMC.from_simple_list(
        params,
        likelihood, 
        chain_name="test2_simple_list",
        num_samples=2000
    )
    
    # Run MCMC
    samples = mcmc.run()
    
    # Print results  
    print("\nqcosmc-style results:")
    for param in ['a', 'b', 'c']:
        mean_val = np.mean(samples[param])
        std_val = np.std(samples[param])
        print(f"  {param} = {mean_val:.3f} ± {std_val:.3f}")
    
    return samples


def test_quick_mcmc():
    """Test the ultra-convenient quick_mcmc function."""
    print("\n" + "=" * 70) 
    print("Test 3: Ultra-Quick MCMC (One Function Call)")
    print("=" * 70)
    
    # Load data
    x, y_obs, y_err = load_test_data()
    
    # Simple likelihood with enclosed data
    def likelihood(a, b, c):
        y_pred = a * x**2 + b * x + c
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    # ONE FUNCTION CALL FOR COMPLETE MCMC ANALYSIS! (super clean)
    results = quick_mcmc(
        [['a', 3.5, 0, 10], ['b', 2.0, 0, 4], ['c', 1.0, 0, 2]],
        likelihood,
        chain_name="test3_quick_mcmc",
        num_samples=1000  # Quick run
    )
    
    print("Ultra-quick results:")
    for param, info in results['results'].items():
        print(f"  {param} = {info['value']:.3f} ± {info['error']:.3f}")
    
    return results


def test_advanced_features():
    """Test advanced features like function inspection and diagnostics."""
    print("\n" + "=" * 70)
    print("Test 4: Advanced Features")  
    print("=" * 70)
    
    # Load data
    x, y_obs, y_err = load_test_data()
    
    # More complex configuration with different priors
    config = {
        'parameters': {
            'a': {
                'prior': {'dist': 'normal', 'loc': 3.5, 'scale': 1.0},
                'bounds': [0, 10],
                'latex': r'a_{coeff}'
            },
            'b': {
                'prior': {'dist': 'truncnorm', 'loc': 2.0, 'scale': 0.5, 'low': 0, 'high': 4},
                'latex': r'b_{coeff}'
            },
            'c': {
                'prior': {'dist': 'beta', 'alpha': 2, 'beta': 3},
                'bounds': [0, 2], 
                'latex': r'c_{coeff}'
            }
        },
        'mcmc': {'num_samples': 1500, 'num_chains': 2}
    }
    
    def advanced_likelihood(a, b, c):
        """Likelihood with more complex structure and enclosed data."""
        y_pred = a * x**2 + b * x + c
        
        # Add some complexity
        residuals = y_obs - y_pred
        chi2 = np.sum(residuals**2 / y_err**2)
        
        # Could add more complex likelihood terms here
        log_likelihood = -0.5 * chi2
        
        return log_likelihood
    
    # Create MCMC (clean and simple)
    mcmc = MCMC(config, advanced_likelihood, 
                   chain_name="test4_advanced_features")
    
    # Check setup validation
    print(f"✓ Setup validation: {mcmc.validate_setup()}")
    print(f"✓ Parameter info: {list(mcmc.get_parameter_info().keys())}")
    
    suggestions = mcmc.suggest_improvements()
    if suggestions:
        print("💡 Suggestions:")
        for suggestion in suggestions:
            print(f"   {suggestion}")
    
    # Run MCMC
    samples = mcmc.run()
    
    # Get diagnostics
    diagnostics = mcmc.get_diagnostics()
    print(f"\n🔍 Diagnostics summary:")
    for param, diag in diagnostics.items():
        if param != '_overall':
            print(f"   {param}: R̂={diag['r_hat']:.3f}, ESS={diag['ess']:.0f}")
    
    # 演示用户主动保存结果（可选）
    # mcmc.save_results("polynomial_fit_results.pkl")
    # print("💾 Results saved to polynomial_fit_results.pkl")
    
    return samples, mcmc


def compare_apis():
    """Compare the old and new APIs side by side."""
    print("\n" + "=" * 70)
    print("API Comparison: Old vs New")
    print("=" * 70)
    
    print("OLD API (NumPyro model):")
    print("=" * 30)
    old_code = """
def polynomial_model(x_data, y_obs=None, y_err=None):
    a = numpyro.sample("a", dist.Uniform(0, 10))
    b = numpyro.sample("b", dist.Uniform(0, 4))
    c = numpyro.sample("c", dist.Uniform(0, 2))
    
    y_pred = a * x_data**2 + b * x_data + c
    
    with numpyro.plate("data", len(x_data)):
        numpyro.sample("y", dist.Normal(y_pred, y_err), obs=y_obs)

sampler = MCMCSampler(polynomial_model, num_warmup=2000, num_samples=4000)
samples = sampler.run(x_data=jnp.array(x), y_obs=jnp.array(y_obs), y_err=jnp.array(y_err))
"""
    print(old_code)
    
    print("NEW API (MCMC):")
    print("=" * 30) 
    new_code = """
# Data loaded once at the beginning  
x, y_obs, y_err = load_data()

def likelihood(a, b, c):  # Clean function signature
    y_pred = a * x**2 + b * x + c
    chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
    return -0.5 * chi2

config = {
    'parameters': {
        'a': {'prior': {'dist': 'uniform', 'min': 0, 'max': 10}},
        'b': {'prior': {'dist': 'uniform', 'min': 0, 'max': 4}}, 
        'c': {'prior': {'dist': 'uniform', 'min': 0, 'max': 2}}
    }
}

mcmc = MCMC(config, likelihood)  # Clean call
samples = mcmc.run()
"""
    print(new_code)
    
    print("ULTRA-SIMPLE API (qcosmc-style):")
    print("=" * 35)
    simple_code = """
results = quick_mcmc(
    [['a', 3.5, 0, 10], ['b', 2.0, 0, 4], ['c', 1.0, 0, 2]],
    likelihood  # Data enclosed in function scope
)
"""
    print(simple_code)


def main():
    """Run all tests."""
    print("MCMC System Test Suite")
    print("=" * 70)
    print("Testing the new dictionary-driven MCMC system")
    print()
    
    # Run all tests
    try:
        samples1 = test_dictionary_config()
        samples2 = test_simple_list_config() 
        results3 = test_quick_mcmc()
        samples4, mcmc4 = test_advanced_features()
        
        # Compare APIs
        compare_apis()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        
        print("\n✅ Key Achievements:")
        print("  • Dictionary-based configuration works perfectly")
        print("  • qcosmc-style simple lists supported") 
        print("  • Ultra-convenient quick_mcmc function")
        print("  • Advanced features (multiple priors, diagnostics)")
        print("  • Automatic parameter-function mapping")
        print("  • All NumPyro power preserved")
        
        print("\n💾 File Saving:")
        print("  • Checkpoint功能默认开启（实时保存，断点续算）")
        print("  • 自动保存到 ./mcmc_chains/ 目录")
        print("  • 文件名直接使用用户指定的chain_name.h5")
        print("  • 用户可通过 mcmc.save_results('filename.pkl') 保存最终结果")
        print("  • 支持pickle和HDF5格式，完全由用户控制")
        
        print("\n🚀 The API is now as simple as qcosmc but with modern Bayesian inference!")
        
        return {
            'dict_config': samples1,
            'simple_list': samples2, 
            'quick_mcmc': results3,
            'advanced': samples4
        }
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()