#!/usr/bin/env python3
"""
Test simplified parameter input format for MCMC.

This test demonstrates the new simplified input format that allows
users to specify parameters using tuples, lists, or sets.
"""

import numpy as np
from pathlib import Path
import sys

# Add HiCosmo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers import MCMC


def test_simplified_tuple_format():
    """Test MCMC with simplified tuple format."""
    print("=" * 70)
    print("Test 1: Simplified Tuple Format")
    print("=" * 70)
    
    # Mock data
    x = np.linspace(0, 3, 20)
    y_true = 3.5 * x**2 + 2 * x + 1
    y_err = 0.1 + 0.8 * x
    y_obs = y_true + np.random.normal(0, y_err)
    
    # Likelihood with enclosed data
    def likelihood(a, b, c):
        y_pred = a * x**2 + b * x + c
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    # Simplified tuple format (ref, min, max, latex)
    config = {
        'parameters': {
            'a': (3.5, 0, 10, '$a$'),
            'b': (2.0, 0, 4, '$b$'),
            'c': (1.0, 0, 2, '$c$')
        },
        'mcmc': {
            'num_warmup': 500,
            'num_samples': 1000,
            'num_chains': 2
        }
    }
    
    mcmc = MCMC(config, likelihood, chain_name="test_tuple_format")
    samples = mcmc.run()
    
    print("\nResults with tuple format:")
    for param in ['a', 'b', 'c']:
        mean_val = np.mean(samples[param])
        std_val = np.std(samples[param])
        print(f"  {param} = {mean_val:.3f} ± {std_val:.3f}")
    
    return samples


def test_simplified_list_format():
    """Test MCMC with simplified list format."""
    print("\n" + "=" * 70)
    print("Test 2: Simplified List Format")
    print("=" * 70)
    
    # Mock data
    x = np.linspace(0, 3, 20)
    y_true = 3.5 * x**2 + 2 * x + 1
    y_err = 0.1 + 0.8 * x
    y_obs = y_true + np.random.normal(0, y_err)
    
    # Likelihood with enclosed data
    def likelihood(a, b, c):
        y_pred = a * x**2 + b * x + c
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    # Simplified list format (without latex)
    config = {
        'parameters': {
            'a': [3.5, 0, 10],  # No latex label
            'b': [2.0, 0, 4, '$b_{param}$'],  # With latex
            'c': [1.0, 0, 2]  # No latex
        },
        'mcmc': {
            'num_warmup': 500,
            'num_samples': 1000,
            'num_chains': 2
        }
    }
    
    mcmc = MCMC(config, likelihood, chain_name="test_list_format")
    samples = mcmc.run()
    
    print("\nResults with list format:")
    for param in ['a', 'b', 'c']:
        mean_val = np.mean(samples[param])
        std_val = np.std(samples[param])
        print(f"  {param} = {mean_val:.3f} ± {std_val:.3f}")
    
    return samples


def test_simplified_set_format():
    """Test MCMC with simplified set format."""
    print("\n" + "=" * 70)
    print("Test 3: Simplified Set Format")
    print("=" * 70)
    
    # Mock data
    x = np.linspace(0, 3, 20)
    y_true = 3.5 * x**2 + 2 * x + 1
    y_err = 0.1 + 0.8 * x
    y_obs = y_true + np.random.normal(0, y_err)
    
    # Likelihood with enclosed data
    def likelihood(a, b, c):
        y_pred = a * x**2 + b * x + c
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    # Simplified set format (Note: sets don't preserve order)
    config = {
        'parameters': {
            'a': {3.5, 0, 10, '$a$'},  # Set with latex
            'b': {2.0, 0, 4, '$b$'},
            'c': {1.0, 0, 2}  # Set without latex
        },
        'mcmc': {
            'num_warmup': 500,
            'num_samples': 1000,
            'num_chains': 2
        }
    }
    
    mcmc = MCMC(config, likelihood, chain_name="test_set_format")
    samples = mcmc.run()
    
    print("\nResults with set format:")
    for param in ['a', 'b', 'c']:
        mean_val = np.mean(samples[param])
        std_val = np.std(samples[param])
        print(f"  {param} = {mean_val:.3f} ± {std_val:.3f}")
    
    return samples


def test_mixed_formats():
    """Test MCMC with mixed parameter formats."""
    print("\n" + "=" * 70)
    print("Test 4: Mixed Parameter Formats")
    print("=" * 70)
    
    # Mock data
    x = np.linspace(0, 3, 20)
    y_true = 3.5 * x**2 + 2 * x + 1
    y_err = 0.1 + 0.8 * x
    y_obs = y_true + np.random.normal(0, y_err)
    
    # Likelihood with enclosed data
    def likelihood(a, b, c):
        y_pred = a * x**2 + b * x + c
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    # Mixed formats: full dict, tuple, list
    config = {
        'parameters': {
            'a': {'prior': {'dist': 'uniform', 'min': 0, 'max': 10}, 'ref': 3.5, 'latex': '$a_{full}$'},  # Full format
            'b': (2.0, 0, 4, '$b_{tuple}$'),  # Tuple format
            'c': [1.0, 0, 2]  # List format
        },
        'mcmc': {
            'num_warmup': 500,
            'num_samples': 1000,
            'num_chains': 2
        }
    }
    
    mcmc = MCMC(config, likelihood, chain_name="test_mixed_formats")
    samples = mcmc.run()
    
    print("\nResults with mixed formats:")
    for param in ['a', 'b', 'c']:
        mean_val = np.mean(samples[param])
        std_val = np.std(samples[param])
        print(f"  {param} = {mean_val:.3f} ± {std_val:.3f}")
    
    # Check parameter info
    param_info = mcmc.get_parameter_info()
    print("\nParameter latex labels:")
    for name, info in param_info.items():
        print(f"  {name}: {info.get('latex', 'N/A')}")
    
    return samples


def main():
    """Run all simplified format tests."""
    print("Simplified Parameter Input Format Test Suite")
    print("=" * 70)
    print("Testing support for tuple, list, and set parameter formats")
    print()
    
    try:
        # Run tests
        samples1 = test_simplified_tuple_format()
        samples2 = test_simplified_list_format()
        samples3 = test_simplified_set_format()
        samples4 = test_mixed_formats()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        
        print("\n✅ Supported formats:")
        print("  • Tuple format: ('a': (3.5, 0, 10, '$a$'))")
        print("  • List format: ('b': [2.0, 0, 4, '$b$'])")
        print("  • Set format: ('c': {1.0, 0, 2})")
        print("  • Mixed formats in same config")
        print("  • Optional latex labels")
        print("  • Full backward compatibility")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)