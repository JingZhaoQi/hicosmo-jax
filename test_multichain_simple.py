#!/usr/bin/env python3
"""
Simple test for MultiChain and enhanced plot_corner functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt

print("Testing MultiChain and enhanced plot_corner functionality")
print("=" * 55)

# Generate test data
np.random.seed(42)

# Chain 1: Planck-like
chain1 = {
    'H0': np.random.normal(70.0, 2.0, 1000),
    'Omega_m': np.random.normal(0.30, 0.02, 1000)
}

# Chain 2: SH0ES-like
chain2 = {
    'H0': np.random.normal(73.5, 1.5, 1000),
    'Omega_m': np.random.normal(0.28, 0.015, 1000)
}

print("Generated test chains")

# Test enhanced plot_corner with multi-chain support
try:
    from hicosmo.visualization import plot_corner

    print("Testing plot_corner multi-chain support...")

    # Test multi-chain
    fig = plot_corner([chain1, chain2],
                     params=['H0', 'Omega_m'],
                     labels=['Planck', 'SH0ES'],
                     filename='test_multichain.pdf')
    plt.close(fig)
    print("✅ Multi-chain plot_corner works")

    # Test MultiChain class
    from hicosmo.visualization import MultiChain

    print("Testing MultiChain class...")
    mc = MultiChain([chain1, chain2], ['Planck', 'SH0ES'])

    # Test parameter indices
    fig1 = mc.plot_corner([0, 1], filename='test_mc_indices.pdf')
    plt.close(fig1)
    print("✅ MultiChain with parameter indices works")

    # Test parameter names
    fig2 = mc.plot_corner(['H0', 'Omega_m'], filename='test_mc_names.pdf')
    plt.close(fig2)
    print("✅ MultiChain with parameter names works")

    print("\n📁 Files created in results/:")
    results_dir = Path(__file__).parent / 'results'
    if results_dir.exists():
        for f in results_dir.glob('test_*.pdf'):
            print(f"  ✅ {f.name}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")