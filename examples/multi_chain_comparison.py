#!/usr/bin/env python3
"""
HIcosmo Multi-Chain Comparison Example

Demonstrates the new multi-chain visualization capability:
- Multiple MCMC chains comparison in single corner plot
- Custom labels and legend
- Different color schemes for each chain
- Auto-save to script's results/ directory

Author: Jingzhao Qi
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

print("🔗 HIcosmo Multi-Chain Comparison Demo")
print("="*50)

# Step 1: Generate multiple MCMC chains with different characteristics
print("\n📝 Step 1: Generate Multiple MCMC Chains")
print("-"*40)

np.random.seed(42)

# Chain 1: Standard cosmology (Planck-like)
print("🔸 Chain 1: Planck-like cosmology")
mean1 = [70.0, 0.30]  # H0, Omega_m
cov1 = [[4.0, 0.1], [0.1, 0.001]]
chain1_data = {
    'H0': np.random.multivariate_normal(mean1, cov1, 2000)[:, 0],
    'Omega_m': np.random.multivariate_normal(mean1, cov1, 2000)[:, 1]
}

# Chain 2: High-H0 (SH0ES-like)
print("🔸 Chain 2: SH0ES-like (high H0)")
mean2 = [73.5, 0.28]  # Higher H0, slightly lower Omega_m
cov2 = [[2.5, 0.05], [0.05, 0.002]]
chain2_data = {
    'H0': np.random.multivariate_normal(mean2, cov2, 1500)[:, 0],
    'Omega_m': np.random.multivariate_normal(mean2, cov2, 1500)[:, 1]
}

# Chain 3: Alternative cosmology
print("🔸 Chain 3: Alternative model")
mean3 = [68.5, 0.32]  # Lower H0, higher Omega_m
cov3 = [[6.0, -0.2], [-0.2, 0.003]]
chain3_data = {
    'H0': np.random.multivariate_normal(mean3, cov3, 1800)[:, 0],
    'Omega_m': np.random.multivariate_normal(mean3, cov3, 1800)[:, 1]
}

print(f"Generated chains with {len(chain1_data['H0'])}, {len(chain2_data['H0'])}, {len(chain3_data['H0'])} samples")

# Step 2: Import visualization system
print("\n📝 Step 2: Import Multi-Chain Visualization")
print("-"*40)

from hicosmo.visualization import plot_corner

print("✅ Imported plot_corner with multi-chain support")

# Step 3: Single chain plot (baseline)
print("\n📝 Step 3: Single Chain Plot (Baseline)")
print("-"*40)

print("🔹 Single chain corner plot:")
fig_single = plot_corner(chain1_data,
                        params=['H0', 'Omega_m'],
                        style='modern',
                        filename='single_chain_baseline.pdf')
plt.close(fig_single)
print("  ✅ Baseline plot saved")

# Step 4: Two-chain comparison
print("\n📝 Step 4: Two-Chain Comparison")
print("-"*40)

print("🔹 Planck vs SH0ES comparison:")
fig_two = plot_corner([chain1_data, chain2_data],
                     params=['H0', 'Omega_m'],
                     labels=['Planck 2018', 'SH0ES 2022'],
                     style='modern',
                     filename='planck_vs_shoes.pdf')
plt.close(fig_two)
print("  ✅ Two-chain comparison saved")

# Step 5: Three-chain comparison
print("\n📝 Step 5: Three-Chain Comparison")
print("-"*40)

print("🔹 Full three-chain comparison:")
fig_three = plot_corner([chain1_data, chain2_data, chain3_data],
                       params=['H0', 'Omega_m'],
                       labels=['Planck 2018', 'SH0ES 2022', 'Alternative Model'],
                       style='modern',
                       filename='three_chain_comparison.pdf')
plt.close(fig_three)
print("  ✅ Three-chain comparison saved")

# Step 6: Classic style comparison
print("\n📝 Step 6: Classic Style Comparison")
print("-"*40)

print("🔹 Classic color scheme:")
fig_classic = plot_corner([chain1_data, chain2_data],
                         params=['H0', 'Omega_m'],
                         labels=['Standard', 'High-H0'],
                         style='classic',
                         filename='classic_style_comparison.pdf')
plt.close(fig_classic)
print("  ✅ Classic style saved")

# Step 7: Advanced multi-parameter comparison
print("\n📝 Step 7: Advanced Multi-Parameter Analysis")
print("-"*40)

# Add derived parameter
for chain_data in [chain1_data, chain2_data, chain3_data]:
    # Age of universe approximation: t0 ≈ 13.8 * (0.7/H0) * (0.3/Omega_m)^0.3
    t0_approx = 13.8 * (70.0/chain_data['H0']) * (0.3/chain_data['Omega_m'])**0.3
    chain_data['t0'] = t0_approx

print("🔹 Three-parameter comparison (H0, Omega_m, t0):")
fig_advanced = plot_corner([chain1_data, chain2_data, chain3_data],
                          params=['H0', 'Omega_m', 't0'],
                          labels=['Planck', 'SH0ES', 'Alternative'],
                          style='modern',
                          filename='advanced_three_param.pdf')
plt.close(fig_advanced)
print("  ✅ Advanced comparison saved")

# Step 8: Performance test with multiple chains
print("\n📝 Step 8: Multi-Chain Performance Test")
print("-"*40)

import time

print("🔹 Performance benchmarking:")
start_time = time.time()

# Test with different chain combinations
test_combinations = [
    ([chain1_data], ['Single']),
    ([chain1_data, chain2_data], ['Chain1', 'Chain2']),
    ([chain1_data, chain2_data, chain3_data], ['Chain1', 'Chain2', 'Chain3'])
]

performance_results = {}
for i, (chains, labels) in enumerate(test_combinations):
    test_start = time.time()
    fig = plot_corner(chains, params=['H0', 'Omega_m'], labels=labels)
    test_end = time.time()
    plt.close(fig)

    duration = test_end - test_start
    performance_results[len(chains)] = duration
    print(f"  📊 {len(chains)} chain{'s' if len(chains) > 1 else ''}: {duration:.3f}s")

total_time = time.time() - start_time
print(f"  ⚡ Total benchmark time: {total_time:.3f}s")

# Step 9: Verification and summary
print("\n📝 Step 9: Output Verification")
print("-"*40)

# Check all generated files
output_files = [
    'single_chain_baseline.pdf',
    'planck_vs_shoes.pdf',
    'three_chain_comparison.pdf',
    'classic_style_comparison.pdf',
    'advanced_three_param.pdf'
]

results_dir = Path(__file__).parent / 'results'
print(f"📁 Checking outputs in: {results_dir}")

successful_outputs = 0
for filename in output_files:
    filepath = results_dir / filename
    if filepath.exists():
        size = filepath.stat().st_size
        print(f"  ✅ {filename} ({size} bytes)")
        successful_outputs += 1
    else:
        print(f"  ❌ {filename} (missing)")

print(f"\n📊 Success Rate: {successful_outputs}/{len(output_files)} files created")

# Step 10: Feature summary
print("\n📝 Step 10: Multi-Chain Feature Summary")
print("-"*40)

features_demonstrated = [
    "✅ Single chain plotting (backward compatibility)",
    "✅ Two-chain comparison with custom labels",
    "✅ Three-chain comparison with legend",
    "✅ Modern and classic color schemes",
    "✅ Multi-parameter analysis (3+ parameters)",
    "✅ Automatic results/ directory creation",
    "✅ Performance scaling with chain count",
    "✅ Professional GetDist integration"
]

print("🎯 Multi-Chain Features Demonstrated:")
for feature in features_demonstrated:
    print(f"   {feature}")

print(f"\n⚡ Performance Summary:")
print(f"   Single chain: {performance_results.get(1, 0):.3f}s")
print(f"   Two chains:   {performance_results.get(2, 0):.3f}s")
print(f"   Three chains: {performance_results.get(3, 0):.3f}s")

scale_factor = performance_results.get(3, 0) / performance_results.get(1, 1)
print(f"   Scaling factor: {scale_factor:.1f}x")

# Usage recommendations
print(f"\n📖 Usage Recommendations:")
print(f"   1. Use descriptive labels: ['Planck 2018', 'SH0ES 2022']")
print(f"   2. Limit to 3-4 chains for clarity")
print(f"   3. Choose 'modern' style for publications")
print(f"   4. Files auto-save to script's results/ directory")
print(f"   5. Backward compatible with single-chain code")

print("\n" + "="*50)
print("🔗 Multi-Chain Comparison Demo Completed")
print("="*50)