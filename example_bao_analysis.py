#!/usr/bin/env python3
"""
HIcosmo BAO Analysis Example

This script demonstrates:
1. Using individual BAO datasets
2. Combining multiple BAO datasets
3. Joint SNe+BAO analysis
4. Creating publication-quality plots
"""

import os
import numpy as np
from hicosmo.samplers import init_hicosmo, MCMC, ParameterConfig
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods import (
    PantheonPlusLikelihood,
    SDSSDR12BAO, BOSSDR12BAO, DESI2024BAO,
    BAOCollection, get_available_datasets
)
from hicosmo.visualization import HIcosmoViz

# Initialize HIcosmo
init_hicosmo()

print("=" * 60)
print("HIcosmo BAO Analysis Example")
print("=" * 60)

# Show available datasets
print("\n📊 Available BAO datasets:")
for dataset in get_available_datasets():
    print(f"   - {dataset}")

# 1. Single BAO dataset analysis
print("\n" + "=" * 60)
print("Example 1: DESI 2024 BAO Only")
print("=" * 60)

desi_bao = DESI2024BAO(verbose=True)

params_bao = {
    'H0': {70, 55, 85},
    'Omega_m': {0.3, 0.1, 0.6}
}

def desi_likelihood(H0, Omega_m):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return desi_bao.log_likelihood(model)

print("\n🚀 Running MCMC for DESI BAO...")
config = ParameterConfig(params_bao, mcmc={'num_samples': 10000, 'num_chains': 4})
samples_desi = MCMC(config, desi_likelihood).run()

print("\n📈 DESI 2024 Results:")
for param in params_bao:
    mean = np.mean(samples_desi[param])
    std = np.std(samples_desi[param])
    median = np.median(samples_desi[param])
    print(f"   {param}: {median:.2f}_{{{median-np.percentile(samples_desi[param], 16):.2f}}}^{{+{np.percentile(samples_desi[param], 84)-median:.2f}}}")

# 2. Multiple BAO datasets combined
print("\n" + "=" * 60)
print("Example 2: Combined BAO (SDSS + BOSS + 6dF)")
print("=" * 60)

bao_collection = BAOCollection(
    datasets=['sdss_dr12', 'boss_dr12', 'sixdf'],
    verbose=True
)

def combined_bao_likelihood(H0, Omega_m):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return bao_collection.log_likelihood(model)

print("\n🚀 Running MCMC for combined BAO...")
samples_combined = MCMC(
    ParameterConfig(params_bao, mcmc={'num_samples': 20000, 'num_chains': 4}),
    combined_bao_likelihood
).run()

print("\n📈 Combined BAO Results:")
for param in params_bao:
    median = np.median(samples_combined[param])
    lower = median - np.percentile(samples_combined[param], 16)
    upper = np.percentile(samples_combined[param], 84) - median
    print(f"   {param}: {median:.2f}_{{{lower:.2f}}}^{{+{upper:.2f}}}")

# 3. Joint SNe + BAO analysis
print("\n" + "=" * 60)
print("Example 3: Joint PantheonPlus + DESI 2024")
print("=" * 60)

data_path = os.path.join(os.path.dirname(__file__), "data", "DataRelease")
pantheon = PantheonPlusLikelihood(data_path, include_shoes=False, marginalize_M_B=False)

params_joint = {
    'H0': {70, 60, 80},
    'Omega_m': {0.3, 0.2, 0.4},
    'M_B': {-19.25, -20, -18}
}

def joint_likelihood(H0, Omega_m, M_B):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    log_like_sne = pantheon.log_likelihood(model, M_B=M_B)
    log_like_bao = desi_bao.log_likelihood(model)
    return log_like_sne + log_like_bao

print("\n🚀 Running joint SNe+BAO MCMC...")
config_joint = ParameterConfig(params_joint, mcmc={'num_samples': 10000, 'num_chains': 4})
samples_joint = MCMC(config_joint, joint_likelihood).run()

print("\n📈 Joint SNe+BAO Results:")
for param in params_joint:
    median = np.median(samples_joint[param])
    lower = median - np.percentile(samples_joint[param], 16)
    upper = np.percentile(samples_joint[param], 84) - median
    print(f"   {param}: {median:.2f}_{{{lower:.2f}}}^{{+{upper:.2f}}}")

# 4. Visualization
print("\n📊 Creating visualization...")
viz = HIcosmoViz()

# Corner plot for joint analysis
viz.corner(
    samples_joint,
    params=list(params_joint.keys()),
    filename="joint_sne_bao_corner.pdf"
)

# Comparison plot (corner_compare not yet implemented)
# viz.corner_compare(
#     [samples_desi, samples_combined, samples_joint],
#     params=['H0', 'Omega_m'],
#     labels=['DESI only', 'Combined BAO', 'SNe+DESI'],
#     filename="bao_comparison_corner.pdf"
# )

# Summary statistics
print("\n" + "=" * 60)
print("Constraint Improvements")
print("=" * 60)

def print_improvement(name1, samples1, name2, samples2):
    for param in ['H0', 'Omega_m']:
        if param in samples1 and param in samples2:
            std1 = np.std(samples1[param])
            std2 = np.std(samples2[param])
            improvement = (1 - std2/std1) * 100
            print(f"{param}: {name1} σ={std1:.3f} → {name2} σ={std2:.3f} ({improvement:+.1f}%)")

print("\nAdding BAO datasets:")
print_improvement("DESI only", samples_desi, "Combined BAO", samples_combined)

print("\nAdding SNe to BAO:")
print_improvement("DESI only", samples_desi, "SNe+DESI", samples_joint)

# Scientific interpretation
print("\n" + "=" * 60)
print("Scientific Summary")
print("=" * 60)

h0_joint = np.median(samples_joint['H0'])
h0_err = np.std(samples_joint['H0'])
om_joint = np.median(samples_joint['Omega_m'])
om_err = np.std(samples_joint['Omega_m'])

print(f"""
The joint SNe+BAO analysis yields:
• H₀ = {h0_joint:.1f} ± {h0_err:.1f} km/s/Mpc
• Ωₘ = {om_joint:.3f} ± {om_err:.3f}

Key insights:
1. BAO provides strong constraints on the matter density Ωₘ
2. SNe primarily constrain the absolute distance scale (via M_B)
3. The combination breaks parameter degeneracies effectively
4. DESI 2024 data extends BAO measurements to z>2

Generated plots:
• joint_sne_bao_corner.pdf - Full parameter constraints
""")