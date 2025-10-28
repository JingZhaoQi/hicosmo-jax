#!/usr/bin/env python3
"""
Test BAO likelihood with MCMC sampling.

This script demonstrates:
1. Using individual BAO datasets
2. Combining multiple BAO datasets
3. Joint analysis with SNe+BAO
"""

import os
import numpy as np
from hicosmo.samplers import init_hicosmo, MCMC, ParameterConfig
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods import (
    PantheonPlusLikelihood,
    SDSSDR12BAO, BOSSDR12BAO, DESI2024BAO, SixDFBAO,
    BAOCollection, get_available_datasets
)
from hicosmo.visualization import HIcosmoViz

# Initialize HIcosmo
init_hicosmo()

print("=" * 60)
print("HIcosmo BAO Likelihood Test")
print("=" * 60)

# Show available BAO datasets
print("\n📊 Available BAO datasets:")
for dataset in get_available_datasets():
    print(f"   - {dataset}")

# Test 1: Single BAO dataset (SDSS DR12)
print("\n" + "=" * 60)
print("Test 1: SDSS DR12 BAO Only")
print("=" * 60)

sdss_bao = SDSSDR12BAO(verbose=True)

# Define parameters for BAO-only analysis
params_bao = {
    'H0': {70, 55, 85},
    'Omega_m': {0.3, 0.2, 0.4}
}

# Create likelihood function
def sdss_likelihood(H0, Omega_m):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return sdss_bao.log_likelihood(model)

# Run MCMC
config = ParameterConfig(params_bao, mcmc={'num_samples': 500, 'num_chains': 2})
samples_sdss = MCMC(config, sdss_likelihood).run()

# Show results
print("\n📈 SDSS DR12 Results:")
for param in params_bao:
    mean = np.mean(samples_sdss[param])
    std = np.std(samples_sdss[param])
    print(f"   {param}: {mean:.2f} ± {std:.2f}")

# Test 2: Multiple BAO datasets combined
print("\n" + "=" * 60)
print("Test 2: Combined BAO (BOSS + 6dF)")
print("=" * 60)

# Create collection of BAO datasets
bao_collection = BAOCollection(
    datasets=['boss_dr12', 'sixdf'],
    verbose=True
)

# Likelihood function for combined BAO
def combined_bao_likelihood(H0, Omega_m):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return bao_collection.log_likelihood(model)

# Run MCMC with combined BAO
samples_combined = MCMC(
    ParameterConfig(params_bao, mcmc={'num_samples': 500, 'num_chains': 2}),
    combined_bao_likelihood
).run()

print("\n📈 Combined BAO Results:")
for param in params_bao:
    mean = np.mean(samples_combined[param])
    std = np.std(samples_combined[param])
    print(f"   {param}: {mean:.2f} ± {std:.2f}")

# Test 3: Joint SNe + BAO analysis
print("\n" + "=" * 60)
print("Test 3: Joint PantheonPlus + DESI 2024 BAO")
print("=" * 60)

# Load data
data_path = os.path.join(os.path.dirname(__file__), "data", "DataRelease")
pantheon = PantheonPlusLikelihood(data_path, include_shoes=False, marginalize_M_B=False)
desi_bao = DESI2024BAO(verbose=True)

# Parameters including M_B for SNe
params_joint = {
    'H0': {70, 60, 80},
    'Omega_m': {0.3, 0.2, 0.4},
    'M_B': {-19.25, -20, -18}
}

# Joint likelihood
def joint_likelihood(H0, Omega_m, M_B):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    log_like_sne = pantheon.log_likelihood(model, M_B=M_B)
    log_like_bao = desi_bao.log_likelihood(model)
    return log_like_sne + log_like_bao

# Run MCMC with joint analysis
print("\n🚀 Running joint SNe+BAO MCMC...")
config_joint = ParameterConfig(params_joint, mcmc={'num_samples': 1000, 'num_chains': 4})
samples_joint = MCMC(config_joint, joint_likelihood).run()

print("\n📈 Joint SNe+BAO Results:")
for param in params_joint:
    mean = np.mean(samples_joint[param])
    std = np.std(samples_joint[param])
    print(f"   {param}: {mean:.2f} ± {std:.2f}")

# Test 4: Using all available BAO datasets
print("\n" + "=" * 60)
print("Test 4: All BAO Datasets Combined")
print("=" * 60)

# Create collection with all datasets
all_bao = BAOCollection(
    datasets=get_available_datasets(),
    verbose=True
)

def all_bao_likelihood(H0, Omega_m):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return all_bao.log_likelihood(model)

# Quick test with fewer samples
config_all = ParameterConfig(params_bao, mcmc={'num_samples': 200, 'num_chains': 2})
samples_all = MCMC(config_all, all_bao_likelihood).run()

print("\n📈 All BAO Results:")
for param in params_bao:
    mean = np.mean(samples_all[param])
    std = np.std(samples_all[param])
    print(f"   {param}: {mean:.2f} ± {std:.2f}")

# Create visualization
viz = HIcosmoViz()
print("\n📊 Creating corner plots...")

# Corner plot for joint analysis
viz.corner(
    samples_joint,
    params=list(params_joint.keys()),
    filename="bao_joint_corner.pdf"
)

# Corner plot comparing different BAO combinations
viz.corner_compare(
    [samples_sdss, samples_combined, samples_joint],
    params=['H0', 'Omega_m'],
    labels=['SDSS DR12', 'BOSS+6dF', 'SNe+DESI'],
    filename="bao_comparison.pdf"
)

print("\n✅ BAO likelihood test complete!")
print("Generated plots:")
print("   - bao_joint_corner.pdf: Joint SNe+BAO constraints")
print("   - bao_comparison.pdf: Comparison of different BAO combinations")

# Summary statistics
print("\n" + "=" * 60)
print("Summary: Constraint Improvements")
print("=" * 60)

def print_constraint_comparison(name1, samples1, name2, samples2, param='H0'):
    std1 = np.std(samples1[param])
    std2 = np.std(samples2[param])
    improvement = (1 - std2/std1) * 100
    print(f"{param}: {name1} σ={std1:.2f} → {name2} σ={std2:.2f} "
          f"({improvement:+.1f}% improvement)")

print("\n📊 Adding BAO to cosmological constraints:")
print_constraint_comparison("SDSS alone", samples_sdss,
                           "Combined BAO", samples_combined, 'H0')
print_constraint_comparison("SDSS alone", samples_sdss,
                           "Combined BAO", samples_combined, 'Omega_m')

print("\n✨ Test demonstrates:")
print("   1. Individual BAO dataset usage")
print("   2. Multiple BAO dataset combination")
print("   3. Joint SNe+BAO analysis")
print("   4. Constraint improvements from data combination")