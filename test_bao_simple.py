#!/usr/bin/env python3
"""
Simple BAO + SNe joint analysis example.
"""

import os
import numpy as np
from hicosmo.samplers import init_hicosmo, MCMC, ParameterConfig
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods import PantheonPlusLikelihood, DESI2024BAO

# Initialize
init_hicosmo()

print("HIcosmo Joint SNe+BAO Analysis")
print("=" * 40)

# Load datasets
data_path = os.path.join(os.path.dirname(__file__), "data", "DataRelease")
sne = PantheonPlusLikelihood(data_path, include_shoes=False, marginalize_M_B=False)
bao = DESI2024BAO(verbose=True)

# Parameters
params = {
    'H0': {70, 60, 80},
    'Omega_m': {0.3, 0.2, 0.4},
    'M_B': {-19.25, -20, -18}
}

# Joint likelihood
def joint_likelihood(H0, Omega_m, M_B):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return sne.log_likelihood(model, M_B=M_B) + bao.log_likelihood(model)

# Run MCMC
print("\n🚀 Running MCMC...")
config = ParameterConfig(params, mcmc={'num_samples': 500, 'num_chains': 2})
samples = MCMC(config, joint_likelihood).run()

# Results
print("\n📊 Joint Constraints:")
for param in params:
    mean = np.mean(samples[param])
    std = np.std(samples[param])
    print(f"  {param}: {mean:.2f} ± {std:.2f}")

print("\n✅ Joint analysis complete!")