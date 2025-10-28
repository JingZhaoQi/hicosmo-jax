#!/usr/bin/env python3
"""HIcosmo H0LiCOW likelihood quick-look analysis."""

import numpy as np

from hicosmo.likelihoods.h0licow import H0LiCOWLikelihood
from hicosmo.models.lcdm import LCDM

print("=" * 60)
print("HIcosmo H0LiCOW Analysis")
print("=" * 60)

likelihood = H0LiCOWLikelihood()
print(f"Loaded {len(likelihood.lenses)} lens systems: {[lens.name for lens in likelihood.lenses]}")

h_grid = np.linspace(60.0, 85.0, 251)
loglikes = np.array([likelihood.log_likelihood(LCDM(H0=float(h0), Omega_m=0.3)) for h0 in h_grid])

best_index = int(np.argmax(loglikes))
best_h0 = float(h_grid[best_index])
max_loglike = float(loglikes[best_index])

# Estimate 1-sigma range from delta log-likelihood = 0.5 (equivalent to Δχ² = 1)
deltas = max_loglike - loglikes

sigma_minus = np.nan
for idx in range(best_index, -1, -1):
    if idx != best_index and deltas[idx] > 0.5:
        break
    sigma_minus = best_h0 - float(h_grid[idx])

sigma_plus = np.nan
for idx in range(best_index, len(h_grid)):
    if idx != best_index and deltas[idx] > 0.5:
        break
    sigma_plus = float(h_grid[idx]) - best_h0

print("\nGrid scan results (Ω_m fixed to 0.3):")
print(f"  Max log-likelihood at H0 = {best_h0:.2f} km/s/Mpc")
print(f"  Approximate 1σ range: -{sigma_minus:.2f} / +{sigma_plus:.2f} km/s/Mpc")

print("\nReference (Wong et al. 2020, H0LiCOW): H0 = 73.3 ± 1.8 km/s/Mpc")
print("This simple grid scan should fall within that interval.")
