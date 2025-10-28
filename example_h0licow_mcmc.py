#!/usr/bin/env python3
"""Ensemble MCMC on H0 and Omega_m using the H0LiCOW likelihood."""

import os
import numpy as np
import emcee

from hicosmo.samplers import init_hicosmo
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods import H0LiCOWLikelihood
from hicosmo.visualization import HIcosmoViz


def log_probability(theta, likelihood):
    H0, Omega_m = theta
    if not (60.0 < H0 < 85.0 and 0.15 < Omega_m < 0.5):
        return -np.inf
    model = LCDM(H0=float(H0), Omega_m=float(Omega_m))
    return float(likelihood.log_likelihood(model))


def run_sampling():
    init_hicosmo()
    likelihood = H0LiCOWLikelihood()

    ndim = 2
    nwalkers = 32
    burn_in = 1000
    production = 4000

    # Initialize walkers around the best-fit grid value
    initial = np.array([73.5, 0.30])
    spread = np.array([0.5, 0.02])
    rng = np.random.default_rng()
    p0 = initial + spread * rng.standard_normal((nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(likelihood,))

    print("Running burn-in...")
    state = sampler.run_mcmc(p0, burn_in, progress=True)
    sampler.reset()

    print("Running production...")
    sampler.run_mcmc(state, production, progress=True)

    samples = sampler.get_chain(flat=True)
    result = {
        "H0": samples[:, 0],
        "Omega_m": samples[:, 1],
    }

    print("\nPosterior summary (median ± 1σ):")
    for key, values in result.items():
        median = float(np.median(values))
        lower = float(median - np.percentile(values, 16))
        upper = float(np.percentile(values, 84) - median)
        print(f"  {key}: {median:.2f} -{lower:.2f} +{upper:.2f}")

    os.makedirs("results", exist_ok=True)
    viz = HIcosmoViz()
    viz.corner(result, params=["H0", "Omega_m"], filename="results/h0licow_corner.pdf")
    print("\nContour plot saved to results/h0licow_corner.pdf")


if __name__ == "__main__":
    run_sampling()
