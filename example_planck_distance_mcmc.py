#!/usr/bin/env python3
"""MCMC constraints on H0, Omega_m, and Omega_b using Planck 2018 distance priors."""

from hicosmo.samplers import init_hicosmo, MCMC, ParameterConfig
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods import Planck2018DistancePriorsLikelihood
from hicosmo.visualization import HIcosmoViz
import numpy as np


def main():
    init_hicosmo()
    like = Planck2018DistancePriorsLikelihood()

    def loglike(H0, Omega_m, Omega_b):
        model = LCDM(H0=H0, Omega_m=Omega_m, Omega_b=Omega_b)
        return like.log_likelihood(model)

    params = {
        "H0": {67.0, 60.0, 75.0},
        "Omega_m": {0.32, 0.2, 0.4},
        "Omega_b": {0.048, 0.03, 0.07},
    }

    config = ParameterConfig(params, mcmc={"num_samples": 6000, "num_chains": 4})
    sampler = MCMC(config, loglike)
    samples = sampler.run()

    print("Posterior summary (median ±1σ):")
    for key in samples:
        vals = samples[key]
        median = np.median(vals)
        lower = median - np.percentile(vals, 16)
        upper = np.percentile(vals, 84) - median
        print(f"  {key}: {median:.3f} -{lower:.3f} +{upper:.3f}")

    viz = HIcosmoViz()
    viz.corner(samples, params=["H0", "Omega_m", "Omega_b"], filename="results/planck_distance_corner.pdf")
    print("Corner plot saved to results/planck_distance_corner.pdf")


if __name__ == "__main__":
    main()
