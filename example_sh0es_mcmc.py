#!/usr/bin/env python3
"""MCMC on H0 using the SH0ES Gaussian prior likelihood."""

import numpy as np

from hicosmo.samplers import init_hicosmo, MCMC, ParameterConfig
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods import SH0ESLikelihood
from hicosmo.visualization import HIcosmoViz


def main():
    init_hicosmo()
    like = SH0ESLikelihood()

    def loglike(H0):
        model = LCDM(H0=H0, Omega_m=0.3, Omega_b=0.05)
        return like.log_likelihood(model)

    params = {
        "H0": {73.0, 65.0, 80.0},
    }

    config = ParameterConfig(params, mcmc={"num_samples": 4000, "num_chains": 4})
    sampler = MCMC(config, loglike)
    samples = sampler.run()

    h0_samples = samples["H0"]
    median = np.median(h0_samples)
    lower = median - np.percentile(h0_samples, 16)
    upper = np.percentile(h0_samples, 84) - median
    print(f"H0 = {median:.2f} -{lower:.2f} +{upper:.2f} km/s/Mpc")

    viz = HIcosmoViz()
    viz.corner(samples, params=["H0"], filename="results/sh0es_h0_corner.pdf")
    print("Corner plot saved to results/sh0es_h0_corner.pdf")


if __name__ == "__main__":
    main()
