#!/usr/bin/env python
"""
Simple MCMC Example: Polynomial Fitting (y = a*x² + b*x + c)

"""
import hicosmo as hc
hc.init()
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from hicosmo.samplers import MCMC
from hicosmo.visualization import Plotter

# Load data
data_path = Path(__file__).parent / 'data' / 'sim_data.txt'
x, y_obs, y_err = np.loadtxt(data_path, unpack=True)
x, y_obs, y_err = jnp.asarray(x), jnp.asarray(y_obs), jnp.asarray(y_err)


def log_likelihood(a, b, c):
    """Log-likelihood: -0.5 * chi²"""
    y_th = a * x**2 + b * x + c
    return -0.5 * jnp.sum((y_obs - y_th)**2 / y_err**2)


# Parameters: {name: (initial, min, max, latex)}
params = {
    'a': (3.5, 0.0, 10.0, r'$a$'),
    'b': (1.0, 0.0, 4.0,  r'$b$'),
    'c': (1.0, 0.0, 3.0,  r'$c$'),
}

if __name__ == '__main__':
    # Run MCMC
    mcmc = MCMC(params, log_likelihood, chain_name='polynomial_fit')
    mcmc.run(num_samples=20000)


    # Plot results

    plotter = Plotter('polynomial_fit')
    plotter.corner()
    plotter.plot_2D(1,2)
    plotter.report()
