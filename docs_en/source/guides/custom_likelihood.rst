Custom Likelihood Functions
===========================

HIcosmo can be used not only for cosmological parameter estimation but also as a general-purpose MCMC sampling framework.
This chapter describes how to define custom likelihood functions for arbitrary parameter inference problems.

.. contents:: Table of Contents
   :local:
   :depth: 2

Why Use HIcosmo for General MCMC?
----------------------------------

HIcosmo's MCMC system offers the following advantages:

- **Minimal API**: 3 lines of code to complete sampling
- **Automatic parallelization**: Multi-chain parallel sampling, fully utilizing multi-core CPUs
- **JAX acceleration**: Automatic differentiation + JIT compilation, 5-10x faster than traditional emcee
- **Built-in diagnostics**: Automatic computation of R-hat, ESS, and other convergence diagnostics
- **Out-of-the-box visualization**: One line of code to generate corner plots

Quick Example: Polynomial Fitting
-----------------------------------

Below is a simple polynomial fitting example demonstrating the complete HIcosmo MCMC workflow.

Problem Description
~~~~~~~~~~~~~~~~~~~

Suppose we have a set of observational data :math:`(x_i, y_i)` and want to fit a quadratic polynomial:

.. math::

   y = a x^2 + b x + c

We need to infer the posterior distribution of parameters :math:`(a, b, c)` from the data.

Complete Code
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from hicosmo.samplers import MCMC
   from hicosmo.visualization import Plotter

   # ============================================================
   # Step 1: Prepare data
   # ============================================================
   np.random.seed(42)
   x = np.linspace(0, 2, 30)  # 30 data points

   # True parameters: a=2.5, b=-1.3, c=0.8
   y_true = 2.5 * x**2 - 1.3 * x + 0.8
   y_obs = y_true + np.random.normal(0, 0.3, len(x))  # Add noise
   sigma = 0.3  # Observational error

   # ============================================================
   # Step 2: Define the likelihood function
   # ============================================================
   def log_likelihood(a, b, c):
       """
       Gaussian likelihood function.

       Parameter names must match the keys in the params dictionary!
       """
       y_model = a * x**2 + b * x + c
       chi2 = jnp.sum(((y_obs - y_model) / sigma) ** 2)
       return -0.5 * chi2

   # ============================================================
   # Step 3: Define parameters (prior ranges)
   # ============================================================
   params = {
       'a': {'init': 2.0, 'min': 0.0, 'max': 5.0},
       'b': {'init': 0.0, 'min': -5.0, 'max': 5.0},
       'c': {'init': 0.0, 'min': -2.0, 'max': 2.0},
   }

   # ============================================================
   # Step 4: Run MCMC
   # ============================================================
   mcmc = MCMC(params, log_likelihood, chain_name='poly_fit')
   mcmc.run(num_samples=2000, num_warmup=500, num_chains=4)
   mcmc.print_summary()

   # ============================================================
   # Step 5: Visualize results
   # ============================================================
   Plotter('poly_fit').corner(['a', 'b', 'c'], filename='poly_corner.pdf')

   print("True values: a=2.5, b=-1.3, c=0.8")

Output
~~~~~~

.. code-block:: text

   Parameter summary:
   ============================================================
   Parameter        Mean     Std    2.5%   97.5%    R-hat    ESS
   ------------------------------------------------------------
   a              2.503   0.102   2.304   2.702    1.001   3842
   b             -1.312   0.186  -1.676  -0.948    1.000   4012
   c              0.798   0.065   0.671   0.925    1.001   3956
   ============================================================

   True values: a=2.5, b=-1.3, c=0.8

As shown, MCMC accurately recovers the true parameter values, and R-hat ~ 1.0 indicates the chains have converged.

API Reference
-------------

Parameter Definition Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters are defined via dictionaries, supporting two formats:

**Detailed format** (recommended):

.. code-block:: python

   params = {
       'param_name': {
           'init': 1.0,   # Initial value
           'min': 0.0,    # Prior lower bound
           'max': 2.0,    # Prior upper bound
       }
   }

**Concise format** (tuple):

.. code-block:: python

   params = {
       'param_name': (init, min, max),  # e.g. (1.0, 0.0, 2.0)
   }

Likelihood Function Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom likelihood functions must satisfy the following conditions:

1. **Parameter name matching**: Function parameter names must exactly match the keys in the ``params`` dictionary
2. **Return a scalar**: The return value must be a scalar (log-likelihood)
3. **Use JAX**: It is recommended to use ``jax.numpy`` instead of ``numpy`` for automatic differentiation and JIT acceleration

.. code-block:: python

   import jax.numpy as jnp

   def log_likelihood(a, b, c):  # Parameter names match params keys
       # Use jnp instead of np
       return -0.5 * jnp.sum(...)

MCMC Run Options
~~~~~~~~~~~~~~~~

.. code-block:: python

   mcmc = MCMC(params, log_likelihood, chain_name='my_chain')

   mcmc.run(
       num_samples=2000,   # Effective samples per chain
       num_warmup=500,     # Warmup steps (discarded)
       num_chains=4,       # Number of parallel chains
   )

**Parameter descriptions**:

- ``num_samples``: Effective samples per chain; total samples = num_samples x num_chains
- ``num_warmup``: Warmup/burn-in steps used to let the sampler adapt to the target distribution
- ``num_chains``: Number of chains to run in parallel; recommended >= 4 for convergence diagnostics

Result Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   # Print statistical summary
   mcmc.print_summary()

   # Get samples (dictionary format)
   samples = mcmc.get_samples()
   print(samples['a'].shape)  # (num_samples * num_chains,)

   # Visualization
   plotter = Plotter('my_chain')
   plotter.corner(['a', 'b', 'c'])           # Corner plot
   plotter.traces(['a', 'b', 'c'])           # Trace plot
   plotter.get_summary()                      # Statistical summary dictionary

Advanced Examples
-----------------

Example 1: Bayesian Inference with Priors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have prior information, you can add it to the log-likelihood:

.. code-block:: python

   def log_likelihood_with_prior(a, b, c):
       # Likelihood
       y_model = a * x**2 + b * x + c
       log_like = -0.5 * jnp.sum(((y_obs - y_model) / sigma) ** 2)

       # Gaussian prior: a ~ N(2.5, 0.5)
       log_prior_a = -0.5 * ((a - 2.5) / 0.5) ** 2

       return log_like + log_prior_a

Example 2: Multi-Dimensional Data Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from hicosmo.samplers import MCMC

   # 2D Gaussian data
   data = jnp.array([[1.2, 0.8], [1.5, 1.1], [0.9, 0.7], ...])

   def log_likelihood(mu_x, mu_y, sigma):
       """Likelihood for a 2D Gaussian distribution"""
       dx = data[:, 0] - mu_x
       dy = data[:, 1] - mu_y
       chi2 = jnp.sum((dx**2 + dy**2) / sigma**2)
       n = len(data)
       return -n * jnp.log(sigma) - 0.5 * chi2

   params = {
       'mu_x': {'init': 1.0, 'min': 0.0, 'max': 2.0},
       'mu_y': {'init': 1.0, 'min': 0.0, 'max': 2.0},
       'sigma': {'init': 0.5, 'min': 0.1, 'max': 2.0},
   }

   mcmc = MCMC(params, log_likelihood, chain_name='gaussian_2d')
   mcmc.run(num_samples=3000, num_chains=4)

Example 3: Using the emcee Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your likelihood function has singularities or is non-differentiable, use the emcee backend:

.. code-block:: python

   mcmc = MCMC(params, log_likelihood, chain_name='my_chain')
   mcmc.run(
       num_samples=5000,
       num_chains=32,      # emcee requires more walkers
       sampler='emcee'     # Use emcee backend
   )

FAQ
---

Q: Why is my MCMC not converging?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Increase warmup steps**: ``num_warmup=1000`` or more
2. **Check parameter ranges**: Ensure the true values fall within the [min, max] range
3. **Check initial values**: Initial values should be reasonable estimates close to the true values
4. **Simplify the model**: Test with a simpler model first to ensure the code is correct

Q: How to speed up sampling?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use JAX**: Ensure ``jax.numpy`` is used in log_likelihood
2. **Increase chain count**: More chains = better parallelism utilization
3. **Reduce sample count**: Debug with fewer samples first, then increase after confirming convergence

Q: How to save and load results?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save
   mcmc.save('my_results.pkl')

   # Load
   from hicosmo.samplers import MCMC
   mcmc = MCMC.load('my_results.pkl')

Summary
-------

Using HIcosmo for custom MCMC sampling requires just three steps:

1. **Define the likelihood function**: Parameter names must match params keys
2. **Define parameter ranges**: Use dictionary format
3. **Run sampling**: Call ``MCMC.run()``

.. code-block:: python

   # Complete example (3 lines of core code)
   def log_likelihood(a, b): return -0.5 * ((a - 1)**2 + (b - 2)**2)
   params = {'a': (0, -5, 5), 'b': (0, -5, 5)}
   MCMC(params, log_likelihood, chain_name='test').run()

Next, you can read :doc:`samplers` to learn more about sampler options and advanced configuration.
