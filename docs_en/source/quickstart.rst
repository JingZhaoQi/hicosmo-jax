Quick Start
===========

This guide will get you up and running with HIcosmo for cosmological parameter estimation.

Inference in 3 Lines of Code
----------------------------

HIcosmo's design philosophy is **3 lines of code for core functionality**:

.. code-block:: python

   from hicosmo import hicosmo

   inf = hicosmo(cosmology="LCDM", likelihood="sn", free_params=["H0", "Omega_m"])
   samples = inf.run()

After completion, you will obtain posterior distribution samples for :math:`H_0` and :math:`\Omega_m`.

.. tip::

   **Parallel acceleration**: Add the following code at the very beginning of your script to enable multi-device parallelism:

   .. code-block:: python

      import hicosmo as hc
      hc.init(8)  # 8 parallel devices = 8 parallel chains = ~800% CPU

   This must be called before importing any other modules!

Basic Examples
--------------

Single-Probe Analysis (Supernovae)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo import hicosmo

   # Constrain LCDM model using Pantheon+ supernova data
   inf = hicosmo(
       cosmology="LCDM",
       likelihood="sn",           # Pantheon+ data
       free_params=["H0", "Omega_m"],
       num_samples=4000,          # Total number of samples
       num_chains=4               # Number of parallel chains
   )

   # Run MCMC sampling
   samples = inf.run()

   # View results summary
   inf.summary()

   # Generate corner plot
   inf.corner_plot('corner_sn.pdf')

Multi-Probe Joint Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo import hicosmo

   # Joint SN + BAO + CMB constraints
   inf = hicosmo(
       cosmology="LCDM",
       likelihood=["sn", "bao", "cmb"],  # Multi-probe joint analysis
       free_params=["H0", "Omega_m"],
       num_samples=8000,
       num_chains=4
   )

   samples = inf.run()
   inf.summary()
   inf.corner_plot('corner_joint.pdf')

Supported Likelihood Strings
-----------------------------

HIcosmo supports the following convenient likelihood strings:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - String
     - Corresponding Class
     - Description
   * - ``sn``
     - ``SN_likelihood("pantheon+")``
     - Pantheon+ 1701 SNe Ia
   * - ``sn_shoes``
     - ``SN_likelihood("pantheon+shoes")``
     - With Cepheid calibration
   * - ``bao``
     - ``BAO_likelihood("desi2024")``
     - DESI 2024 BAO
   * - ``bao_sdss_dr12``
     - ``BAO_likelihood("sdss_dr12")``
     - SDSS DR12 BAO
   * - ``cmb``
     - ``Planck2018DistancePriorsLikelihood``
     - Planck 2018 compressed likelihood
   * - ``h0licow``
     - ``H0LiCOWLikelihood``
     - Strong lensing time delay
   * - ``tdcosmo``
     - ``TDCOSMOLikelihood``
     - TDCOSMO hierarchical Bayesian
   * - ``sh0es``
     - ``SH0ESLikelihood``
     - SH0ES local :math:`H_0`

Extended Dark Energy Models
---------------------------

**wCDM model** (constant dark energy equation of state):

.. code-block:: python

   inf = hicosmo(
       cosmology="wCDM",
       likelihood=["sn", "bao", "cmb"],
       free_params=["H0", "Omega_m", "w0"],
       num_samples=10000
   )
   samples = inf.run()

**CPL parameterization** (time-varying dark energy):

.. math::

   w(a) = w_0 + w_a (1 - a)

.. code-block:: python

   inf = hicosmo(
       cosmology="CPL",
       likelihood=["sn", "bao", "cmb"],
       free_params=["H0", "Omega_m", "w0", "wa"],
       num_samples=16000
   )
   samples = inf.run()

Custom Likelihood Functions
---------------------------

HIcosmo can be used not only for cosmological analysis but also for arbitrary parameter fitting problems. The following example demonstrates how to fit a quadratic polynomial :math:`y = ax^2 + bx + c`:

.. code-block:: python

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
       plotter.report()

**Parameter format explanation**:

.. code-block:: python

   # Dict format: {name: (initial, min, max, latex)}
   params = {
       'a': (3.5, 0.0, 10.0, r'$a$'),  # Parameter a: initial 3.5, range [0, 10]
       'b': (1.0, 0.0, 4.0,  r'$b$'),  # Parameter b: initial 1.0, range [0, 4]
       'c': (1.0, 0.0, 3.0,  r'$c$'),  # Parameter c: initial 1.0, range [0, 3]
   }

**Output**:

.. code-block:: text

   INFO:hicosmo.samplers.init:✅ HIcosmo: 8 devices ready for parallel MCMC
   INFO:hicosmo.samplers.inference:MCMC completed:
   INFO:hicosmo.samplers.inference:  Start time: 2026-01-08 16:54:43
   INFO:hicosmo.samplers.inference:  End time:   2026-01-08 16:54:45
   INFO:hicosmo.samplers.inference:  Duration:   1.94s
   INFO:hicosmo.samplers.inference:  Saved to:   mcmc_chains/polynomial_fit.h5

.. tip::

   The generated report (``plotter.report()``) includes:

   - Parameter constraints (mean, median, 68%/95% confidence intervals)
   - LaTeX-formatted output (ready for papers)
   - MCMC diagnostics (ESS, R-hat)
   - Model selection criteria (AIC, Bayesian evidence)

Using the Class Interface (Advanced)
------------------------------------

For scenarios requiring more control, you can use the class interface directly:

.. code-block:: python

   # 1. Initialize first (must be at the very beginning!)
   import hicosmo as hc
   hc.init(8)  # 8 parallel devices

   # 2. Import other modules
   from hicosmo.models import LCDM
   from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
   from hicosmo.samplers import MCMC

   # 3. Create likelihood objects
   sne = SN_likelihood(LCDM, "pantheon+", M_B="marginalize")
   bao = BAO_likelihood(LCDM, "desi2024")

   # Joint likelihood
   joint_likelihood = sne + bao

   # 4. Configure parameters
   params = {
       "H0": (70.0, 60.0, 80.0),       # (reference, min, max)
       "Omega_m": (0.3, 0.1, 0.5),
   }

   # 5. Create MCMC and run (num_chains defaults to device count = 8)
   mcmc = MCMC(params, joint_likelihood, chain_name="lcdm_sn_bao")
   samples = mcmc.run(num_samples=4000)

   # 6. Output results
   mcmc.print_summary()

Visualizing Results
-------------------

**Corner Plot**:

.. code-block:: python

   from hicosmo.visualization import Plotter

   # Load from saved chains
   plotter = Plotter("lcdm_sn_bao")
   plotter.corner(["H0", "Omega_m"], filename="corner.pdf")

**Multi-chain comparison**:

.. code-block:: python

   plotter = Plotter(
       ["chain_planck", "chain_shoes", "chain_bao"],
       chain_labels=["Planck", "SH0ES", "DESI BAO"]
   )
   plotter.corner(["H0", "Omega_m"], filename="h0_tension.pdf")

Saving and Loading Results
--------------------------

.. code-block:: python

   # Save results
   inf.save_results("results/my_analysis.pkl")

   # Load and continue analysis
   from hicosmo.visualization import Plotter
   plotter = Plotter("results/my_analysis.pkl")
   plotter.corner(["H0", "Omega_m"])

YAML Configuration-Driven
--------------------------

For reproducible analyses, YAML configuration files are recommended:

**config.yaml**:

.. code-block:: yaml

   name: joint_analysis
   theory: LCDM

   likelihood:
     - name: sn
       dataset: pantheon+
     - name: bao
       dataset: desi2024
     - name: cmb
       dataset: planck2018

   params:
     H0:
       prior: {dist: uniform, min: 50, max: 100}
       ref: 70.0
     Omega_m:
       prior: {dist: uniform, min: 0.1, max: 0.5}
       ref: 0.3

   sampler:
     name: numpyro
     num_samples: 8000
     num_chains: 4
     num_warmup: 1000

   output:
     root: results
     chain_name: joint_lcdm

**Running from configuration**:

.. code-block:: python

   from hicosmo import run_from_config

   run_from_config("config.yaml")

Next Steps
----------

- Read `Core Concepts <concepts.html>`_ to understand the HIcosmo architecture
- See `Likelihood Functions <guides/likelihoods.html>`_ for more datasets
- Learn about `Visualization <guides/visualization.html>`_ for publication-quality plots
- Explore `Fisher Forecasting <guides/fisher.html>`_ for survey predictions
