Sampling and Inference
======================

HIcosmo provides a flexible MCMC sampling system with support for multiple backend samplers, automatic parameter management, and a checkpoint system.

Sampler Overview
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Sampler
     - Method
     - Features
   * - **NumPyro NUTS**
     - Hamiltonian Monte Carlo
     - Default, gradient-based, efficient
   * - **emcee**
     - Ensemble sampler
     - Robust, gradient-free, handles NaN/Inf

Quick Start
-----------

**4 lines to complete inference** (with parallel initialization):

.. code-block:: python

   import hicosmo as hc
   hc.init(8)  # 8 parallel devices -> 8 chains running simultaneously

   from hicosmo import hicosmo
   inf = hicosmo(cosmology="LCDM", likelihood="sn", free_params=["H0", "Omega_m"])
   samples = inf.run()  # Automatic num_chains=8

.. note::

   **New API (v2.0+)**: ``hc.init(N)`` directly sets N parallel devices, defaulting ``num_chains=N``.

   - ``hc.init(8)`` -> 8 devices, 8 parallel chains, ~800% CPU usage
   - ``hc.init()`` -> Auto-detect (up to 8 devices)
   - ``hc.init("GPU")`` -> GPU mode

   **Lazy loading**: ``import hicosmo`` does not trigger JAX import, ensuring ``hc.init()`` can correctly set the device count.

High-Level API
--------------

``hicosmo()`` Function
~~~~~~~~~~~~~~~~~~~~~~

``hicosmo()`` is HIcosmo's core entry point, providing the most concise API:

.. code-block:: python

   from hicosmo import hicosmo

   inf = hicosmo(
       cosmology="LCDM",                    # Cosmological model
       likelihood=["sn", "bao", "cmb"],    # Likelihood functions (can be a list)
       free_params=["H0", "Omega_m"],      # Free parameters
       num_samples=4000,                   # Total number of samples
       num_chains=4                        # Number of parallel chains
   )

   # Run sampling
   samples = inf.run()

   # View results
   inf.summary()

   # Save corner plot
   inf.corner_plot("corner.pdf")

**Available cosmology strings**:

- ``"LCDM"``: Standard :math:`\Lambda\text{CDM}` model
- ``"wCDM"``: Constant dark energy equation of state
- ``"CPL"``: Chevallier-Polarski-Linder parameterization
- ``"ILCDM"``: Interacting dark energy model

**Available likelihood strings**:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - String
     - Corresponding Likelihood
   * - ``"sn"``
     - Pantheon+ supernovae
   * - ``"sn_shoes"``
     - Pantheon+SH0ES
   * - ``"bao"``
     - DESI 2024 BAO
   * - ``"cmb"``
     - Planck 2018 distance priors
   * - ``"h0licow"``
     - H0LiCOW strong lensing
   * - ``"tdcosmo"``
     - TDCOSMO hierarchical Bayesian

MCMC Class Interface
---------------------

For scenarios requiring more control, use the ``MCMC`` class:

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from hicosmo.samplers import MCMC
   from hicosmo.models import LCDM
   from hicosmo.likelihoods import SN_likelihood

   # Create likelihood
   sn = SN_likelihood(LCDM, "pantheon+")

   # Parameter configuration: (reference value, minimum, maximum)
   params = {
       "H0": (70.0, 60.0, 80.0),
       "Omega_m": (0.3, 0.1, 0.5),
   }

   # Create MCMC
   mcmc = MCMC(params, sn, chain_name="lcdm_sn")

   # Run
   samples = mcmc.run(num_samples=4000, num_chains=4)

   # View results
   mcmc.print_summary()

Parameter Configuration Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HIcosmo supports multiple parameter configuration formats:

**Tuple format** (simplest):

.. code-block:: python

   params = {
       'H0': (70.0, 60.0, 80.0),      # (reference value, minimum, maximum)
       'Omega_m': (0.3, 0.1, 0.5),
   }

**Dictionary format** (full control):

.. code-block:: python

   params = {
       'H0': {
           'prior': {'dist': 'uniform', 'min': 60, 'max': 80},
           'ref': 70.0,
           'latex': r'$H_0$'
       },
       'Omega_m': {
           'prior': {'dist': 'normal', 'loc': 0.3, 'scale': 0.05},
           'bounds': [0.1, 0.5],
           'latex': r'$\Omega_m$'
       }
   }

**Supported prior distributions**:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Distribution
     - Parameters
     - Description
   * - ``uniform``
     - ``min``, ``max``
     - Uniform distribution
   * - ``normal``
     - ``loc``, ``scale``
     - Normal distribution
   * - ``truncated_normal``
     - ``loc``, ``scale``, ``low``, ``high``
     - Truncated normal distribution

Multiple Sampler Backends
--------------------------

NumPyro NUTS (Default)
~~~~~~~~~~~~~~~~~~~~~~

NumPyro NUTS is the default sampler, using Hamiltonian Monte Carlo:

.. code-block:: python

   mcmc = MCMC(
       params, likelihood,
       sampler='numpyro',     # Default
       chain_method='auto'    # Automatic parallelization method
   )

**Features**:

- Gradient-based NUTS sampler
- JAX JIT compilation acceleration
- Efficient handling of high-dimensional parameter spaces
- Automatic step size and mass matrix tuning

emcee Sampler
~~~~~~~~~~~~~

emcee is an ensemble sampler suitable for complex likelihood functions:

.. code-block:: python

   mcmc = MCMC(
       params, likelihood,
       sampler='emcee'
   )

**Features**:

- No gradient information required
- Robust handling of NaN/Inf
- Suitable for multimodal distributions
- More forgiving of numerical instability in likelihood functions

**Selection guide**:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Scenario
     - Recommended Sampler
     - Reason
   * - Standard cosmological analysis
     - NumPyro NUTS
     - Efficient, fast convergence
   * - Unstable likelihood function
     - emcee
     - Robust handling of outliers
   * - High-dimensional parameters (>20)
     - NumPyro NUTS
     - Gradient information accelerates convergence
   * - Multimodal distribution
     - emcee
     - Ensemble sampling avoids getting trapped in local optima

Run Configuration
-----------------

Sampling Parameters
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   samples = mcmc.run(
       num_samples=4000,      # Total number of samples (all chains combined)
       num_chains=4,          # Number of parallel chains
       num_warmup=1000,       # Warmup steps PER CHAIN
       progress_bar=True      # Show progress bar
   )

**The two count parameters have different semantics**:

- ``num_samples`` is the **total** across all chains: 4 chains with 4000
  samples means 1000 samples per chain.
- ``num_warmup`` is **per chain**: warmup is each chain's independent
  mass-matrix and step-size adaptation and cannot be amortized across
  chains. When unspecified it defaults to
  ``max(200, min(1000, samples_per_chain))``.

.. note::

   When ``progress_bar`` is not given explicitly, the bar is shown only on
   interactive terminals (TTY) and auto-disabled when output is redirected
   or running in batch jobs. NumPyro's progress bar triggers a host
   callback every step, costing 5–20% for millisecond-scale likelihoods,
   and floods log files.

Chain Parallelization Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   mcmc = MCMC(
       params, likelihood,
       chain_method='auto'    # Automatic selection
   )

**Options**:

- ``'auto'``: Automatic selection (recommended)
- ``'vectorized'``: Parallelize using vmap (multi-core CPU)
- ``'sequential'``: Sequential execution (single-threaded)
- ``'parallel'``: Use pmap (multi-GPU)

Parallel Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. important::

   **New API (v2.0+)**: ``hc.init(N)`` directly sets N parallel devices, ``num_chains`` defaults to match the device count.

**Recommended usage**:

.. code-block:: python

   import hicosmo as hc

   # Most common: 8 devices, 8 parallel chains
   hc.init(8)

   # Auto-detect (up to 8 devices)
   hc.init()

   # GPU mode
   hc.init("GPU")

   # Then import other modules
   from hicosmo import hicosmo

**Complete example**:

.. code-block:: python

   import hicosmo as hc
   hc.init(8)  # Must be at the very beginning!

   from hicosmo.samplers import MCMC
   from hicosmo.models import LCDM
   from hicosmo.likelihoods import SN_likelihood

   likelihood = SN_likelihood(LCDM, "pantheon+")
   params = {'H0': (70.0, 60.0, 80.0), 'Omega_m': (0.3, 0.1, 0.5)}

   mcmc = MCMC(params, likelihood)
   # num_chains defaults to 8 (matching device count)
   samples = mcmc.run(num_samples=8000)

.. warning::

   **JAX already imported warning**: If JAX has already been imported, the device count cannot be changed.
   Ensure ``hc.init()`` is called at the very beginning of the script (before any ``import jax``).

**API reference table**:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Call
     - Effect
   * - ``hc.init(8)``
     - 8 devices, default 8 parallel chains
   * - ``hc.init(4)``
     - 4 devices, default 4 parallel chains
   * - ``hc.init()``
     - Auto-detect (up to 8)
   * - ``hc.init("GPU")``
     - GPU mode, auto-detect number of GPUs

Checkpoint System
-----------------

HIcosmo provides a complete checkpoint and resume system (with time-driven saving).

Automatic Checkpoints
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   mcmc = MCMC(
       params, likelihood,
       enable_checkpoints=True,           # Enable checkpoints
       checkpoint_interval_seconds=600,   # Save every 10 minutes (recommended)
       checkpoint_dir="checkpoints"       # Save directory
   )

.. note::

   ``checkpoint_interval`` is still available, but it represents the **total** number of samples across all chains.
   If using ``checkpoint_interval``, estimate in conjunction with ``num_chains``.

Flowchart (Time-Driven Saving)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   [Start]
      |
   [MCMC.run()]
      |
   [Segmented sampling chunk]
      |
   [Time interval reached?]
      +-- No -> Continue sampling
      +-- Yes -> Save checkpoint (chain_step_N.h5)
      |
   [Total samples reached]
      |
   [Save final checkpoint + latest alias chain.h5]
      |
   [End]

Resume from Checkpoint
~~~~~~~~~~~~~~~~~~~~~~

The recommended approach is to use the explicit ``MCMC.resume``:

.. code-block:: python

   # Resume from checkpoint (requires providing likelihood)
   mcmc = MCMC.resume("checkpoints/chain_step_500.h5", likelihood)
   samples = mcmc.run()  # Continue until remaining samples are completed

.. note::

   If the checkpoint contains internal state (NumPyro/NUTS), **true resumption** is achieved; otherwise, new samples are appended after existing ones.

Manual Checkpoint Loading
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # List available checkpoints (requires chain_name and checkpoint_dir)
   mcmc = MCMC(params, likelihood, chain_name="chain", checkpoint_dir="checkpoints")
   mcmc.list_checkpoints()

   # Resume from checkpoint (.h5)
   mcmc = MCMC.resume("checkpoints/chain_step_500.h5", likelihood)
   mcmc.run()

Convergence Diagnostics
------------------------

Gelman-Rubin Statistic
~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \hat{R} = \sqrt{\frac{\hat{V}}{W}}

where :math:`\hat{V}` is the total variance estimate and :math:`W` is the within-chain variance.

**Convergence criterion**: :math:`\hat{R} < 1.01`

.. code-block:: python

   # Print diagnostic information (includes R-hat)
   mcmc.print_summary()

   # Output example:
   # Parameter   Mean    Std    R-hat   ESS
   # -----------------------------------------
   # H0         67.36   0.42   1.002  2341
   # Omega_m    0.315   0.007  1.001  2156

Effective Sample Size (ESS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The effective sample size reflects the number of independent samples:

.. math::

   \text{ESS} = \frac{N}{1 + 2\sum_{k=1}^K \rho_k}

where :math:`\rho_k` is the autocorrelation coefficient at lag :math:`k`.

**Recommendation**: ESS > 100 per parameter

Likelihood Diagnostics
~~~~~~~~~~~~~~~~~~~~~~

Check the numerical stability of the likelihood function before running:

.. code-block:: python

   from hicosmo.samplers import LikelihoodDiagnostics

   diagnostics = LikelihoodDiagnostics(likelihood, params)
   result = diagnostics.run(n_tests=100)
   diagnostics.print_report(result)

   # If success rate is low, consider using emcee
   if result.success_rate < 0.5:
       mcmc = MCMC(params, likelihood, sampler='emcee')

Result Handling
---------------

Getting Samples
~~~~~~~~~~~~~~~

.. code-block:: python

   # Run MCMC
   samples = mcmc.run(num_samples=4000, num_chains=4)

   # Get parameter samples
   H0_samples = samples['H0']
   Omega_m_samples = samples['Omega_m']

   # Compute statistics
   import numpy as np
   print(f"H0 = {np.mean(H0_samples):.2f} +/- {np.std(H0_samples):.2f}")

Saving Results
~~~~~~~~~~~~~~

.. code-block:: python

   # Save to file
   mcmc.save_results("results/lcdm_sn.pkl")

   # Load results
   from hicosmo.samplers import MCMC
   loaded = MCMC.load_results("results/lcdm_sn.pkl")

GetDist Compatibility
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from getdist import MCSamples

   # Convert to GetDist format
   gd_samples = MCSamples(
       samples=[samples['H0'], samples['Omega_m']],
       names=['H0', 'Omega_m'],
       labels=[r'H_0', r'\Omega_m']
   )

   # Plot using GetDist
   from getdist import plots
   g = plots.get_subplot_plotter()
   g.triangle_plot(gd_samples, ['H0', 'Omega_m'])

Automatic Nuisance Parameter Collection
-----------------------------------------

MCMC automatically collects nuisance parameters from likelihood functions:

.. code-block:: python

   from hicosmo.likelihoods import TDCOSMO
   from hicosmo.samplers import MCMC

   tdcosmo = TDCOSMO(LCDM)

   # Only specify cosmological parameters
   cosmo_params = {
       'H0': (70.0, 60.0, 80.0),
       'Omega_m': (0.3, 0.1, 0.5),
   }

   # MCMC automatically collects TDCOSMO's nuisance parameters
   mcmc = MCMC(cosmo_params, tdcosmo)

   # Print all parameters (including automatically collected ones)
   print(mcmc.param_names)
   # ['H0', 'Omega_m', 'lambda_int_mean', 'lambda_int_sigma', ...]

Performance Optimization
-------------------------

Initialization Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For complex problems, use optimization to find the initial point:

.. code-block:: python

   mcmc = MCMC(
       params, likelihood,
       optimize_init=True,            # Optimize initial point
       max_opt_iterations=500,        # Maximum iterations
       opt_learning_rate=0.01         # Learning rate
   )

**Applicable scenarios**:

- Likelihood function evaluation > 10ms
- Parameter dimensionality > 20
- Multimodal problems

JIT Warmup
~~~~~~~~~~

JAX compiles functions on first run, so it is recommended to:

.. code-block:: python

   # Warmup (optional but recommended)
   _ = likelihood(H0=70, Omega_m=0.3)

   # Actual run
   samples = mcmc.run()

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Configuration
     - qcosmc (scipy)
     - HIcosmo (JAX)
     - Speedup
   * - LCDM + Pantheon+ (10k samples)
     - 180s
     - 45s
     - **4x**
   * - CPL + BAO + SN (10k samples)
     - 420s
     - 85s
     - **5x**
   * - 4 parallel chains (8-core CPU)
     - N/A
     - Automatic
     - --

Complete Example
----------------

**Complete example using the new API**:

.. code-block:: python

   # 1. Initialize first (must be at the very beginning!)
   import hicosmo as hc
   hc.init(8)  # 8 parallel devices = 8 parallel chains

   # 2. Import other modules
   from hicosmo.samplers import MCMC
   from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
   from hicosmo.models import LCDM

   # 3. Create joint likelihood
   sne = SN_likelihood(LCDM, "pantheon+")
   bao = BAO_likelihood(LCDM, "desi2024")

   def joint_likelihood(**params):
       return sne(**params) + bao(**params)

   # 4. Parameter configuration
   params = {
       'H0': {
           'prior': {'dist': 'uniform', 'min': 60, 'max': 80},
           'ref': 70.0,
           'latex': r'$H_0$'
       },
       'Omega_m': {
           'prior': {'dist': 'uniform', 'min': 0.1, 'max': 0.5},
           'ref': 0.3,
           'latex': r'$\Omega_m$'
       }
   }

   # 5. Create MCMC
   mcmc = MCMC(
       params,
       joint_likelihood,
       chain_name="lcdm_joint",
       enable_checkpoints=True
   )

   # 6. Run sampling (num_chains defaults to device count = 8)
   samples = mcmc.run(num_samples=8000)

   # 7. View results
   mcmc.print_summary()
   mcmc.save_results("results/lcdm_joint.pkl")

**Minimal example** (3 lines of code):

.. code-block:: python

   import hicosmo as hc
   hc.init(8)  # 8 parallel devices

   from hicosmo import hicosmo
   inf = hicosmo("LCDM", ["sn", "bao"], ["H0", "Omega_m"])
   samples = inf.run()

Next Steps
----------

- `Visualization <visualization.html>`_: Generate publication-quality plots
- `Fisher Forecasts <fisher.html>`_: Perform survey predictions
