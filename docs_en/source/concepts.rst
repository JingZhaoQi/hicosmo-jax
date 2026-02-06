Core Concepts
=============

HIcosmo is a high-performance framework for cosmological parameter estimation. This chapter introduces its core design philosophy and architecture.

Design Philosophy
-----------------

API Simplicity First Principle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HIcosmo's design follows the principle that **API simplicity directly determines project success**:

.. code-block:: text

   API Simplicity -> Lower Learning Curve -> Higher User Adoption -> Stronger User Retention -> Project Success

**Design Standards**:

- **3 lines of code for core functionality** -- this is our API design target
- **Smart defaults** -- 90% of use cases need no extra configuration
- **Progressive complexity** -- simple tasks stay simple; complexity is only exposed for advanced use
- **Never add required parameters for "completeness"** -- each new parameter is a risk of user churn

JAX-First Architecture
~~~~~~~~~~~~~~~~~~~~~~

HIcosmo is built on JAX with the following characteristics:

- **Pure functions first**: all cosmological calculations are side-effect-free pure functions
- **Automatic differentiation**: leveraging JAX's autograd for exact gradient computation
- **JIT compilation**: critical computation paths are JIT-compiled for 4-7x speedup
- **GPU-ready**: transparent GPU acceleration support
- **Vectorization**: efficient batch computation via vmap

Six-Layer Architecture
----------------------

HIcosmo employs a layered architecture design:

.. code-block:: text

   +---------------------------------------------------------+
   |  Layer 6: Visualization                                 |
   |  Plotter, plot_corner, plot_chains, GetDist integration  |
   +---------------------------------------------------------+
   |  Layer 5: Fisher Matrix Forecasting                     |
   |  IntensityMappingFisher, FisherMatrix, Survey Forecasts  |
   +---------------------------------------------------------+
   |  Layer 4: Samplers                                      |
   |  MCMC, NumPyro NUTS, emcee, Checkpoint System           |
   +---------------------------------------------------------+
   |  Layer 3: Likelihoods                                   |
   |  SN, BAO, CMB, H0LiCOW, TDCOSMO, SH0ES                 |
   +---------------------------------------------------------+
   |  Layer 2: Models (Cosmological Models)                  |
   |  LCDM, wCDM, CPL, ILCDM                                |
   +---------------------------------------------------------+
   |  Layer 1: Core (Foundation)                             |
   |  CosmologyBase, Distance Calculations, Unified Params   |
   +---------------------------------------------------------+

Layer 1: Core (Foundation)
~~~~~~~~~~~~~~~~~~~~~~~~~~

``hicosmo/core/`` contains:

- **CosmologyBase**: abstract base class for all cosmological models
- **compute_distances_from_E_z()**: JIT-compiled distance calculation engine
- **make_compute_grid_traced()**: factory function for generating traced-aware computations

Layer 2: Models (Cosmological Models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hicosmo/models/`` contains:

- **LCDM**: standard :math:`\Lambda\text{CDM}` model
- **wCDM**: constant dark energy equation of state
- **CPL**: Chevallier-Polarski-Linder parameterization
- **ILCDM**: interacting dark energy model

Layer 3: Likelihoods
~~~~~~~~~~~~~~~~~~~~

``hicosmo/likelihoods/`` contains:

- **PantheonPlus**: distance measurements from 1701 SNe Ia
- **BAO**: BAO constraints from DESI 2024, BOSS, SDSS
- **Planck**: CMB distance priors
- **H0LiCOW / TDCOSMO**: strong gravitational lensing time delays

Layer 4: Samplers
~~~~~~~~~~~~~~~~~

``hicosmo/samplers/`` contains:

- **MCMC**: high-level MCMC interface
- **NumPyro Backend**: NUTS sampler wrapper
- **emcee Backend**: ensemble sampler (alternative)
- **Persistence**: checkpoint and recovery system

Layer 5: Fisher Matrix
~~~~~~~~~~~~~~~~~~~~~~

``hicosmo/fisher/`` contains:

- **IntensityMappingFisher**: 21cm intensity mapping Fisher forecasting
- **FisherMatrix**: general-purpose Fisher matrix computation

Layer 6: Visualization
~~~~~~~~~~~~~~~~~~~~~~

``hicosmo/visualization/`` contains:

- **Plotter**: unified visualization class
- **plot_corner, plot_chains**: function interfaces
- **GetDist integration**: publication-quality plots

Module Responsibility Boundaries
--------------------------------

**Core principle: each module does one thing and never crosses boundaries!**

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Module
     - Computation Nature
     - Input -> Output
   * - ``models/``
     - **Cosmological predictions**
     - :math:`(H_0, \Omega_m, z) \to (d_L, D_M, H(z), r_d)`
   * - ``likelihoods/``
     - **Statistical inference**
     - :math:`(\text{theory}, \text{data}, C) \to \chi^2`
   * - ``samplers/``
     - **Parameter exploration**
     - :math:`(\text{prior}, \mathcal{L}) \to \text{posterior samples}`
   * - ``visualization/``
     - **Results presentation**
     - :math:`\text{samples} \to \text{plots}`

**Determining which module code belongs to**:

- Ask: "What is this code computing?"
- If it is "computing physical quantities from cosmological parameters" -> ``models/``
- If it is "comparing theoretical predictions with observational data" -> ``likelihoods/``

Parameter System
----------------

HIcosmo uses a unified parameter management system.

Parameter Class
~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.parameters import Parameter

   param = Parameter(
       name='H0',
       value=70.0,                    # Reference/initial value
       free=True,                     # Whether to sample
       prior={
           'dist': 'uniform',
           'min': 60, 'max': 80
       },
       latex_label=r'$H_0$',          # For plotting
       description='Hubble constant'
   )

ParameterRegistry
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.parameters import ParameterRegistry

   # Create from presets
   registry = ParameterRegistry.from_defaults('planck2018')

   # Set free parameters
   registry.set_free(['H0', 'Omega_m'])

   # Add nuisance parameters from a likelihood
   registry.add_from_likelihood(tdcosmo)

Parameter Presets
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 55

   * - Preset
     - :math:`H_0`
     - :math:`\Omega_m`
     - Description
   * - ``planck2018``
     - 67.36
     - 0.3153
     - Planck 2018 TT,TE,EE+lowE+lensing
   * - ``wmap9``
     - 70.0
     - 0.279
     - WMAP 9-year data

Distance Calculations
---------------------

HIcosmo provides comprehensive cosmological distance calculations, supporting both flat and non-flat universes.

Basic Distance Quantities
~~~~~~~~~~~~~~~~~~~~~~~~~

**Comoving distance** :math:`d_C`:

.. math::

   d_C(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}

**Transverse comoving distance** :math:`D_M` (accounting for curvature):

.. math::

   D_M = \begin{cases}
   D_H / \sqrt{\Omega_k} \sinh(\sqrt{\Omega_k} \, d_C / D_H) & \Omega_k > 0 \\
   d_C & \Omega_k = 0 \\
   D_H / \sqrt{|\Omega_k|} \sin(\sqrt{|\Omega_k|} \, d_C / D_H) & \Omega_k < 0
   \end{cases}

**Luminosity distance**:

.. math::

   d_L(z) = (1+z) \, D_M(z)

**Angular diameter distance**:

.. math::

   D_A(z) = \frac{D_M(z)}{1+z}

**Distance modulus**:

.. math::

   \mu = 5 \log_{10}(d_L / \text{Mpc}) + 25

Traced-Aware Interface
~~~~~~~~~~~~~~~~~~~~~~

For NumPyro MCMC sampling, HIcosmo provides a traced-aware batch computation interface:

.. code-block:: python

   from hicosmo.models import LCDM
   import jax.numpy as jnp

   z_grid = jnp.linspace(0.01, 2.0, 1000)
   params = {'H0': 70.0, 'Omega_m': 0.3}

   # Batch compute all distances
   grid = LCDM.compute_grid_traced(z_grid, params)

   # Returns a dict containing: d_L, D_M, D_H, dVc_dz, E_z, d_C

Data Management
---------------

HIcosmo manages data directories and the configuration system through the ``runner`` module.

Data Directory Search Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Environment variable ``HICOSMO_DATA``
2. ``data.root`` in the configuration
3. Defaults to ``data/`` in the project root directory

Dataset Structure
~~~~~~~~~~~~~~~~~

.. code-block:: text

   data/
   ├── sne/                    # Supernova data
   │   ├── Pantheon+SH0ES.dat
   │   └── Pantheon+SH0ES_STAT+SYS.cov
   ├── bao_data/               # BAO data
   │   ├── desi_2024/
   │   ├── boss_dr12/
   │   └── sdss_dr12/
   ├── h0licow/                # Strong lensing data
   └── tdcosmo/                # TDCOSMO data

Performance Characteristics
---------------------------

HIcosmo is deeply optimized for performance:

.. list-table::
   :header-rows: 1
   :widths: 35 20 20 25

   * - Operation
     - qcosmc (scipy)
     - HIcosmo (JAX)
     - Speedup
   * - Distance calculation (1000 pts)
     - 0.15s
     - 0.02s
     - **7.5x**
   * - MCMC sampling (10k samples)
     - 180s
     - 45s
     - **4.0x**
   * - BAO likelihood (JIT)
     - 21.8ms
     - 1.24ms
     - **17.6x**

Next Steps
----------

- `Cosmological Models <guides/models.html>`_: learn about LCDM, wCDM, CPL, and more
- `Likelihood Functions <guides/likelihoods.html>`_: learn about SN, BAO, CMB, and other data
- `Samplers <guides/samplers.html>`_: learn about MCMC configuration and diagnostics
