Likelihood Functions
====================

HIcosmo provides a rich set of observational data likelihood functions, covering supernovae, BAO, CMB, strong gravitational lensing, and other cosmological probes. All likelihood functions follow a unified API design and support JAX JIT compilation optimization.

Likelihood Function Overview
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Type
     - Dataset
     - Description
   * - **SNe Ia**
     - Pantheon+ (2022)
     - 1701 SNe Ia, redshift :math:`z < 2.3`
   * - **BAO**
     - DESI 2024
     - Latest BAO constraints, 6 redshift bins, uses :math:`H_0 r_d` as nuisance by default
   * - **BAO**
     - SDSS/BOSS DR12
     - Classic BAO datasets
   * - **CMB**
     - Planck 2018
     - Compressed distance priors :math:`(R, l_a, \omega_b h^2)`
   * - **Strong Lensing**
     - H0LiCOW
     - Time-delay distances from 6 lens systems
   * - **Strong Lensing**
     - TDCOSMO
     - Hierarchical Bayesian analysis with MST effects
   * - **Local H0**
     - SH0ES
     - Cepheid distance ladder

Unified API Design
-------------------

All likelihood functions follow a unified interface design:

**Shortcut interface** (recommended):

.. code-block:: python

   from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
   from hicosmo.models import LCDM

   # Specify dataset by string
   sne = SN_likelihood(LCDM, "pantheon+")
   bao = BAO_likelihood(LCDM, "desi2024")  # Default H0_rd mode

   # Call directly to compute log-likelihood
   # Note: BAO requires the H0_rd nuisance parameter
   log_L = sne(H0=70, Omega_m=0.3) + bao(H0=70, Omega_m=0.3, H0_rd=100)

**Class interface** (advanced usage):

.. code-block:: python

   from hicosmo.likelihoods import PantheonPlusLikelihood, DESI2024BAO

   sne = PantheonPlusLikelihood(cosmology_class=LCDM)
   bao = DESI2024BAO(cosmology_class=LCDM)

Type Ia Supernovae (SNe Ia)
----------------------------

Pantheon+ Dataset
~~~~~~~~~~~~~~~~~

Pantheon+ (Brout et al. 2022) contains 1701 Type Ia supernovae and is currently the largest SNe Ia sample.

**Likelihood function**:

.. math::

   -2 \ln \mathcal{L} = \Delta \boldsymbol{\mu}^T \, C^{-1} \, \Delta \boldsymbol{\mu}

where:

- :math:`\Delta \boldsymbol{\mu} = \mu_{\rm obs} - \mu_{\rm theory}`
- :math:`\mu_{\rm theory} = 5 \log_{10}(d_L/\text{Mpc}) + 25`
- :math:`C` is the full covariance matrix (statistical + systematic errors)

**Basic usage**:

.. code-block:: python

   from hicosmo.likelihoods import SN_likelihood
   from hicosmo.models import LCDM

   # Pantheon+ (without SH0ES Cepheid calibration)
   sne = SN_likelihood(LCDM, "pantheon+", M_B="marginalize")

   # Pantheon+SH0ES (with Cepheid calibration)
   sne_shoes = SN_likelihood(LCDM, "pantheon+shoes", M_B="marginalize")

   # Compute log-likelihood
   log_L = sne(H0=70.0, Omega_m=0.3)
   print(f"log L = {log_L:.2f}")

**Absolute magnitude handling**:

Supernova analysis requires handling the absolute magnitude :math:`\mathcal{M}_B = M_B + 5\log_{10}(c/H_0) + 25`:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Option
     - Description
   * - ``M_B="marginalize"``
     - Analytic marginalization (recommended, default)
   * - ``M_B="free"``
     - Sample as a free parameter

.. code-block:: python

   # Analytic marginalization of M_B (default, recommended)
   sne = SN_likelihood(LCDM, "pantheon+", M_B="marginalize")
   log_L = sne(H0=70, Omega_m=0.3)  # No need to provide M_B

   # M_B as a free parameter
   sne = SN_likelihood(LCDM, "pantheon+", M_B="free")
   log_L = sne(H0=70, Omega_m=0.3, M_B=-19.3)  # Must provide M_B

**Advanced options**:

.. code-block:: python

   from hicosmo.likelihoods import PantheonPlusLikelihood

   sne = PantheonPlusLikelihood(
       cosmology_class=LCDM,
       include_shoes=True,          # Use SH0ES Cepheid data
       include_systematics=True,    # Include systematic errors (default)
       z_min=0.01,                  # Minimum redshift cutoff
       z_max=None,                  # Maximum redshift cutoff
       marginalize_M_B=True         # Marginalize absolute magnitude
   )

Union3 Dataset
~~~~~~~~~~~~~~

Union3 (Rubin et al. 2023) compresses 2087 Type Ia supernovae into 22
redshift-binned distance-modulus measurements through the UNITY framework.
The absolute magnitude is already marginalized within UNITY, so no ``M_B``
handling is needed:

.. code-block:: python

   from hicosmo.likelihoods import SN_likelihood
   from hicosmo.models import LCDM

   union3 = SN_likelihood(LCDM, "union3")
   log_L = union3(H0=70.0, Omega_m=0.3)   # no M_B parameter

DESY5 Dataset
~~~~~~~~~~~~~

DES Year 5 (DES Collaboration 2024) contains 1829 photometrically
classified Type Ia supernovae (including an external low-z sample). The
magnitude offset ``M`` is handled analogously to Pantheon+'s ``M_B``:

.. code-block:: python

   # Analytic marginalization over M (default, matches the official
   # Dovekie pipeline)
   desy5 = SN_likelihood(LCDM, "desy5")
   log_L = desy5(H0=70.0, Omega_m=0.3)

   # M as a free parameter
   desy5_free = SN_likelihood(LCDM, "desy5", M="free")
   log_L = desy5_free(H0=70.0, Omega_m=0.3, M=-19.3)

.. note::

   All three SN datasets (``pantheon+`` / ``union3`` / ``desy5``) implement
   the ``CombinedLikelihood`` shared-distance-grid protocol, so joint
   analyses compute the distance integral only once.

BAO (Baryon Acoustic Oscillations)
-----------------------------------

BAO provides precise measurements of :math:`H(z) r_d` and :math:`D_M(z)/r_d`, serving as a key link in the cosmological distance ladder.

Observables
~~~~~~~~~~~

HIcosmo supports the following BAO observables:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Observable
     - Physical Meaning
   * - :math:`D_M/r_d`
     - Transverse comoving distance / sound horizon
   * - :math:`D_H/r_d = c/(H(z) r_d)`
     - Radial Hubble distance / sound horizon
   * - :math:`D_V/r_d`
     - Spherically averaged distance :math:`[z D_M^2 D_H]^{1/3} / r_d`

**Spherically averaged distance**:

.. math::

   D_V(z) = \left[ z \, D_M^2(z) \, D_H(z) \right]^{1/3}

Available Datasets
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Dataset
     - Year
     - Reference
   * - ``desi2024``
     - 2024
     - DESI Collaboration 2024 (DR1, arXiv:2404.03002)
   * - ``desi_dr2``
     - 2025
     - DESI Collaboration 2025 (DR2, arXiv:2503.14738)
   * - ``sdss_dr12``
     - 2017
     - Alam et al. 2017, MNRAS 470, 2617
   * - ``sdss_dr16``
     - 2020
     - eBOSS Collaboration 2020
   * - ``boss_dr12``
     - 2017
     - Alam et al. 2017
   * - ``sixdf``
     - 2011
     - Beutler et al. 2011

DESI 2024
~~~~~~~~~

DESI 2024 provides BAO measurements in 6 redshift bins:

.. code-block:: python

   from hicosmo.likelihoods import BAO_likelihood
   from hicosmo.models import LCDM

   bao = BAO_likelihood(LCDM, "desi2024")

   # Compute log-likelihood (requires the H0_rd nuisance parameter)
   log_L = bao(H0=67.36, Omega_m=0.3, H0_rd=100.0)
   print(f"DESI BAO log L = {log_L:.2f}")

**Sound horizon handling mode (omega_b_mode)**:

BAO measurements constrain :math:`D_M/r_d` and :math:`D_H/r_d`, where the sound horizon :math:`r_d` depends on :math:`\Omega_b`.
BAO data alone can only constrain the combination :math:`H_0 r_d`, not :math:`H_0` individually.

HIcosmo provides multiple handling modes:

.. list-table::
   :header-rows: 1
   :widths: 18 20 62

   * - Mode
     - Nuisance Parameter
     - Description
   * - ``'h0rd'`` (default)
     - ``H0_rd``
     - Directly sample :math:`H_0 r_d / (100\,{\rm km\,s^{-1}})`, recommended for BAO-only analysis
   * - ``'bbn_prior'``
     - ``Omega_b``
     - Use BBN prior (:math:`\omega_b = 0.02218 \pm 0.00055`)
   * - ``'fixed'``
     - None
     - Fix :math:`\Omega_b` to Planck 2018 value (0.0493)

**Usage examples**:

.. code-block:: python

   from hicosmo.likelihoods import BAO_likelihood
   from hicosmo.models import LCDM

   # Default mode: H0_rd as nuisance parameter (recommended for BAO-only)
   bao = BAO_likelihood(LCDM, "desi2024")
   # nuisance: H0_rd, prior: [95, 110]

   # Use BBN prior
   bao_bbn = BAO_likelihood(LCDM, "desi2024", omega_b_mode='bbn_prior')
   # nuisance: Omega_b, with BBN constraint on Omega_b*h^2

   # Fixed Omega_b (Planck value)
   bao_fixed = BAO_likelihood(LCDM, "desi2024", omega_b_mode='fixed')
   # No nuisance parameters

.. note::

   In ``'bbn_prior'`` mode the BBN prior is applied exactly once, inside
   the JIT closure: evaluating through ``bao(**params)``,
   ``log_likelihood_from_params``, or the ``CombinedLikelihood``
   shared-grid path gives identical results (guaranteed by the
   regression tests in ``tests/test_desi_dr2.py``).

DESI DR2
~~~~~~~~

DESI DR2 (2025, arXiv:2503.14738) provides 13 BAO measurements across 7
redshift bins covering :math:`0.295 \le z \le 2.330`, currently the most
constraining BAO dataset. The interface is identical to ``desi2024``
(inheriting all ``omega_b_mode`` options and shared-grid support):

.. code-block:: python

   from hicosmo.likelihoods import BAO_likelihood
   from hicosmo.models import LCDM

   bao = BAO_likelihood(LCDM, "desi_dr2")
   log_L = bao(H0=70.0, Omega_m=0.3, H0_rd=101.0)

**Physical explanation**:

The typical value of :math:`H_0 r_d` is approximately :math:`10000\,{\rm km\,s^{-1}}` (corresponding to :math:`H_0 \approx 70`, :math:`r_d \approx 147\,{\rm Mpc}`).
The ``H0_rd`` parameter is defined as :math:`H_0 r_d / (100\,{\rm km\,s^{-1}})`, with a numerical value around 100.

**Redshift distribution**:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Redshift Bin
     - Effective Redshift
     - Observables
   * - BGS
     - :math:`z = 0.30`
     - :math:`D_V/r_d`
   * - LRG1
     - :math:`z = 0.51`
     - :math:`D_M/r_d, D_H/r_d`
   * - LRG2
     - :math:`z = 0.71`
     - :math:`D_M/r_d, D_H/r_d`
   * - LRG3+ELG
     - :math:`z = 0.93`
     - :math:`D_M/r_d, D_H/r_d`
   * - ELG
     - :math:`z = 1.32`
     - :math:`D_M/r_d, D_H/r_d`
   * - QSO
     - :math:`z = 1.49`
     - :math:`D_M/r_d, D_H/r_d`

Multi-Dataset Combination
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.likelihoods import BAO_likelihood

   # Individual use
   desi = BAO_likelihood(LCDM, "desi2024")
   sdss = BAO_likelihood(LCDM, "sdss_dr12")

   # Joint constraints
   log_L = desi(H0=67, Omega_m=0.3) + sdss(H0=67, Omega_m=0.3)

CMB Distance Priors
--------------------

Planck 2018
~~~~~~~~~~~

HIcosmo uses Planck 2018 compressed distance priors, containing three observables:

**Compressed likelihood observables**:

.. math::

   \boldsymbol{x} = \begin{pmatrix} R \\ l_a \\ \omega_b h^2 \end{pmatrix}

where:

- :math:`R = \sqrt{\Omega_m H_0^2} \, D_M(z_*) / c`: Shift parameter
- :math:`l_a = \pi D_M(z_*) / r_s(z_*)`: Acoustic scale
- :math:`\omega_b h^2 = \Omega_b h^2`: Baryon density parameter

:math:`z_* \approx 1090` is the redshift of the last scattering surface.

**Planck 2018 central values**:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Observable
     - Value
     - Description
   * - :math:`R`
     - 1.7502
     - Shift parameter
   * - :math:`l_a`
     - 301.47
     - Acoustic scale
   * - :math:`\omega_b h^2`
     - 0.02236
     - Baryon density

**Basic usage**:

.. code-block:: python

   from hicosmo.likelihoods import Planck
   from hicosmo.models import LCDM

   cmb = Planck(LCDM)

   # Requires Omega_b
   log_L = cmb(H0=67.36, Omega_m=0.315, Omega_b=0.049)
   print(f"Planck log L = {log_L:.2f}")

**Note**: Planck distance priors require the ``Omega_b`` parameter.

Strong Gravitational Lensing
-----------------------------

Strong gravitational lensing time-delay cosmography provides :math:`H_0` measurements independent of the distance ladder.

H0LiCOW
~~~~~~~~

H0LiCOW (Wong et al. 2019) uses time-delay distances from 6 lens systems to constrain :math:`H_0`.

**Time-delay distance**:

.. math::

   D_{\Delta t} = (1 + z_{\rm lens}) \frac{D_{\rm lens} D_{\rm source}}{D_{\rm lens,source}}

where :math:`D_{\rm lens,source}` is the angular diameter distance from the lens to the source.

**Basic usage**:

.. code-block:: python

   from hicosmo.likelihoods import H0LiCOW
   from hicosmo.models import LCDM

   lensing = H0LiCOW(LCDM)

   log_L = lensing(H0=73.3, Omega_m=0.3)
   print(f"H0LiCOW log L = {log_L:.2f}")

**Lens systems**:

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - System Name
     - :math:`z_{\rm lens}`
     - :math:`z_{\rm source}`
     - Type
   * - B1608+656
     - 0.630
     - 1.394
     - KDE
   * - RXJ1131-1231
     - 0.295
     - 0.654
     - Skewed log-normal
   * - HE0435-1223
     - 0.454
     - 1.693
     - KDE
   * - SDSS1206+4332
     - 0.745
     - 1.789
     - KDE
   * - WFI2033-4723
     - 0.661
     - 1.662
     - KDE
   * - PG1115+080
     - 0.311
     - 1.722
     - KDE

TDCOSMO
~~~~~~~

TDCOSMO provides hierarchical Bayesian analysis, properly handling the mass-sheet transform (MST) and stellar velocity dispersion systematic errors.

**Key parameters**:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``lambda_int_mean``
     - 1.0
     - Internal MST population mean
   * - ``lambda_int_sigma``
     - 0.05
     - Internal MST scatter
   * - ``a_ani_mean``
     - 1.0
     - Stellar anisotropy mean
   * - ``a_ani_sigma``
     - 0.1
     - Anisotropy scatter

**Basic usage**:

.. code-block:: python

   from hicosmo.likelihoods import TDCOSMO
   from hicosmo.models import LCDM

   tdcosmo = TDCOSMO(LCDM)

   # Get nuisance parameters
   for p in tdcosmo.nuisance_parameters:
       print(f"{p.name}: {p.value} ({p.prior['min']}, {p.prior['max']})")

   # Compute log-likelihood (with nuisance parameters)
   log_L = tdcosmo(
       H0=73.0,
       Omega_m=0.3,
       lambda_int_mean=1.0,
       a_ani_mean=1.0
   )

SH0ES
------

SH0ES provides local :math:`H_0` measurements based on Cepheid variables.

.. code-block:: python

   from hicosmo.likelihoods import SH0ESLikelihood
   from hicosmo.models import LCDM

   shoes = SH0ESLikelihood(cosmology_class=LCDM)

   log_L = shoes(H0=73.0)
   print(f"SH0ES log L = {log_L:.2f}")

**SH0ES constraint**:

.. math::

   H_0 = 73.04 \pm 1.04 \text{ km/s/Mpc}

Joint Likelihood
-----------------

HIcosmo supports multiple ways to combine likelihood functions.

Addition Operator
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.likelihoods import SN_likelihood, BAO_likelihood, Planck
   from hicosmo.models import LCDM

   # Create individual likelihoods
   sne = SN_likelihood(LCDM, "pantheon+")
   bao = BAO_likelihood(LCDM, "desi2024")  # Default H0_rd mode
   cmb = Planck(LCDM)

   # Joint likelihood = sum of individual likelihoods
   # Note: BAO requires H0_rd, CMB requires Omega_b
   params = {'H0': 67.36, 'Omega_m': 0.315, 'Omega_b': 0.049, 'H0_rd': 100.0}
   log_L_joint = (
       sne(**params) +
       bao(**params) +
       cmb(**params)
   )

CombinedLikelihood Class
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.likelihoods import CombinedLikelihood

   joint = CombinedLikelihood([sne, bao, cmb])
   log_L = joint(H0=67.36, Omega_m=0.315, Omega_b=0.049, H0_rd=100.0)

Nuisance Parameters
--------------------

Some likelihood functions include nuisance parameters that need to be sampled.

Getting Nuisance Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.likelihoods import BAO_likelihood, TDCOSMO
   from hicosmo.models import LCDM

   # BAO uses H0_rd as a nuisance parameter by default
   bao = BAO_likelihood(LCDM, "desi2024")
   for p in bao.nuisance_parameters:
       print(f"{p.name}: {p.value}, prior: {p.prior}")
   # Output: H0_rd: 102.5, prior: {'dist': 'uniform', 'min': 95.0, 'max': 110.0}

   # TDCOSMO has multiple nuisance parameters
   tdcosmo = TDCOSMO(LCDM)
   for p in tdcosmo.nuisance_parameters:
       print(f"{p.name}:")
       print(f"  Default: {p.value}")
       print(f"  Prior: {p.prior}")
       print(f"  LaTeX: {p.latex_label}")

**Automatic nuisance parameter collection**:

During MCMC sampling, nuisance parameters are automatically collected:

.. code-block:: python

   from hicosmo.samplers import MCMC

   # Cosmological parameters
   params = {
       'H0': (70.0, 60.0, 80.0),
       'Omega_m': (0.3, 0.1, 0.5),
   }

   # MCMC automatically collects nuisance parameters from the likelihood
   mcmc = MCMC(params, tdcosmo)
   samples = mcmc.run(num_samples=2000)

Integration with MCMC
----------------------

Complete MCMC Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo import hicosmo

   # Shortcut API: 3 lines to completion
   inf = hicosmo(
       cosmology="LCDM",
       likelihood=["sn", "bao", "cmb"],
       free_params=["H0", "Omega_m"]
   )
   samples = inf.run()
   inf.summary()

Class Interface Example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
   from hicosmo.samplers import MCMC
   from hicosmo.models import LCDM

   # Create joint likelihood
   sne = SN_likelihood(LCDM, "pantheon+")
   bao = BAO_likelihood(LCDM, "desi2024")

   def joint_likelihood(**params):
       return sne(**params) + bao(**params)

   # Configure MCMC
   params = {
       'H0': (70.0, 60.0, 80.0),
       'Omega_m': (0.3, 0.1, 0.5),
   }

   mcmc = MCMC(params, joint_likelihood, chain_name="sn_bao")
   samples = mcmc.run(num_samples=4000, num_chains=4)

   mcmc.print_summary()

Traced-Aware Interface
~~~~~~~~~~~~~~~~~~~~~~

Likelihood functions internally use the traced-aware interface for NumPyro compatibility:

.. code-block:: python

   import numpyro
   import numpyro.distributions as dist
   from hicosmo.models import LCDM

   def numpyro_model(z_obs, mu_obs, mu_err):
       H0 = numpyro.sample('H0', dist.Uniform(60, 80))
       Omega_m = numpyro.sample('Omega_m', dist.Uniform(0.1, 0.5))

       params = {'H0': H0, 'Omega_m': Omega_m}
       grid = LCDM.compute_grid_traced(z_obs, params)
       d_L = grid['d_L']
       mu_theory = 5 * jnp.log10(d_L) + 25

       numpyro.sample('obs', dist.Normal(mu_theory, mu_err), obs=mu_obs)

Performance Optimization
-------------------------

JIT Compilation
~~~~~~~~~~~~~~~

All likelihood functions' core computations are JAX JIT compiled:

.. code-block:: python

   # First call: includes compilation
   %timeit sne(H0=70, Omega_m=0.3)  # ~100ms

   # Subsequent calls: compiled
   %timeit sne(H0=70, Omega_m=0.3)  # ~1ms

Covariance Matrix Pre-computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The covariance matrix inverse is pre-computed at initialization:

.. code-block:: python

   # Covariance matrix inverse computed only once
   sne = PantheonPlusLikelihood()  # Pre-computes C^{-1}

   # Subsequent calls use it directly
   log_L = sne(H0=70, Omega_m=0.3)

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Likelihood Function
     - First Call
     - Subsequent Calls
     - Data Size
   * - Pantheon+
     - 150 ms
     - 1.2 ms
     - 1701 SNe
   * - DESI BAO
     - 50 ms
     - 0.5 ms
     - 12 points
   * - Planck
     - 30 ms
     - 0.3 ms
     - 3 points

Adding Custom Likelihood Functions
------------------------------------

HIcosmo's base class design makes creating custom likelihood functions straightforward.

Minimal Implementation
~~~~~~~~~~~~~~~~~~~~~~

Simply inherit from ``Likelihood`` and implement the ``log_likelihood`` method:

.. code-block:: python

   from hicosmo.likelihoods import Likelihood
   from hicosmo.models import LCDM
   import jax.numpy as jnp

   class MyDistanceModulusLikelihood(Likelihood):
       """Custom distance modulus likelihood function example."""

       def __init__(self, cosmology_class=LCDM, **kwargs):
           super().__init__(**kwargs)
           self._cosmology_class = cosmology_class

           # Your observational data
           self.z_data = jnp.array([0.5, 1.0, 1.5, 2.0])
           self.mu_obs = jnp.array([42.5, 44.1, 45.2, 46.0])
           self.sigma = jnp.array([0.1, 0.15, 0.12, 0.18])

       def log_likelihood(self, model, **kwargs):
           """Compute log-likelihood.

           Parameters
           ----------
           model : CosmologyBase
               Cosmological model instance (automatically built from parameters)
           **kwargs
               Additional parameters (such as nuisance parameters)

           Returns
           -------
           float
               Log-likelihood value
           """
           mu_theory = model.distance_modulus(self.z_data)
           chi2 = jnp.sum(((self.mu_obs - mu_theory) / self.sigma)**2)
           return -0.5 * chi2

   # Usage
   my_lik = MyDistanceModulusLikelihood()
   log_L = my_lik(H0=70, Omega_m=0.3)  # Cosmological model built automatically

**Key points**:

1. Setting ``self._cosmology_class`` enables the base class to automatically build the cosmological model
2. ``log_likelihood(model, **kwargs)`` receives the already-built model instance
3. No need to override ``__call__``, ``__add__``, or other methods -- the base class provides them

With Nuisance Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Adding nuisance parameters that need to be sampled:

.. code-block:: python

   from hicosmo.likelihoods import Likelihood, NuisanceList
   from hicosmo.parameters import Parameter
   from hicosmo.models import LCDM
   import jax.numpy as jnp

   class SNeLikelihoodWithMB(Likelihood):
       """Supernova likelihood with absolute magnitude nuisance parameter."""

       def __init__(self, cosmology_class=LCDM, **kwargs):
           super().__init__(**kwargs)
           self._cosmology_class = cosmology_class

           # Observational data
           self.z_data = jnp.array([0.1, 0.3, 0.5, 0.8, 1.0])
           self.m_obs = jnp.array([23.5, 26.2, 27.8, 29.1, 29.8])  # Apparent magnitudes
           self.sigma = jnp.array([0.15, 0.12, 0.10, 0.12, 0.15])

       @property
       def nuisance_parameters(self):
           """Declare nuisance parameters.

           The MCMC sampler will automatically collect and sample these parameters.
           """
           return NuisanceList([
               Parameter(
                   name='M_B',
                   value=-19.3,
                   free=True,
                   prior={'dist': 'uniform', 'min': -20.0, 'max': -18.0},
                   latex_label=r'$\mathcal{M}_B$',
                   description='SNe Ia absolute magnitude'
               )
           ])

       def log_likelihood(self, model, M_B=-19.3, **kwargs):
           """Compute log-likelihood.

           Parameters
           ----------
           model : CosmologyBase
               Cosmological model instance
           M_B : float
               Absolute magnitude (nuisance parameter)
           """
           mu_theory = model.distance_modulus(self.z_data)
           m_theory = mu_theory + M_B
           chi2 = jnp.sum(((self.m_obs - m_theory) / self.sigma)**2)
           return -0.5 * chi2

   # Usage
   sne = SNeLikelihoodWithMB()

   # Get nuisance parameter information
   for p in sne.nuisance_parameters:
       print(f"{p.name}: {p.value}, prior: {p.prior}")

   # MCMC automatically samples nuisance parameters
   from hicosmo.samplers import MCMC

   params = {
       'H0': (70.0, 60.0, 80.0),
       'Omega_m': (0.3, 0.1, 0.5),
   }
   mcmc = MCMC(params, sne)  # M_B automatically added to sampling
   samples = mcmc.run(num_samples=2000)

Joint Likelihood
~~~~~~~~~~~~~~~~

Custom likelihoods can be combined with built-in likelihoods:

.. code-block:: python

   from hicosmo.likelihoods import BAO_likelihood

   # Combine custom + built-in
   my_lik = MyDistanceModulusLikelihood()
   bao = BAO_likelihood(LCDM, "desi2024")

   joint = my_lik + bao

   # MCMC sampling
   mcmc = MCMC(params, joint)
   samples = mcmc.run()

Using a Covariance Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~

For likelihoods requiring a full covariance matrix:

.. code-block:: python

   class CorrelatedDataLikelihood(Likelihood):
       """Likelihood function with covariance matrix."""

       def __init__(self, cosmology_class=LCDM, **kwargs):
           super().__init__(**kwargs)
           self._cosmology_class = cosmology_class

           # Data
           self.z_data = jnp.array([0.3, 0.5, 0.7, 0.9])
           self.obs = jnp.array([1.02, 1.05, 1.08, 1.12])

           # Covariance matrix
           cov = jnp.array([
               [0.01, 0.002, 0.001, 0.0],
               [0.002, 0.012, 0.003, 0.001],
               [0.001, 0.003, 0.015, 0.002],
               [0.0, 0.001, 0.002, 0.018]
           ])
           self.inv_cov = jnp.linalg.inv(cov)

       def log_likelihood(self, model, **kwargs):
           # Theoretical prediction
           theory = model.E_z(self.z_data)

           # Chi-squared with covariance
           diff = self.obs - theory
           chi2 = diff @ self.inv_cov @ diff
           return -0.5 * chi2

Advanced: JIT Compilation Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For performance-critical likelihoods, use the traced interface:

.. code-block:: python

   from jax import jit
   import jax.numpy as jnp

   class FastTracedLikelihood(Likelihood):
       """High-performance likelihood using the traced interface."""

       def __init__(self, cosmology_class=LCDM, **kwargs):
           super().__init__(**kwargs)
           self._cosmology_class = cosmology_class

           self.z_data = jnp.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
           self.mu_obs = jnp.array([38.5, 41.2, 42.8, 43.9, 44.9, 46.1, 46.8])
           self.sigma = jnp.array([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.2])

           # Pre-compile JIT function
           self._build_jit_loglike()

       def _build_jit_loglike(self):
           z = self.z_data
           mu_obs = self.mu_obs
           sigma = self.sigma
           cosmo_class = self._cosmology_class

           @jit
           def _loglike_impl(params):
               # Compute directly using the traced interface
               grid = cosmo_class.compute_grid_traced(z, params)
               d_L = grid['d_L']
               mu_theory = 5 * jnp.log10(d_L) + 25
               chi2 = jnp.sum(((mu_obs - mu_theory) / sigma)**2)
               return -0.5 * chi2

           self._loglike_jit = _loglike_impl

       def __call__(self, **params):
           # Prepare JAX parameters
           params_jax = self._prepare_params_for_jax(params)
           return self._loglike_jit(params_jax)

Design Principles
~~~~~~~~~~~~~~~~~

When creating custom likelihoods, follow these principles:

1. **Only override necessary methods**
   - Required: ``__init__`` and ``log_likelihood``
   - Optional: ``nuisance_parameters`` (if nuisance parameters exist)

2. **Use JAX arrays**
   - Store data using ``jnp.array``
   - Compute using JAX functions

3. **Set cosmology_class**
   - ``self._cosmology_class = cosmology_class``
   - Enables base class ``__call__`` to automatically build the model

4. **Don't reinvent the wheel**
   - ``__add__``/``__radd__`` are already provided by the base class
   - Covariance handling can use ``_prepare_params_for_jax``

5. **Nuisance parameters must have LaTeX labels**
   - Used for plot display
   - See ``nuisance-parameters.md`` specification

Common Patterns
~~~~~~~~~~~~~~~

**BAO type (observable / r_d)**:

.. code-block:: python

   def log_likelihood(self, model, **kwargs):
       DM = model.transverse_comoving_distance(self.z)
       DH = model.hubble_distance(self.z)
       rd = model.sound_horizon_drag()

       DM_rd_theory = DM / rd
       DH_rd_theory = DH / rd

       chi2 = ...
       return -0.5 * chi2

**CMB distance prior type**:

.. code-block:: python

   def log_likelihood(self, model, **kwargs):
       z_star = model.recombination_redshift()
       DA = model.angular_diameter_distance(z_star)
       rs = model.sound_horizon(z_star)

       # Compute R, l_a, etc.
       ...

**Strong lensing type**:

.. code-block:: python

   def log_likelihood(self, model, **kwargs):
       Ddt = model.time_delay_distance(z_lens, z_source)
       # Compare with observed Ddt
       ...

Next Steps
----------

- `Samplers <samplers.html>`_: Learn about MCMC configuration and execution
- `Visualization <visualization.html>`_: Generate publication-quality plots
- `Fisher Forecasts <fisher.html>`_: Perform survey predictions
