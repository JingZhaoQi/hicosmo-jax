Configuration-Driven Workflow
=============================

HIcosmo supports Cobaya-like YAML configuration.

Parallel Initialization
-----------------------

Before running any configuration, initialize the parallel environment:

.. code-block:: python

   import hicosmo as hc
   hc.init(8)  # 8 parallel devices

.. note::

   ``hc.init(N)`` creates N JAX logical devices, enabling true parallelism for N chains.

   - ``hc.init(8)`` - 8 parallel devices (recommended)
   - ``hc.init()`` - Auto-detect (up to 8)
   - ``hc.init("GPU")`` - GPU mode

YAML Configuration Example
---------------------------

.. code-block:: yaml

   name: joint_sn_bao
   theory: LCDM
   likelihood:
     - name: sn
       dataset: pantheon+
     - name: bao
       dataset: sdss_dr12

   params:
     H0: {prior: {dist: uniform, min: 50, max: 100}, ref: 70.0}
     Omega_m: {prior: {dist: uniform, min: 0.1, max: 0.5}, ref: 0.3}

   sampler:
     name: numpyro
     num_samples: 1000
     num_chains: 8         # Default = number of devices
     num_warmup: 300

   output:
     root: results
     chain_name: joint_sn_bao

Running a Configuration
-----------------------

.. code-block:: python

   # 1. Initialize first
   import hicosmo as hc
   hc.init(8)

   # 2. Run the configuration
   from hicosmo import run_from_config
   run_from_config("examples/tutorials/configs/joint_sn_bao.yaml")
