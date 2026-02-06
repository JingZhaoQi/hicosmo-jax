Parallelism and Acceleration Configuration
==========================================

HIcosmo is built on JAX, with native support for multi-core CPU parallelism and GPU acceleration.
This chapter explains in detail how to configure the parallel environment to fully leverage your hardware.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start
-----------

**Configure parallelism in one line**:

.. code-block:: python

   import hicosmo as hc

   # Most common: 8-core parallelism
   hc.init(8)

   # Auto-detect optimal configuration
   hc.init()

   # GPU mode
   hc.init("GPU")

.. important::

   **You must call** ``hc.init()`` **before importing JAX!**

   JAX's device configuration is fixed at the time of first import and cannot be changed afterwards.
   Therefore ``hc.init()`` should be the very first line of your script.

   .. code-block:: python

      # ✅ Correct order
      import hicosmo as hc
      hc.init(8)

      from hicosmo.models import LCDM  # JAX is imported here
      from hicosmo.samplers import MCMC

      # ❌ Wrong order
      from hicosmo.models import LCDM  # JAX already imported, device count fixed to 1
      import hicosmo as hc
      hc.init(8)  # ⚠️ Warning: configuration ignored

Initialization API
------------------

``hc.init()`` Function
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import hicosmo as hc

   hc.init(
       num_devices='auto',  # Number of devices
       device=None,         # Device type ('GPU' or None)
       verbose=True,        # Whether to display configuration info
       force=True           # Whether to force-override existing configuration
   )

**Parameter Description**:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Type
     - Description
   * - ``num_devices``
     - int / str / None
     - | Number of parallel devices:
       | - ``int``: Specified count (e.g., 8)
       | - ``'auto'``: Auto-detect (default, up to 8)
       | - ``'GPU'``: GPU mode
       | - ``None``: Single-device vectorized mode
   * - ``device``
     - str / None
     - | Explicit device type:
       | - ``None``: CPU mode (default)
       | - ``'GPU'`` / ``'cuda'``: GPU mode
   * - ``verbose``
     - bool
     - Whether to print configuration info
   * - ``force``
     - bool
     - Whether to override existing XLA_FLAGS

Multi-Core CPU Parallelism
--------------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import hicosmo as hc

   # Use 8 CPU cores
   hc.init(8)

   # Running MCMC will automatically use 8 parallel chains
   from hicosmo.samplers import MCMC
   mcmc = MCMC(params, likelihood, chain_name='test')
   mcmc.run(num_samples=2000, num_chains=8)  # 8 chains run in parallel

How It Works
~~~~~~~~~~~~

When you call ``hc.init(8)``, HIcosmo will:

1. **Set XLA_FLAGS**: Configure JAX to use 8 logical CPU devices
2. **Configure thread pool**: Assign CPU cores to each device
3. **Set up NumPyro**: Inform NumPyro to use 8 devices

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                     CPU (8 cores)                       │
   ├─────────┬─────────┬─────────┬─────────┬─────────────────┤
   │ Device 0│ Device 1│ Device 2│ Device 3│ ... Device 7    │
   │ (Chain 1)│ (Chain 2)│ (Chain 3)│ (Chain 4)│ ... (Chain 8)│
   └─────────┴─────────┴─────────┴─────────┴─────────────────┘

Recommended Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Scenario
     - Recommendation
     - Notes
   * - Personal laptop (4 cores)
     - ``hc.init(4)``
     - Fully utilize all cores
   * - Workstation (8-16 cores)
     - ``hc.init(8)``
     - 8 chains are sufficient for convergence diagnostics
   * - Server (32+ cores)
     - ``hc.init(8)``
     - More chains are not necessarily better
   * - Debugging/Development
     - ``hc.init(1)``
     - Single device is easier to debug

.. note::

   **Why are 8 chains recommended?**

   - Sufficient for computing R-hat convergence diagnostics (requires >= 4 chains)
   - More chains do not significantly improve sampling efficiency
   - Too many chains may cause memory pressure

GPU Acceleration
----------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import hicosmo as hc

   # Auto-detect GPU
   hc.init("GPU")

   # Or specify explicitly
   hc.init(device="GPU")

Multi-GPU Configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import hicosmo as hc

   # Use 4 GPUs (if available)
   hc.init(4, device="GPU")

   # Auto-detect all available GPUs
   hc.init("GPU")  # Automatically detects GPU count

GPU vs CPU Selection
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Scenario
     - Recommendation
     - Reason
   * - Simple likelihoods (SNe, BAO)
     - CPU
     - GPU startup overhead exceeds computation benefit
   * - Complex likelihoods (CMB power spectrum)
     - GPU
     - Large-scale matrix operations benefit from GPU
   * - Large-scale Fisher matrices
     - GPU
     - Matrix inversion is well-suited for GPU
   * - Long MCMC runs (>100k samples)
     - GPU
     - Cumulative benefit is significant

GPU Environment Setup
~~~~~~~~~~~~~~~~~~~~~

Make sure the JAX GPU version is installed:

.. code-block:: bash

   # CUDA 12
   pip install jax[cuda12]

   # CUDA 11
   pip install jax[cuda11_pip]

Verify GPU availability:

.. code-block:: python

   import jax
   print(jax.devices())
   # [cuda(id=0), cuda(id=1), ...]  # GPU available
   # [CpuDevice(id=0)]              # CPU only

Cluster Configuration
---------------------

SLURM Cluster
~~~~~~~~~~~~~

Running HIcosmo on a SLURM cluster:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=hicosmo
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --time=24:00:00

   # Activate environment
   source activate hicosmo

   # Run script
   python my_mcmc_script.py

Corresponding Python script:

.. code-block:: python

   import hicosmo as hc

   # Use the 8 cores allocated by SLURM
   hc.init(8)

   # ... rest of the code

PBS/Torque Cluster
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash
   #PBS -N hicosmo
   #PBS -l nodes=1:ppn=8
   #PBS -l mem=32gb
   #PBS -l walltime=24:00:00

   cd $PBS_O_WORKDIR
   source activate hicosmo
   python my_mcmc_script.py

GPU Cluster
~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=hicosmo_gpu
   #SBATCH --nodes=1
   #SBATCH --gres=gpu:4
   #SBATCH --mem=64G
   #SBATCH --time=24:00:00

   source activate hicosmo
   python my_mcmc_script.py

.. code-block:: python

   import hicosmo as hc

   # Use 4 GPUs
   hc.init(4, device="GPU")

Environment Variable Configuration
-----------------------------------

HIcosmo automatically manages the following environment variables; manual setup is usually not needed:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Environment Variable
     - Description
   * - ``XLA_FLAGS``
     - JAX/XLA compiler configuration, including device count
   * - ``JAX_ENABLE_X64``
     - Enable 64-bit precision (default True)
   * - ``JAX_NUM_THREADS``
     - Number of threads per device

Manual Configuration (Advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For finer-grained control:

.. code-block:: bash

   # Set in shell
   export XLA_FLAGS="--xla_force_host_platform_device_count=8"
   export JAX_ENABLE_X64=True
   python my_script.py

.. code-block:: python

   import os
   os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

   # Must be after setting environment variables and before importing JAX
   import hicosmo as hc
   hc.init(8, force=False)  # Do not override already-set XLA_FLAGS

Viewing Current Configuration
-----------------------------

.. code-block:: python

   import hicosmo as hc

   hc.init(8)

   # View configuration status
   status = hc.Config.status()
   print(status)
   # {
   #     'initialized': True,
   #     'config': {'num_devices': 8, 'device_type': 'cpu', ...},
   #     'system_cores': 16,
   #     'jax_devices': 8,
   #     'jax_device_list': ['CpuDevice(id=0)', 'CpuDevice(id=1)', ...]
   # }

   # View JAX devices
   import jax
   print(f"Available devices: {len(jax.devices())}")
   for d in jax.devices():
       print(f"  {d}")

Performance Tuning Tips
-----------------------

1. **Match the number of chains to the number of devices**

   .. code-block:: python

      hc.init(8)
      # num_chains should be equal to or less than the device count
      mcmc.run(num_samples=2000, num_chains=8)  # ✅ Optimal
      mcmc.run(num_samples=2000, num_chains=4)  # ✅ OK, but wastes 4 devices
      mcmc.run(num_samples=2000, num_chains=16) # ⚠️ Exceeds device count, will queue

2. **Warm up JIT compilation**

   .. code-block:: python

      # The first call triggers JIT compilation (slower)
      result = likelihood(H0=70, Omega_m=0.3)

      # Subsequent calls use the cache (fast)
      result = likelihood(H0=71, Omega_m=0.3)

3. **Memory management**

   .. code-block:: python

      # Monitor memory during large-scale sampling
      import jax
      jax.clear_caches()  # Clear JIT caches to free memory

4. **Avoid frequent initialization**

   .. code-block:: python

      # ❌ Do not initialize inside a loop
      for i in range(10):
          hc.init(8)  # Will print a warning each time
          ...

      # ✅ Initialize only once
      hc.init(8)
      for i in range(10):
          ...

FAQ
---

Q: Why was my configuration ignored?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most common cause is that JAX has already been imported:

.. code-block:: python

   from hicosmo.models import LCDM  # ❌ JAX is imported here
   import hicosmo as hc
   hc.init(8)  # ⚠️ Warning: configuration ignored

Solution: Make sure ``hc.init()`` is called before all JAX-related imports.

Q: GPU not detected?
~~~~~~~~~~~~~~~~~~~~~

1. Check if the JAX GPU version is installed:

   .. code-block:: python

      import jax
      print(jax.devices())  # Should show cuda devices

2. Check the CUDA environment:

   .. code-block:: bash

      nvidia-smi  # Should display GPU information

3. Reinstall the JAX GPU version:

   .. code-block:: bash

      pip uninstall jax jaxlib
      pip install jax[cuda12]

Q: How to use in a Jupyter Notebook?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the first cell of the notebook:

.. code-block:: python

   # Cell 1 - Must be the first cell!
   import hicosmo as hc
   hc.init(8)

   # Cell 2 - Then import other modules
   from hicosmo.models import LCDM
   from hicosmo.samplers import MCMC

.. warning::

   If you restart the kernel, you need to re-run ``hc.init()``.

Q: What to do about out-of-memory errors?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Reduce the number of parallel chains:

   .. code-block:: python

      hc.init(4)  # Instead of 8

2. Reduce the number of samples:

   .. code-block:: python

      mcmc.run(num_samples=1000)  # Instead of 5000

3. Be especially careful when using 64-bit precision:

   .. code-block:: python

      # 32-bit precision can halve memory usage
      os.environ['JAX_ENABLE_X64'] = 'False'

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Scenario
     - Recommended Configuration
   * - Quick testing
     - ``hc.init(1)`` or do not call
   * - Daily use
     - ``hc.init()`` auto-configure
   * - Production run
     - ``hc.init(8)`` 8-core parallelism
   * - GPU acceleration
     - ``hc.init("GPU")``
   * - Cluster submission
     - ``hc.init(N)`` at the beginning of the script

Remember: **``hc.init()`` must be called before importing JAX!**
