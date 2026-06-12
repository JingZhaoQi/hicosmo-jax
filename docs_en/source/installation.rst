Installation Guide
==================

System Requirements
-------------------

**Operating System**:

- Linux (recommended for best performance)
- macOS (fully supported)
- Windows (supported via WSL2)

**Python Version**:

- Python 3.9 - 3.11 (3.10 recommended)
- Python 3.12+ has not been fully tested

**Hardware Recommendations**:

- CPU: 4+ cores (for parallel MCMC sampling)
- RAM: 8GB+ (16GB+ recommended for large-scale MCMC)
- GPU (optional): NVIDIA CUDA 11.8+ for JAX GPU acceleration

Basic Installation
------------------

**Install from source (recommended)**:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourname/hicosmo.git
   cd hicosmo

   # Create a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or venv\Scripts\activate  # Windows

   # Install
   pip install -e .

**Using conda**:

.. code-block:: bash

   conda create -n hicosmo python=3.10
   conda activate hicosmo
   pip install -e .

Development Installation
------------------------

If you need to develop or debug HIcosmo:

.. code-block:: bash

   pip install -e ".[dev]"

This installs additional development dependencies:

- ``pytest``: testing framework
- ``pytest-cov``: coverage reporting
- ``black``: code formatting
- ``isort``: import sorting
- ``mypy``: type checking
- ``flake8``: code style checking

GPU Acceleration
----------------

HIcosmo is built on JAX and supports GPU acceleration. For NVIDIA GPUs:

.. code-block:: bash

   # First install the CPU version
   pip install -e .

   # Then install JAX GPU version (CUDA 11.8+)
   pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # Or use CUDA 12
   pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

**Verify GPU installation**:

.. code-block:: python

   import jax
   print(jax.devices())
   # Should display something like: [GpuDevice(id=0, process_index=0)]

Core Dependencies
-----------------

HIcosmo's core dependencies include:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Package
     - Version
     - Purpose
   * - jax
     - >=0.4.20
     - High-performance numerical computing and automatic differentiation
   * - jaxlib
     - >=0.4.20
     - JAX compilation backend
   * - numpyro
     - >=0.13.0
     - Probabilistic programming and NUTS sampler
   * - numpy
     - >=1.24.0
     - Array operations
   * - scipy
     - >=1.10.0
     - Scientific computing
   * - astropy
     - >=5.0
     - Astronomical constants and units
   * - getdist
     - >=1.4.0
     - MCMC visualization (corner plots)
   * - matplotlib
     - >=3.7.0
     - Plotting
   * - pyyaml
     - >=6.0
     - Configuration file parsing
   * - rich
     - >=13.0.0
     - Enhanced terminal output

Optional Dependencies
---------------------

**emcee sampler** (alternative sampler):

.. code-block:: bash

   pip install emcee

**Documentation building**:

.. code-block:: bash

   pip install sphinx sphinx_rtd_theme

**Jupyter support**:

.. code-block:: bash

   pip install jupyter ipykernel

Verifying Installation
----------------------

After installation, run the following commands to verify:

.. code-block:: bash

   # Run the test suite
   pytest tests/ -v --tb=short

**Python verification**:

.. code-block:: python

   import hicosmo
   print(hicosmo.__version__)

   # Verify JAX and precision configuration
   import jax
   import jax.numpy as jnp
   x = jnp.array([1.0, 2.0, 3.0])
   print(f"JAX working: {x.sum()}")
   print(f"float64 enabled: {jax.config.jax_enable_x64}")  # should be True

.. note::

   Cosmological distance integrals and covariance solves need float64,
   so ``import hicosmo`` automatically enables JAX's x64 mode. Import
   hicosmo before running any JAX computation. If you genuinely need
   float32 (e.g. tight GPU memory), set the environment variable
   ``HICOSMO_DISABLE_X64=1`` before importing.

   # Verify NumPyro
   import numpyro
   print(f"NumPyro version: {numpyro.__version__}")

   # Quick cosmology calculation test
   from hicosmo.models import LCDM
   model = LCDM(H0=70, Omega_m=0.3)
   print(f"d_L(z=1) = {model.luminosity_distance(1.0):.2f} Mpc")

Troubleshooting
---------------

**JAX version conflicts**:

If you encounter JAX-related errors, ensure ``jax`` and ``jaxlib`` versions match:

.. code-block:: bash

   pip install --upgrade jax jaxlib

**NumPyro import errors**:

Make sure the correct version of NumPyro is installed:

.. code-block:: bash

   pip install --upgrade numpyro

**GetDist plotting issues**:

If corner plots fail, check the matplotlib backend:

.. code-block:: python

   import matplotlib
   matplotlib.use('Agg')  # Non-interactive backend

**Out of memory**:

For large-scale MCMC, configure JAX memory pre-allocation:

.. code-block:: bash

   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

CI/CD Environment Configuration
--------------------------------

For CI environments, the following settings are recommended:

.. code-block:: bash

   # Force CPU usage (avoid GPU-related issues)
   export JAX_PLATFORM_NAME=cpu

   # Disable JAX pre-allocation
   export XLA_PYTHON_CLIENT_PREALLOCATE=false

   # Set random seed for reproducibility
   export PYTHONHASHSEED=42

Read the Docs Build
-------------------

This project's English documentation is built from the ``docs_en/`` directory.

The configuration file is located at ``docs_en/source/conf.py``.

Building the documentation:

.. code-block:: bash

   cd docs_en
   make html
   # Output in docs_en/build/html/
