Configuration Runner API
========================

The ``hicosmo.runner`` module provides a Cobaya-style YAML configuration system that integrates:

- **Configuration loading/validation**: Load configurations from YAML/JSON files
- **Component registry**: String-to-class mapping for Theory, Sampler, and Likelihood
- **Dataset management**: Data path resolution and automatic downloading

.. automodule:: hicosmo.runner
   :members:
   :undoc-members:
   :show-inheritance:

Quick Start
-----------

.. code-block:: python

   from hicosmo import run_from_config

   # Run inference from a YAML configuration
   result = run_from_config("analysis.yaml")
   samples = result["samples"]

Component Registry
------------------

.. code-block:: python

   from hicosmo.runner import THEORY_REGISTRY, LIKELIHOOD_REGISTRY

   # List available cosmological models
   print(THEORY_REGISTRY.list())  # ['lcdm', 'wcdm', 'cpl', 'ilcdm']

   # List available likelihood functions
   print(LIKELIHOOD_REGISTRY.list())

Dataset Management
------------------

.. code-block:: python

   from hicosmo.runner import ensure_dataset, resolve_data_root

   # Ensure datasets exist
   ensure_dataset("pantheon_plus")
   ensure_dataset("desi2024")

   # Get the data root directory
   data_root = resolve_data_root()
