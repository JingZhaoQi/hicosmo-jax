High-Level API
==============

This page describes the "user-facing" API of HIcosmo, including:
- ``hicosmo`` main entry point
- ``InferenceRunner`` runner
- ``run_from_config`` configuration-driven entry point
- ``list_likelihoods`` / ``list_cosmologies``

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   hicosmo.hicosmo.hicosmo
   hicosmo.hicosmo.InferenceRunner
   hicosmo.run_from_config
   hicosmo.list_likelihoods
   hicosmo.list_cosmologies

Detailed Interface
------------------

.. autofunction:: hicosmo.hicosmo.hicosmo

.. autoclass:: hicosmo.hicosmo.InferenceRunner
   :members:
   :undoc-members:

.. autofunction:: hicosmo.run_from_config

.. autofunction:: hicosmo.list_likelihoods

.. autofunction:: hicosmo.list_cosmologies
