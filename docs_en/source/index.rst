HIcosmo Documentation
=====================

HIcosmo is a high-performance framework for cosmological parameter estimation and forecasting, with core goals of:
**minimal API, smart defaults, performance-first, and extensibility**.

Quick Links
-----------

- Getting started: `Quick Start <quickstart.html>`_
- Technical foundation: `JAX Technical Guide <jax_introduction.html>`_
- Parameters and likelihoods: `Likelihood Functions <guides/likelihoods.html>`_
- Configuration-driven: `Configuration Guide <guides/config.html>`_
- Complete API: `API Reference <api/index.html>`_

This documentation follows the Read the Docs style inspired by Cobaya, providing clear structured navigation:
- Installation and getting started
- Core concepts and usage guides
- Detailed API reference (with entry points for every interface)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   concepts
   jax_introduction

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/data
   guides/models
   guides/likelihoods
   guides/samplers
   guides/custom_likelihood
   guides/parallel
   guides/config
   guides/visualization
   guides/model_selection
   guides/fisher
   guides/jax_tools

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
