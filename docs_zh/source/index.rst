HIcosmo 中文文档
================

HIcosmo 是面向宇宙学参数估计与预报的高性能框架，核心目标是：
**API 极简、默认智能、性能优先、可扩展**。

快速入口
--------

- 新手入门：`快速开始 <quickstart.html>`_
- 技术基础：`JAX 技术详解 <jax_introduction.html>`_
- 参数与似然：`似然函数 <guides/likelihoods.html>`_
- 配置驱动：`配置驱动 <guides/config.html>`_
- 完整 API：`API 接口详细介绍 <api/index.html>`_

本中文文档参考 Cobaya 的 Read the Docs 风格，提供清晰的结构化导航：
- 入门与安装
- 核心概念与使用指南
- 详细 API 接口说明（每个接口都有入口）

.. toctree::
   :maxdepth: 2
   :caption: 入门

   installation
   quickstart
   concepts
   jax_introduction

.. toctree::
   :maxdepth: 2
   :caption: 使用指南

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
   :caption: API 接口（完整）

   api/index
