自定义似然函数
==============

HIcosmo 不仅可以用于宇宙学参数估计，还可以作为通用的 MCMC 采样框架。
本章介绍如何定义自定义似然函数，用于任意参数推断问题。

.. contents:: 本章目录
   :local:
   :depth: 2

为什么使用 HIcosmo 做通用 MCMC？
-------------------------------

HIcosmo 的 MCMC 系统具有以下优势：

- **极简 API**：3 行代码完成采样
- **自动并行**：多链并行采样，充分利用多核 CPU
- **JAX 加速**：自动微分 + JIT 编译，比传统 emcee 快 5-10 倍
- **内置诊断**：自动计算 R-hat、ESS 等收敛诊断
- **开箱即用的可视化**：一行代码生成 corner plot

快速示例：多项式拟合
-------------------

下面用一个简单的多项式拟合示例，展示 HIcosmo MCMC 的完整流程。

问题描述
~~~~~~~~

假设我们有一组观测数据 :math:`(x_i, y_i)`，想要拟合一个二次多项式：

.. math::

   y = a x^2 + b x + c

我们需要从数据推断参数 :math:`(a, b, c)` 的后验分布。

完整代码
~~~~~~~~

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from hicosmo.samplers import MCMC
   from hicosmo.visualization import Plotter

   # ============================================================
   # 步骤 1：准备数据
   # ============================================================
   np.random.seed(42)
   x = np.linspace(0, 2, 30)  # 30 个数据点

   # 真实参数：a=2.5, b=-1.3, c=0.8
   y_true = 2.5 * x**2 - 1.3 * x + 0.8
   y_obs = y_true + np.random.normal(0, 0.3, len(x))  # 添加噪声
   sigma = 0.3  # 观测误差

   # ============================================================
   # 步骤 2：定义似然函数
   # ============================================================
   def log_likelihood(a, b, c):
       """
       高斯似然函数。

       参数名必须与 params 字典中的键名一致！
       """
       y_model = a * x**2 + b * x + c
       chi2 = jnp.sum(((y_obs - y_model) / sigma) ** 2)
       return -0.5 * chi2

   # ============================================================
   # 步骤 3：定义参数（先验范围）
   # ============================================================
   params = {
       'a': {'init': 2.0, 'min': 0.0, 'max': 5.0},
       'b': {'init': 0.0, 'min': -5.0, 'max': 5.0},
       'c': {'init': 0.0, 'min': -2.0, 'max': 2.0},
   }

   # ============================================================
   # 步骤 4：运行 MCMC
   # ============================================================
   mcmc = MCMC(params, log_likelihood, chain_name='poly_fit')
   mcmc.run(num_samples=2000, num_warmup=500, num_chains=4)
   mcmc.print_summary()

   # ============================================================
   # 步骤 5：可视化结果
   # ============================================================
   Plotter('poly_fit').corner(['a', 'b', 'c'], filename='poly_corner.pdf')

   print("真实值: a=2.5, b=-1.3, c=0.8")

运行结果
~~~~~~~~

.. code-block:: text

   参数统计摘要:
   ============================================================
   Parameter        Mean     Std    2.5%   97.5%    R-hat    ESS
   ------------------------------------------------------------
   a              2.503   0.102   2.304   2.702    1.001   3842
   b             -1.312   0.186  -1.676  -0.948    1.000   4012
   c              0.798   0.065   0.671   0.925    1.001   3956
   ============================================================

   真实值: a=2.5, b=-1.3, c=0.8

可以看到，MCMC 准确恢复了真实参数值，且 R-hat ≈ 1.0 表示链已收敛。

API 详解
--------

参数定义格式
~~~~~~~~~~~~

参数通过字典定义，支持两种格式：

**详细格式**（推荐）：

.. code-block:: python

   params = {
       'param_name': {
           'init': 1.0,   # 初始值
           'min': 0.0,    # 先验下界
           'max': 2.0,    # 先验上界
       }
   }

**简洁格式**（元组）：

.. code-block:: python

   params = {
       'param_name': (init, min, max),  # 例如 (1.0, 0.0, 2.0)
   }

似然函数要求
~~~~~~~~~~~~

自定义似然函数需要满足以下条件：

1. **参数名匹配**：函数参数名必须与 ``params`` 字典中的键名完全一致
2. **返回标量**：返回值必须是一个标量（log-likelihood）
3. **使用 JAX**：建议使用 ``jax.numpy`` 而非 ``numpy``，以获得自动微分和 JIT 加速

.. code-block:: python

   import jax.numpy as jnp

   def log_likelihood(a, b, c):  # 参数名与 params 键名一致
       # 使用 jnp 而非 np
       return -0.5 * jnp.sum(...)

MCMC 运行选项
~~~~~~~~~~~~~

.. code-block:: python

   mcmc = MCMC(params, log_likelihood, chain_name='my_chain')

   mcmc.run(
       num_samples=2000,   # 每条链的有效样本数
       num_warmup=500,     # 预热步数（会被丢弃）
       num_chains=4,       # 并行链数
   )

**参数说明**：

- ``num_samples``：每条链的有效样本数，总样本数 = num_samples × num_chains
- ``num_warmup``：预热/burn-in 步数，用于让采样器适应目标分布
- ``num_chains``：并行运行的链数，建议 ≥ 4 以便计算收敛诊断

结果分析
~~~~~~~~

.. code-block:: python

   # 打印统计摘要
   mcmc.print_summary()

   # 获取样本（字典格式）
   samples = mcmc.get_samples()
   print(samples['a'].shape)  # (num_samples * num_chains,)

   # 可视化
   plotter = Plotter('my_chain')
   plotter.corner(['a', 'b', 'c'])           # corner plot
   plotter.traces(['a', 'b', 'c'])           # trace plot
   plotter.get_summary()                      # 统计摘要字典

进阶示例
--------

示例 1：带先验的贝叶斯推断
~~~~~~~~~~~~~~~~~~~~~~~~~~

如果你有先验信息，可以将其加入 log-likelihood：

.. code-block:: python

   def log_likelihood_with_prior(a, b, c):
       # 似然
       y_model = a * x**2 + b * x + c
       log_like = -0.5 * jnp.sum(((y_obs - y_model) / sigma) ** 2)

       # 高斯先验：a ~ N(2.5, 0.5)
       log_prior_a = -0.5 * ((a - 2.5) / 0.5) ** 2

       return log_like + log_prior_a

示例 2：多维数据拟合
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from hicosmo.samplers import MCMC

   # 二维高斯数据
   data = jnp.array([[1.2, 0.8], [1.5, 1.1], [0.9, 0.7], ...])

   def log_likelihood(mu_x, mu_y, sigma):
       """二维高斯分布的似然"""
       dx = data[:, 0] - mu_x
       dy = data[:, 1] - mu_y
       chi2 = jnp.sum((dx**2 + dy**2) / sigma**2)
       n = len(data)
       return -n * jnp.log(sigma) - 0.5 * chi2

   params = {
       'mu_x': {'init': 1.0, 'min': 0.0, 'max': 2.0},
       'mu_y': {'init': 1.0, 'min': 0.0, 'max': 2.0},
       'sigma': {'init': 0.5, 'min': 0.1, 'max': 2.0},
   }

   mcmc = MCMC(params, log_likelihood, chain_name='gaussian_2d')
   mcmc.run(num_samples=3000, num_chains=4)

示例 3：使用 emcee 后端
~~~~~~~~~~~~~~~~~~~~~~~

如果你的似然函数有奇异点或不可微，可以使用 emcee 后端：

.. code-block:: python

   mcmc = MCMC(params, log_likelihood, chain_name='my_chain')
   mcmc.run(
       num_samples=5000,
       num_chains=32,      # emcee 需要更多 walkers
       sampler='emcee'     # 使用 emcee 后端
   )

常见问题
--------

Q: 为什么我的 MCMC 不收敛？
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **增加预热步数**：``num_warmup=1000`` 或更多
2. **检查参数范围**：确保真实值在 [min, max] 范围内
3. **检查初始值**：初始值应该接近真实值的合理估计
4. **简化模型**：先用简单模型测试，确保代码正确

Q: 如何加速采样？
~~~~~~~~~~~~~~~~~

1. **使用 JAX**：确保 log_likelihood 中使用 ``jax.numpy``
2. **增加链数**：更多链 = 更好的并行利用
3. **减少样本数**：先用少量样本调试，确认收敛后再增加

Q: 如何保存和加载结果？
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 保存
   mcmc.save('my_results.pkl')

   # 加载
   from hicosmo.samplers import MCMC
   mcmc = MCMC.load('my_results.pkl')

小结
----

使用 HIcosmo 进行自定义 MCMC 采样只需三步：

1. **定义似然函数**：参数名与 params 键名一致
2. **定义参数范围**：使用字典格式
3. **运行采样**：调用 ``MCMC.run()``

.. code-block:: python

   # 完整示例（3 行核心代码）
   def log_likelihood(a, b): return -0.5 * ((a - 1)**2 + (b - 2)**2)
   params = {'a': (0, -5, 5), 'b': (0, -5, 5)}
   MCMC(params, log_likelihood, chain_name='test').run()

下一步，可以阅读 :doc:`samplers` 了解更多采样器选项和高级配置。
