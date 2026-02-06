快速开始
========

本指南将带你快速上手 HIcosmo，实现宇宙学参数估计。

3 行代码完成推断
----------------

HIcosmo 的设计哲学是 **3 行代码完成核心功能**：

.. code-block:: python

   from hicosmo import hicosmo

   inf = hicosmo(cosmology="LCDM", likelihood="sn", free_params=["H0", "Omega_m"])
   samples = inf.run()

运行完毕后，你将获得 :math:`H_0` 和 :math:`\Omega_m` 的后验分布样本。

.. tip::

   **并行加速**：在脚本最开头添加以下代码启用多设备并行：

   .. code-block:: python

      import hicosmo as hc
      hc.init(8)  # 8 个并行设备 = 8 条并行链 = ~800% CPU

   必须在导入其他模块之前调用！

基本示例
--------

单探针分析（超新星）
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo import hicosmo

   # 使用 Pantheon+ 超新星数据约束 ΛCDM 模型
   inf = hicosmo(
       cosmology="LCDM",
       likelihood="sn",           # Pantheon+ 数据
       free_params=["H0", "Omega_m"],
       num_samples=4000,          # 总样本数
       num_chains=4               # 并行链数
   )

   # 运行 MCMC 采样
   samples = inf.run()

   # 查看结果摘要
   inf.summary()

   # 生成 corner plot
   inf.corner_plot('corner_sn.pdf')

多探针联合分析
~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo import hicosmo

   # 联合 SN + BAO + CMB 约束
   inf = hicosmo(
       cosmology="LCDM",
       likelihood=["sn", "bao", "cmb"],  # 多探针联合
       free_params=["H0", "Omega_m"],
       num_samples=8000,
       num_chains=4
   )

   samples = inf.run()
   inf.summary()
   inf.corner_plot('corner_joint.pdf')

支持的 Likelihood 字符串
------------------------

HIcosmo 支持以下便捷的 likelihood 字符串：

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - 字符串
     - 对应类
     - 说明
   * - ``sn``
     - ``SN_likelihood("pantheon+")``
     - Pantheon+ 1701个 SNe Ia
   * - ``sn_shoes``
     - ``SN_likelihood("pantheon+shoes")``
     - 含 Cepheid 标校
   * - ``bao``
     - ``BAO_likelihood("desi2024")``
     - DESI 2024 BAO
   * - ``bao_sdss_dr12``
     - ``BAO_likelihood("sdss_dr12")``
     - SDSS DR12 BAO
   * - ``cmb``
     - ``Planck2018DistancePriorsLikelihood``
     - Planck 2018 压缩似然
   * - ``h0licow``
     - ``H0LiCOWLikelihood``
     - 强透镜时延
   * - ``tdcosmo``
     - ``TDCOSMOLikelihood``
     - TDCOSMO 层次贝叶斯
   * - ``sh0es``
     - ``SH0ESLikelihood``
     - SH0ES 局部 :math:`H_0`

扩展暗能量模型
--------------

**wCDM 模型**（常数暗能量状态方程）：

.. code-block:: python

   inf = hicosmo(
       cosmology="wCDM",
       likelihood=["sn", "bao", "cmb"],
       free_params=["H0", "Omega_m", "w0"],
       num_samples=10000
   )
   samples = inf.run()

**CPL 参数化**（时变暗能量）：

.. math::

   w(a) = w_0 + w_a (1 - a)

.. code-block:: python

   inf = hicosmo(
       cosmology="CPL",
       likelihood=["sn", "bao", "cmb"],
       free_params=["H0", "Omega_m", "w0", "wa"],
       num_samples=16000
   )
   samples = inf.run()

自定义似然函数
--------------

HIcosmo 不仅可以用于宇宙学分析，还可以用于任意参数拟合问题。以下示例展示如何拟合二次多项式 :math:`y = ax^2 + bx + c`：

.. code-block:: python

   #!/usr/bin/env python
   """
   Simple MCMC Example: Polynomial Fitting (y = a*x² + b*x + c)

   """
   import hicosmo as hc
   hc.init()
   import numpy as np
   import jax.numpy as jnp
   from pathlib import Path

   from hicosmo.samplers import MCMC
   from hicosmo.visualization import Plotter

   # Load data
   data_path = Path(__file__).parent / 'data' / 'sim_data.txt'
   x, y_obs, y_err = np.loadtxt(data_path, unpack=True)
   x, y_obs, y_err = jnp.asarray(x), jnp.asarray(y_obs), jnp.asarray(y_err)


   def log_likelihood(a, b, c):
       """Log-likelihood: -0.5 * chi²"""
       y_th = a * x**2 + b * x + c
       return -0.5 * jnp.sum((y_obs - y_th)**2 / y_err**2)


   # Parameters: {name: (initial, min, max, latex)}
   params = {
       'a': (3.5, 0.0, 10.0, r'$a$'),
       'b': (1.0, 0.0, 4.0,  r'$b$'),
       'c': (1.0, 0.0, 3.0,  r'$c$'),
   }

   if __name__ == '__main__':
       # Run MCMC
       mcmc = MCMC(params, log_likelihood, chain_name='polynomial_fit')
       mcmc.run(num_samples=20000)

       # Plot results
       plotter = Plotter('polynomial_fit')
       plotter.corner()
       plotter.report()

**参数格式说明**：

.. code-block:: python

   # 字典格式: {name: (initial, min, max, latex)}
   params = {
       'a': (3.5, 0.0, 10.0, r'$a$'),  # 参数 a: 初值 3.5，范围 [0, 10]
       'b': (1.0, 0.0, 4.0,  r'$b$'),  # 参数 b: 初值 1.0，范围 [0, 4]
       'c': (1.0, 0.0, 3.0,  r'$c$'),  # 参数 c: 初值 1.0，范围 [0, 3]
   }

**运行结果**：

.. code-block:: text

   INFO:hicosmo.samplers.init:✅ HIcosmo: 8 devices ready for parallel MCMC
   INFO:hicosmo.samplers.inference:MCMC completed:
   INFO:hicosmo.samplers.inference:  Start time: 2026-01-08 16:54:43
   INFO:hicosmo.samplers.inference:  End time:   2026-01-08 16:54:45
   INFO:hicosmo.samplers.inference:  Duration:   1.94s
   INFO:hicosmo.samplers.inference:  Saved to:   mcmc_chains/polynomial_fit.h5

.. tip::

   生成的报告 (``plotter.report()``) 包含：

   - 参数约束（均值、中位数、68%/95% 置信区间）
   - LaTeX 格式输出（可直接用于论文）
   - MCMC 诊断（ESS、R̂）
   - 模型选择准则（AIC、贝叶斯证据）

使用类接口（高级）
------------------

对于需要更多控制的场景，可以直接使用类接口：

.. code-block:: python

   # 1. 首先初始化（必须在最前面！）
   import hicosmo as hc
   hc.init(8)  # 8 个并行设备

   # 2. 导入其他模块
   from hicosmo.models import LCDM
   from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
   from hicosmo.samplers import MCMC

   # 3. 创建 likelihood 对象
   sne = SN_likelihood(LCDM, "pantheon+", M_B="marginalize")
   bao = BAO_likelihood(LCDM, "desi2024")

   # 联合 likelihood
   joint_likelihood = sne + bao

   # 4. 配置参数
   params = {
       "H0": (70.0, 60.0, 80.0),       # (参考值, 最小值, 最大值)
       "Omega_m": (0.3, 0.1, 0.5),
   }

   # 5. 创建 MCMC 并运行（num_chains 默认 = 设备数 = 8）
   mcmc = MCMC(params, joint_likelihood, chain_name="lcdm_sn_bao")
   samples = mcmc.run(num_samples=4000)

   # 6. 输出结果
   mcmc.print_summary()

可视化结果
----------

**Corner Plot**：

.. code-block:: python

   from hicosmo.visualization import Plotter

   # 从保存的链加载
   plotter = Plotter("lcdm_sn_bao")
   plotter.corner(["H0", "Omega_m"], filename="corner.pdf")

**多链对比**：

.. code-block:: python

   plotter = Plotter(
       ["chain_planck", "chain_shoes", "chain_bao"],
       chain_labels=["Planck", "SH0ES", "DESI BAO"]
   )
   plotter.corner(["H0", "Omega_m"], filename="h0_tension.pdf")

保存和加载结果
--------------

.. code-block:: python

   # 保存结果
   inf.save_results("results/my_analysis.pkl")

   # 加载并继续分析
   from hicosmo.visualization import Plotter
   plotter = Plotter("results/my_analysis.pkl")
   plotter.corner(["H0", "Omega_m"])

YAML 配置驱动
-------------

对于可复现的分析，推荐使用 YAML 配置文件：

**config.yaml**：

.. code-block:: yaml

   name: joint_analysis
   theory: LCDM

   likelihood:
     - name: sn
       dataset: pantheon+
     - name: bao
       dataset: desi2024
     - name: cmb
       dataset: planck2018

   params:
     H0:
       prior: {dist: uniform, min: 50, max: 100}
       ref: 70.0
     Omega_m:
       prior: {dist: uniform, min: 0.1, max: 0.5}
       ref: 0.3

   sampler:
     name: numpyro
     num_samples: 8000
     num_chains: 4
     num_warmup: 1000

   output:
     root: results
     chain_name: joint_lcdm

**运行配置**：

.. code-block:: python

   from hicosmo import run_from_config

   run_from_config("config.yaml")

下一步
------

- 阅读 `核心概念 <concepts.html>`_ 了解 HIcosmo 架构
- 查看 `似然函数 <guides/likelihoods.html>`_ 了解更多数据集
- 学习 `可视化 <guides/visualization.html>`_ 生成出版级图表
- 探索 `Fisher 预测 <guides/fisher.html>`_ 进行巡天预测
