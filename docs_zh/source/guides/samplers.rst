采样与推断
==========

HIcosmo 提供灵活的 MCMC 采样系统，支持多种后端采样器、自动参数管理和检查点系统。

采样器概览
----------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 采样器
     - 方法
     - 特点
   * - **NumPyro NUTS**
     - 哈密顿蒙特卡洛
     - 默认，基于梯度，高效
   * - **emcee**
     - 集合采样器
     - 鲁棒，无需梯度，处理 NaN/Inf

快速开始
--------

**4 行完成推断**（含并行初始化）：

.. code-block:: python

   import hicosmo as hc
   hc.init(8)  # 8 个并行设备 → 8 条链同时运行

   from hicosmo import hicosmo
   inf = hicosmo(cosmology="LCDM", likelihood="sn", free_params=["H0", "Omega_m"])
   samples = inf.run()  # 自动 num_chains=8

.. note::

   **新 API（v2.0+）**：``hc.init(N)`` 直接设置 N 个并行设备，默认 ``num_chains=N``。

   - ``hc.init(8)`` → 8 个设备，8 条链并行，CPU 占用 ~800%
   - ``hc.init()`` → 自动检测（最多 8 设备）
   - ``hc.init("GPU")`` → GPU 模式

   **懒加载**：``import hicosmo`` 不会触发 JAX 导入，确保 ``hc.init()`` 能正确设置设备数。

高层 API
--------

``hicosmo()`` 函数
~~~~~~~~~~~~~~~~~~

``hicosmo()`` 是 HIcosmo 的核心入口，提供最简洁的 API：

.. code-block:: python

   from hicosmo import hicosmo

   inf = hicosmo(
       cosmology="LCDM",                    # 宇宙学模型
       likelihood=["sn", "bao", "cmb"],    # 似然函数（可以是列表）
       free_params=["H0", "Omega_m"],      # 自由参数
       num_samples=4000,                   # 总样本数
       num_chains=4                        # 并行链数
   )

   # 运行采样
   samples = inf.run()

   # 查看结果
   inf.summary()

   # 保存 corner plot
   inf.corner_plot("corner.pdf")

**可用 cosmology 字符串**：

- ``"LCDM"``：标准 :math:`\Lambda\text{CDM}` 模型
- ``"wCDM"``：常数暗能量状态方程
- ``"CPL"``：Chevallier-Polarski-Linder 参数化
- ``"ILCDM"``：相互作用暗能量模型

**可用 likelihood 字符串**：

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 字符串
     - 对应似然
   * - ``"sn"``
     - Pantheon+ 超新星
   * - ``"sn_shoes"``
     - Pantheon+SH0ES
   * - ``"bao"``
     - DESI 2024 BAO
   * - ``"cmb"``
     - Planck 2018 距离先验
   * - ``"h0licow"``
     - H0LiCOW 强透镜
   * - ``"tdcosmo"``
     - TDCOSMO 层次贝叶斯

MCMC 类接口
-----------

对于需要更多控制的场景，使用 ``MCMC`` 类：

基本用法
~~~~~~~~

.. code-block:: python

   from hicosmo.samplers import MCMC
   from hicosmo.models import LCDM
   from hicosmo.likelihoods import SN_likelihood

   # 创建似然
   sn = SN_likelihood(LCDM, "pantheon+")

   # 参数配置：(参考值, 最小值, 最大值)
   params = {
       "H0": (70.0, 60.0, 80.0),
       "Omega_m": (0.3, 0.1, 0.5),
   }

   # 创建 MCMC
   mcmc = MCMC(params, sn, chain_name="lcdm_sn")

   # 运行
   samples = mcmc.run(num_samples=4000, num_chains=4)

   # 查看结果
   mcmc.print_summary()

参数配置格式
~~~~~~~~~~~~

HIcosmo 支持多种参数配置格式：

**元组格式**（最简）：

.. code-block:: python

   params = {
       'H0': (70.0, 60.0, 80.0),      # (参考值, 最小值, 最大值)
       'Omega_m': (0.3, 0.1, 0.5),
   }

**字典格式**（完整控制）：

.. code-block:: python

   params = {
       'H0': {
           'prior': {'dist': 'uniform', 'min': 60, 'max': 80},
           'ref': 70.0,
           'latex': r'$H_0$'
       },
       'Omega_m': {
           'prior': {'dist': 'normal', 'loc': 0.3, 'scale': 0.05},
           'bounds': [0.1, 0.5],
           'latex': r'$\Omega_m$'
       }
   }

**支持的先验分布**：

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 分布
     - 参数
     - 描述
   * - ``uniform``
     - ``min``, ``max``
     - 均匀分布
   * - ``normal``
     - ``loc``, ``scale``
     - 正态分布
   * - ``truncated_normal``
     - ``loc``, ``scale``, ``low``, ``high``
     - 截断正态分布

多采样器后端
------------

NumPyro NUTS（默认）
~~~~~~~~~~~~~~~~~~~~

NumPyro NUTS 是默认采样器，使用哈密顿蒙特卡洛方法：

.. code-block:: python

   mcmc = MCMC(
       params, likelihood,
       sampler='numpyro',     # 默认
       chain_method='auto'    # 自动选择并行方式
   )

**特点**：

- 基于梯度的 NUTS 采样器
- JAX JIT 编译加速
- 高效处理高维参数空间
- 自动选择步长和质量矩阵

emcee 采样器
~~~~~~~~~~~~

emcee 是集合采样器，适合处理复杂似然函数：

.. code-block:: python

   mcmc = MCMC(
       params, likelihood,
       sampler='emcee'
   )

**特点**：

- 无需梯度信息
- 鲁棒处理 NaN/Inf
- 适合多模态分布
- 对似然函数数值不稳定更宽容

**选择建议**：

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - 场景
     - 推荐采样器
     - 原因
   * - 标准宇宙学分析
     - NumPyro NUTS
     - 高效，收敛快
   * - 似然函数不稳定
     - emcee
     - 鲁棒处理异常值
   * - 高维参数 (>20)
     - NumPyro NUTS
     - 梯度信息加速收敛
   * - 多模态分布
     - emcee
     - 集合采样避免陷入局部极值

运行配置
--------

采样参数
~~~~~~~~

.. code-block:: python

   samples = mcmc.run(
       num_samples=4000,      # 总样本数（所有链合计）
       num_chains=4,          # 并行链数
       num_warmup=1000,       # 预热步数
       progress_bar=True      # 显示进度条
   )

**注意**：``num_samples`` 是所有链的**总样本数**。4 链 4000 样本意味着每链 1000 样本。

链并行方式
~~~~~~~~~~

.. code-block:: python

   mcmc = MCMC(
       params, likelihood,
       chain_method='auto'    # 自动选择
   )

**可选值**：

- ``'auto'``：自动选择（推荐）
- ``'vectorized'``：使用 vmap 并行化（多核 CPU）
- ``'sequential'``：顺序运行（单线程）
- ``'parallel'``：使用 pmap（多 GPU）

并行配置
~~~~~~~~

.. important::

   **新 API（v2.0+）**：``hc.init(N)`` 直接设置 N 个并行设备，``num_chains`` 默认与设备数一致。

**推荐用法**：

.. code-block:: python

   import hicosmo as hc

   # 最常用：8 个设备，8 条链并行
   hc.init(8)

   # 自动检测（最多 8 设备）
   hc.init()

   # GPU 模式
   hc.init("GPU")

   # 之后导入其他模块
   from hicosmo import hicosmo

**完整示例**：

.. code-block:: python

   import hicosmo as hc
   hc.init(8)  # 必须在最开始！

   from hicosmo.samplers import MCMC
   from hicosmo.models import LCDM
   from hicosmo.likelihoods import SN_likelihood

   likelihood = SN_likelihood(LCDM, "pantheon+")
   params = {'H0': (70.0, 60.0, 80.0), 'Omega_m': (0.3, 0.1, 0.5)}

   mcmc = MCMC(params, likelihood)
   # num_chains 默认 = 8（与设备数一致）
   samples = mcmc.run(num_samples=8000)

.. warning::

   **JAX 已导入警告**：如果 JAX 已经被导入，设备数无法再改变。
   确保 ``hc.init()`` 在脚本最开始调用（在任何 ``import jax`` 之前）。

**API 对照表**：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 调用
     - 效果
   * - ``hc.init(8)``
     - 8 设备，默认 8 链并行
   * - ``hc.init(4)``
     - 4 设备，默认 4 链并行
   * - ``hc.init()``
     - 自动检测（最多 8）
   * - ``hc.init("GPU")``
     - GPU 模式，自动检测卡数

检查点系统
----------

HIcosmo 提供完整的检查点和断点续跑功能（支持时间驱动保存）。

自动检查点
~~~~~~~~~~

.. code-block:: python

   mcmc = MCMC(
       params, likelihood,
       enable_checkpoints=True,           # 启用检查点
       checkpoint_interval_seconds=600,   # 每 10 分钟保存一次（推荐）
       checkpoint_dir="checkpoints"       # 保存目录
   )

.. note::

   ``checkpoint_interval`` 仍然可用，但它表示**所有链合计**的样本数（总样本数）。
   如果使用 ``checkpoint_interval``，建议配合 ``num_chains`` 做估算。

流程图（时间驱动保存）
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   [开始] 
      ↓
   [MCMC.run()]
      ↓
   [分段采样 chunk]
      ↓
   [是否到达时间间隔?]
      ├─ 否 → 继续采样
      └─ 是 → 保存 checkpoint (chain_step_N.h5)
      ↓
   [达到总样本数]
      ↓
   [保存最终 checkpoint + 最新别名 chain.h5]
      ↓
   [结束]

断点续跑
~~~~~~~~

推荐使用显式的 ``MCMC.resume``：

.. code-block:: python

   # 从检查点恢复（需要提供 likelihood）
   mcmc = MCMC.resume("checkpoints/chain_step_500.h5", likelihood)
   samples = mcmc.run()  # 继续直到完成剩余样本

.. note::

   若检查点包含内部状态（NumPyro/NUTS），将实现**真正续跑**；否则会在已有样本后追加新样本。

手动加载检查点
~~~~~~~~~~~~~~

.. code-block:: python

   # 列出可用检查点（需要 chain_name 与 checkpoint_dir）
   mcmc = MCMC(params, likelihood, chain_name="chain", checkpoint_dir="checkpoints")
   mcmc.list_checkpoints()

   # 从检查点恢复（.h5）
   mcmc = MCMC.resume("checkpoints/chain_step_500.h5", likelihood)
   mcmc.run()

收敛诊断
--------

Gelman-Rubin 统计量
~~~~~~~~~~~~~~~~~~~

.. math::

   \hat{R} = \sqrt{\frac{\hat{V}}{W}}

其中 :math:`\hat{V}` 是总方差估计，:math:`W` 是链内方差。

**收敛标准**：:math:`\hat{R} < 1.01`

.. code-block:: python

   # 打印诊断信息（包含 R̂）
   mcmc.print_summary()

   # 输出示例：
   # Parameter   Mean    Std    R̂      ESS
   # -----------------------------------------
   # H0         67.36   0.42   1.002  2341
   # Omega_m    0.315   0.007  1.001  2156

有效样本数 (ESS)
~~~~~~~~~~~~~~~~

有效样本数反映独立样本的数量：

.. math::

   \text{ESS} = \frac{N}{1 + 2\sum_{k=1}^K \rho_k}

其中 :math:`\rho_k` 是滞后 :math:`k` 的自相关系数。

**建议**：ESS > 100 per parameter

似然诊断
~~~~~~~~

运行前检查似然函数的数值稳定性：

.. code-block:: python

   from hicosmo.samplers import LikelihoodDiagnostics

   diagnostics = LikelihoodDiagnostics(likelihood, params)
   result = diagnostics.run(n_tests=100)
   diagnostics.print_report(result)

   # 如果成功率低，建议使用 emcee
   if result.success_rate < 0.5:
       mcmc = MCMC(params, likelihood, sampler='emcee')

结果处理
--------

获取样本
~~~~~~~~

.. code-block:: python

   # 运行 MCMC
   samples = mcmc.run(num_samples=4000, num_chains=4)

   # 获取参数样本
   H0_samples = samples['H0']
   Omega_m_samples = samples['Omega_m']

   # 计算统计量
   import numpy as np
   print(f"H0 = {np.mean(H0_samples):.2f} ± {np.std(H0_samples):.2f}")

保存结果
~~~~~~~~

.. code-block:: python

   # 保存到文件
   mcmc.save_results("results/lcdm_sn.pkl")

   # 加载结果
   from hicosmo.samplers import MCMC
   loaded = MCMC.load_results("results/lcdm_sn.pkl")

GetDist 兼容
~~~~~~~~~~~~

.. code-block:: python

   from getdist import MCSamples

   # 转换为 GetDist 格式
   gd_samples = MCSamples(
       samples=[samples['H0'], samples['Omega_m']],
       names=['H0', 'Omega_m'],
       labels=[r'H_0', r'\Omega_m']
   )

   # 使用 GetDist 绘图
   from getdist import plots
   g = plots.get_subplot_plotter()
   g.triangle_plot(gd_samples, ['H0', 'Omega_m'])

Nuisance 参数自动收集
---------------------

MCMC 自动从似然函数收集 nuisance 参数：

.. code-block:: python

   from hicosmo.likelihoods import TDCOSMO
   from hicosmo.samplers import MCMC

   tdcosmo = TDCOSMO(LCDM)

   # 只需指定宇宙学参数
   cosmo_params = {
       'H0': (70.0, 60.0, 80.0),
       'Omega_m': (0.3, 0.1, 0.5),
   }

   # MCMC 自动收集 TDCOSMO 的 nuisance 参数
   mcmc = MCMC(cosmo_params, tdcosmo)

   # 打印所有参数（包括自动收集的）
   print(mcmc.param_names)
   # ['H0', 'Omega_m', 'lambda_int_mean', 'lambda_int_sigma', ...]

性能优化
--------

初始化优化
~~~~~~~~~~

对于复杂问题，可使用优化找到初始点：

.. code-block:: python

   mcmc = MCMC(
       params, likelihood,
       optimize_init=True,            # 优化初始点
       max_opt_iterations=500,        # 最大迭代次数
       opt_learning_rate=0.01         # 学习率
   )

**适用场景**：

- 似然函数评估 > 10ms
- 参数维度 > 20
- 多模态问题

JIT 预热
~~~~~~~~

首次运行时 JAX 会编译函数，建议：

.. code-block:: python

   # 预热（可选，但推荐）
   _ = likelihood(H0=70, Omega_m=0.3)

   # 正式运行
   samples = mcmc.run()

性能基准
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - 配置
     - qcosmc (scipy)
     - HIcosmo (JAX)
     - 加速比
   * - LCDM + Pantheon+ (10k 样本)
     - 180s
     - 45s
     - **4x**
   * - CPL + BAO + SN (10k 样本)
     - 420s
     - 85s
     - **5x**
   * - 4 链并行（8 核 CPU）
     - N/A
     - 自动
     - —

完整示例
--------

**使用新 API 的完整示例**：

.. code-block:: python

   # 1. 首先初始化（必须在最前面！）
   import hicosmo as hc
   hc.init(8)  # 8 个并行设备 = 8 条并行链

   # 2. 导入其他模块
   from hicosmo.samplers import MCMC
   from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
   from hicosmo.models import LCDM

   # 3. 创建联合似然
   sne = SN_likelihood(LCDM, "pantheon+")
   bao = BAO_likelihood(LCDM, "desi2024")

   def joint_likelihood(**params):
       return sne(**params) + bao(**params)

   # 4. 参数配置
   params = {
       'H0': {
           'prior': {'dist': 'uniform', 'min': 60, 'max': 80},
           'ref': 70.0,
           'latex': r'$H_0$'
       },
       'Omega_m': {
           'prior': {'dist': 'uniform', 'min': 0.1, 'max': 0.5},
           'ref': 0.3,
           'latex': r'$\Omega_m$'
       }
   }

   # 5. 创建 MCMC
   mcmc = MCMC(
       params,
       joint_likelihood,
       chain_name="lcdm_joint",
       enable_checkpoints=True
   )

   # 6. 运行采样（num_chains 默认 = 设备数 = 8）
   samples = mcmc.run(num_samples=8000)

   # 7. 查看结果
   mcmc.print_summary()
   mcmc.save_results("results/lcdm_joint.pkl")

**极简示例**（3 行代码）：

.. code-block:: python

   import hicosmo as hc
   hc.init(8)  # 8 个并行设备

   from hicosmo import hicosmo
   inf = hicosmo("LCDM", ["sn", "bao"], ["H0", "Omega_m"])
   samples = inf.run()

下一步
------

- `可视化 <visualization.html>`_：生成出版级图表
- `Fisher 预报 <fisher.html>`_：进行巡天预测
