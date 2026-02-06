核心概念
========

HIcosmo 是一个面向宇宙学参数估计的高性能框架，本章介绍其核心设计理念和架构。

设计哲学
--------

API 简洁性第一原则
~~~~~~~~~~~~~~~~~~

HIcosmo 的设计遵循 **API 简洁性直接决定项目成败** 的原则：

.. code-block:: text

   API简洁性 → 降低学习成本 → 提高用户选择性 → 增强用户粘度 → 项目成功

**设计标准**：

- ✅ **3 行代码完成核心功能** — 这是我们的 API 设计目标
- ✅ **智能默认参数** — 90% 的使用场景不需要额外配置
- ✅ **渐进式复杂度** — 简单任务简单做，复杂任务才暴露复杂性
- ❌ **绝不为了"完整性"增加必选参数** — 每个新参数都是用户流失的风险

JAX-First 架构
~~~~~~~~~~~~~~

HIcosmo 基于 JAX 构建，具有以下特性：

- **纯函数优先**：所有宇宙学计算为无副作用的纯函数
- **自动微分**：利用 JAX 的 autograd 进行精确梯度计算
- **JIT 编译**：关键计算路径 JIT 编译获得 4-7 倍性能提升
- **GPU 就绪**：透明 GPU 加速支持
- **向量化**：vmap 实现高效批量计算

六层架构
--------

HIcosmo 采用分层架构设计：

.. code-block:: text

   ┌─────────────────────────────────────────────────────┐
   │  Layer 6: Visualization（可视化）                   │
   │  Plotter, plot_corner, plot_chains, GetDist集成     │
   ├─────────────────────────────────────────────────────┤
   │  Layer 5: Fisher Matrix（Fisher矩阵预测）           │
   │  IntensityMappingFisher, FisherMatrix, 巡天预测    │
   ├─────────────────────────────────────────────────────┤
   │  Layer 4: Samplers（采样器）                        │
   │  MCMC, NumPyro NUTS, emcee, 检查点系统             │
   ├─────────────────────────────────────────────────────┤
   │  Layer 3: Likelihoods（似然函数）                   │
   │  SN, BAO, CMB, H0LiCOW, TDCOSMO, GW               │
   ├─────────────────────────────────────────────────────┤
   │  Layer 2: Models（宇宙学模型）                      │
   │  LCDM, wCDM, CPL, ILCDM                            │
   ├─────────────────────────────────────────────────────┤
   │  Layer 1: Core（核心基础）                          │
   │  CosmologyBase, 距离计算, 统一参数系统             │
   └─────────────────────────────────────────────────────┘

Layer 1: Core（核心基础）
~~~~~~~~~~~~~~~~~~~~~~~~~

``hicosmo/core/`` 包含：

- **CosmologyBase**：所有宇宙学模型的抽象基类
- **compute_distances_from_E_z()**：JIT 编译的距离计算引擎
- **make_compute_grid_traced()**：工厂函数生成 traced-aware 计算

Layer 2: Models（宇宙学模型）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hicosmo/models/`` 包含：

- **LCDM**：标准 :math:`\Lambda\text{CDM}` 模型
- **wCDM**：常数暗能量状态方程
- **CPL**：Chevallier-Polarski-Linder 参数化
- **ILCDM**：相互作用暗能量模型

Layer 3: Likelihoods（似然函数）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hicosmo/likelihoods/`` 包含：

- **PantheonPlus**：1701 个 SNe Ia 的距离测量
- **BAO**：DESI 2024, BOSS, SDSS 的 BAO 约束
- **Planck**：CMB 距离先验
- **H0LiCOW / TDCOSMO**：强引力透镜时延

Layer 4: Samplers（采样器）
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hicosmo/samplers/`` 包含：

- **MCMC**：高层 MCMC 接口
- **NumPyro Backend**：NUTS 采样器封装
- **emcee Backend**：集合采样器（备用）
- **Persistence**：检查点和恢复系统

Layer 5: Fisher Matrix
~~~~~~~~~~~~~~~~~~~~~~

``hicosmo/fisher/`` 包含：

- **IntensityMappingFisher**：21cm 强度映射 Fisher 预测
- **FisherMatrix**：通用 Fisher 矩阵计算

Layer 6: Visualization
~~~~~~~~~~~~~~~~~~~~~~

``hicosmo/visualization/`` 包含：

- **Plotter**：统一的可视化类
- **plot_corner, plot_chains**：函数接口
- **GetDist 集成**：出版级图表

模块职责边界
------------

**核心原则：每个模块只做一件事，绝不越界！**

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - 模块
     - 计算本质
     - 输入 → 输出
   * - ``models/``
     - **宇宙学预测**
     - :math:`(H_0, \Omega_m, z) \to (d_L, D_M, H(z), r_d)`
   * - ``likelihoods/``
     - **统计推断**
     - :math:`(\text{theory}, \text{data}, C) \to \chi^2`
   * - ``samplers/``
     - **参数探索**
     - :math:`(\text{prior}, \mathcal{L}) \to \text{posterior samples}`
   * - ``visualization/``
     - **结果展示**
     - :math:`\text{samples} \to \text{plots}`

**判断代码属于哪个模块**：

- 问："这段代码在计算什么？"
- 如果是 "从宇宙学参数计算物理量" → ``models/``
- 如果是 "比较理论值和观测值" → ``likelihoods/``

参数系统
--------

HIcosmo 使用统一的参数管理系统。

Parameter 类
~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.parameters import Parameter

   param = Parameter(
       name='H0',
       value=70.0,                    # 参考值/初值
       free=True,                     # 是否采样
       prior={
           'dist': 'uniform',
           'min': 60, 'max': 80
       },
       latex_label=r'$H_0$',          # 用于绑图
       description='Hubble constant'
   )

ParameterRegistry
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.parameters import ParameterRegistry

   # 从预设创建
   registry = ParameterRegistry.from_defaults('planck2018')

   # 设置自由参数
   registry.set_free(['H0', 'Omega_m'])

   # 从 likelihood 添加 nuisance 参数
   registry.add_from_likelihood(tdcosmo)

预设参数集
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 55

   * - 预设
     - :math:`H_0`
     - :math:`\Omega_m`
     - 说明
   * - ``planck2018``
     - 67.36
     - 0.3153
     - Planck 2018 TT,TE,EE+lowE+lensing
   * - ``wmap9``
     - 70.0
     - 0.279
     - WMAP 9年数据

距离计算
--------

HIcosmo 提供完整的宇宙学距离计算，支持平直和非平直宇宙。

基本距离量
~~~~~~~~~~

**共动距离** :math:`d_C`：

.. math::

   d_C(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}

**横向共动距离** :math:`D_M`（考虑曲率）：

.. math::

   D_M = \begin{cases}
   D_H / \sqrt{\Omega_k} \sinh(\sqrt{\Omega_k} \, d_C / D_H) & \Omega_k > 0 \\
   d_C & \Omega_k = 0 \\
   D_H / \sqrt{|\Omega_k|} \sin(\sqrt{|\Omega_k|} \, d_C / D_H) & \Omega_k < 0
   \end{cases}

**光度距离**：

.. math::

   d_L(z) = (1+z) \, D_M(z)

**角直径距离**：

.. math::

   D_A(z) = \frac{D_M(z)}{1+z}

**距离模数**：

.. math::

   \mu = 5 \log_{10}(d_L / \text{Mpc}) + 25

Traced-Aware 接口
~~~~~~~~~~~~~~~~~

对于 NumPyro MCMC 采样，HIcosmo 提供 traced-aware 的批量计算接口：

.. code-block:: python

   from hicosmo.models import LCDM
   import jax.numpy as jnp

   z_grid = jnp.linspace(0.01, 2.0, 1000)
   params = {'H0': 70.0, 'Omega_m': 0.3}

   # 批量计算所有距离
   grid = LCDM.compute_grid_traced(z_grid, params)

   # 返回字典包含：d_L, D_M, D_H, dVc_dz, E_z, d_C

数据管理
--------

HIcosmo 通过 ``runner`` 模块管理数据目录和配置系统。

数据目录查找顺序
~~~~~~~~~~~~~~~~

1. 环境变量 ``HICOSMO_DATA``
2. 配置中的 ``data.root``
3. 默认使用项目根目录的 ``data/``

数据集结构
~~~~~~~~~~

.. code-block:: text

   data/
   ├── sne/                    # 超新星数据
   │   ├── Pantheon+SH0ES.dat
   │   └── Pantheon+SH0ES_STAT+SYS.cov
   ├── bao_data/               # BAO 数据
   │   ├── desi_2024/
   │   ├── boss_dr12/
   │   └── sdss_dr12/
   ├── h0licow/                # 强透镜数据
   └── tdcosmo/                # TDCOSMO 数据

性能特性
--------

HIcosmo 针对性能进行了深度优化：

.. list-table::
   :header-rows: 1
   :widths: 35 20 20 25

   * - 操作
     - qcosmc (scipy)
     - HIcosmo (JAX)
     - 加速比
   * - 距离计算 (1000点)
     - 0.15s
     - 0.02s
     - **7.5x**
   * - MCMC 采样 (10k样本)
     - 180s
     - 45s
     - **4.0x**
   * - BAO likelihood (JIT)
     - 21.8ms
     - 1.24ms
     - **17.6x**

下一步
------

- `宇宙学模型 <guides/models.html>`_：了解 LCDM, wCDM, CPL 等模型
- `似然函数 <guides/likelihoods.html>`_：了解 SN, BAO, CMB 等数据
- `采样器 <guides/samplers.html>`_：了解 MCMC 配置和诊断
