可视化
======

HIcosmo 提供基于 GetDist 的专业级可视化系统，支持 MCMC 样本可视化、多链比较和 Fisher 预报绘图。

可视化概览
----------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 图表类型
     - 用途
     - 适用场景
   * - **Corner Plot**
     - 参数联合约束
     - 展示参数间相关性和边缘分布
   * - **2D Contour**
     - 两参数约束
     - 重点展示特定参数对的置信区间
   * - **1D Marginal**
     - 单参数约束
     - 展示各参数的后验分布
   * - **Trace Plot**
     - 收敛诊断
     - 检查 MCMC 链的收敛性
   * - **Fisher Contour**
     - 预报约束
     - 可视化 Fisher 矩阵预测结果

快速开始
--------

**3 行绘制 corner 图**：

.. code-block:: python

   from hicosmo.visualization import Plotter

   plotter = Plotter('my_chain')
   plotter.corner(['H0', 'Omega_m'], filename='corner.pdf')

Plotter 类接口
--------------

``Plotter`` 类是 HIcosmo 可视化的统一入口，支持多种数据输入格式和丰富的绘图功能。

初始化
~~~~~~

.. code-block:: python

   from hicosmo.visualization import Plotter

   # 方式 1：从链名称加载（最简单）
   plotter = Plotter('my_chain')  # 自动从 mcmc_chains/my_chain.h5 加载

   # 方式 2：多链比较
   plotter = Plotter(
       ['chain1', 'chain2', 'chain3'],
       labels=['Planck', 'SH0ES', 'DESI']  # 图例标签
   )

   # 方式 3：从字典创建
   plotter = Plotter({
       'H0': samples_h0,      # 1D array
       'Omega_m': samples_om  # 1D array
   })

   # 方式 4：从 MCMC 对象创建
   plotter = Plotter.from_mcmc(mcmc)

   # 方式 5：从 Fisher 结果创建
   plotter = Plotter.from_fisher(
       mean=[67.36, 0.315],
       covariance=[[0.5, 0.01], [0.01, 0.007]],
       param_names=['H0', 'Omega_m']
   )

**初始化参数**：

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - 参数
     - 类型
     - 说明
   * - ``data``
     - 多种格式
     - 链名称、路径、字典、数组或 MCSamples
   * - ``labels``
     - dict 或 list
     - 参数 LaTeX 标签
   * - ``ranges``
     - dict
     - 绘图范围 ``{name: (min, max)}``
   * - ``chain_labels``
     - list
     - 多链模式的图例标签
   * - ``style``
     - str
     - 配色方案（``'modern'`` 或 ``'classic'``）
   * - ``results_dir``
     - str 或 Path
     - 结果保存目录

Corner 图
~~~~~~~~~

Corner 图（三角图）是宇宙学参数约束可视化的标准方式，同时展示参数的边缘分布和二维联合分布。

.. code-block:: python

   # 绘制所有参数
   fig = plotter.corner()

   # 选择特定参数（按名称）
   fig = plotter.corner(['H0', 'Omega_m', 'rd_h'])

   # 选择特定参数（按索引，0-based）
   fig = plotter.corner([0, 1, 2])  # 前三个参数

   # 保存到文件
   fig = plotter.corner(['H0', 'Omega_m'], filename='corner.pdf')

**关键特性**：

- 对角线：1D 边缘分布（填充密度）
- 非对角线：2D 等高线（68% 和 95% 置信区间）
- 自动 LaTeX 标签（从链文件元数据读取）
- 智能刻度优化（防止标签重叠）

2D 等高线图
~~~~~~~~~~~

单独绘制两个参数的 2D 约束：

.. code-block:: python

   # 绘制两参数等高线
   fig = plotter.plot_2d('H0', 'Omega_m', filename='h0_om.pdf')

   # 使用参数索引
   fig = plotter.plot_2d(0, 1)

   # 不填充
   fig = plotter.plot_2d('w0', 'wa', filled=False)

1D 边缘分布
~~~~~~~~~~~

绘制单参数的后验分布：

.. code-block:: python

   # 单参数分布
   fig = plotter.plot_1d('H0', filename='h0_posterior.pdf')

   # 多参数网格布局
   fig = plotter.marginals(['H0', 'Omega_m', 'Omega_b'])

链轨迹图
~~~~~~~~

用于 MCMC 收敛诊断的轨迹图：

.. code-block:: python

   # 默认显示前 6 个参数
   fig = plotter.traces()

   # 选择特定参数
   fig = plotter.traces(['H0', 'Omega_m'], filename='traces.pdf')

**收敛诊断要点**：

- 良好收敛：链应该在均值附近随机波动
- 未收敛信号：明显的趋势或周期性
- 预热问题：初始部分明显偏离平衡态

统计摘要
~~~~~~~~

获取参数的统计信息：

.. code-block:: python

   summary = plotter.get_summary()

   # 返回结构：
   # {
   #     'H0': {'mean': 67.36, 'std': 0.42, 'median': 67.35, 'q16': 66.94, 'q84': 67.78},
   #     'Omega_m': {'mean': 0.315, 'std': 0.007, ...}
   # }

   # 打印结果
   for param, stats in summary.items():
       print(f"{param}: {stats['mean']:.3f} ± {stats['std']:.3f}")

分析报告
~~~~~~~~

生成包含参数约束和诊断信息的 Markdown 报告：

.. code-block:: python

   # 单链模式：生成一份报告
   plotter = Plotter('my_chain')
   plotter.report()  # 自动保存为 my_chain_report.md

   # 自定义标题
   plotter.report(title='LCDM Analysis Report')

   # 不包含 LaTeX 格式约束
   plotter.report(include_latex=False)

**报告内容**：

- **运行信息**：链名称、采样器、链数、预热步数、样本数
- **似然函数**：类名、签名、描述
- **宇宙学参数**：先验、边界、是否自由
- **Nuisance 参数**：先验、边界、描述
- **参数约束**：均值±σ、中位数、68%/95% 置信区间
- **LaTeX 格式**：1σ 和 2σ 约束的 LaTeX 表达式
- **MCMC 诊断**：总样本数、有效样本数 (ESS)

多链比较
--------

HIcosmo 支持多条 MCMC 链的对比可视化，这在比较不同数据集或模型时非常有用。

基本用法
~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import Plotter

   # 从多个链名称创建
   plotter = Plotter(
       ['planck_chain', 'shoes_chain', 'desi_chain'],
       labels=['Planck 2018', 'SH0ES', 'DESI DR1']
   )

   # 绘制对比 corner 图
   fig = plotter.corner(['H0', 'Omega_m'], filename='comparison.pdf')

自动文件命名
~~~~~~~~~~~~

多链模式下，Corner 图自动使用 ``+`` 连接所有链名生成文件名：

.. code-block:: python

   plotter = Plotter(['test_sne', 'test_bao'], labels=['Pantheon+', 'DESI R1'])
   plotter.corner()  # 自动保存为 test_sne+test_bao_corner.pdf

自动参数收集
~~~~~~~~~~~~

调用 ``corner()`` 时不指定参数，会自动绘制 **所有链的所有参数**，包括 nuisance 参数：

.. code-block:: python

   # 假设 test_sne 有 [H0, Omega_m]
   # 假设 test_bao 有 [H0, Omega_m, rd_fid_Mpc]（含 nuisance 参数）

   plotter = Plotter(['test_sne', 'test_bao'], labels=['Pantheon+', 'DESI R1'])
   plotter.corner()  # 自动绘制 [H0, Omega_m, rd_fid_Mpc] 三个参数

   # 如需指定参数，显式传递
   plotter.corner(['H0', 'Omega_m'])  # 只绘制这两个参数

多链报告生成
~~~~~~~~~~~~

多链模式下，``report()`` 为每条链 **分别生成独立报告**：

.. code-block:: python

   plotter = Plotter(['test_sne', 'test_bao'], labels=['Pantheon+', 'DESI R1'])
   plotter.report()
   # 生成两个文件：
   #   - results/test_sne_report.md
   #   - results/test_bao_report.md

这样设计的原因是每条链可能使用不同的似然函数、不同的 nuisance 参数，合并报告会造成混淆。

从字典创建
~~~~~~~~~~

.. code-block:: python

   # 准备多链数据
   chain1 = {'H0': samples1_h0, 'Omega_m': samples1_om}
   chain2 = {'H0': samples2_h0, 'Omega_m': samples2_om}
   chain3 = {'H0': samples3_h0, 'Omega_m': samples3_om}

   plotter = Plotter(
       [chain1, chain2, chain3],
       chain_labels=['CMB only', 'CMB + BAO', 'CMB + BAO + SN']
   )

   fig = plotter.corner(filename='joint_comparison.pdf')

配色方案
~~~~~~~~

HIcosmo 提供两套专业配色方案：

**Modern（默认）**：

.. code-block:: python

   MODERN_COLORS = [
       '#2E86AB',  # 深蓝
       '#A23B72',  # 紫红
       '#F18F01',  # 橙色
       '#C73E1D',  # 红色
       '#592E83',  # 紫色
       '#0F7173',  # 青色
       '#7B2D26',  # 棕色
   ]

**Classic**：

.. code-block:: python

   CLASSIC_COLORS = [
       '#348ABD',  # 蓝色
       '#7A68A6',  # 紫色
       '#E24A33',  # 红色
       '#467821',  # 绿色
       '#ffb3a6',  # 粉色
       '#188487',  # 青色
       '#A60628',  # 深红
   ]

使用方式：

.. code-block:: python

   plotter = Plotter('my_chain', style='classic')

Fisher 预报可视化
-----------------

HIcosmo 支持从 Fisher 矩阵结果生成预报等高线图。

从 Fisher 结果创建
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import Plotter
   import numpy as np

   # 假设有 Fisher 分析结果
   mean = np.array([67.36, 0.315])  # 基准值
   covariance = np.array([
       [0.5, 0.01],
       [0.01, 0.007]
   ])  # 协方差矩阵

   # 创建 Plotter
   plotter = Plotter.from_fisher(
       mean=mean,
       covariance=covariance,
       param_names=['H0', 'Omega_m']
   )

   # 绘制预报约束
   fig = plotter.corner(filename='fisher_forecast.pdf')

**工作原理**：

1. 从高斯分布 :math:`\mathcal{N}(\mu, \Sigma)` 生成样本
2. 默认生成 200,000 个样本以确保等高线光滑
3. 使用 GetDist 绘制等高线

暗能量 Figure of Merit
~~~~~~~~~~~~~~~~~~~~~~

暗能量状态方程的约束能力用 Figure of Merit (FoM) 衡量：

.. math::

   \text{FoM} = \frac{1}{\sqrt{\det(\text{Cov}_{w_0, w_a})}}

.. code-block:: python

   from hicosmo.fisher import IntensityMappingFisher

   # 获取 w0-wa 子矩阵
   cov_de = fisher.marginalize(['w0', 'wa']).covariance
   fom = 1.0 / np.sqrt(np.linalg.det(cov_de))

   print(f"Dark Energy FoM: {fom:.1f}")

函数接口
--------

除了 ``Plotter`` 类，HIcosmo 还提供独立函数接口，适合快速绘图。

plot_corner
~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import plot_corner

   # 单链
   fig = plot_corner(
       'results/mcmc_chain.pkl',
       params=['H0', 'Omega_m'],
       style='modern',
       filename='corner.pdf'
   )

   # 多链对比
   fig = plot_corner(
       [chain1, chain2],
       labels=['Dataset A', 'Dataset B'],
       filename='comparison.pdf'
   )

plot_chains
~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import plot_chains

   fig = plot_chains(
       'results/mcmc_chain.pkl',
       params=[0, 1, 2],  # 前三个参数
       filename='traces.pdf'
   )

plot_1d
~~~~~~~

.. code-block:: python

   from hicosmo.visualization import plot_1d

   fig = plot_1d(
       'results/mcmc_chain.pkl',
       filename='marginals.pdf'
   )

plot_fisher_contours
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import plot_fisher_contours

   fig = plot_fisher_contours(
       mean=[67.36, 0.315],
       covariance=[[0.5, 0.01], [0.01, 0.007]],
       param_names=['H0', 'Omega_m'],
       filename='forecast.pdf'
   )

数据格式支持
------------

HIcosmo 自动识别多种数据格式：

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 格式
     - 扩展名
     - 说明
   * - **HDF5**
     - ``.h5``, ``.hdf5``
     - 推荐格式，支持元数据
   * - **Pickle**
     - ``.pkl``, ``.pickle``
     - Python 原生格式
   * - **NumPy**
     - ``.npy``, ``.npz``
     - 数组格式
   * - **文本**
     - ``.txt``, ``.dat``
     - ASCII 格式
   * - **字典**
     - 内存
     - ``{'param': array}``
   * - **MCSamples**
     - 内存
     - GetDist 原生对象

元数据自动提取
~~~~~~~~~~~~~~

使用 ``MCMC.save_results()`` 保存的链文件包含完整元数据：

.. code-block:: python

   # 保存链（元数据自动包含）
   mcmc.save_results('my_chain')  # 保存到 mcmc_chains/my_chain.h5

   # 加载时自动提取
   plotter = Plotter('my_chain')
   # - LaTeX 标签从文件读取
   # - 参数范围从文件读取
   # - 派生参数（如 rd_h）自动可用

样式定制
--------

全局样式
~~~~~~~~

HIcosmo 自动应用专业样式，包括：

- **字体大小**：轴标签 14pt，刻度 11pt，图例 12pt
- **线宽**：等高线 2pt，网格 1.2pt
- **背景**：白色（出版友好）
- **图例**：无边框

手动样式调整
~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # 修改全局样式
   plt.rcParams.update({
       'axes.labelsize': 16,
       'legend.fontsize': 14,
       'lines.linewidth': 2.5,
   })

   # 创建绘图器
   plotter = Plotter('my_chain')
   fig = plotter.corner()

   # 进一步调整（返回的是 matplotlib Figure）
   for ax in fig.axes:
       ax.tick_params(labelsize=10)

GetDist 直接集成
~~~~~~~~~~~~~~~~

对于需要更多 GetDist 控制的高级用户：

.. code-block:: python

   from getdist import plots, MCSamples
   import numpy as np

   # 创建 GetDist MCSamples
   samples = MCSamples(
       samples=np.column_stack([h0_samples, om_samples]),
       names=['H0', 'Omega_m'],
       labels=[r'H_0', r'\Omega_m']
   )

   # 使用 GetDist 原生 API
   g = plots.get_subplot_plotter()
   g.settings.figure_legend_frame = False
   g.triangle_plot(samples, filled=True)

保存选项
--------

文件格式
~~~~~~~~

.. code-block:: python

   # PDF（默认，矢量图，推荐出版）
   plotter.corner(filename='corner.pdf')

   # PNG（栅格图，网页展示）
   plotter.corner(filename='corner.png')

   # SVG（矢量图，可编辑）
   plotter.corner(filename='corner.svg')

分辨率设置
~~~~~~~~~~

默认 DPI 为 300，适合出版。可通过 matplotlib 调整：

.. code-block:: python

   fig = plotter.corner()
   fig.savefig('corner.png', dpi=600)  # 高分辨率

完整示例
--------

MCMC 结果可视化
~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo import hicosmo
   from hicosmo.visualization import Plotter

   # 运行 MCMC
   inf = hicosmo(
       cosmology='LCDM',
       likelihood=['sn', 'bao'],
       free_params=['H0', 'Omega_m'],
       num_samples=4000,
       num_chains=4
   )
   samples = inf.run()

   # 保存链
   inf.mcmc.save_results('lcdm_sn_bao')

   # 可视化
   plotter = Plotter('lcdm_sn_bao')

   # Corner 图
   plotter.corner(['H0', 'Omega_m'], filename='lcdm_corner.pdf')

   # 收敛诊断
   plotter.traces(filename='lcdm_traces.pdf')

   # 1D 分布
   plotter.marginals(filename='lcdm_marginals.pdf')

   # 打印统计
   for param, stats in plotter.get_summary().items():
       print(f"{param}: {stats['mean']:.4f} ± {stats['std']:.4f}")

多探针比较
~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import Plotter

   # 运行多组分析（假设已完成）
   # ...

   # 创建对比图
   plotter = Plotter(
       ['sn_only', 'bao_only', 'cmb_only', 'joint_all'],
       labels=[
           'Pantheon+',
           'DESI BAO',
           'Planck 2018',
           'Joint'
       ]
   )

   # 绘制对比（自动包含所有参数，包括 nuisance）
   plotter.corner()  # 自动保存为 sn_only+bao_only+cmb_only+joint_all_corner.pdf

   # 或只绘制宇宙学参数
   plotter.corner(['H0', 'Omega_m'], filename='multi_probe_comparison.pdf')

   # 单独对比 H0
   plotter.plot_1d('H0', filename='h0_tension.pdf')

   # 生成各自的分析报告
   plotter.report()
   # 生成四个独立报告：
   #   - sn_only_report.md
   #   - bao_only_report.md
   #   - cmb_only_report.md
   #   - joint_all_report.md

Fisher 预报可视化
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.fisher import IntensityMappingFisher, load_survey
   from hicosmo.models import CPL
   from hicosmo.visualization import Plotter
   import numpy as np

   # Fisher 分析
   survey = load_survey('ska1_mid_band2')
   fiducial = CPL(H0=67.36, Omega_m=0.3153, w0=-1.0, wa=0.0)
   fisher = IntensityMappingFisher(survey, fiducial)
   result = fisher.parameter_forecast(['w0', 'wa'])

   # 可视化
   plotter = Plotter.from_fisher(
       mean=result.mean,
       covariance=result.covariance,
       param_names=['w0', 'wa']
   )

   plotter.corner(filename='ska1_forecast.pdf')

   # 计算 FoM
   fom = 1.0 / np.sqrt(np.linalg.det(result.covariance))
   print(f"SKA1 Dark Energy FoM: {fom:.1f}")

常见问题
--------

Q: GetDist 未安装怎么办？
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install getdist

Q: 如何修改等高线颜色？
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 使用 classic 配色
   plotter = Plotter('my_chain', style='classic')

   # 或自定义（需要使用 GetDist API）
   from getdist import plots
   g = plots.get_subplot_plotter()
   g.triangle_plot(samples, contour_colors=['red', 'blue'])

Q: 如何添加真值标记？
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig = plotter.corner(['H0', 'Omega_m'])

   # 添加真值线
   import matplotlib.pyplot as plt
   for ax in fig.axes:
       # 在每个子图添加参考线
       pass  # 需要根据具体子图位置添加

下一步
------

- `Fisher 预报 <fisher.html>`_：进行巡天预测
- `采样器 <samplers.html>`_：了解 MCMC 配置
