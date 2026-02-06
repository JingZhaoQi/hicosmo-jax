Fisher 预测
===========

HIcosmo 提供专业的 Fisher 矩阵分析工具，支持 21cm 强度映射巡天预报、多探针联合分析和参数约束预测。

Fisher 矩阵基础
----------------

Fisher 信息矩阵
~~~~~~~~~~~~~~~

Fisher 信息矩阵量化似然函数在参数空间中的曲率，定义为：

.. math::

   F_{ij} = -\left\langle \frac{\partial^2 \ln \mathcal{L}}{\partial \theta_i \partial \theta_j} \right\rangle

对于高斯似然，Fisher 矩阵可以通过协方差矩阵 :math:`C` 和理论预测 :math:`\mu` 计算：

.. math::

   F_{ij} = \frac{1}{2} \text{Tr}\left[ C^{-1} \frac{\partial C}{\partial \theta_i} C^{-1} \frac{\partial C}{\partial \theta_j} \right] + \frac{\partial \mu^T}{\partial \theta_i} C^{-1} \frac{\partial \mu}{\partial \theta_j}

参数约束
~~~~~~~~

参数的 :math:`1\sigma` 约束由协方差矩阵给出：

.. math::

   \text{Cov}(\theta) = F^{-1}

.. math::

   \sigma(\theta_i) = \sqrt{(F^{-1})_{ii}}

**边际化约束**：当存在 nuisance 参数时，目标参数的约束需要对 nuisance 参数积分：

.. math::

   \text{Cov}_{\text{target}} = (F^{-1})_{\text{target block}}

**条件约束**：固定其他参数时的约束更紧：

.. math::

   \sigma_{\text{cond}}(\theta_i) = \frac{1}{\sqrt{F_{ii}}}

快速开始
--------

**3 行完成 Fisher 预报**：

.. code-block:: python

   from hicosmo.models import CPL
   from hicosmo.fisher import IntensityMappingFisher

   result = IntensityMappingFisher.forecast('ska1_mid_band2', CPL(), ['w0', 'wa'])
   print(result)

IntensityMappingFisher 类
-------------------------

``IntensityMappingFisher`` 是 21cm 强度映射 Fisher 预报的核心类，实现 Bull et al. (2016) 方法。

类方法接口（推荐）
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import ILCDM
   from hicosmo.fisher import IntensityMappingFisher

   # 3 行完成预报
   ilcdm = ILCDM(H0=67.36, Omega_m=0.3153, beta=0.001)
   result = IntensityMappingFisher.forecast('tianlai', ilcdm, ['beta', 'H0'])
   print(result)

**输出示例**：

.. code-block:: text

   ======================================================================
   Fisher Forecast Results: tianlai
   ======================================================================
   Cosmology: ILCDM
   Redshift bins: 3
   Parameters: beta, H0

   1σ Constraints:
   ----------------------------------------------------------------------
     σ(beta      ) =    1.340000e-02
     σ(H0        ) =    8.250000e-01
   ======================================================================

实例接口
~~~~~~~~

当需要多次预报时使用实例接口：

.. code-block:: python

   from hicosmo.fisher import IntensityMappingFisher, load_survey
   from hicosmo.models import CPL

   # 加载巡天配置
   survey = load_survey('ska1_mid_band2')
   cosmology = CPL(H0=67.36, Omega_m=0.3153, w0=-1.0, wa=0.0)

   # 创建 Fisher 计算器
   fisher = IntensityMappingFisher(survey, cosmology)

   # 多次预报
   result1 = fisher.parameter_forecast(['w0', 'wa'])
   result2 = fisher.parameter_forecast(['H0', 'Omega_m'])

带边际化
~~~~~~~~

边际化 nuisance 参数：

.. code-block:: python

   result = IntensityMappingFisher.forecast(
       'ska1_mid_band2',
       cpl_model,
       params=['w0', 'wa'],
       marginalize_over=['H0', 'Omega_m', 'gamma']
   )

   print(f"σ(w0) = {result.errors['w0']:.4f}")
   print(f"σ(wa) = {result.errors['wa']:.4f}")

FisherForecastResult 对象
~~~~~~~~~~~~~~~~~~~~~~~~~

``forecast()`` 返回的结果对象包含：

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 属性
     - 类型
     - 说明
   * - ``params``
     - list
     - 参数名称列表
   * - ``errors``
     - dict
     - :math:`1\sigma` 约束 ``{param: σ}``
   * - ``fisher_matrix``
     - ndarray
     - Fisher 矩阵
   * - ``covariance``
     - ndarray
     - 协方差矩阵
   * - ``correlation``
     - ndarray
     - 相关矩阵
   * - ``survey_name``
     - str
     - 巡天名称
   * - ``n_bins``
     - int
     - 红移 bin 数

.. code-block:: python

   # 访问结果
   print(f"σ(w0) = {result.errors['w0']:.4f}")
   print(f"w0-wa 相关系数: {result.correlation[0,1]:.3f}")

   # 转换为字典（兼容旧 API）
   data = result.to_dict()

巡天配置
--------

HIcosmo 内置多个 21cm 强度映射巡天配置。

可用巡天
~~~~~~~~

.. code-block:: python

   from hicosmo.fisher import list_available_surveys

   surveys = list_available_surveys()
   print(surveys)
   # ['ska1_mid_band2', 'ska1_wide_band1', 'tianlai', 'bingo', 'meerkat', 'chime', ...]

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - 巡天
     - 红移范围
     - 特点
   * - **ska1_mid_band2**
     - 0.0 - 0.5
     - SKA1-MID 中等深度，Band 2
   * - **ska1_wide_band1**
     - 0.4 - 1.0
     - SKA1-MID 宽视场，Band 1
   * - **tianlai**
     - 0.8 - 1.2
     - 天籁实验，柱面阵列
   * - **bingo**
     - 0.13 - 0.45
     - 巴西 21cm 项目
   * - **meerkat**
     - 0.0 - 0.6
     - 南非 MeerKAT 阵列
   * - **chime**
     - 0.8 - 2.5
     - 加拿大 CHIME 柱面阵列

加载巡天
~~~~~~~~

.. code-block:: python

   from hicosmo.fisher import load_survey

   survey = load_survey('ska1_mid_band2')

   # 查看巡天信息
   print(f"名称: {survey.name}")
   print(f"描述: {survey.description}")
   print(f"红移 bins: {len(survey.redshift_bins)}")
   print(f"天线数: {survey.instrument.ndish}")
   print(f"巡天面积: {survey.instrument.survey_area_deg2} deg²")

巡天配置格式
~~~~~~~~~~~~

巡天使用 YAML 格式配置（位于 ``hicosmo/fisher/surveys/``）：

.. code-block:: yaml

   name: ska1_mid_band2
   description: "SKA1-MID Medium-Deep Band 2 survey"

   # 硬件配置
   instrument:
     telescope_type: single_dish
     ndish: 133
     dish_diameter_m: 15.0
     nbeam: 1

   # 观测策略
   observing:
     survey_area_deg2: 5000.0
     total_time_hours: 10000.0
     channel_width_hz: 50000.0

   # 噪声参数
   noise:
     system_temperature_K: 13.5
     sky_temperature:
       model: power_law
       T_ref_K: 25.0
       nu_ref_MHz: 408.0
       beta: 2.75

   # HI 示踪体
   hi_tracers:
     bias:
       model: polynomial
       coefficients: [0.67, 0.18, 0.05]
     density:
       model: polynomial
       coefficients: [0.00048, 0.00039, -0.000065]

   # 红移分 bin
   redshift_bins:
     default_delta_z: 0.1
     centers: [0.05, 0.15, 0.25, 0.35, 0.45]

21cm 物理
---------

亮温度
~~~~~~

21cm 亮温度公式（Santos et al. 2015）：

.. math::

   T_b(z) = 27 \, \text{mK} \times h \times \frac{\Omega_{\text{HI}}}{10^{-3}} \times \frac{(1+z)^2}{E(z)}

其中 :math:`\Omega_{\text{HI}}` 是 HI 密度参数，:math:`E(z) = H(z)/H_0`。

功率谱
~~~~~~

21cm 功率谱为：

.. math::

   P_{21}(k, \mu, z) = T_b^2 \, b^2 \, (1 + \beta \mu^2)^2 \, P_m(k, z)

其中：
- :math:`b(z)` 是 HI 偏差因子
- :math:`\beta = f/b` 是红移空间畸变参数
- :math:`f = d\ln D/d\ln a` 是增长率
- :math:`\mu = \cos\theta` 是视线方向余弦
- :math:`P_m(k,z)` 是物质功率谱

有效体积
~~~~~~~~

Fisher 信息加权的有效体积：

.. math::

   V_{\text{eff}}(k, \mu, z) = V_{\text{survey}} \left[ \frac{P_{\text{signal}}}{P_{\text{signal}} + P_{\text{noise}}} \right]^2

噪声功率谱
~~~~~~~~~~

仪器噪声功率谱：

.. math::

   P_{\text{noise}} = \frac{\sigma_{\text{pix}}^2 V_{\text{pix}}}{W^2(k_\perp)}

其中 :math:`W(k_\perp)` 是波束窗函数，:math:`\sigma_{\text{pix}}` 是像素噪声。

FisherMatrix 类
---------------

对于通用 Fisher 矩阵计算（非 21cm 特定），使用 ``FisherMatrix`` 类。

基本用法
~~~~~~~~

.. code-block:: python

   from hicosmo.fisher import FisherMatrix, FisherMatrixConfig

   # 配置
   config = FisherMatrixConfig(
       differentiation_method='automatic',  # 使用 JAX 自动微分
       regularization=True
   )

   # 创建计算器
   calculator = FisherMatrix(config)

   # 计算 Fisher 矩阵
   fisher_matrix, param_names = calculator.compute_fisher_matrix(
       likelihood_func=my_likelihood,
       parameters=params,
       fiducial_values={'H0': 67.36, 'Omega_m': 0.315}
   )

配置选项
~~~~~~~~

.. code-block:: python

   from hicosmo.fisher import FisherMatrixConfig

   config = FisherMatrixConfig(
       # 微分方法
       differentiation_method='automatic',  # 'automatic', 'numerical', 'hybrid'
       step_size=1e-6,
       fallback_to_numerical=True,

       # 正则化
       regularization=True,
       reg_lambda=1e-10,
       svd_threshold=1e-12,

       # 验证
       check_symmetry=True,
       check_positive_definite=True,

       # 性能
       vectorize_derivatives=True,
       cache_derivatives=True
   )

矩阵合并
~~~~~~~~

合并多探针 Fisher 矩阵：

.. code-block:: python

   # 计算各探针 Fisher 矩阵
   fisher_bao, params_bao = calculator.compute_fisher_matrix(bao_likelihood, ...)
   fisher_sn, params_sn = calculator.compute_fisher_matrix(sn_likelihood, ...)

   # 合并
   combined_fisher, combined_params = calculator.combine_fisher_matrices(
       [(fisher_bao, params_bao), (fisher_sn, params_sn)],
       probe_names=['BAO', 'SN']
   )

参数边际化
~~~~~~~~~~

.. code-block:: python

   # 对 nuisance 参数边际化
   marginalized_fisher, remaining_params = calculator.marginalize_parameters(
       fisher_matrix,
       param_names,
       marginalize_over=['M_B', 'delta_M']
   )

暗能量 Figure of Merit
----------------------

暗能量实验的约束能力用 Figure of Merit (FoM) 量化：

.. math::

   \text{FoM} = \frac{1}{\sqrt{\det(\text{Cov}_{w_0, w_a})}}

这等价于 :math:`w_0`-:math:`w_a` 平面上 :math:`1\sigma` 等高线所围面积的倒数。

.. code-block:: python

   from hicosmo.models import CPL
   from hicosmo.fisher import IntensityMappingFisher
   import numpy as np

   # 预报 w0-wa 约束
   cpl = CPL(H0=67.36, Omega_m=0.3153, w0=-1.0, wa=0.0)
   result = IntensityMappingFisher.forecast('ska1_mid_band2', cpl, ['w0', 'wa'])

   # 计算 FoM
   fom = 1.0 / np.sqrt(np.linalg.det(result.covariance))
   print(f"Dark Energy FoM: {fom:.1f}")

增长参数预报
------------

HIcosmo 支持修改引力理论的增长参数预报。

增长参数化
~~~~~~~~~~

增长率参数化为：

.. math::

   f(z) = \Omega_m(z)^{\gamma_0 + \gamma_1 z}

偏离参数为：

.. math::

   \eta(z) = \eta_0 + \eta_1 z

在广义相对论中：:math:`\gamma_0 = 0.55`, :math:`\gamma_1 = \eta_0 = \eta_1 = 0`。

增长场景
~~~~~~~~

HIcosmo 实现 Bull et al. (2016) 的四种增长场景：

.. code-block:: python

   from hicosmo.fisher.intensity_mapping import GrowthScenario

   # 可用场景
   scenarios = [
       'gr',           # 广义相对论（无自由增长参数）
       'gamma0_free',  # γ₀ 自由，其余固定
       'gamma1_free',  # γ₁ 自由，其余固定
       'eta0_free',    # η₀ 自由，其余固定
       'eta1_free',    # η₁ 自由，其余固定
       'all_free',     # 所有增长参数自由
   ]

   # 使用增长场景
   scenario = GrowthScenario('gamma0_free')
   print(f"自由参数: {scenario.free_params}")
   print(f"固定参数: {scenario.fixed_params}")

可视化
------

Fisher 预报结果可视化：

.. code-block:: python

   from hicosmo.visualization import Plotter, plot_fisher_contours
   import numpy as np

   # 方式 1：使用 Plotter.from_fisher
   plotter = Plotter.from_fisher(
       mean=result.cosmology_summary['fiducial'],
       covariance=result.covariance,
       param_names=result.params
   )
   plotter.corner(filename='forecast.pdf')

   # 方式 2：使用独立函数
   plot_fisher_contours(
       mean=[-1.0, 0.0],
       covariance=result.covariance,
       param_names=['w0', 'wa'],
       filename='de_forecast.pdf'
   )

多巡天比较
~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import CPL
   from hicosmo.fisher import IntensityMappingFisher
   from hicosmo.visualization import Plotter

   cpl = CPL()
   surveys = ['ska1_mid_band2', 'tianlai', 'bingo']
   results = []

   for survey in surveys:
       result = IntensityMappingFisher.forecast(survey, cpl, ['w0', 'wa'])
       results.append(result)

   # 对比图（使用 Plotter.from_fisher 多次）
   # 或手动生成高斯样本后使用 Plotter

完整示例
--------

SKA1 暗能量预报
~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import CPL
   from hicosmo.fisher import IntensityMappingFisher, list_available_surveys
   import numpy as np

   # 定义宇宙学模型
   fiducial = CPL(
       H0=67.36,
       Omega_m=0.3153,
       Omega_b=0.0493,
       w0=-1.0,
       wa=0.0,
       sigma8=0.811,
       n_s=0.9649
   )

   # 预报 w0-wa 约束
   result = IntensityMappingFisher.forecast(
       'ska1_mid_band2',
       fiducial,
       params=['w0', 'wa'],
       marginalize_over=['H0', 'Omega_m']
   )

   # 打印结果
   print(result)

   # 计算 FoM
   fom = 1.0 / np.sqrt(np.linalg.det(result.covariance))
   print(f"\nDark Energy FoM: {fom:.1f}")

   # 相关系数
   print(f"w0-wa correlation: {result.correlation[0,1]:.3f}")

相互作用暗能量预报
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import ILCDM
   from hicosmo.fisher import IntensityMappingFisher

   # 相互作用暗能量模型
   ilcdm = ILCDM(
       H0=67.36,
       Omega_m=0.3153,
       beta=0.001  # 相互作用强度
   )

   # 天籁实验对 β 的约束
   result = IntensityMappingFisher.forecast(
       'tianlai',
       ilcdm,
       params=['beta', 'H0', 'Omega_m']
   )

   print(f"天籁实验对相互作用参数的约束:")
   print(f"σ(β) = {result.errors['beta']:.4e}")

多巡天联合分析
~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import CPL
   from hicosmo.fisher import IntensityMappingFisher, load_survey, FisherMatrix
   import numpy as np

   cpl = CPL()
   surveys = ['ska1_mid_band2', 'tianlai', 'bingo']

   # 收集各巡天 Fisher 矩阵
   fisher_matrices = []
   for survey_name in surveys:
       result = IntensityMappingFisher.forecast(
           survey_name, cpl, ['w0', 'wa', 'H0', 'Omega_m']
       )
       fisher_matrices.append((result.fisher_matrix, result.params))

   # 合并 Fisher 矩阵
   calculator = FisherMatrix()
   combined_fisher, combined_params = calculator.combine_fisher_matrices(
       fisher_matrices,
       probe_names=surveys
   )

   # 计算联合约束
   combined_cov = np.linalg.inv(combined_fisher)
   combined_errors = np.sqrt(np.diag(combined_cov))

   print("联合约束:")
   for i, param in enumerate(combined_params):
       print(f"  σ({param}) = {combined_errors[i]:.4e}")

性能基准
--------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - 操作
     - 时间
     - 备注
   * - 单巡天 Fisher 预报
     - ~0.5s
     - 5 红移 bins
   * - 完整参数预报（含边际化）
     - ~1.2s
     - 6 参数
   * - 自动微分 Hessian
     - ~0.1s
     - JAX JIT

常见问题
--------

Q: 如何自定义巡天配置？
~~~~~~~~~~~~~~~~~~~~~~~

创建 YAML 文件并使用 ``IntensityMappingSurvey.from_file()``：

.. code-block:: python

   from hicosmo.fisher import IntensityMappingSurvey

   survey = IntensityMappingSurvey.from_file('my_survey.yaml')

Q: Fisher 矩阵不正定怎么办？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

启用正则化：

.. code-block:: python

   from hicosmo.fisher import FisherMatrixConfig

   config = FisherMatrixConfig(
       regularization=True,
       reg_lambda=1e-8
   )

Q: 如何比较不同宇宙学模型？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用相同巡天配置，不同模型：

.. code-block:: python

   from hicosmo.models import LCDM, wCDM, CPL

   survey = 'ska1_mid_band2'

   result_lcdm = IntensityMappingFisher.forecast(survey, LCDM(), ['H0', 'Omega_m'])
   result_wcdm = IntensityMappingFisher.forecast(survey, wCDM(), ['H0', 'Omega_m', 'w0'])
   result_cpl = IntensityMappingFisher.forecast(survey, CPL(), ['H0', 'Omega_m', 'w0', 'wa'])

下一步
------

- `可视化 <visualization.html>`_：展示 Fisher 预报结果
- `宇宙学模型 <models.html>`_：了解可用模型
