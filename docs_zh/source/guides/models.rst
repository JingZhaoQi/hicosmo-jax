宇宙学模型
==========

HIcosmo 提供多种宇宙学模型，涵盖标准宇宙学模型及其动态暗能量扩展。所有模型都基于 JAX 构建，支持 JIT 编译和自动微分。

模型概览
--------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - 模型
     - 暗能量状态方程
     - 描述
   * - **LCDM**
     - :math:`w = -1`
     - 标准 :math:`\Lambda\text{CDM}` 模型，宇宙学常数
   * - **wCDM**
     - :math:`w = w_0`
     - 常数暗能量状态方程
   * - **CPL**
     - :math:`w(a) = w_0 + w_a(1-a)`
     - Chevallier-Polarski-Linder 参数化
   * - **ILCDM**
     - :math:`w = -1` + 相互作用
     - 相互作用暗能量模型

基础理论
--------

无量纲 Hubble 参数
~~~~~~~~~~~~~~~~~~

所有模型的核心是无量纲 Hubble 参数 :math:`E(z) = H(z)/H_0`。它决定了宇宙的膨胀历史和所有距离测量。

**通用形式**：

.. math::

   E^2(z) = \Omega_m (1+z)^3 + \Omega_r (1+z)^4 + \Omega_k (1+z)^2 + \Omega_{\rm DE}(z)

其中：

- :math:`\Omega_m`：物质密度参数
- :math:`\Omega_r`：辐射密度参数
- :math:`\Omega_k`：曲率密度参数
- :math:`\Omega_{\rm DE}(z)`：暗能量密度演化

**归一化条件**：在 :math:`z=0` 时，:math:`E(0) = 1`，这要求：

.. math::

   \Omega_m + \Omega_r + \Omega_k + \Omega_\Lambda = 1

距离计算
~~~~~~~~

从 :math:`E(z)` 可以推导所有宇宙学距离：

共动距离：

.. math::

   d_C(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}

横向共动距离（考虑曲率）：

.. math::

   D_M(z) = \begin{cases}
   D_H / \sqrt{\Omega_k} \sinh(\sqrt{\Omega_k} \, d_C / D_H) & \Omega_k > 0 \\
   d_C & \Omega_k = 0 \\
   D_H / \sqrt{|\Omega_k|} \sin(\sqrt{|\Omega_k|} \, d_C / D_H) & \Omega_k < 0
   \end{cases}

其中 :math:`D_H = c/H_0` 是 Hubble 距离。

光度距离：

.. math::

   d_L(z) = (1+z) \, D_M(z)

角直径距离：

.. math::

   D_A(z) = \frac{D_M(z)}{1+z}

距离模数：

.. math::

   \mu = 5 \log_{10}\left(\frac{d_L}{\text{Mpc}}\right) + 25

LCDM 模型
---------

标准 :math:`\Lambda\text{CDM}` 模型假设暗能量是宇宙学常数，:math:`w = -1`。

LCDM 数学形式
~~~~~~~~~~~~~

Hubble 参数：

.. math::

   E^2(z) = \Omega_m (1+z)^3 + \Omega_r (1+z)^4 + \Omega_k (1+z)^2 + \Omega_\Lambda

暗能量状态方程：

.. math::

   w(z) = -1 \quad \text{（常数）}

LCDM 参数说明
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 12 15 55

   * - 参数
     - 默认值
     - 范围
     - 描述
   * - ``H0``
     - 67.36
     - [50, 100]
     - Hubble 常数，单位 km/s/Mpc
   * - ``Omega_m``
     - 0.3153
     - [0.01, 0.99]
     - 总物质密度参数
   * - ``Omega_b``
     - 0.0493
     - [0.01, 0.1]
     - 重子密度参数
   * - ``Omega_k``
     - 0.0
     - [-0.3, 0.3]
     - 曲率密度参数（0 表示平直宇宙）
   * - ``Omega_r``
     - 自动计算
     - —
     - 辐射密度参数，由 :math:`T_{\rm CMB}` 计算
   * - ``sigma8``
     - 0.8111
     - [0.5, 1.2]
     - 物质功率谱归一化
   * - ``n_s``
     - 0.9649
     - [0.8, 1.1]
     - 原初功率谱标量指数
   * - ``T_cmb``
     - 2.7255
     - —
     - CMB 温度 (K)
   * - ``N_eff``
     - 3.046
     - —
     - 有效中微子数目

LCDM 基本使用
~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import LCDM

   # 使用 Planck 2018 默认参数
   model = LCDM()

   # 或指定参数
   model = LCDM(H0=70.0, Omega_m=0.3, Omega_k=0.01)

   # 计算距离
   z = 1.0
   d_L = model.luminosity_distance(z)      # 光度距离 [Mpc]
   D_A = model.angular_diameter_distance(z)  # 角直径距离 [Mpc]
   D_M = model.transverse_comoving_distance(z)  # 横向共动距离 [Mpc]
   mu = model.distance_modulus(z)          # 距离模数

   print(f"d_L(z=1) = {d_L:.2f} Mpc")
   print(f"D_A(z=1) = {D_A:.2f} Mpc")
   print(f"mu(z=1) = {mu:.2f}")

参数一致性与更新
~~~~~~~~~~~~~~~~~~~~~~

所有模型的参数都以 ``model.params`` 为单一真相（内部为 ``CosmologicalParameters``）。
更新参数的推荐方式是**重建实例**：这与 JAX 的不可变风格一致，也避免已经
JIT 编译的距离方法继续引用旧参数。

.. code-block:: python

   from hicosmo.models import LCDM, wCDM, CPL, ILCDM

   model = LCDM(H0=70.0, Omega_m=0.3)

   # 注意：wCDM 的状态方程参数名是 w0（不是 w）。
   # 拼错的参数名会被静默忽略，似然层会发出 UserWarning 提示。
   w_model = wCDM(w0=-1.0)
   w_model = wCDM(w0=-0.9)          # 更新参数 = 重建实例

   cpl = CPL(w0=-0.95, wa=0.2)

   ilcdm = ILCDM(beta=0.05)

.. warning::

   不要用属性赋值（如 ``w_model.w0 = -0.9``）更新参数：它只会创建一个
   与模型无关的普通实例属性，``params`` 中的值并不会改变。

声音视界计算
~~~~~~~~~~~~

LCDM 模型包含拖拽时刻声音视界 :math:`r_d` 的计算，这是 BAO 分析的关键量。

声音视界（Eisenstein & Hu 1998 拟合公式）：

.. math::

   r_d = \frac{44.5 \ln(9.83 / \Omega_m h^2)}{\sqrt{1 + 10 (\Omega_b h^2)^{3/4}}} \text{ Mpc}

.. code-block:: python

   # 获取声音视界
   rd = model.sound_horizon()  # Mpc
   rd_h = model.rd_h           # r_d × h (无量纲)

   print(f"r_d = {rd:.2f} Mpc")

增长函数
~~~~~~~~

LCDM 模型提供线性增长因子和增长率的计算。

增长因子 D(z)（Carroll, Press & Turner 1992 近似）：

.. math::

   D(z) \propto \frac{5 \Omega_m E(z)}{2} \int_z^\infty \frac{(1+z') dz'}{E^3(z')}

增长率：

.. math::

   f(z) = \frac{d \ln D}{d \ln a} \approx \Omega_m(z)^\gamma, \quad \gamma \approx 0.55

.. code-block:: python

   # 增长因子（归一化到 z=0）
   D = model.growth_factor(z=0.5)

   # 增长率
   f = model.growth_rate(z=0.5)

   # f*sigma8 参数（RSD 分析常用）
   fsig8 = model.fsigma8(z=0.5)

   print(f"D(z=0.5) = {D:.4f}")
   print(f"f(z=0.5) = {f:.4f}")
   print(f"f*sigma8(z=0.5) = {fsig8:.4f}")

wCDM 模型
---------

wCDM 模型将暗能量状态方程推广为常数 :math:`w \neq -1`。

wCDM 数学形式
~~~~~~~~~~~~~

Hubble 参数：

.. math::

   E^2(z) = \Omega_m (1+z)^3 + \Omega_r (1+z)^4 + \Omega_k (1+z)^2 + \Omega_\Lambda (1+z)^{3(1+w)}

暗能量状态方程：

.. math::

   w(z) = w_0 \quad \text{（常数）}

wCDM 参数说明
~~~~~~~~~~~~~

wCDM 继承 LCDM 所有参数，并增加：

.. list-table::
   :header-rows: 1
   :widths: 18 12 20 50

   * - 参数
     - 默认值
     - 先验范围
     - 描述
   * - ``w`` (或 ``w0``)
     - -1.0
     - [-2.5, 0.0]
     - 暗能量状态方程常数

物理约束：

- :math:`w = -1`：宇宙学常数（LCDM）
- :math:`w > -1`：Quintessence 类暗能量
- :math:`w < -1`：Phantom 暗能量（违反能量条件）

wCDM 基本使用
~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import wCDM

   # 创建 wCDM 模型
   model = wCDM(H0=70.0, Omega_m=0.3, w=-0.9)

   # 所有距离方法继承自 LCDM
   d_L = model.luminosity_distance(1.0)

   # 查看状态方程
   w = model.w_z(0.5)  # 返回常数 w
   print(f"w(z=0.5) = {w:.2f}")

   # E(z) 现在包含 w 参数
   E = model.E_z(1.0)
   print(f"E(z=1) = {E:.4f}")

CPL 模型
--------

CPL (Chevallier-Polarski-Linder) 参数化允许暗能量状态方程随时间演化。

CPL 数学形式
~~~~~~~~~~~~

状态方程参数化：

.. math::

   w(a) = w_0 + w_a (1 - a) = w_0 + w_a \frac{z}{1+z}

其中 :math:`a = 1/(1+z)` 是尺度因子。

暗能量密度演化：

.. math::

   f_{\rm DE}(z) = \exp\left( \frac{3 w_a z}{1+z} \right) \cdot (1+z)^{3(1 + w_0 + w_a)}

Hubble 参数：

.. math::

   E^2(z) = \Omega_m (1+z)^3 + \Omega_r (1+z)^4 + \Omega_k (1+z)^2 + \Omega_\Lambda \, f_{\rm DE}(z)

CPL 参数说明
~~~~~~~~~~~~

CPL 继承 LCDM 所有参数，并增加：

.. list-table::
   :header-rows: 1
   :widths: 15 12 18 55

   * - 参数
     - 默认值
     - 先验范围
     - 描述
   * - ``w0``
     - -1.0
     - [-2.5, 0.5]
     - 今日 (:math:`z=0`) 的暗能量状态方程
   * - ``wa``
     - 0.0
     - [-3.0, 2.0]
     - 状态方程演化参数

特殊情况：

- :math:`w_0 = -1, w_a = 0`：LCDM
- :math:`w_a = 0`：wCDM（常数 :math:`w`）
- :math:`w_a \neq 0`：动态暗能量

CPL 基本使用
~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import CPL

   # 创建 CPL 模型
   model = CPL(H0=67.36, Omega_m=0.3, w0=-1.0, wa=0.1)

   # 查看演化的状态方程
   z_values = [0.0, 0.5, 1.0, 2.0]
   for z in z_values:
       w = model.w_z(z)
       print(f"w(z={z}) = {w:.3f}")

   # 输出:
   # w(z=0.0) = -1.000
   # w(z=0.5) = -0.967
   # w(z=1.0) = -0.950
   # w(z=2.0) = -0.933

   # 距离计算
   d_L = model.luminosity_distance(1.0)
   print(f"d_L(z=1) = {d_L:.2f} Mpc")

暗能量品质因子
~~~~~~~~~~~~~~

CPL 参数化常用于计算暗能量品质因子（Figure of Merit）：

.. math::

   \text{FoM} = \frac{1}{\sqrt{\det(\text{Cov}(w_0, w_a))}}

FoM 越大，对暗能量状态方程的约束越好。

ILCDM 模型
----------

ILCDM (Interacting Lambda-CDM) 模型假设暗物质和暗能量之间存在能量交换。

物理图像
~~~~~~~~

在标准 LCDM 中，暗物质和暗能量各自独立演化。ILCDM 引入相互作用项：

.. math::

   Q = \beta H \rho_c

其中 :math:`\beta` 是无量纲耦合常数，:math:`\rho_c` 是冷暗物质密度。

能量守恒方程：

.. math::

   \dot{\rho}_c + 3H\rho_c = -Q = -\beta H \rho_c

.. math::

   \dot{\rho}_\Lambda = +Q = \beta H \rho_c

密度演化：

.. math::

   \rho_c(z) = \rho_{c,0} (1+z)^{3 - \beta}

ILCDM 数学形式
~~~~~~~~~~~~~~

Hubble 参数：

.. math::

   E^2(z) = \Omega_\Lambda + \Omega_b (1+z)^3 + \Omega_r (1+z)^4 + \Omega_c \left[ \frac{\beta}{\beta - 1} + \frac{1}{1-\beta} (1+z)^{3-3\beta} \right]

其中 :math:`\Omega_c = \Omega_m - \Omega_b` 是冷暗物质密度参数。

ILCDM 参数说明
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 12 18 55

   * - 参数
     - 默认值
     - 先验范围
     - 描述
   * - ``beta``
     - 0.0
     - [-1.0, 1.0]
     - 暗能量-暗物质耦合常数

物理意义：

- :math:`\beta = 0`：标准 LCDM
- :math:`\beta > 0`：暗物质衰变为暗能量
- :math:`\beta < 0`：暗能量衰变为暗物质

**注意**：ILCDM 假设平直宇宙（:math:`\Omega_k = 0`）。

ILCDM 基本使用
~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import ILCDM

   # 创建 ILCDM 模型
   model = ILCDM(H0=67.36, Omega_m=0.3, beta=0.05)

   # 距离计算
   d_L = model.luminosity_distance(1.0)

   # 比较不同 beta 值
   for beta in [-0.1, 0.0, 0.1]:
       m = ILCDM(H0=67.36, Omega_m=0.3, beta=beta)
       d_L = m.luminosity_distance(1.0)
       print(f"beta={beta:+.1f}: d_L(z=1) = {d_L:.2f} Mpc")

Traced-Aware 接口
-----------------

为了与 NumPyro MCMC 采样器兼容，所有模型都提供 ``compute_grid_traced`` 静态方法，支持 JAX tracer 传播。

基本用法
~~~~~~~~

.. code-block:: python

   from hicosmo.models import LCDM, CPL
   import jax.numpy as jnp

   # 定义红移网格
   z_grid = jnp.linspace(0.01, 2.0, 1000)

   # 参数字典（用于 MCMC 采样）
   # 只需提供 H0 / Omega_m 以及模型特有参数
   params = {'H0': 70.0, 'Omega_m': 0.3}

   # 批量计算所有距离
   grid = LCDM.compute_grid_traced(z_grid, params)

   # 返回字典包含多个量
   d_L = grid['d_L']     # 光度距离
   D_M = grid['D_M']     # 横向共动距离
   D_H = grid['D_H']     # Hubble 距离 (c/H(z))
   E_z = grid['E_z']     # 无量纲 Hubble 参数
   d_C = grid['d_C']     # 共动距离
   dVc_dz = grid['dVc_dz']  # 共动体积元

CPL 模型示例
~~~~~~~~~~~~

.. code-block:: python

   # CPL 需要额外参数
   cpl_params = {
       'H0': 67.36,
       'Omega_m': 0.3,
       'w0': -1.0,
       'wa': 0.1
   }

   grid = CPL.compute_grid_traced(z_grid, cpl_params)
   d_L = grid['d_L']

wCDM / ILCDM 参数字典示例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import wCDM, ILCDM

   w_params = {'H0': 70.0, 'Omega_m': 0.3, 'w': -0.9}
   grid_w = wCDM.compute_grid_traced(z_grid, w_params)

   ilcdm_params = {'H0': 70.0, 'Omega_m': 0.3, 'beta': 0.05}
   grid_ilcdm = ILCDM.compute_grid_traced(z_grid, ilcdm_params)

在 NumPyro 中使用
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpyro
   import numpyro.distributions as dist

   def model(z_obs, mu_obs, mu_err):
       # 采样宇宙学参数
       H0 = numpyro.sample('H0', dist.Uniform(60, 80))
       Omega_m = numpyro.sample('Omega_m', dist.Uniform(0.1, 0.5))

       params = {'H0': H0, 'Omega_m': Omega_m}

       # 计算理论距离模数
       grid = LCDM.compute_grid_traced(z_obs, params)
       d_L = grid['d_L']
       mu_theory = 5 * jnp.log10(d_L) + 25

       # 似然
       numpyro.sample('obs', dist.Normal(mu_theory, mu_err), obs=mu_obs)

预设参数
--------

HIcosmo 提供常用宇宙学参数预设：

Planck 2018
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 值
     - 说明
   * - :math:`H_0`
     - 67.36
     - Hubble 常数 [km/s/Mpc]
   * - :math:`\Omega_m`
     - 0.3153
     - 物质密度参数
   * - :math:`\Omega_b`
     - 0.0493
     - 重子密度参数
   * - :math:`\sigma_8`
     - 0.8111
     - 物质功率谱归一化
   * - :math:`n_s`
     - 0.9649
     - 原初功率谱标量指数

.. code-block:: python

   # 使用 Planck 2018 默认参数
   model = LCDM()  # 默认就是 Planck 2018

   # 或显式使用预设
   from hicosmo.parameters import ParameterRegistry
   registry = ParameterRegistry.from_defaults('planck2018')

WMAP9
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - 参数
     - 值
     - 说明
   * - :math:`H_0`
     - 70.0
     - Hubble 常数 [km/s/Mpc]
   * - :math:`\Omega_m`
     - 0.279
     - 物质密度参数
   * - :math:`\Omega_b`
     - 0.046
     - 重子密度参数

添加新宇宙学模型
----------------

HIcosmo 采用极简的模型架构设计，让用户只需定义核心物理公式即可创建新的宇宙学模型。本节详细介绍如何添加自定义模型。

设计原则
~~~~~~~~

HIcosmo 模型架构遵循以下核心原则：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 原则
     - 描述
   * - **最小化子类**
     - 子类只定义差异性物理 + 必要参数
   * - **统一签名**
     - 所有核心计算函数使用 ``func(z, params)`` 签名
   * - **用户优先命名**
     - 公开 API 使用直觉名称 ``E_z``
   * - **装饰器生成**
     - 样板代码由装饰器自动生成
   * - **物理与框架分离**
     - 子类只管物理，框架逻辑在基类/装饰器

模型继承层次
~~~~~~~~~~~~

.. code-block:: text

   CosmologyBase (抽象基类)
       │
       └── LCDM (参考实现)
               │
               ├── wCDM (常数 w)
               ├── CPL  (演化 w(z))
               └── ILCDM (相互作用)

所有自定义模型应继承自 ``LCDM`` 或其子类。

最小子类模式
~~~~~~~~~~~~

一个最简的宇宙学模型只需要以下内容：

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - 必须定义
     - 可选定义
     - 绝不定义
   * - ``__init__``
     - ``w_z()`` (如果不同)
     - ``_E_z_static``
   * - ``@staticmethod E_z(z, params)``
     - ``get_parameters()`` (如果有新参数)
     - 距离计算方法
   * -
     -
     - 工厂方法

**核心思想**：只定义与父类不同的物理公式，其他一切由装饰器和基类自动处理。

完整示例：创建 Quintessence 模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下面创建一个简单的 Quintessence 暗能量模型，其状态方程为：

.. math::

   w(z) = w_0 + w_1 \ln(1+z)

**步骤 1：创建模型类**

.. code-block:: python

   # hicosmo/models/quintessence.py

   from typing import Dict, Union
   import jax.numpy as jnp

   from .lcdm import LCDM
   from .base import register_cosmology_model


   @register_cosmology_model
   class Quintessence(LCDM):
       """
       Quintessence 暗能量模型。

       状态方程参数化：w(z) = w0 + w1 * ln(1+z)

       物理背景：
       - 标量场慢滚近似下的有效状态方程
       - w1 > 0：早期宇宙暗能量更"硬"
       - w1 < 0：早期宇宙暗能量更"软"
       """

       def __init__(self, w0: float = -1.0, w1: float = 0.0, **kwargs):
           """
           初始化 Quintessence 模型。

           Parameters
           ----------
           w0 : float
               今日 (z=0) 的暗能量状态方程，默认 -1.0
           w1 : float
               状态方程的对数演化系数，默认 0.0
           **kwargs
               传递给 LCDM 的其他参数 (H0, Omega_m, etc.)
           """
           super().__init__(w0=w0, w1=w1, **kwargs)

       @staticmethod
       def E_z(z: Union[float, jnp.ndarray], params: Dict) -> jnp.ndarray:
           """
           计算无量纲 Hubble 参数 E(z) = H(z)/H0。

           这是模型的核心：只需定义物理公式！

           数学形式：
           对于 w(z) = w0 + w1*ln(1+z)，暗能量密度演化为：
           f_DE(z) = (1+z)^{3(1+w0)} * exp(3*w1*[(1+z)*ln(1+z) - z])
           """
           z_arr = jnp.asarray(z)
           one_plus_z = 1.0 + z_arr

           # 从 params 字典获取参数
           Omega_m = params['Omega_m']
           Omega_r = params.get('Omega_r', 0.0)
           Omega_k = params.get('Omega_k', 0.0)
           Omega_Lambda = params.get('Omega_Lambda', 1.0 - Omega_m - Omega_k - Omega_r)
           w0 = params.get('w0', -1.0)
           w1 = params.get('w1', 0.0)

           # 暗能量密度演化因子
           # 对 w(z) = w0 + w1*ln(1+z) 积分得到
           log_term = one_plus_z * jnp.log(one_plus_z) - z_arr
           f_DE = one_plus_z**(3.0 * (1.0 + w0)) * jnp.exp(3.0 * w1 * log_term)

           # E^2(z) = Omega_m*(1+z)^3 + Omega_r*(1+z)^4 + Omega_k*(1+z)^2 + Omega_Lambda*f_DE
           E_squared = (
               Omega_m * one_plus_z**3 +
               Omega_r * one_plus_z**4 +
               Omega_k * one_plus_z**2 +
               Omega_Lambda * f_DE
           )
           return jnp.sqrt(E_squared)

       def w_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
           """
           暗能量状态方程 w(z) = w0 + w1 * ln(1+z)。

           只有当状态方程与 LCDM 不同时才需要重写此方法。
           """
           z_arr = jnp.asarray(z)
           w0 = self.params.get('w0', -1.0)
           w1 = self.params.get('w1', 0.0)
           return w0 + w1 * jnp.log(1.0 + z_arr)

       @classmethod
       def get_parameters(cls):
           """
           返回模型参数列表。

           只有当模型引入新参数时才需要重写此方法。
           """
           from ..parameters import Parameter

           # 获取父类参数
           params = LCDM.get_parameters()

           # 添加新参数
           params.extend([
               Parameter(
                   name='w0',
                   value=-1.0,
                   free=False,
                   prior={'dist': 'uniform', 'min': -2.0, 'max': 0.0},
                   latex_label=r'$w_0$',
                   description='Dark energy equation of state at z=0'
               ),
               Parameter(
                   name='w1',
                   value=0.0,
                   free=False,
                   prior={'dist': 'uniform', 'min': -1.0, 'max': 1.0},
                   latex_label=r'$w_1$',
                   description='Logarithmic evolution coefficient'
               ),
           ])
           return params

**步骤 2：注册模型（可选）**

如果想让模型在 ``hicosmo.models`` 中可用，需要在 ``hicosmo/models/__init__.py`` 中添加：

.. code-block:: python

   from .quintessence import Quintessence

   __all__ = ['LCDM', 'wCDM', 'CPL', 'ILCDM', 'Quintessence']

**步骤 3：使用新模型**

.. code-block:: python

   from hicosmo.models import Quintessence

   # 创建模型实例
   model = Quintessence(H0=67.36, Omega_m=0.3, w0=-0.9, w1=0.1)

   # 所有距离方法自动可用（继承自 LCDM）
   d_L = model.luminosity_distance(1.0)
   D_A = model.angular_diameter_distance(1.0)

   # 查看演化的状态方程
   import jax.numpy as jnp
   z = jnp.array([0.0, 0.5, 1.0, 2.0])
   w = model.w_z(z)
   print(f"w(z) = {w}")

   # 在 MCMC 中使用
   params = {'H0': 70.0, 'Omega_m': 0.3, 'w0': -0.9, 'w1': 0.1}
   z_grid = jnp.linspace(0.01, 2.0, 100)
   grid = Quintessence.compute_grid_traced(z_grid, params)
   d_L_array = grid['d_L']

装饰器的作用
~~~~~~~~~~~~

``@register_cosmology_model`` 装饰器会自动为你的模型生成：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 生成的内容
     - 说明
   * - ``_E_z_kernel``
     - 从用户定义的 ``E_z`` 转换而来
   * - ``_E_z_static``
     - JIT 编译版本，用于高性能计算
   * - ``E_z(self, z)``
     - 实例方法包装，自动传入 ``self.params``
   * - ``compute_grid_traced``
     - MCMC 工厂方法，支持 JAX tracer
   * - ``sound_horizon_traced``
     - 声音视界计算工厂方法
   * - ``sound_horizon_drag_traced``
     - 拖拽时刻声视界工厂方法
   * - ``recombination_redshift_traced``
     - 重组红移计算工厂方法

**核心优势**：你只需写物理公式，装饰器处理所有框架代码！

参数字典规范
~~~~~~~~~~~~

``E_z(z, params)`` 中的 ``params`` 是一个字典，包含所有宇宙学参数：

**必须参数**：

.. code-block:: python

   params = {
       'Omega_m': 0.3,    # 必须：物质密度参数
   }

**可选参数**（有默认值）：

.. code-block:: python

   params = {
       'Omega_m': 0.3,
       'Omega_r': 0.0,              # 辐射密度，默认 0
       'Omega_k': 0.0,              # 曲率，默认 0（平直）
       'Omega_Lambda': 0.7,         # 暗能量密度，默认 1-Omega_m-Omega_k-Omega_r
       # 模型特有参数
       'w': -1.0,                   # wCDM
       'w0': -1.0, 'wa': 0.0,       # CPL
       'beta': 0.0,                 # ILCDM
   }

**推荐写法**：使用 ``params.get('key', default)`` 提供默认值：

.. code-block:: python

   Omega_r = params.get('Omega_r', 0.0)  # 如果不存在，使用 0.0

测试新模型
~~~~~~~~~~

创建新模型后，应编写测试验证其正确性：

.. code-block:: python

   # tests/test_quintessence.py

   import pytest
   import jax.numpy as jnp
   from hicosmo.models import Quintessence, LCDM


   class TestQuintessence:
       """Quintessence 模型测试套件"""

       def test_reduces_to_lcdm(self):
           """w0=-1, w1=0 时应退化为 LCDM"""
           quint = Quintessence(H0=67.36, Omega_m=0.3, w0=-1.0, w1=0.0)
           lcdm = LCDM(H0=67.36, Omega_m=0.3)

           z = jnp.array([0.5, 1.0, 2.0])

           # E(z) 应该相同
           E_quint = quint.E_z(z)
           E_lcdm = lcdm.E_z(z)
           assert jnp.allclose(E_quint, E_lcdm, rtol=1e-10)

           # 距离也应该相同
           d_L_quint = quint.luminosity_distance(1.0)
           d_L_lcdm = lcdm.luminosity_distance(1.0)
           assert jnp.isclose(d_L_quint, d_L_lcdm, rtol=1e-10)

       def test_equation_of_state(self):
           """测试状态方程 w(z) = w0 + w1*ln(1+z)"""
           quint = Quintessence(w0=-0.9, w1=0.1)

           # z=0 时 w = w0
           assert jnp.isclose(quint.w_z(0.0), -0.9, rtol=1e-10)

           # z=e-1 时 w = w0 + w1*ln(e) = w0 + w1
           z_e = jnp.e - 1
           assert jnp.isclose(quint.w_z(z_e), -0.9 + 0.1, rtol=1e-6)

       def test_compute_grid_traced(self):
           """测试 traced 接口"""
           params = {'H0': 70.0, 'Omega_m': 0.3, 'w0': -0.9, 'w1': 0.1}
           z_grid = jnp.linspace(0.01, 2.0, 100)

           grid = Quintessence.compute_grid_traced(z_grid, params)

           # 检查返回值
           assert 'd_L' in grid
           assert 'D_M' in grid
           assert 'E_z' in grid

           # 检查形状
           assert grid['d_L'].shape == z_grid.shape

       def test_jax_gradients(self):
           """测试 JAX 自动微分"""
           from jax import grad

           def loss(H0):
               params = {'H0': H0, 'Omega_m': 0.3, 'w0': -1.0, 'w1': 0.0}
               grid = Quintessence.compute_grid_traced(jnp.array([1.0]), params)
               return grid['d_L'][0]

           # 梯度应该存在且有限
           grad_H0 = grad(loss)(70.0)
           assert jnp.isfinite(grad_H0)

运行测试：

.. code-block:: bash

   pytest tests/test_quintessence.py -v

常见错误与解决
~~~~~~~~~~~~~~

**错误 1：忘记使用装饰器**

.. code-block:: python

   # ❌ 错误：没有装饰器
   class MyModel(LCDM):
       @staticmethod
       def E_z(z, params):
           ...

   # ✅ 正确：使用装饰器
   @register_cosmology_model
   class MyModel(LCDM):
       @staticmethod
       def E_z(z, params):
           ...

**错误 2：``E_z`` 不是静态方法**

.. code-block:: python

   # ❌ 错误：实例方法
   @register_cosmology_model
   class MyModel(LCDM):
       def E_z(self, z, params):  # 不应该有 self
           ...

   # ✅ 正确：静态方法
   @register_cosmology_model
   class MyModel(LCDM):
       @staticmethod
       def E_z(z, params):
           ...

**错误 3：参数签名错误**

.. code-block:: python

   # ❌ 错误：展开参数
   @staticmethod
   def E_z(z, Omega_m, Omega_Lambda, w):
       ...

   # ✅ 正确：使用 params 字典
   @staticmethod
   def E_z(z, params):
       Omega_m = params['Omega_m']
       Omega_Lambda = params.get('Omega_Lambda', 1.0 - Omega_m)
       w = params.get('w', -1.0)
       ...

**错误 4：在子类中重复定义框架方法**

.. code-block:: python

   # ❌ 错误：不需要重新定义这些方法
   @register_cosmology_model
   class MyModel(LCDM):
       @staticmethod
       def E_z(z, params):
           ...

       @staticmethod
       def _E_z_static(z, params):  # ❌ 装饰器会生成
           ...

       def compute_grid_traced(self, z, params):  # ❌ 装饰器会生成
           ...

性能优化
--------

JIT 编译
~~~~~~~~

所有模型的核心计算都经过 JAX JIT 编译优化：

.. code-block:: python

   from jax import jit
   import jax.numpy as jnp
   from hicosmo.models import LCDM

   model = LCDM()

   # 首次调用：编译
   %timeit model.luminosity_distance(1.0)  # ~100ms (含编译)

   # 后续调用：已编译
   %timeit model.luminosity_distance(1.0)  # ~0.01ms

向量化计算
~~~~~~~~~~

使用 ``vmap`` 实现高效批量计算：

.. code-block:: python

   from jax import vmap

   z_array = jnp.linspace(0.01, 2.0, 1000)

   # 向量化距离计算
   d_L_array = vmap(model.luminosity_distance)(z_array)

或者直接使用 ``compute_grid_traced``：

.. code-block:: python

   grid = LCDM.compute_grid_traced(z_array, {'H0': 70.0, 'Omega_m': 0.3})
   d_L_array = grid['d_L']

性能基准
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - 操作
     - scipy (qcosmc)
     - HIcosmo (JAX)
     - 加速比
   * - 距离计算 (1000点)
     - 150 ms
     - 20 ms
     - **7.5x**
   * - E(z) 计算 (10000点)
     - 50 ms
     - 2 ms
     - **25x**
   * - 增长因子
     - 120 ms
     - 15 ms
     - **8x**

最佳实践
--------

1. **选择合适的模型**

   - 标准分析使用 LCDM
   - 探索暗能量性质使用 CPL
   - 研究暗物质-暗能量相互作用使用 ILCDM

2. **利用预设参数**

   .. code-block:: python

      # 使用 Planck 先验
      model = LCDM()  # 默认 Planck 2018

3. **批量计算优化**

   .. code-block:: python

      # ✅ 推荐：一次计算多个红移
      grid = LCDM.compute_grid_traced(z_array, params)

      # ❌ 避免：循环调用
      for z in z_array:
          d_L = model.luminosity_distance(z)

4. **MCMC 使用 traced 接口**

   .. code-block:: python

      # 在 NumPyro 模型中使用静态方法
      grid = LCDM.compute_grid_traced(z_obs, params)

下一步
------

- `似然函数 <likelihoods.html>`_：了解如何使用观测数据
- `采样器 <samplers.html>`_：学习 MCMC 参数估计
- `Fisher 预报 <fisher.html>`_：进行巡天预测
