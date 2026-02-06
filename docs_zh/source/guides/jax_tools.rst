JAX 数值工具参考
================

``hicosmo.utils.jax_tools`` 模块提供了一套完整的 JAX 优化数值工具，包括积分、微分、求根和ODE求解器。
所有函数都支持 JIT 编译和自动微分。

.. contents:: 目录
   :local:
   :depth: 2

快速开始
--------

.. code-block:: python

   from hicosmo.utils import (
       integrate_simpson,      # Simpson积分
       gauss_legendre_integrate,  # Gauss-Legendre积分
       cumulative_trapezoid,   # 累积梯形积分
       gradient_1d,            # 一维梯度
       newton_root,            # 牛顿法求根
       odeint_rk4,             # RK4 ODE求解器
   )


积分工具
--------

梯形积分
^^^^^^^^

.. function:: trapezoid(y, x, axis=-1)

   JAX 梯形积分包装器。

   :param y: 函数值数组
   :param x: 自变量数组
   :param axis: 积分轴
   :return: 积分结果

   **示例**:

   .. code-block:: python

      import jax.numpy as jnp
      from hicosmo.utils import trapezoid

      x = jnp.linspace(0, 1, 100)
      y = x**2
      result = trapezoid(y, x)  # ≈ 1/3


Simpson积分
^^^^^^^^^^^

.. function:: simpson(y, x)

   复合Simpson积分，用于离散采样数据。

   :param y: 函数值数组（至少3个点）
   :param x: 自变量数组
   :return: 积分结果
   :raises ValueError: 如果点数少于3个

.. function:: integrate_simpson(func, a, b, *, num=512)

   在区间 [a, b] 上使用Simpson法则积分函数。

   :param func: 被积函数 f(x)
   :param a: 积分下限
   :param b: 积分上限
   :param num: 采样点数（默认512，自动调整为偶数）
   :return: 积分结果

   **示例**:

   .. code-block:: python

      from hicosmo.utils import integrate_simpson
      import jax.numpy as jnp

      # 计算 ∫₀¹ x² dx = 1/3
      result = integrate_simpson(lambda x: x**2, 0, 1)
      print(f"Result: {result:.6f}")  # ≈ 0.333333


对数空间积分
^^^^^^^^^^^^

.. function:: integrate_logspace(func, k_min, k_max, *, num=512)

   在对数空间网格上积分，适用于正定义域的幂律函数。

   :param func: 被积函数 f(k)
   :param k_min: 积分下限（必须 > 0）
   :param k_max: 积分上限
   :param num: 采样点数
   :return: 积分结果

   **示例**:

   .. code-block:: python

      from hicosmo.utils import integrate_logspace

      # 对幂律函数积分
      result = integrate_logspace(lambda k: k**(-2), 0.01, 100)


Gauss-Legendre 积分
^^^^^^^^^^^^^^^^^^^

高精度求积方法，对光滑函数非常有效。

.. function:: gauss_legendre_nodes_weights(order)

   返回指定阶数的 Gauss-Legendre 节点和权重。

   :param order: 阶数（支持 8, 12, 16）
   :return: (nodes, weights) 元组

.. function:: gauss_legendre_integrate(integrand, a, b, *, order=12)

   在区间 [a, b] 上进行 Gauss-Legendre 积分。

   :param integrand: 被积函数
   :param a: 积分下限
   :param b: 积分上限
   :param order: 积分阶数（8, 12, 或 16）
   :return: 积分结果

.. function:: gauss_legendre_integrate_batch(integrand, z_array, *, z_min=0.0, order=12)

   批量 Gauss-Legendre 积分，对多个上限同时计算。

   :param integrand: 被积函数
   :param z_array: 上限数组
   :param z_min: 积分下限
   :param order: 积分阶数
   :return: 积分结果数组

   **示例**:

   .. code-block:: python

      from hicosmo.utils import gauss_legendre_integrate_batch
      import jax.numpy as jnp

      # 计算多个上限的积分
      z_values = jnp.array([0.5, 1.0, 2.0])
      results = gauss_legendre_integrate_batch(
          lambda z: 1.0 / jnp.sqrt(0.3 * (1 + z)**3 + 0.7),
          z_values,
          z_min=0.0,
          order=12
      )


累积积分
^^^^^^^^

.. function:: cumulative_trapezoid(y, x)

   累积梯形积分，返回从第一个点开始的累积积分值。

   :param y: 函数值数组
   :param x: 自变量数组
   :return: 累积积分数组（第一个元素为0）

   **示例**:

   .. code-block:: python

      from hicosmo.utils import cumulative_trapezoid
      import jax.numpy as jnp

      x = jnp.linspace(0, 1, 100)
      y = jnp.ones_like(x)  # 常数函数
      cumulative = cumulative_trapezoid(y, x)
      # cumulative[-1] ≈ 1.0


.. function:: integrate_batch_cumulative(func, z_array, *, z_min=0.0, z_max=None, n_grid=4096)

   使用累积网格和插值进行批量积分。适用于大量查询点。

   :param func: 被积函数
   :param z_array: 查询点数组
   :param z_min: 积分下限
   :param z_max: 网格上限（默认为 z_array 的最大值）
   :param n_grid: 网格点数
   :return: 积分结果数组

   **性能提示**: 当查询点很多时，此方法比 ``gauss_legendre_integrate_batch`` 更快。


分段积分
^^^^^^^^

.. function:: integrate_segmented(func, a, b, *, n_segments=8, order=12)

   分段 Gauss-Legendre 积分，适用于大积分区间。

   :param func: 被积函数
   :param a: 积分下限
   :param b: 积分上限
   :param n_segments: 分段数
   :param order: 每段的 Gauss-Legendre 阶数
   :return: 积分结果

   **使用场景**: 当积分区间很大（如 [0, 1000]）时，单次 Gauss-Legendre
   积分精度不足，使用分段积分可以提高精度。


自适应Simpson积分
^^^^^^^^^^^^^^^^^

.. function:: integrate_adaptive_simpson(func, a, b, *, num=64, max_num=4096, tol=1e-6)

   自适应 Simpson 积分，自动调整网格密度。

   :param func: 被积函数
   :param a: 积分下限
   :param b: 积分上限
   :param num: 初始网格点数（必须为偶数）
   :param max_num: 最大网格点数（必须是 num 的整数倍）
   :param tol: 收敛容差
   :return: 积分结果


微分工具
--------

一维梯度
^^^^^^^^

.. function:: gradient_1d(values, coords)

   计算一维网格上的导数，支持非均匀网格。

   :param values: 函数值数组
   :param coords: 坐标数组
   :return: 导数数组

   **特点**:
   - 端点使用3点Lagrange插值（二阶精度）
   - 内部点使用二阶中心差分
   - 支持非均匀网格

   **示例**:

   .. code-block:: python

      from hicosmo.utils import gradient_1d
      import jax.numpy as jnp

      x = jnp.linspace(0, 2 * jnp.pi, 100)
      y = jnp.sin(x)
      dy_dx = gradient_1d(y, x)  # ≈ cos(x)


有限差分梯度
^^^^^^^^^^^^

.. function:: finite_difference_grad(func, x, *, step=1e-5)

   使用中心有限差分计算标量函数的梯度。

   :param func: 标量函数 f(x) -> scalar
   :param x: 求值点（数组）
   :param step: 差分步长
   :return: 梯度数组

   **注意**: 对于可微函数，推荐使用 ``jax.grad`` 进行自动微分。


自动微分包装器
^^^^^^^^^^^^^^

.. function:: grad(func)

   返回函数的梯度函数（``jax.grad`` 包装器）。

.. function:: jacobian(func)

   返回函数的雅可比矩阵函数（``jax.jacobian`` 包装器）。

.. function:: hessian(func)

   返回函数的 Hessian 矩阵函数（``jax.hessian`` 包装器）。

   **示例**:

   .. code-block:: python

      from hicosmo.utils import grad, hessian
      import jax.numpy as jnp

      def f(x):
          return jnp.sum(x**2)

      grad_f = grad(f)
      hess_f = hessian(f)

      x = jnp.array([1.0, 2.0])
      print(grad_f(x))  # [2., 4.]


求根工具
--------

Newton-Raphson 法
^^^^^^^^^^^^^^^^^

.. function:: newton_root(func, x0, *, tol=1e-8, max_iter=64, deriv=None)

   使用 Newton-Raphson 方法求解 f(x) = 0。

   :param func: 目标函数 f(x)
   :param x0: 初始猜测值
   :param tol: 收敛容差
   :param max_iter: 最大迭代次数
   :param deriv: 导数函数（可选，默认使用自动微分）
   :return: 根的近似值

   **示例**:

   .. code-block:: python

      from hicosmo.utils import newton_root
      import jax.numpy as jnp

      # 求解 x² - 2 = 0
      root = newton_root(lambda x: x**2 - 2, jnp.array(1.5))
      print(f"√2 ≈ {root:.10f}")  # 1.4142135624


二分法
^^^^^^

.. function:: bisection_root(func, a, b, *, tol=1e-8, max_iter=128)

   使用二分法求解 f(x) = 0（要求区间端点符号相反）。

   :param func: 目标函数 f(x)
   :param a: 区间左端点
   :param b: 区间右端点
   :param tol: 收敛容差
   :param max_iter: 最大迭代次数
   :return: 根的近似值（如果端点同号则返回 NaN）

   **示例**:

   .. code-block:: python

      from hicosmo.utils import bisection_root
      import jax.numpy as jnp

      # 求解 x³ - x - 1 = 0
      root = bisection_root(
          lambda x: x**3 - x - 1,
          jnp.array(1.0),
          jnp.array(2.0)
      )


ODE 求解器
----------

RK4 单步
^^^^^^^^

.. function:: rk4_step(func, t, y, dt)

   执行一步四阶 Runge-Kutta 积分。

   :param func: ODE 右端项 f(t, y)
   :param t: 当前时间
   :param y: 当前状态
   :param dt: 时间步长
   :return: 下一时刻的状态


RK4 积分器
^^^^^^^^^^

.. function:: odeint_rk4(func, y0, t_grid)

   在固定网格上使用 RK4 方法求解 ODE y' = f(t, y)。

   :param func: ODE 右端项 f(t, y)
   :param y0: 初始条件
   :param t_grid: 时间网格
   :return: 解数组（每个时间点一行）

   **示例**:

   .. code-block:: python

      from hicosmo.utils import odeint_rk4
      import jax.numpy as jnp

      # 求解 y' = -y, y(0) = 1 (解析解: y = e^(-t))
      def rhs(t, y):
          return -y

      t = jnp.linspace(0, 5, 100)
      y = odeint_rk4(rhs, jnp.array(1.0), t)


.. function:: solve_ivp_rk4(func, y0, t0, t1, *, n_steps=256)

   在区间 [t0, t1] 上使用固定步长 RK4 求解初值问题。

   :param func: ODE 右端项 f(t, y)
   :param y0: 初始条件
   :param t0: 起始时间
   :param t1: 终止时间
   :param n_steps: 步数
   :return: (t_grid, y_grid) 元组


宇宙学专用工具
--------------

声视界计算
^^^^^^^^^^

基于 Eisenstein & Hu (1998) 拟合公式。

.. function:: sound_horizon_drag_eh98(H0, Omega_m, Omega_b, T_cmb)

   计算拖拽时刻（drag epoch）的声视界 r_d。

   :param H0: 哈勃常数 [km/s/Mpc]
   :param Omega_m: 物质密度参数
   :param Omega_b: 重子密度参数
   :param T_cmb: CMB 温度 [K]
   :return: 声视界 r_d [Mpc]

   **示例**:

   .. code-block:: python

      from hicosmo.utils.jax_tools import sound_horizon_drag_eh98

      r_d = sound_horizon_drag_eh98(
          H0=67.36,
          Omega_m=0.3153,
          Omega_b=0.0493,
          T_cmb=2.7255
      )
      print(f"r_d = {r_d:.2f} Mpc")  # ≈ 147 Mpc


.. function:: sound_horizon_eh98(H0, Omega_m, Omega_b, T_cmb, z)

   计算任意红移 z 处的声视界。

   :param H0: 哈勃常数 [km/s/Mpc]
   :param Omega_m: 物质密度参数
   :param Omega_b: 重子密度参数
   :param T_cmb: CMB 温度 [K]
   :param z: 红移
   :return: 声视界 r_s(z) [Mpc]


性能对比
--------

各积分方法的适用场景：

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - 方法
     - 精度
     - 适用场景
   * - ``trapezoid``
     - O(h²)
     - 快速粗略估计
   * - ``simpson``
     - O(h⁴)
     - 光滑函数的标准选择
   * - ``gauss_legendre``
     - 高
     - 小区间高精度积分
   * - ``integrate_batch_cumulative``
     - O(h²)
     - 大量查询点的批量积分
   * - ``integrate_segmented``
     - 高
     - 大区间高精度积分


JIT 编译注意事项
----------------

所有函数都兼容 JAX JIT 编译：

.. code-block:: python

   from jax import jit
   from hicosmo.utils import integrate_simpson

   @jit
   def my_function(params):
       result = integrate_simpson(
           lambda x: params[0] * x**2 + params[1],
           0, 1
       )
       return result

**注意事项**:

1. 积分区间 [a, b] 可以是动态值
2. 网格点数 (num, n_grid) 必须是静态值
3. 使用 ``static_argnums`` 处理需要静态化的参数


完整示例
--------

计算共动距离
^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   from hicosmo.utils import integrate_batch_cumulative

   # 宇宙学参数
   H0 = 70.0  # km/s/Mpc
   Omega_m = 0.3
   c = 299792.458  # km/s

   def E_z(z):
       """Hubble 参数 E(z) = H(z)/H0"""
       return jnp.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

   # 计算多个红移的共动距离
   z_array = jnp.array([0.1, 0.5, 1.0, 2.0])
   integrals = integrate_batch_cumulative(
       lambda z: 1.0 / E_z(z),
       z_array,
       n_grid=4096
   )
   d_c = (c / H0) * integrals

   for z, d in zip(z_array, d_c):
       print(f"z = {z:.1f}: d_c = {d:.1f} Mpc")


参考文献
--------

- Eisenstein, D. J., & Hu, W. (1998). Baryonic Features in the Matter Transfer Function. ApJ, 496, 605.
