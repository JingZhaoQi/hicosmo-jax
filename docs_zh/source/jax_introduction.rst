JAX 技术详解
============

本章详细介绍 JAX 的核心概念、技术优势，以及 HIcosmo 选择 JAX 作为计算引擎的原因。
如果你对 JAX 不熟悉，本章将帮助你理解为什么 JAX 是现代科学计算的革命性工具。

什么是 JAX？
------------

JAX 是 Google 开发的一个高性能数值计算库，可以理解为 **"可微分、可编译、可并行的 NumPy"**。
它的名字来源于 **J**\ ust-in-time compilation + **A**\ utomatic differentiation + **X**\ LA (加速线性代数)。

**官方定义**：

   JAX is NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research.

**通俗解释**：

- 如果你会用 NumPy，你就会用 JAX（API 几乎相同）
- JAX 可以自动计算任何函数的导数（包括高阶导数）
- JAX 可以自动将代码编译为高效的机器码
- JAX 可以透明地在 CPU、GPU、TPU 上运行

JAX 被视为一种 **可微分编程语言**，它将 Python 作为构建 XLA 计算图的"元编程语言"。

函数式编程：JAX 的核心哲学
--------------------------

JAX 的设计核心在于 **函数式编程（Functional Programming, FP）**，这与传统的面向对象编程（OOP）框架（如 PyTorch）有着本质的区别。

为什么选择函数式编程？
~~~~~~~~~~~~~~~~~~~~~~

JAX 采用函数式编程的主要原因包括：

1. **数学一致性**：自动微分在数学上本质上是函数的变换。函数式编程允许 JAX 将 ``grad``（求导）、``jit``（编译）、``vmap``（向量化）作为高阶函数自由嵌套和组合，例如：``jit(vmap(grad(f)))``

2. **便于编译优化**：XLA 编译器需要一个静态的、无副作用的计算图来执行算子融合和内存优化。面向对象中的状态突变（如修改类成员变量）会破坏这种静态图的构建

3. **确定性随机状态**：传统 OOP 框架常使用全局随机状态，而 JAX 强制要求显式传递随机数生成器（PRNG）密钥，确保代码的可复现性

4. **状态与逻辑分离**：在函数式范式下，模型参数作为函数的输入参数传入，而不是存储在对象内部，使得参数分发和更新更加透明

纯函数 (Pure Functions)
~~~~~~~~~~~~~~~~~~~~~~~

纯函数是 JAX 能够实现高性能计算的基石。

**定义**：纯函数是指 **不改变外部状态** 且 **没有副作用** 的函数。

**特性**：对于相同的输入，纯函数 **必须始终产生相同的输出**。

**为什么重要**：纯函数的可预测性允许 JAX 的 XLA 编译器对代码进行深度优化、实现即时编译（JIT）以及在 GPU/TPU 上轻松实现算子并行和分片。

.. code-block:: python

   import jax.numpy as jnp

   # ✅ 纯函数：相同输入 → 相同输出，无副作用
   def pure_function(x, y):
       return jnp.sin(x) + jnp.cos(y)

   # ❌ 非纯函数：依赖外部状态
   global_state = 0
   def impure_function(x):
       global global_state
       global_state += 1  # 副作用：修改外部状态
       return x + global_state

不可变数组 (Immutable Arrays)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 JAX 中，数组一旦创建就 **不能被修改**。

**与 NumPy 的区别**：传统 NumPy 允许原地（in-place）更新数组，JAX 数组是不可变的。

**操作方式**：如果需要修改数组中的某个元素，JAX 不会直接覆盖原内存，而是创建一个包含修改后内容的新数组。

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp

   # NumPy：原地修改
   np_arr = np.array([1, 2, 3])
   np_arr[0] = 99  # ✅ 可以直接修改

   # JAX：不可变，需要创建新数组
   jax_arr = jnp.array([1, 2, 3])
   # jax_arr[0] = 99  # ❌ TypeError!
   jax_arr = jax_arr.at[0].set(99)  # ✅ 创建新数组

**设计原因**：不可变性解锁了编译优化，使 XLA 能够安全地进行算子融合和内存重用。

梯度计算对比：PyTorch vs JAX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下是 PyTorch（面向对象）与 JAX（函数式）在处理梯度时的核心区别：

**PyTorch (面向对象范式)**：

梯度被视为张量对象的一个属性，通过修改状态来实现。

.. code-block:: python

   import torch

   def f(x):
       return x**2 + 3*x + 5

   x = torch.tensor([2.0], requires_grad=True)
   loss = f(x)
   loss.backward()  # 触发 AutoGrad 并在计算图中回溯

   # 梯度被"存储"在变量 x 的 .grad 属性中（状态改变）
   print(x.grad.item())  # 输出: 7.0

**JAX (函数式范式)**：

求导是一个 **函数变换**，不会改变任何输入或对象的状态。

.. code-block:: python

   import jax

   def f(x):
       return x**2 + 3*x + 5

   # jax.grad 接收一个函数并返回另一个函数（函数变换）
   df_dx = jax.grad(f)

   x_value = 2.0
   # 梯度直接作为返回值返回，不修改原变量
   derivative_value = df_dx(x_value)

   print(derivative_value)  # 输出: 7.0

**对比总结**：

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 特性
     - 面向对象 (PyTorch)
     - 函数式 (JAX)
   * - **状态管理**
     - 状态存储在对象内部，允许原地修改
     - 纯函数，状态通过参数传递，不可变
   * - **梯度处理**
     - 通过 ``loss.backward()`` 存储在 ``.grad`` 属性
     - 使用 ``jax.grad`` 变换函数，直接返回梯度值
   * - **随机数**
     - 依赖隐式全局随机状态
     - 必须显式传递 PRNG 密钥
   * - **底层核心**
     - 动态计算图，易于调试
     - XLA 编译的静态计算图，追求极致性能

JAX 的四大核心特性
------------------

1. NumPy 兼容 API
~~~~~~~~~~~~~~~~~

JAX 提供与 NumPy 几乎完全兼容的 API：

.. code-block:: python

   # NumPy 代码
   import numpy as np

   x = np.array([1.0, 2.0, 3.0])
   y = np.sin(x) + np.cos(x)
   z = np.dot(x, y)

   # JAX 代码（只需改变 import）
   import jax.numpy as jnp

   x = jnp.array([1.0, 2.0, 3.0])
   y = jnp.sin(x) + jnp.cos(x)
   z = jnp.dot(x, y)

**主要区别**：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 特性
     - NumPy
     - JAX
   * - **可变性**
     - 数组可变 (``x[0] = 5``)
     - 数组不可变 (``x = x.at[0].set(5)``)
   * - **随机数**
     - 全局状态 (``np.random.rand()``)
     - 显式 key (``jax.random.uniform(key)``)
   * - **执行模式**
     - 即时执行
     - 可延迟执行（traced）
   * - **硬件支持**
     - 仅 CPU
     - CPU / GPU / TPU

2. 自动微分 (Automatic Differentiation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX 的自动微分是其最强大的特性之一。它可以自动计算任意复杂函数的精确导数。

**什么是自动微分？**

在科学计算中，我们经常需要计算函数的导数（梯度）。传统方法有三种：

1. **符号微分**：用数学公式手动推导，容易出错，无法处理复杂函数
2. **数值微分**：用有限差分近似，:math:`f'(x) \approx \frac{f(x+h) - f(x)}{h}`，精度低，效率低
3. **自动微分**：通过链式法则自动追踪计算图，精确高效

**JAX 的自动微分示例**：

.. code-block:: python

   import jax
   import jax.numpy as jnp

   # 定义一个复杂函数
   def f(x):
       return jnp.sin(x) * jnp.exp(-x**2) + jnp.log(1 + x**2)

   # 自动获取导数函数
   df = jax.grad(f)      # 一阶导数
   d2f = jax.grad(df)    # 二阶导数
   d3f = jax.grad(d2f)   # 三阶导数

   # 计算导数值
   x = 1.0
   print(f"f(x)   = {f(x):.6f}")
   print(f"f'(x)  = {df(x):.6f}")
   print(f"f''(x) = {d2f(x):.6f}")
   print(f"f'''(x)= {d3f(x):.6f}")

**向量函数的 Jacobian 和 Hessian**：

.. code-block:: python

   # Hessian 矩阵（用于优化和 Fisher 矩阵）
   def scalar_func(x):
       return jnp.sum(x**2)

   hessian = jax.hessian(scalar_func)

**自动微分的优势**：

- **精确**：结果是精确的数学导数，不是数值近似
- **高效**：复杂度与原函数相同，不需要多次函数评估
- **通用**：支持任意复杂的函数组合，包括条件语句和循环

3. JIT 编译与 XLA 编译器
~~~~~~~~~~~~~~~~~~~~~~~~

JIT 编译是 JAX 性能的核心来源。它将 Python 代码编译为优化的机器码。

**XLA 编译器工作原理**

XLA（加速线性代数，Accelerated Linear Algebra）是一个专门针对线性代数运算的编译器，它将 JAX 编写的程序转化为针对特定硬件（CPU、GPU 或 TPU）优化过的可执行内核。

工作流程分为以下几个阶段：

1. **追踪 (Tracing)**：当你对一个 Python 函数应用 ``jax.jit`` 并首次调用它时，JAX 会通过"追踪"机制记录该函数在处理具有特定形状和类型的数组时的操作序列

2. **构建计算图**：追踪得到的信息会被转化为 XLA 计算图（HLO，高级优化器表示）

3. **硬件编译与优化**：XLA 编译器会对该图进行全方位的分析，执行包括死代码消除、算子融合等优化策略，最后将其编译为直接在硬件加速器上运行的二进制机器码

**算子融合 (Operator Fusion)**

算子融合是 XLA 最具杀伤力的优化技术之一。

**原理**：在传统的向量化编程中，每一步简单操作（如 ``A * B + C``）通常会产生大量的中间临时数组。这些中间结果需要写回显存然后再读入，造成了严重的内存带宽浪费。

**实现方式**：XLA 会在计算图中识别出连续的操作，并将其"融合"为一个单一的 GPU 内核。这意味着中间计算结果会直接保留在处理器的快速寄存器或缓存中，而不是频繁地与主存交换数据。

**收益**：显著降低了内存带宽占用并减少了内存分配压力，在处理高维宇宙学数据时尤其能避免 OOM（内存溢出）错误。

**比喻说明**：

   你可以把传统的 Python 运算想象成快餐店点餐：你每点一个汉堡，服务员都要跑回厨房做（调用 C 库），做完拿给你，你再点一根薯条，服务员再跑一次。

   而 XLA 编译器和 JIT 就像是一个智能主厨：你告诉他你要点一个套餐（整个函数），他会分析菜单，决定同时炸薯条和煎肉饼（算子融合），最后一次性把热气腾腾的完整套餐端给你。

   这种"一站式"处理极大地减少了路途上的等待时间（Python 开销和内存读写）。

**JIT 编译示例**：

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import time

   # 定义一个计算密集型函数
   def slow_function(x):
       for _ in range(100):
           x = jnp.sin(x) + jnp.cos(x)
       return x

   # JIT 编译版本
   fast_function = jax.jit(slow_function)

   x = jnp.ones(10000)

   # 第一次调用：包含编译时间
   start = time.time()
   _ = fast_function(x)
   print(f"第一次调用（含编译）: {time.time() - start:.4f}s")

   # 后续调用：使用缓存的编译结果
   start = time.time()
   for _ in range(100):
       _ = fast_function(x)
   print(f"后续100次调用: {time.time() - start:.4f}s")

4. 向量化 (Vectorization with vmap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``vmap`` 是 JAX 的向量化变换，可以自动将标量函数转换为批量函数。

**示例**：

.. code-block:: python

   import jax
   import jax.numpy as jnp

   # 定义一个处理单个样本的函数
   def single_sample_loglike(theta, data_point):
       mu, sigma = theta
       return -0.5 * ((data_point - mu) / sigma)**2 - jnp.log(sigma)

   # 使用 vmap 自动批量化
   batch_loglike = jax.vmap(single_sample_loglike, in_axes=(None, 0))

   theta = (0.0, 1.0)
   data = jnp.array([0.1, 0.5, -0.3, 0.8, -0.2])

   # 一次性计算所有数据点的对数似然
   log_likes = batch_loglike(theta, data)

**性能对比** （1000 个数据点）：

- Python 循环：~50ms
- NumPy 向量化：~2ms
- JAX vmap + JIT：~0.1ms

PRNG 随机数管理
---------------

JAX 的伪随机数生成器（PRNG）管理机制是其函数式编程哲学的核心体现。与 NumPy 或 PyTorch 等传统框架使用的"全局随机状态"不同，JAX 采用的是一种 **显式、无状态且可分支** 的随机数管理系统。

核心机制：显式密钥与分支
~~~~~~~~~~~~~~~~~~~~~~~~

在 JAX 中，随机性不是由一个随时间变化的全局种子（Seed）控制的，而是通过一个 **随机密钥（PRNGKey）** 来管理的。

- **PRNGKey**：这是一个包含随机状态的数组对象。所有的随机函数（如 ``jax.random.normal``）都需要接收一个 Key 作为输入，并根据该 Key 生成确定的输出

- **密钥分支（Splitting）**：为了生成多个不相关的随机数流，JAX 使用 ``jax.random.split`` 函数将一个主 Key 分解为多个子 Key。这类似于树状结构：从一个根种子开始，不断分叉出独立的随机支流

为什么需要显式传递随机密钥？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX 要求显式传递密钥，根本原因在于其对 **纯函数** 的追求：

1. **消除副作用**：传统的全局随机状态本质上是一个"全局变量"。每次调用随机函数都会改变这个全局状态，这属于"副作用"。为了实现 JIT 和算子融合，函数必须是"纯"的

2. **可复现性**：显式密钥确保了无论计算在何处、以何种顺序执行，只要 Key 相同，结果就完全一致

3. **并行化与矢量化**：在大规模并行计算（如 ``pmap`` 或 ``vmap``）中，全局状态会导致竞争和难以预料的结果。显式密钥使得 PRNG 的生成过程可以轻松地在多个硬件设备上进行并行化

**对比**：

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 特性
     - 传统框架 (PyTorch/NumPy)
     - JAX
   * - **状态管理**
     - 隐式、全局。存在一个后台全局状态
     - 显式、局部。密钥作为参数在函数间传递
   * - **副作用**
     - 有。每次调用随机函数都会改变全局状态
     - 无。调用函数不改变输入密钥，只返回结果
   * - **并行安全**
     - 困难。并行环境下的全局状态同步复杂
     - 原生支持。通过 Split 为每个分支提供独立 Key
   * - **确定性**
     - 依赖于执行顺序
     - 仅依赖于传递的 Key，与执行顺序无关

代码示例
~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp

   # 1. 创建初始密钥
   key = jax.random.PRNGKey(42)

   # 2. 如果直接多次使用同一个 key，会得到相同的结果
   val1 = jax.random.normal(key)
   val2 = jax.random.normal(key)
   assert jnp.all(val1 == val2)  # 相同！

   # 3. 正确做法：使用 split 分支密钥
   key, subkey = jax.random.split(key)
   random_val = jax.random.normal(subkey)

   # 4. 生成多个独立的随机数
   key, *subkeys = jax.random.split(key, 5)
   random_vals = [jax.random.normal(k) for k in subkeys]

**形象比喻**：

   传统的随机生成器像一个 **"自动取款机"**，你每取一次钱，银行后台的余额（全局状态）就会减少，你无法回到取钱前的状态。

   而 JAX 的 PRNG 机制像是一张 **"可复制的藏宝图"**，密钥就是地图上的坐标。如果你把地图复制一份（Split）给另一个人，你们按照相同的坐标和步骤走，最终找到的宝藏（随机数）必然是一模一样的。

   这使得在大规模分布式搜索（并行计算）中，每个人都能清晰地知道自己负责哪一块区域而不会发生混乱。

JAX 在宇宙学中的应用
--------------------

JAX 在宇宙学计算中的应用正在引发一场范式革命，推动该领域从传统的"数值积分驱动"向 **"全流程可微分推断"（Differentiable Universe）** 转变。

具体应用领域
~~~~~~~~~~~~

JAX 通过其自动微分和 GPU 加速能力，渗透到了宇宙学研究的各个核心环节：

1. **高维贝叶斯推断**：在处理下一代天文观测数据（如 Stage IV 巡天）时，模型参数往往超过 150 个。JAX 使整个似然函数变得完全可导，支持高效的高维空间探索

2. **宇宙学预测仿真器（Emulators）**：使用基于 JAX 的神经网络（如 CosmoPower-JAX）替代传统的玻尔兹曼求解器（如 CAMB/CLASS）。这些仿真器能以亚毫秒级的速度预测物质功率谱等观测效应

3. **可微分宇宙模拟**：JAX 被用于编写完全可微的 N 体模拟和引力透镜模拟（如 JAXtronomy）。这意味着研究人员可以进行"场级推断"（Field-level inference）

4. **Fisher 矩阵计算**：传统方法依赖不稳定的数值差分。JAX 可以通过 ``jax.hessian`` 自动计算精确的二阶导数

如何加速参数推断和 MCMC 采样？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX 的加速并非单一维度的提升，而是通过硬件优化和算法改进的结合实现的：

- **从"随机盲走"到"梯度导航"**：传统的 MCMC（如 Metropolis-Hastings）在参数空间中随机试探，效率随维度增加而急剧下降。JAX 允许使用 **哈密顿蒙特卡罗（HMC）** 或 NUTS 采样器，利用梯度引导采样器沿着高概率区域"滑行"

- **XLA 编译器与算子融合**：通过 ``jax.jit``，XLA 编译器将 Python 代码转化为针对 GPU/TPU 优化的机器码，消除 Python 的解释延迟

- **神经网络加速**：像 CosmoPower-JAX 这样的神经网络仿真器本质上是线性代数操作，这正是 GPU 所擅长的

**比喻理解**：

   传统宇宙学采样就像是在浓雾笼罩的山谷中蒙眼摸路（随机撒点），每走一步都要耗费大量体力去试探深度。

   而 JAX 就像是为研究者提供了一套 **GPS 导航系统和强力聚光灯**，梯度信息指明了通往山顶的最快路径，而 GPU 硬件加速则像是为研究者换上了高速滑板，让原本需要数年才能走完的路程在几天内即可完成。

宇宙学 JAX 项目生态系统
~~~~~~~~~~~~~~~~~~~~~~~

目前，宇宙学界已经形成了丰富的 JAX 软件生态系统：

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - 项目名称
     - 核心用途
     - 特点
   * - **jax-cosmo**
     - 核心宇宙学库
     - 提供可微分的背景演化、功率谱和似然函数计算
   * - **CosmoPower-JAX**
     - 神经网络仿真器
     - 高速预测 CMB 和物质功率谱，支持几百个参数的 HMC 采样
   * - **candl**
     - CMB 似然分析
     - 专门用于分析 CMB 功率谱测量（如 SPT、ACT）
   * - **JAXtronomy**
     - 引力透镜模拟
     - lenstronomy 的 JAX 移植版，支持 GPU 加速
   * - **microJAX**
     - 微引力透镜模拟
     - 首个完全可微的微透镜建模框架
   * - **DISCO-DJ**
     - 可微 N 体模拟
     - 实现完全自动微分的宇宙学模拟
   * - **PyBird-JAX**
     - 有效场论 (EFT)
     - 用于快速处理 LSS 数据的 EFT 预测

性能数据与基准测试
------------------

根据研究数据，JAX/XLA 展示了远超传统框架的加速能力：

宇宙学推断加速
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 25 25 10

   * - 任务
     - 传统方法
     - JAX 方法
     - 加速比
   * - 高维推断（157 参数）
     - 12 年（48 核 CPU，嵌套采样）
     - 8 天（24 GPU，梯度采样）
     - **10^5 倍**
   * - 宇宙剪切分析（37 参数）
     - 数周
     - 数小时
     - **~1000 倍**
   * - 智利央行经济模型
     - 12 小时（工业服务器）
     - 几秒（消费级 GPU）
     - **~1000 倍**

科学计算加速
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 20 20 10

   * - 操作
     - NumPy/SciPy
     - JAX JIT
     - 加速比
   * - 超新星似然函数计算
     - ~50 ms
     - ~0.05 ms
     - **1000 倍**
   * - 距离计算 (1000 点)
     - 150 ms
     - 20 ms
     - **7.5 倍**
   * - JAXtronomy 射线追踪 (vs CPU)
     - 基准
     - GPU 加速
     - **120-140 倍**
   * - 求解万维 PDE
     - 基准
     - JAX 优化
     - **1000 倍 + 30x 内存节省**

为什么 HIcosmo 选择 JAX？
-------------------------

1. MCMC 采样需要高效的梯度计算
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

现代 MCMC 采样器（如 NUTS）依赖于哈密顿蒙特卡洛方法，需要在每一步计算参数的梯度。

**传统数值微分的问题**：

- 对于 10 个参数：需要 20 次函数评估
- 对于 100 个参数：需要 200 次函数评估
- 精度受 epsilon 选择影响

**JAX 自动微分的优势**：

- 无论多少参数，只需要 ~2 次等效函数评估
- 结果是精确的数学导数
- NumPyro NUTS 采样器直接使用 JAX 的自动微分

2. 宇宙学计算需要高精度数值积分
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

宇宙学距离计算涉及积分：

.. math::

   d_C(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}

**HIcosmo 方法**：预计算高精度积分表，JIT 编译后极快

.. code-block:: python

   from hicosmo.models import LCDM
   import jax.numpy as jnp

   z_grid = jnp.linspace(0.01, 2.0, 1000)
   params = {'H0': 70.0, 'Omega_m': 0.3}

   # 第一次调用：编译 ~100ms
   grid = LCDM.compute_grid_traced(z_grid, params)

   # 后续调用：使用缓存 ~0.1ms
   grid = LCDM.compute_grid_traced(z_grid, params)

3. Fisher 矩阵需要二阶导数
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fisher 矩阵是似然函数的 Hessian 矩阵的期望值：

.. math::

   F_{ij} = -\left\langle \frac{\partial^2 \ln L}{\partial \theta_i \partial \theta_j} \right\rangle

**传统方法**：数值二阶导数需要 :math:`O(n^2)` 次函数评估且不稳定

**JAX 方法**：

.. code-block:: python

   import jax

   # 自动获取 Hessian 矩阵
   hessian = jax.hessian(log_likelihood)

   # Fisher 矩阵
   fisher_matrix = -hessian(best_fit_params)

JAX vs 其他框架
---------------

与 NumPy 对比
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 特性
     - NumPy
     - JAX
   * - **API**
     - 原生 Python
     - 与 NumPy 几乎相同
   * - **自动微分**
     - 不支持
     - 完整支持
   * - **JIT 编译**
     - 不支持
     - 支持
   * - **GPU 支持**
     - 需要 CuPy
     - 透明支持
   * - **学习曲线**
     - 低
     - 低（会 NumPy 即可）

与 TensorFlow/PyTorch 对比
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - 特性
     - TensorFlow
     - PyTorch
     - JAX
   * - **设计目标**
     - 深度学习
     - 深度学习
     - 通用科学计算
   * - **API 风格**
     - 计算图
     - 动态图
     - 函数变换
   * - **科学计算友好**
     - 一般
     - 一般
     - **优秀**
   * - **与 NumPy 兼容**
     - 部分
     - 部分
     - **完全**
   * - **函数式编程**
     - 有限
     - 有限
     - **核心设计**

**为什么 HIcosmo 不用 TensorFlow/PyTorch？**

1. **API 复杂度**：TensorFlow/PyTorch 的 API 针对神经网络设计，对于纯数值计算过于复杂
2. **NumPy 兼容性**：JAX 可以直接使用 NumPy 风格的代码，迁移成本低
3. **函数式编程**：宇宙学计算是纯函数，JAX 的函数变换理念完美匹配
4. **编译效率**：XLA 编译器对数值计算有专门优化

JAX 常见陷阱与解决方案
----------------------

1. 数组不可变
~~~~~~~~~~~~~

.. code-block:: python

   # ❌ 错误：JAX 数组不可变
   x = jnp.array([1, 2, 3])
   x[0] = 5  # TypeError!

   # ✅ 正确：使用 .at[].set()
   x = x.at[0].set(5)

2. 随机数需要显式 key
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ❌ 错误：没有全局随机状态
   x = jax.random.normal()  # Error!

   # ✅ 正确：显式传递 key
   key = jax.random.PRNGKey(42)
   x = jax.random.normal(key, shape=(10,))

   # 生成新的 key
   key, subkey = jax.random.split(key)
   y = jax.random.normal(subkey, shape=(10,))

3. JIT 中的动态形状
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ❌ 错误：JIT 函数中使用动态形状
   @jax.jit
   def bad_func(x):
       return jnp.zeros(len(x))  # len(x) 在编译时未知

   # ✅ 正确：使用 x.shape
   @jax.jit
   def good_func(x):
       return jnp.zeros(x.shape)

4. Python 控制流
~~~~~~~~~~~~~~~~

.. code-block:: python

   # ❌ 错误：JIT 中的 Python if
   @jax.jit
   def bad_func(x, flag):
       if flag:  # Python 条件
           return x + 1
       return x

   # ✅ 正确：使用 jnp.where
   @jax.jit
   def good_func(x, flag):
       return jnp.where(flag, x + 1, x)

HIcosmo 的 JAX 使用模式
-----------------------

基本使用
~~~~~~~~

.. code-block:: python

   # HIcosmo 对 JAX 进行了封装，用户通常不需要直接使用 JAX
   from hicosmo import hicosmo

   # 这行代码背后使用了 JAX 的：
   # - JIT 编译加速距离计算
   # - 自动微分提供 NUTS 采样器的梯度
   # - vmap 并行化多条 MCMC 链
   inf = hicosmo("LCDM", "sn", ["H0", "Omega_m"])
   samples = inf.run()

高级使用
~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from hicosmo.models import LCDM

   # 直接使用 JAX 的自动微分
   def my_likelihood(params):
       grid = LCDM.compute_grid_traced(z_obs, params)
       d_L = grid['d_L']
       mu_theory = 5 * jnp.log10(d_L) + 25
       chi2 = jnp.sum(((mu_obs - mu_theory) / mu_err)**2)
       return -0.5 * chi2

   # 自动获取梯度
   grad_likelihood = jax.grad(my_likelihood)

   # 自动获取 Hessian（用于 Fisher 矩阵）
   hessian_likelihood = jax.hessian(my_likelihood)

学习资源
--------

**官方文档**：

- `JAX 官方文档 <https://jax.readthedocs.io/>`_
- `JAX 快速入门 <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_
- `JAX 常见陷阱 <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_

**推荐教程**：

- `Thinking in JAX <https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html>`_
- `JAX 101 <https://jax.readthedocs.io/en/latest/jax-101/index.html>`_

**社区资源**：

- `Awesome JAX <https://github.com/n2cholas/awesome-jax>`_：JAX 资源合集
- `NumPyro <https://num.pyro.ai/>`_：基于 JAX 的概率编程库（HIcosmo 使用）

**宇宙学 JAX 项目**：

- `jax-cosmo <https://github.com/DifferentiableUniverseInitiative/jax_cosmo>`_：可微分宇宙学
- `CosmoPower-JAX <https://github.com/dpiras/cosmopower-jax>`_：神经网络仿真器
- `JAXtronomy <https://github.com/Autolens/JAXtronomy>`_：引力透镜模拟

总结
----

JAX 为 HIcosmo 提供了以下核心能力：

1. **自动微分**：NUTS 采样器无需手动梯度，支持任意复杂似然函数
2. **JIT 编译**：10-1000 倍性能提升，XLA 算子融合消除内存瓶颈
3. **向量化**：vmap 实现高效批量计算和多链并行
4. **GPU 支持**：代码无需修改，透明加速
5. **可组合性**：grad/jit/vmap 可以任意组合使用
6. **函数式编程**：纯函数设计确保可复现性和并行安全

通过这种函数式架构，JAX 能够将复杂的物理或数学模型转化为可优化的对象，从而在宇宙学等高维推断任务中实现相较于传统方法 **10^3 到 10^5 倍** 的性能飞跃。

对于用户来说，HIcosmo 已经封装了 JAX 的复杂性。你可以像使用普通 Python 库一样使用 HIcosmo，同时享受 JAX 带来的性能提升。只有在需要自定义似然函数或高级分析时，才需要直接使用 JAX API。

下一步
------

- `核心概念 <concepts.html>`_：了解 HIcosmo 的架构设计
- `采样器 <guides/samplers.html>`_：了解 MCMC 配置
- `Fisher 预测 <guides/fisher.html>`_：了解 Fisher 矩阵分析
