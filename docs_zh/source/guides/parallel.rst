并行与加速配置
==============

HIcosmo 基于 JAX 构建，原生支持多核 CPU 并行和 GPU 加速。
本章详细介绍如何配置并行环境，充分发挥硬件性能。

.. contents:: 本章目录
   :local:
   :depth: 2

快速开始
--------

**一行代码配置并行**：

.. code-block:: python

   import hicosmo as hc

   # 最常用：8 核并行
   hc.init(8)

   # 自动检测最优配置
   hc.init()

   # GPU 模式
   hc.init("GPU")

.. important::

   **必须在导入 JAX 之前调用** ``hc.init()``！

   JAX 的设备配置在首次导入时就会固定，之后无法更改。
   因此 ``hc.init()`` 应该是你脚本的第一行代码。

   .. code-block:: python

      # ✅ 正确顺序
      import hicosmo as hc
      hc.init(8)

      from hicosmo.models import LCDM  # 此时 JAX 被导入
      from hicosmo.samplers import MCMC

      # ❌ 错误顺序
      from hicosmo.models import LCDM  # JAX 已导入，设备数固定为 1
      import hicosmo as hc
      hc.init(8)  # ⚠️ 警告：配置被忽略

初始化 API
----------

``hc.init()`` 函数
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import hicosmo as hc

   hc.init(
       num_devices='auto',  # 设备数量
       device=None,         # 设备类型 ('GPU' 或 None)
       verbose=True,        # 是否显示配置信息
       force=True           # 是否强制覆盖现有配置
   )

**参数说明**：

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - 参数
     - 类型
     - 说明
   * - ``num_devices``
     - int / str / None
     - | 并行设备数量：
       | - ``int``：指定数量（如 8）
       | - ``'auto'``：自动检测（默认，最多 8）
       | - ``'GPU'``：GPU 模式
       | - ``None``：单设备向量化模式
   * - ``device``
     - str / None
     - | 显式设备类型：
       | - ``None``：CPU 模式（默认）
       | - ``'GPU'`` / ``'cuda'``：GPU 模式
   * - ``verbose``
     - bool
     - 是否打印配置信息
   * - ``force``
     - bool
     - 是否覆盖已存在的 XLA_FLAGS

多核 CPU 并行
-------------

基本配置
~~~~~~~~

.. code-block:: python

   import hicosmo as hc

   # 使用 8 个 CPU 核心
   hc.init(8)

   # 运行 MCMC 时会自动使用 8 条并行链
   from hicosmo.samplers import MCMC
   mcmc = MCMC(params, likelihood, chain_name='test')
   mcmc.run(num_samples=2000, num_chains=8)  # 8 条链并行运行

工作原理
~~~~~~~~

当你调用 ``hc.init(8)`` 时，HIcosmo 会：

1. **设置 XLA_FLAGS**：配置 JAX 使用 8 个逻辑 CPU 设备
2. **配置线程池**：将 CPU 核心分配给各设备
3. **设置 NumPyro**：告知 NumPyro 使用 8 个设备

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                     CPU (8 核)                          │
   ├─────────┬─────────┬─────────┬─────────┬─────────────────┤
   │ Device 0│ Device 1│ Device 2│ Device 3│ ... Device 7    │
   │ (Chain 1)│ (Chain 2)│ (Chain 3)│ (Chain 4)│ ... (Chain 8)│
   └─────────┴─────────┴─────────┴─────────┴─────────────────┘

推荐配置
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - 场景
     - 推荐配置
     - 说明
   * - 个人笔记本（4 核）
     - ``hc.init(4)``
     - 充分利用所有核心
   * - 工作站（8-16 核）
     - ``hc.init(8)``
     - 8 条链足够收敛诊断
   * - 服务器（32+ 核）
     - ``hc.init(8)``
     - 更多链不一定更好
   * - 调试/开发
     - ``hc.init(1)``
     - 单设备更容易调试

.. note::

   **为什么推荐 8 条链？**

   - 足够计算 R-hat 收敛诊断（需要 ≥4 条链）
   - 更多链不会显著提高采样效率
   - 过多链可能导致内存压力

GPU 加速
--------

基本配置
~~~~~~~~

.. code-block:: python

   import hicosmo as hc

   # 自动检测 GPU
   hc.init("GPU")

   # 或显式指定
   hc.init(device="GPU")

多 GPU 配置
~~~~~~~~~~~

.. code-block:: python

   import hicosmo as hc

   # 使用 4 个 GPU（如果有）
   hc.init(4, device="GPU")

   # 自动检测所有可用 GPU
   hc.init("GPU")  # 自动检测 GPU 数量

GPU 与 CPU 的选择
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 场景
     - 推荐
     - 原因
   * - 简单似然（SNe, BAO）
     - CPU
     - GPU 启动开销大于计算收益
   * - 复杂似然（CMB 功率谱）
     - GPU
     - 大规模矩阵运算受益于 GPU
   * - 大规模 Fisher 矩阵
     - GPU
     - 矩阵求逆适合 GPU
   * - 长时间 MCMC（>10万样本）
     - GPU
     - 累积收益明显

GPU 环境准备
~~~~~~~~~~~~

确保已安装 JAX GPU 版本：

.. code-block:: bash

   # CUDA 12
   pip install jax[cuda12]

   # CUDA 11
   pip install jax[cuda11_pip]

验证 GPU 可用：

.. code-block:: python

   import jax
   print(jax.devices())
   # [cuda(id=0), cuda(id=1), ...]  # GPU 可用
   # [CpuDevice(id=0)]              # 仅 CPU

集群配置
--------

SLURM 集群
~~~~~~~~~~

在 SLURM 集群上运行 HIcosmo：

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=hicosmo
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --time=24:00:00

   # 激活环境
   source activate hicosmo

   # 运行脚本
   python my_mcmc_script.py

对应的 Python 脚本：

.. code-block:: python

   import hicosmo as hc

   # 使用 SLURM 分配的 8 个核心
   hc.init(8)

   # ... 其余代码

PBS/Torque 集群
~~~~~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash
   #PBS -N hicosmo
   #PBS -l nodes=1:ppn=8
   #PBS -l mem=32gb
   #PBS -l walltime=24:00:00

   cd $PBS_O_WORKDIR
   source activate hicosmo
   python my_mcmc_script.py

GPU 集群
~~~~~~~~

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=hicosmo_gpu
   #SBATCH --nodes=1
   #SBATCH --gres=gpu:4
   #SBATCH --mem=64G
   #SBATCH --time=24:00:00

   source activate hicosmo
   python my_mcmc_script.py

.. code-block:: python

   import hicosmo as hc

   # 使用 4 个 GPU
   hc.init(4, device="GPU")

环境变量配置
------------

HIcosmo 自动管理以下环境变量，通常不需要手动设置：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 环境变量
     - 说明
   * - ``XLA_FLAGS``
     - JAX/XLA 编译器配置，包括设备数
   * - ``JAX_ENABLE_X64``
     - 启用 64 位精度（默认 True）
   * - ``JAX_NUM_THREADS``
     - 每设备线程数

手动配置（高级）
~~~~~~~~~~~~~~~~

如果需要更精细的控制：

.. code-block:: bash

   # 在 shell 中设置
   export XLA_FLAGS="--xla_force_host_platform_device_count=8"
   export JAX_ENABLE_X64=True
   python my_script.py

.. code-block:: python

   import os
   os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

   # 必须在设置环境变量后、导入 JAX 前
   import hicosmo as hc
   hc.init(8, force=False)  # 不覆盖已设置的 XLA_FLAGS

查看当前配置
------------

.. code-block:: python

   import hicosmo as hc

   hc.init(8)

   # 查看配置状态
   status = hc.Config.status()
   print(status)
   # {
   #     'initialized': True,
   #     'config': {'num_devices': 8, 'device_type': 'cpu', ...},
   #     'system_cores': 16,
   #     'jax_devices': 8,
   #     'jax_device_list': ['CpuDevice(id=0)', 'CpuDevice(id=1)', ...]
   # }

   # 查看 JAX 设备
   import jax
   print(f"可用设备: {len(jax.devices())}")
   for d in jax.devices():
       print(f"  {d}")

性能调优建议
------------

1. **匹配链数与设备数**

   .. code-block:: python

      hc.init(8)
      # num_chains 应该等于或小于设备数
      mcmc.run(num_samples=2000, num_chains=8)  # ✅ 最优
      mcmc.run(num_samples=2000, num_chains=4)  # ✅ 可以，但浪费 4 个设备
      mcmc.run(num_samples=2000, num_chains=16) # ⚠️ 超过设备数，会排队

2. **预热 JIT 编译**

   .. code-block:: python

      # 第一次调用会触发 JIT 编译（较慢）
      result = likelihood(H0=70, Omega_m=0.3)

      # 后续调用使用缓存（很快）
      result = likelihood(H0=71, Omega_m=0.3)

3. **内存管理**

   .. code-block:: python

      # 大规模采样时监控内存
      import jax
      jax.clear_caches()  # 清理 JIT 缓存释放内存

4. **避免频繁初始化**

   .. code-block:: python

      # ❌ 不要在循环中初始化
      for i in range(10):
          hc.init(8)  # 每次都会打印警告
          ...

      # ✅ 只初始化一次
      hc.init(8)
      for i in range(10):
          ...

常见问题
--------

Q: 为什么配置被忽略了？
~~~~~~~~~~~~~~~~~~~~~~~

最常见原因是 JAX 已被导入：

.. code-block:: python

   from hicosmo.models import LCDM  # ❌ JAX 在这里被导入
   import hicosmo as hc
   hc.init(8)  # ⚠️ 警告：配置被忽略

解决方法：确保 ``hc.init()`` 在所有 JAX 相关导入之前。

Q: GPU 未被检测到？
~~~~~~~~~~~~~~~~~~~

1. 检查 JAX GPU 版本是否安装：

   .. code-block:: python

      import jax
      print(jax.devices())  # 应该显示 cuda 设备

2. 检查 CUDA 环境：

   .. code-block:: bash

      nvidia-smi  # 应该显示 GPU 信息

3. 重新安装 JAX GPU 版本：

   .. code-block:: bash

      pip uninstall jax jaxlib
      pip install jax[cuda12]

Q: 如何在 Jupyter Notebook 中使用？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 notebook 的第一个单元格：

.. code-block:: python

   # Cell 1 - 必须是第一个单元格！
   import hicosmo as hc
   hc.init(8)

   # Cell 2 - 然后导入其他模块
   from hicosmo.models import LCDM
   from hicosmo.samplers import MCMC

.. warning::

   如果重启 kernel，需要重新运行 ``hc.init()``。

Q: 内存不足怎么办？
~~~~~~~~~~~~~~~~~~~

1. 减少并行链数：

   .. code-block:: python

      hc.init(4)  # 而不是 8

2. 减少样本数：

   .. code-block:: python

      mcmc.run(num_samples=1000)  # 而不是 5000

3. 使用 64 位精度时尤其注意：

   .. code-block:: python

      # 32 位精度可以减少一半内存
      os.environ['JAX_ENABLE_X64'] = 'False'

小结
----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 场景
     - 推荐配置
   * - 快速测试
     - ``hc.init(1)`` 或不调用
   * - 日常使用
     - ``hc.init()`` 自动配置
   * - 生产运行
     - ``hc.init(8)`` 8 核并行
   * - GPU 加速
     - ``hc.init("GPU")``
   * - 集群提交
     - 在脚本开头 ``hc.init(N)``

记住：**``hc.init()`` 必须在导入 JAX 之前调用！**
