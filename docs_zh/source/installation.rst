安装指南
========

系统需求
--------

**操作系统**：

- Linux（推荐，最佳性能）
- macOS（完全支持）
- Windows（通过 WSL2 支持）

**Python 版本**：

- Python 3.9 - 3.11（推荐 3.10）
- Python 3.12+ 尚未完全测试

**硬件建议**：

- CPU：4核以上（MCMC 并行采样）
- 内存：8GB+（大规模 MCMC 推荐 16GB+）
- GPU（可选）：NVIDIA CUDA 11.8+ 用于 JAX GPU 加速

基础安装
--------

**从源码安装（推荐）**：

.. code-block:: bash

   # 克隆仓库
   git clone https://github.com/yourname/hicosmo.git
   cd hicosmo

   # 创建虚拟环境（推荐）
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # 或 venv\Scripts\activate  # Windows

   # 安装
   pip install -e .

**使用 conda**：

.. code-block:: bash

   conda create -n hicosmo python=3.10
   conda activate hicosmo
   pip install -e .

开发模式安装
------------

如果你需要开发或调试 HIcosmo：

.. code-block:: bash

   pip install -e ".[dev]"

这将安装额外的开发依赖：

- ``pytest``：测试框架
- ``pytest-cov``：覆盖率报告
- ``black``：代码格式化
- ``isort``：导入排序
- ``mypy``：类型检查
- ``flake8``：代码风格检查

GPU 加速安装
------------

HIcosmo 基于 JAX，支持 GPU 加速。对于 NVIDIA GPU：

.. code-block:: bash

   # 先安装 CPU 版本
   pip install -e .

   # 再安装 JAX GPU 版本（CUDA 11.8+）
   pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # 或使用 CUDA 12
   pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

**验证 GPU 安装**：

.. code-block:: python

   import jax
   print(jax.devices())
   # 应显示类似: [GpuDevice(id=0, process_index=0)]

核心依赖说明
------------

HIcosmo 的核心依赖包括：

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - 包名
     - 版本
     - 用途
   * - jax
     - >=0.4.20
     - 高性能数值计算和自动微分
   * - jaxlib
     - >=0.4.20
     - JAX 编译后端
   * - numpyro
     - >=0.13.0
     - 概率编程和 NUTS 采样器
   * - numpy
     - >=1.24.0
     - 数组操作
   * - scipy
     - >=1.10.0
     - 科学计算
   * - astropy
     - >=5.0
     - 天文学常数和单位
   * - getdist
     - >=1.4.0
     - MCMC 可视化（corner plots）
   * - matplotlib
     - >=3.7.0
     - 绑图
   * - pyyaml
     - >=6.0
     - 配置文件解析
   * - rich
     - >=13.0.0
     - 终端美化输出

可选依赖
--------

**emcee 采样器**（备用采样器）：

.. code-block:: bash

   pip install emcee

**文档构建**：

.. code-block:: bash

   pip install sphinx sphinx_rtd_theme

**Jupyter 支持**：

.. code-block:: bash

   pip install jupyter ipykernel

验证安装
--------

安装完成后，运行以下命令验证：

.. code-block:: bash

   # 运行测试套件
   pytest tests/ -v --tb=short

**Python 验证**：

.. code-block:: python

   import hicosmo
   print(hicosmo.__version__)

   # 验证 JAX 与精度配置
   import jax
   import jax.numpy as jnp
   x = jnp.array([1.0, 2.0, 3.0])
   print(f"JAX working: {x.sum()}")
   print(f"float64 enabled: {jax.config.jax_enable_x64}")  # 应为 True

.. note::

   宇宙学距离积分和协方差矩阵运算需要 float64 精度，因此
   ``import hicosmo`` 会自动启用 JAX 的 x64 模式。请在任何 JAX
   计算之前先导入 hicosmo。如确需 float32（例如受限的 GPU 显存），
   可在导入前设置环境变量 ``HICOSMO_DISABLE_X64=1``。

   # 验证 NumPyro
   import numpyro
   print(f"NumPyro version: {numpyro.__version__}")

   # 快速宇宙学计算测试
   from hicosmo.models import LCDM
   model = LCDM(H0=70, Omega_m=0.3)
   print(f"d_L(z=1) = {model.luminosity_distance(1.0):.2f} Mpc")

常见问题
--------

**JAX 版本冲突**：

如果遇到 JAX 相关错误，请确保 ``jax`` 和 ``jaxlib`` 版本匹配：

.. code-block:: bash

   pip install --upgrade jax jaxlib

**NumPyro 导入错误**：

确保安装了正确版本的 NumPyro：

.. code-block:: bash

   pip install --upgrade numpyro

**GetDist 绘图问题**：

如果 corner plot 失败，检查 matplotlib 后端：

.. code-block:: python

   import matplotlib
   matplotlib.use('Agg')  # 非交互式后端

**内存不足**：

对于大规模 MCMC，设置 JAX 内存预分配：

.. code-block:: bash

   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

CI/CD 环境配置
--------------

在 CI 环境中，建议设置：

.. code-block:: bash

   # 强制使用 CPU（避免 GPU 相关问题）
   export JAX_PLATFORM_NAME=cpu

   # 禁用 JAX 预分配
   export XLA_PYTHON_CLIENT_PREALLOCATE=false

   # 设置随机种子确保可复现
   export PYTHONHASHSEED=42

Read the Docs 构建
------------------

本项目的中文文档默认从 ``docs_zh/`` 目录构建。

配置文件位于 ``docs_zh/source/conf.py``。

构建文档：

.. code-block:: bash

   cd docs_zh
   make html
   # 输出在 docs_zh/build/html/
