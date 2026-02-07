配置运行系统 API
================

``hicosmo.runner`` 模块提供 Cobaya 风格的 YAML 配置系统，整合了：

- **配置加载/验证**: 从 YAML/JSON 文件加载配置
- **组件注册表**: Theory、Sampler、Likelihood 的字符串→类映射
- **数据集管理**: 数据路径解析和自动下载

.. automodule:: hicosmo.runner
   :members:
   :undoc-members:
   :show-inheritance:

快速开始
--------

.. code-block:: python

   from hicosmo import run_from_config

   # 从 YAML 配置运行推断
   result = run_from_config("analysis.yaml")
   samples = result["samples"]

组件注册表
----------

.. code-block:: python

   from hicosmo.runner import THEORY_REGISTRY, LIKELIHOOD_REGISTRY

   # 列出可用的宇宙学模型
   print(THEORY_REGISTRY.list())  # ['lcdm', 'wcdm', 'cpl', 'ilcdm']

   # 列出可用的似然函数
   print(LIKELIHOOD_REGISTRY.list())

数据集管理
----------

.. code-block:: python

   from hicosmo.runner import ensure_dataset, resolve_data_root

   # 确保数据集存在
   ensure_dataset("pantheon_plus")
   ensure_dataset("desi2024")

   # 获取数据根目录
   data_root = resolve_data_root()
