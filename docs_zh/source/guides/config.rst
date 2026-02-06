配置驱动
========

HIcosmo 支持类似 Cobaya 的 YAML 配置。

并行初始化
----------

在运行任何配置前，先初始化并行环境：

.. code-block:: python

   import hicosmo as hc
   hc.init(8)  # 8 个并行设备

.. note::

   ``hc.init(N)`` 创建 N 个 JAX 逻辑设备，实现 N 条链真正并行。

   - ``hc.init(8)`` - 8 个并行设备（推荐）
   - ``hc.init()`` - 自动检测（最多 8 个）
   - ``hc.init("GPU")`` - GPU 模式

YAML 配置示例
-------------

.. code-block:: yaml

   name: joint_sn_bao
   theory: LCDM
   likelihood:
     - name: sn
       dataset: pantheon+
     - name: bao
       dataset: sdss_dr12

   params:
     H0: {prior: {dist: uniform, min: 50, max: 100}, ref: 70.0}
     Omega_m: {prior: {dist: uniform, min: 0.1, max: 0.5}, ref: 0.3}

   sampler:
     name: numpyro
     num_samples: 1000
     num_chains: 8         # 默认 = 设备数
     num_warmup: 300

   output:
     root: results
     chain_name: joint_sn_bao

运行配置
--------

.. code-block:: python

   # 1. 先初始化
   import hicosmo as hc
   hc.init(8)

   # 2. 运行配置
   from hicosmo import run_from_config
   run_from_config("examples/tutorials/configs/joint_sn_bao.yaml")

