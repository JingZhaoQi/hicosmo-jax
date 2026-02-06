数据集管理
==========

HIcosmo 提供自动发现数据集的功能，用户可以轻松查看所有可用的观测数据。
无需硬编码 —— 当您添加新数据集到数据目录时，系统会自动识别。

查看可用数据集
--------------

快速查看
~~~~~~~~

最简单的方式是调用 ``show_available_datasets()``：

.. code-block:: python

   from hicosmo.likelihoods import show_available_datasets

   # 显示所有可用数据集
   show_available_datasets()

输出示例::

   ======================================================================
                       HIcosmo Available Datasets
   ======================================================================

     BAO (Baryon Acoustic Oscillations)
     --------------------------------------------------
       • desi_2024        : DESI 2024 DR1 BAO measurements (z=0.3-2.3)
       • boss_dr12        : BOSS DR12 Luminous Red Galaxy BAO
       • sdss_dr12        : SDSS DR12 consensus BAO
       • sdss_dr16        : SDSS DR16 LRG and QSO BAO
       • sixdf            : 6dF Galaxy Survey BAO (z=0.1)

     SN (Type Ia Supernovae)
     --------------------------------------------------
       • pantheon+shoes   : Pantheon+ with SH0ES Cepheid calibration

     CMB (Cosmic Microwave Background)
     --------------------------------------------------
       • planck2018_distance : Planck 2018 distance priors (built-in)

     Lensing (Strong Gravitational Lensing)
     --------------------------------------------------
       • h0licow          : H0LiCOW strong lensing time delays (6 lenses)
       • tdcosmo          : TDCOSMO hierarchical strong lensing (7 lenses)
       • tdcosmo2025      : TDCOSMO 2025 updated analysis

   ======================================================================

程序化获取
~~~~~~~~~~

如果需要在代码中使用数据集列表：

.. code-block:: python

   from hicosmo.likelihoods import list_all_datasets

   # 获取所有数据集（字典形式）
   datasets = list_all_datasets()

   # 查看 BAO 数据集
   print(datasets['bao'])
   # ['desi_2024', 'boss_dr12', 'sdss_dr12', 'sdss_dr16', 'sixdf']

   # 查看 SN 数据集
   print(datasets['sn'])
   # ['pantheon+shoes']

   # 动态选择数据集
   for bao_name in datasets['bao']:
       print(f"可用 BAO 数据集: {bao_name}")

按类别查看
~~~~~~~~~~

只查看特定类别的数据集：

.. code-block:: python

   # 只显示 BAO 数据集
   show_available_datasets('bao')

   # 只显示强透镜数据集
   show_available_datasets('lensing')

类方法查询
~~~~~~~~~~

每个 Likelihood 类也提供 ``available_datasets()`` 方法：

.. code-block:: python

   from hicosmo.likelihoods.bao import DESI2024BAO
   from hicosmo.likelihoods.sn import PantheonPlusLikelihood

   # BAO 类的可用数据集
   print(DESI2024BAO.available_datasets())
   # ['desi_2024', 'boss_dr12', 'sdss_dr12', 'sdss_dr16', 'sixdf']

   # SN 类的可用数据集
   print(PantheonPlusLikelihood.available_datasets())
   # ['pantheon+shoes']


数据集分类
----------

HIcosmo 支持以下类别的观测数据：

.. list-table:: 数据集分类
   :header-rows: 1
   :widths: 15 35 50

   * - 类别
     - 全称
     - 包含数据集
   * - ``bao``
     - Baryon Acoustic Oscillations
     - DESI 2024, BOSS DR12, SDSS DR12/16, 6dF
   * - ``sn``
     - Type Ia Supernovae
     - Pantheon+, Pantheon+SH0ES
   * - ``cmb``
     - Cosmic Microwave Background
     - Planck 2018 距离先验
   * - ``h0``
     - Direct H0 Measurements
     - SH0ES
   * - ``lensing``
     - Strong Gravitational Lensing
     - H0LiCOW, TDCOSMO, TDCOSMO 2025


使用发现的数据集
----------------

将发现的数据集用于 MCMC 分析：

.. code-block:: python

   from hicosmo.likelihoods import (
       BAO_likelihood, SN_likelihood,
       list_all_datasets
   )
   from hicosmo.models import LCDM

   # 1. 查看可用数据集
   datasets = list_all_datasets()
   print("BAO 数据集:", datasets['bao'])

   # 2. 选择数据集创建 likelihood
   bao = BAO_likelihood(LCDM, 'desi_2024')
   sn = SN_likelihood(LCDM, 'pantheon+shoes')

   # 3. 联合分析
   joint = bao + sn

   # 4. 运行 MCMC
   from hicosmo.samplers import MCMC
   params = {
       'H0': (70.0, 60.0, 80.0),
       'Omega_m': (0.3, 0.1, 0.5),
   }
   mcmc = MCMC(params, joint, chain_name='joint_analysis')
   mcmc.run(num_samples=2000)


默认数据目录
------------

数据文件存储位置：

- **默认路径**：``hicosmo/data/``
- **环境变量**：可通过 ``HICOSMO_DATA`` 指定自定义路径

目录结构
~~~~~~~~

.. code-block:: text

   hicosmo/data/
   ├── bao_data/
   │   ├── desi_2024/
   │   │   ├── desi_2024_gaussian_bao_ALL_GCcomb_mean.txt
   │   │   └── desi_2024_gaussian_bao_ALL_GCcomb_cov.txt
   │   ├── boss_dr12/
   │   ├── sdss_dr12/
   │   ├── sdss_dr16/
   │   └── sixdf/
   ├── sne/
   │   ├── Pantheon+SH0ES.dat
   │   └── Pantheon+SH0ES_STAT+SYS.cov
   ├── h0licow/
   └── tdcosmo/


添加新数据集
------------

添加新数据集非常简单 —— 只需将数据文件放入正确的目录：

1. **BAO 数据**：在 ``bao_data/`` 下创建新目录
2. **SN 数据**：放入 ``sne/`` 目录
3. **其他**：放入对应类别目录

示例：添加新的 BAO 数据集

.. code-block:: bash

   # 创建新数据集目录
   mkdir hicosmo/data/bao_data/my_new_survey

   # 复制数据文件
   cp my_data.txt hicosmo/data/bao_data/my_new_survey/

然后刷新数据注册表：

.. code-block:: python

   from hicosmo.likelihoods import list_all_datasets

   # 强制刷新，发现新数据集
   datasets = list_all_datasets(refresh=True)
   print(datasets['bao'])
   # [..., 'my_new_survey']  # 新数据集已自动发现！


DataRegistry 高级用法
---------------------

对于更复杂的需求，可以直接使用 ``DataRegistry`` 类：

.. code-block:: python

   from hicosmo.data_registry import DataRegistry

   # 创建注册表实例
   registry = DataRegistry()

   # 获取详细信息
   info = registry.get_info('bao', 'desi_2024')
   print(f"名称: {info.name}")
   print(f"描述: {info.description}")
   print(f"文件数: {len(info.files)}")
   print(f"路径: {info.path}")

   # 导出为字典（可序列化为 JSON）
   data = registry.to_dict()


完整示例
--------

.. code-block:: python

   """数据集发现与使用完整示例"""
   from hicosmo.likelihoods import (
       show_available_datasets,
       list_all_datasets,
       BAO_likelihood,
       SN_likelihood,
   )
   from hicosmo.models import LCDM

   # 1. 查看所有可用数据集
   print("=" * 50)
   print("Step 1: 查看可用数据集")
   print("=" * 50)
   show_available_datasets()

   # 2. 程序化获取
   print("\nStep 2: 获取数据集列表")
   datasets = list_all_datasets()
   for category, names in datasets.items():
       if names:
           print(f"  {category}: {names}")

   # 3. 创建 likelihood
   print("\nStep 3: 创建 Likelihood")
   bao = BAO_likelihood(LCDM, datasets['bao'][0])  # 使用第一个 BAO 数据集
   print(f"  已创建: {bao}")

   # 4. 测试 likelihood
   print("\nStep 4: 测试 Likelihood")
   log_L = bao(H0=70.0, Omega_m=0.3)
   print(f"  log(L) = {log_L:.2f}")
