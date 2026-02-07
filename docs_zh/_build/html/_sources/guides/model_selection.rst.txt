模型选择与信息判据
==================

HIcosmo 提供完整的模型选择工具，包括 AIC、BIC 和贝叶斯证据计算。这些工具帮助比较不同宇宙学模型的拟合优度。

信息判据概览
------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30

   * - 判据
     - 公式
     - 适用场景
     - 解释
   * - **AIC**
     - :math:`\chi^2_{\min} + 2k`
     - 小样本、探索性分析
     - 较小值表示更好的模型
   * - **BIC**
     - :math:`\chi^2_{\min} + k \ln N`
     - 大样本、模型选择
     - 对参数数量惩罚更强
   * - **贝叶斯证据**
     - :math:`\int L(\theta) \pi(\theta) d\theta`
     - 贝叶斯模型比较
     - 考虑先验体积

其中：

- :math:`k` = 自由参数数量
- :math:`N` = 数据点数量
- :math:`\chi^2_{\min}` = 最小卡方值

快速开始
--------

**3 行计算信息判据**：

.. code-block:: python

   from hicosmo.visualization import information_criteria

   result = information_criteria('my_chain')
   print(f"AIC = {result['aic']:.2f}, BIC = {result['bic']:.2f}")

公式详解
--------

卡方统计量
~~~~~~~~~~

对于高斯似然函数：

.. math::

   \log L = -\frac{\chi^2}{2} + \text{normalization constant}

其中归一化常数包括：

- :math:`-\frac{1}{2} \log |C|`：协方差矩阵行列式
- 其他与参数无关的常数

从 log-likelihood 提取 :math:`\chi^2`：

.. math::

   \chi^2 = -2 \times (\log L - \text{normalization constant})

.. note::

   HIcosmo 自动处理归一化常数，确保 AIC/BIC 值正确。用户无需手动干预。

AIC (Akaike Information Criterion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

赤池信息准则：

.. math::

   \text{AIC} = \chi^2_{\min} + 2k

**特点**：

- 惩罚项与样本量无关
- 适合小样本情况
- 趋向于选择更复杂的模型

**解释 ΔAIC**：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ΔAIC
     - 解释
   * - < 2
     - 无显著差异
   * - 2 - 4
     - 弱证据支持较小 AIC 模型
   * - 4 - 7
     - 中等证据
   * - > 10
     - 强证据

BIC (Bayesian Information Criterion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

贝叶斯信息准则：

.. math::

   \text{BIC} = \chi^2_{\min} + k \ln N

**特点**：

- 惩罚项随样本量增加
- 对过拟合惩罚更强
- 一致性估计（样本足够大时选择真实模型）

**Jeffreys 尺度解释 ΔBIC**：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ΔBIC
     - 证据强度
   * - < 2
     - 不足以区分
   * - 2 - 6
     - 正面证据 (Positive)
   * - 6 - 10
     - 强证据 (Strong)
   * - > 10
     - 决定性证据 (Decisive)

贝叶斯证据
~~~~~~~~~~

贝叶斯证据（边际似然）：

.. math::

   Z = \int L(\theta) \pi(\theta) d\theta

其中 :math:`\pi(\theta)` 是先验分布。

**贝叶斯因子**：

.. math::

   B_{12} = \frac{Z_1}{Z_2} = \exp(\log Z_1 - \log Z_2)

**Jeffreys 尺度**：

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Δlog Z
     - 贝叶斯因子 B
     - 证据强度
   * - < 1.0
     - < 2.7
     - 不足以区分
   * - 1.0 - 2.5
     - 2.7 - 12
     - 弱证据
   * - 2.5 - 5.0
     - 12 - 150
     - 中等证据
   * - > 5.0
     - > 150
     - 强证据

.. warning::

   HIcosmo 使用 **Harmonic Mean Estimator** 计算贝叶斯证据。此方法可能不稳定，
   对于精确的贝叶斯证据计算，建议使用 nested sampling（如 dynesty）。

归一化常数的处理
----------------

为什么需要归一化常数？
~~~~~~~~~~~~~~~~~~~~~~

考虑高斯似然函数的完整形式：

.. math::

   \log L = -\frac{1}{2} (d - t)^T C^{-1} (d - t) - \frac{1}{2} \log |C| - \frac{N}{2} \log(2\pi)

其中：

- 第一项 :math:`-\frac{\chi^2}{2}` 依赖于模型参数
- 后两项是归一化常数，不依赖于模型参数

**问题**：如果直接使用 :math:`\log L` 计算 AIC/BIC：

.. code-block:: python

   # 错误！
   aic = 2*k - 2*log_likelihood  # 会得到负值

**解决方案**：提取 :math:`\chi^2` 后再计算：

.. code-block:: python

   # 正确
   chi2 = -2 * (log_likelihood - normalization_constant)
   aic = chi2 + 2*k  # 正值

HIcosmo 的实现
~~~~~~~~~~~~~~

HIcosmo 自动处理归一化常数，用户无需手动干预：

1. **似然函数提供归一化常数**：

.. code-block:: python

   class PantheonPlusLikelihood(Likelihood):
       def get_normalization_constant(self) -> float:
           """返回归一化常数"""
           return -0.5 * self.log_det_cov

2. **MCMC 保存时自动存储**：

.. code-block:: python

   # 保存链时自动包含归一化常数
   mcmc.save_results('my_chain')
   # metadata 中包含 normalization_constant

3. **信息判据自动读取**：

.. code-block:: python

   result = information_criteria('my_chain')
   # 自动从 metadata 读取归一化常数

对模型对比的影响
~~~~~~~~~~~~~~~~

**重要结论**：归一化常数对 **模型对比结果无影响**！

.. math::

   \Delta\text{AIC} = \text{AIC}_1 - \text{AIC}_2 = (\chi^2_1 + 2k_1) - (\chi^2_2 + 2k_2)

归一化常数相同（使用相同数据），在差值中被抵消。

.. code-block:: python

   # 验证
   delta_aic_without_norm = aic1 - aic2
   delta_aic_with_norm = aic1_corrected - aic2_corrected
   # 两者相等！

但是，**绝对值** AIC/BIC 需要正确的归一化才能有意义：

- 约化 :math:`\chi^2 = \chi^2_{\min}/N \approx 1` 表示良好拟合
- AIC、BIC 必须是正数

API 接口
--------

information_criteria 函数
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import information_criteria

   result = information_criteria(
       samples='my_chain',           # 链名称、路径或样本字典
       num_params=None,              # 自由参数数（自动推断）
       num_data=None,                # 数据点数（自动推断）
       log_likelihood_fn=None,       # 似然函数（可选）
       evidence_method='harmonic_mean',  # 证据估计方法
   )

**返回值**：

.. code-block:: python

   {
       'num_params': 4,                    # 自由参数数 k
       'num_data': 1580,                   # 数据点数 N
       'n_samples': 10000,                 # MCMC 样本数
       'log_likelihood_max': 2300.68,      # 最大 log-likelihood
       'normalization_constant': 2994.34,  # 归一化常数
       'chi2_min': 1387.32,                # 最小卡方值
       'aic': 1395.32,                     # AIC
       'bic': 1417.88,                     # BIC
       'log_evidence': 2298.5,             # log(贝叶斯证据)
       'log_evidence_method': 'harmonic_mean',
   }

aic 和 bic 函数
~~~~~~~~~~~~~~~

独立计算 AIC 或 BIC：

.. code-block:: python

   from hicosmo.visualization.information_criteria import aic, bic

   # 计算 AIC
   aic_value = aic(
       log_likelihood_max=2300.68,
       num_params=4,
       normalization_constant=2994.34
   )

   # 计算 BIC
   bic_value = bic(
       log_likelihood_max=2300.68,
       num_params=4,
       num_data=1580,
       normalization_constant=2994.34
   )

完整示例
--------

单模型信息判据
~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo import MCMC
   from hicosmo.models import LCDM
   from hicosmo.likelihoods import SN_likelihood
   from hicosmo.visualization import information_criteria

   # 运行 MCMC
   likelihood = SN_likelihood(LCDM, 'pantheon+')
   params = {
       'H0': (67.0, 60.0, 80.0),
       'Omega_m': (0.3, 0.1, 0.5),
   }

   mcmc = MCMC(params, likelihood, chain_name='lcdm_pantheon')
   mcmc.run(num_samples=5000, num_chains=4)

   # 计算信息判据
   result = information_criteria('lcdm_pantheon')

   print(f"自由参数数 k = {result['num_params']}")
   print(f"数据点数 N = {result['num_data']}")
   print(f"χ²_min = {result['chi2_min']:.2f}")
   print(f"约化 χ² = {result['chi2_min']/result['num_data']:.4f}")
   print(f"AIC = {result['aic']:.2f}")
   print(f"BIC = {result['bic']:.2f}")

多模型比较
~~~~~~~~~~

.. code-block:: python

   from hicosmo import MCMC
   from hicosmo.models import LCDM, wCDM, CPL
   from hicosmo.likelihoods import SN_likelihood
   from hicosmo.visualization import information_criteria
   import numpy as np

   # 定义三个模型
   models = {
       'LCDM': (LCDM, ['H0', 'Omega_m']),
       'wCDM': (wCDM, ['H0', 'Omega_m', 'w']),
       'CPL': (CPL, ['H0', 'Omega_m', 'w0', 'wa']),
   }

   results = {}

   for name, (model_class, free_params) in models.items():
       # 构建参数配置
       params = {p: (67.0 if p == 'H0' else 0.3, 0.1, 0.9) for p in free_params}
       if 'w' in free_params:
           params['w'] = (-1.0, -2.0, 0.0)
       if 'w0' in free_params:
           params['w0'] = (-1.0, -2.0, 0.0)
           params['wa'] = (0.0, -1.0, 1.0)

       # 运行 MCMC
       likelihood = SN_likelihood(model_class, 'pantheon+')
       mcmc = MCMC(params, likelihood, chain_name=f'{name.lower()}_chain')
       mcmc.run(num_samples=5000, num_chains=4)

       # 计算信息判据
       results[name] = information_criteria(f'{name.lower()}_chain')

   # 打印比较表格
   print("\n模型选择结果")
   print("=" * 70)
   print(f"{'模型':<8} {'k':<4} {'χ²_min':<12} {'AIC':<12} {'BIC':<12} {'ΔBIC':<10}")
   print("-" * 70)

   bic_ref = results['LCDM']['bic']
   for name, ic in results.items():
       delta_bic = ic['bic'] - bic_ref
       print(f"{name:<8} {ic['num_params']:<4} {ic['chi2_min']:<12.2f} "
             f"{ic['aic']:<12.2f} {ic['bic']:<12.2f} {delta_bic:<+10.2f}")

   print("-" * 70)

   # 解释结果
   print("\n模型选择解释 (Jeffreys 尺度):")
   for name, ic in results.items():
       if name == 'LCDM':
           continue
       delta = abs(ic['bic'] - bic_ref)
       if delta < 2:
           strength = "不足以区分"
       elif delta < 6:
           strength = "正面证据"
       elif delta < 10:
           strength = "强证据"
       else:
           strength = "决定性证据"
       better = "ΛCDM" if ic['bic'] > bic_ref else name
       print(f"  {name} vs ΛCDM: |ΔBIC| = {delta:.1f} → {strength}支持 {better}")

贝叶斯证据比较
~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import information_criteria

   # 假设已运行多个模型的 MCMC
   chains = ['lcdm_chain', 'wcdm_chain', 'cpl_chain']
   names = ['ΛCDM', 'wCDM', 'CPL']

   print("\n贝叶斯证据比较")
   print("=" * 60)
   print(f"{'模型':<8} {'log(Z)':<15} {'Δlog(Z)':<15} {'B':<10}")
   print("-" * 60)

   results = [information_criteria(c) for c in chains]
   log_z_ref = results[0]['log_evidence']

   for name, ic in zip(names, results):
       delta = ic['log_evidence'] - log_z_ref
       bayes_factor = np.exp(delta) if abs(delta) < 100 else float('inf')
       print(f"{name:<8} {ic['log_evidence']:<15.4f} {delta:<+15.4f} {bayes_factor:<10.2f}")

   print("-" * 60)
   print("\nJeffreys 尺度: |Δlog(Z)| > 5 表示强证据")

常见问题
--------

Q: AIC/BIC 值为负数是否正确？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**不正确**。AIC = χ² + 2k 和 BIC = χ² + k·ln(N) 必须是正数，因为 χ² ≥ 0。

如果您看到负值，可能是使用了未修正的 log-likelihood。HIcosmo 已自动处理此问题。

Q: 应该使用 AIC 还是 BIC？
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **AIC**：适合小样本、探索性分析，对复杂模型惩罚较轻
- **BIC**：适合大样本、模型选择，对过拟合惩罚更强

宇宙学中通常推荐 **BIC**，因为数据集通常很大（N > 1000）。

Q: 贝叶斯证据估计不稳定怎么办？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Harmonic Mean Estimator 确实可能不稳定。建议：

1. 增加 MCMC 样本数
2. 使用多条链取平均
3. 对于精确计算，使用 nested sampling（如 dynesty、polychord）

Q: 归一化常数是否影响模型对比？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**不影响**。当比较使用相同数据的模型时，归一化常数相同，在 ΔAIC、ΔBIC 中被抵消。

但归一化常数对 **绝对值** 有影响：

- 确保 χ² > 0
- 确保 AIC、BIC 为正数
- 约化 χ² ≈ 1 表示良好拟合

下一步
------

- `可视化 <visualization.html>`_：绘制约束图
- `采样器 <samplers.html>`_：了解 MCMC 配置
- `Fisher 预报 <fisher.html>`_：进行巡天预测
