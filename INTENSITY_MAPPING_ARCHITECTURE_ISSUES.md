# Intensity Mapping 模块架构问题总结

**发现日期**: 2025-10-28
**问题严重性**: 🔴 **CRITICAL** - 架构设计根本性错误

---

## 🚨 核心问题

### 问题描述

`hicosmo/configs/surveys/` 目录下的巡天配置文件**混淆了两个完全不同的概念**：

1. **巡天硬件/观测配置** （应该有的）
   - 天线数量 (ndish)
   - 天线口径 (dish_diameter_m)
   - 观测面积 (survey_area_deg2)
   - 观测时间 (total_time_hours)
   - 频道宽度 (channel_width_hz)
   - 系统温度 (system_temperature_K)
   - 等等...

2. **宇宙学模型参数** （不应该在这里）
   - 模型类型 (model: CPL/LCDM/wCDM)
   - Fiducial参数 (H0, Omega_m, w0, wa)
   - 要预测的参数列表 (parameters)

### 具体问题示例

#### 问题1: 模型参数出现在巡天配置中

```yaml
# bingo.yaml - 严重混淆
name: BINGO
model: CPL                    # ❌ 模型类型不应该在巡天配置中
description: "..."
parameters:                   # ❌ 要预测的参数不应该在巡天配置中
  - H0
  - Omega_m
  - w0
  - wa
reference:                    # ❌ Fiducial宇宙学参数不应该在巡天配置中
  H0: 67.36
  Omega_m: 0.3153
  w0: -1.0
  wa: 0.0
# ✅ 应该只有硬件配置...
```

#### 问题2: 有些配置文件完全没有硬件信息

```yaml
# ska1.yaml - 完全是模型信息！
SKA1:
  description: "SKA1-MID Band 1 intensity mapping reference survey"
  model: "LCDM"               # ❌ 模型信息
  parameters: ["H0", "Omega_m"]  # ❌ 参数列表
  reference:                  # ❌ Fiducial参数
    H0: 67.36
    Omega_m: 0.3153
  measurements:               # ❌ 这个是Fisher预测结果，不是配置
    - z: 0.35
      sigma:
        DA_over_rs: 0.025
        H_times_rs: 0.022
        fsigma8: 0.04
# ❌ 完全没有硬件配置！天线在哪里？观测时间是多少？
```

#### 问题3: 混合了正确和错误的信息

```yaml
# ska1_mid_band2.yaml - 部分正确，部分错误
name: ska1_mid_band2
model: cpl                    # ❌ 模型不应该在这里
description: "SKA1-MID Medium-Deep Band 2 survey (Red Book 2018)"
reference:                    # ❌ Fiducial参数不应该在这里
  H0: 67.36
  Omega_m: 0.316
  w0: -1.0
  wa: 0.0
instrument:                   # ✅ 这部分是正确的！
  ndish: 200
  dish_diameter_m: 15.0
  survey_area_deg2: 5000.0
  total_time_hours: 10000.0
  channel_width_hz: 50000.0
  system_temperature_K: 13.5
```

---

## 🎯 正确的架构应该是什么样的？

### 正确的关注点分离

#### 1. 巡天配置 (Survey Configuration)

**文件**: `hicosmo/configs/surveys/*.yaml`

**内容**: 仅包含硬件和观测策略
```yaml
# ska1_mid_band2.yaml - 正确的巡天配置
name: SKA1-MID-Band2
description: "SKA1-MID Medium-Deep Band 2 survey (Red Book 2018)"

# 硬件配置
instrument:
  telescope_type: interferometer  # or single_dish
  ndish: 200
  dish_diameter_m: 15.0
  baseline_min_m: 30.0
  baseline_max_m: 3000.0

# 观测策略
observing:
  survey_area_deg2: 5000.0
  total_time_hours: 10000.0
  frequency_range_MHz: [950, 1420]
  channel_width_MHz: 0.05
  redshift_range: [0.0, 0.5]

# 噪声参数
noise:
  system_temperature_K: 13.5
  sky_temperature:
    model: power_law
    T_ref_K: 25.0
    nu_ref_MHz: 408.0
    beta: 2.75

# HI物理参数（观测量，不是宇宙学参数）
hi_properties:
  bias_model: polynomial
  bias_coefficients: [0.67, 0.18, 0.05]
  density_model: polynomial
  density_coefficients: [0.00048, 0.00039, -0.000065]

# 红移分bin策略
redshift_bins:
  delta_z: 0.1
  bins:
    - z_min: 0.0
      z_max: 0.1
    - z_min: 0.1
      z_max: 0.2
    # ...
```

#### 2. 宇宙学模型参数 (Cosmological Model)

**由分析脚本指定**，不是配置文件！

```python
# 在分析脚本中指定
from hicosmo.models import wCDM
from hicosmo.forecasts import load_survey, IntensityMappingFisher

# 1. 加载巡天配置（只有硬件信息）
survey = load_survey('ska1_mid_band2')

# 2. 指定宇宙学模型和fiducial参数
fiducial_params = {
    'H0': 67.36,
    'Omega_m': 0.3153,
    'Omega_b': 0.0493,
    'w': -1.0,  # wCDM模型
}
model = wCDM(**fiducial_params)

# 3. 运行Fisher预测
fisher = IntensityMappingFisher(survey, model)
result = fisher.forecast(params_to_constrain=['H0', 'Omega_m', 'w'])
```

---

## 🔍 为什么这个设计是错误的？

### 1. 违反单一职责原则

巡天配置应该**只描述硬件和观测策略**，不应该知道任何宇宙学模型的信息。

**类比**：望远镜的说明书不应该包含"宇宙的哈勃常数是67.36"这样的信息。

### 2. 限制了灵活性

当前设计下，如果我想：
- 用同一个巡天预测LCDM、wCDM、CPL等不同模型
- 测试不同的fiducial宇宙学参数
- 预测不同的参数组合

我必须为每种情况创建一个新的配置文件，或者修改代码。这是**完全错误的**。

### 3. 混淆了"输入"和"输出"

```yaml
# ska1.yaml中的这部分是什么？
measurements:
  - z: 0.35
    sigma:
      DA_over_rs: 0.025
      H_times_rs: 0.022
      fsigma8: 0.04
```

这看起来像是**Fisher预测的结果**，而不是输入配置！为什么会出现在配置文件中？

### 4. 无法复现论文结果

Bull et al. (2016) 论文中：
- 表1列出的是**巡天硬件参数**（天线数、面积、时间等）
- 表2-4列出的是**不同宇宙学模型下的预测精度**

如果硬件配置和模型参数混在一起，我们无法清晰地复现论文的分析。

---

## 📋 需要修复的内容

### 1. 重新设计配置文件结构

**分离关注点**：
- `surveys/*.yaml` - 只包含硬件配置
- 模型参数在代码中指定，不在配置文件中

### 2. 重构 `IntensityMappingFisher` 类

当前签名（推测）：
```python
class IntensityMappingFisher:
    def __init__(self, survey):  # survey包含了模型信息 ❌
        self.survey = survey
        self.model = survey.model  # 从配置中读取 ❌
        self.fiducial = survey.reference  # 从配置中读取 ❌
```

应该改为：
```python
class IntensityMappingFisher:
    def __init__(self, survey, cosmology_model):  # 分离！ ✅
        """
        Parameters
        ----------
        survey : IntensityMappingSurvey
            巡天硬件配置（只包含硬件信息）
        cosmology_model : CosmologyBase
            宇宙学模型实例（LCDM, wCDM, CPL等）
        """
        self.survey = survey
        self.model = cosmology_model  # 外部传入 ✅
```

### 3. 清理所有配置文件

需要检查并修复：
- `bingo.yaml`
- `chime.yaml`
- `meerkat.yaml`
- `ska_mid.yaml`
- `ska1_mid_band2.yaml`
- `ska1_wide_band1.yaml`
- `ska1.yaml`
- `tianlai.yaml`

### 4. 更新示例脚本

`examples/run_ska1_forecasts.py` 也需要相应修改。

---

## 📚 需要阅读的参考资料

1. **Bull et al. (2016)** - ApJ 817, 26
   - 文件位置: `/Users/qijingzhao/Programs/hicosmo_new1/Bull_2016_ApJ_817_26.pdf`
   - 重点：Table 1（巡天参数）和 Table 2-4（预测结果）
   - 理解正确的巡天配置应该包含什么

2. **当前代码**:
   - `/Users/qijingzhao/Programs/hicosmo_new1/hicosmo/forecasts/intensity_mapping.py`
   - `/Users/qijingzhao/Programs/hicosmo_new1/examples/run_ska1_forecasts.py`

---

## 🔧 修复优先级

| 任务 | 优先级 | 估计工作量 |
|------|--------|-----------|
| 1. Codex审查整个架构 | P0 | 1小时 |
| 2. 阅读Bull 2016论文 | P0 | 1小时 |
| 3. 设计新的配置文件结构 | P0 | 2小时 |
| 4. 重构IntensityMappingFisher | P0 | 4小时 |
| 5. 更新所有配置文件 | P1 | 3小时 |
| 6. 修改示例脚本 | P1 | 1小时 |
| 7. 实现wCDM预测脚本 | P1 | 2小时 |
| 8. 验证与论文对比 | P1 | 2小时 |

**总计**: ~16小时工作量

---

## 🎯 期望的最终效果

### 使用示例

```python
# 加载巡天配置（只有硬件）
survey = load_survey('ska1_mid_band2')

# 定义要测试的宇宙学模型
models = {
    'LCDM': LCDM(H0=67.36, Omega_m=0.3153),
    'wCDM': wCDM(H0=67.36, Omega_m=0.3153, w=-1.0),
    'CPL': CPL(H0=67.36, Omega_m=0.3153, w0=-1.0, wa=0.0),
}

# 对每个模型运行Fisher预测
results = {}
for model_name, model in models.items():
    fisher = IntensityMappingFisher(survey, model)
    results[model_name] = fisher.forecast(
        params=['H0', 'Omega_m', 'w0', 'wa']  # 根据模型选择
    )

# 绘制不同模型的约束对比
plot_constraints_comparison(results, reference_paper='Bull2016')
```

---

## ✅ 行动计划

1. **立即行动**: 使用codex-review技能审查
   ```bash
   # 让Codex CLI分析整个架构
   - 阅读Bull 2016论文
   - 审查intensity_mapping.py代码
   - 审查所有配置文件
   - 给出详细的修改建议
   ```

2. **根据Codex建议**: 制定详细重构计划

3. **实施重构**: 按优先级逐步修复

4. **验证**: 确保能复现Bull 2016的结果

---

**结论**: 这是一个**架构层面的根本性错误**，必须彻底重构才能正确使用。当前的设计混淆了硬件配置和宇宙学模型两个概念，导致代码既不灵活也不清晰。

**下一步**: 调用codex-review技能进行深度审查。
