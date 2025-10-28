# Intensity Mapping 架构审查报告

**审查日期**: 2025-10-28
**审查工具**: Codex CLI (GPT-5-Codex)
**审查范围**: Intensity Mapping模块架构设计
**问题严重性**: 🔴 **P0 CRITICAL**

---

## 1. 架构问题诊断

### P0 问题（必须立即修复）

#### 问题1: 配置文件混淆硬件与模型
**位置**: `hicosmo/configs/surveys/bingo.yaml:2, :6, :11`
- `model`、`parameters`、`reference` 同时出现在巡天配置中
- **后果**: 单一巡天无法在不同宇宙学模型间复用
- **违反原则**: 单一职责原则，关注点分离

#### 问题2: 配置文件包含分析结果
**位置**: `hicosmo/configs/surveys/ska1.yaml:7` (`measurements` 字段)
- 把Fisher预测结果伪装成输入配置
- **后果**: 误把旧表格当作真实观测数据
- **风险**: 循环依赖，无法独立验证

#### 问题3: IntensityMappingSurvey强制携带宇宙学
**位置**: `hicosmo/forecasts/intensity_mapping.py:168-195`
- `IntensityMappingSurvey` 类强制包含 `model` 和 `reference`
- **后果**: 同一硬件无法同时驱动LCDM与wCDM分析
- **限制**: 破坏了硬件配置的可复用性

#### 问题4: IntensityMappingFisher耦合配置
**位置**: `hicosmo/forecasts/intensity_mapping.py:353-493`
- `IntensityMappingFisher` 直接读取 `self.survey.reference`
- `run_forecast` 通过复制survey进行hack (`intensity_mapping.py:520-535`)
- **后果**: 任何模型切换都需要修改巡天配置文件
- **设计缺陷**: 无法在代码层面灵活切换模型

### P1 问题（应该尽快修复）

#### 问题5: 示例脚本放大配置污染
**位置**: `examples/run_ska1_forecasts.py:34-47`
- 默认把 `survey.reference` 当作fiducial参数
- **后果**: 放大了配置污染的影响，难以追溯参数来源
- **可维护性**: 后续分析难以解释参数选择依据

---

## 2. 与 Bull 2016 论文对比

### P0 偏差（违背论文方法）

#### 偏差1: 配置内容与论文Table 1不符
- **论文正确做法**: Bull 2016 Table 1仅列望远镜及观测策略
  - Ndish, Ddish, Tsys, Sky area, Tintegration
  - **纯硬件参数**
- **当前错误做法**: YAML掺入 `H0`、`w0` 等cosmology参数
- **结论**: 偏离了论文硬件与模型分离的做法

#### 偏差2: 流程顺序混乱
- **论文正确流程** (Section 3):
  ```
  Survey spec → Noise model → Fisher → Cosmology projection
  ```
- **当前错误流程**:
  ```
  Survey spec + Cosmology → Noise model → Fisher
  (提前绑定模型，打乱了分析顺序)
  ```

### P1 偏差（限制功能性）

#### 偏差3: 无法复用pipeline
- **论文要求** (Section 4): 不同实验共用统一理论pipeline
- **当前问题**: 配置捆绑模型，无法重用同一噪声链路测试多模型
- **影响**: 对比分析困难，效率低下

#### 偏差4: 输入输出边界模糊
- **论文方法**: Tables 2-4是分析**输出**
- **当前问题**: 仓库把Fisher结果写回配置 (`measurements` 字段)
- **后果**: 掩盖了回归验证和再现研究所需的清晰I/O边界

---

## 3. 正确的设计方案

### P0: 配置文件新模板

**原则**: 配置文件只包含硬件、观测策略、噪声模型，**不包含**宇宙学参数

```yaml
name: SKA1-MID-Band2
description: "SKA1-MID Medium-Deep Band 2 (Bull 2016)"

# ✅ 硬件配置 (应该有)
instrument:
  telescope_type: single_dish
  ndish: 200
  dish_diameter_m: 15.0
  nbeam: 1

# ✅ 观测策略 (应该有)
observing:
  survey_area_deg2: 5000
  total_time_hours: 10000
  frequency_range_MHz: [950, 1420]
  channel_width_MHz: 0.05

# ✅ 噪声参数 (应该有)
noise:
  system_temperature_K: 13.5
  sky_temperature:
    model: power_law
    T_ref_K: 25.0
    nu_ref_MHz: 408.0
    beta: 2.75

# ✅ HI物理参数 (观测量，不是宇宙学)
hi_tracers:
  bias:
    model: polynomial
    coefficients: [0.67, 0.18, 0.05]
  omega_hi:
    model: polynomial
    coefficients: [4.8e-4, 3.9e-4, -6.5e-5]

# ✅ 红移分bin (观测策略)
redshift_bins:
  default_delta_z: 0.1
  centers: [0.05, 0.15, 0.25, 0.35, 0.45]

# ✅ 元数据 (追溯来源)
metadata:
  reference: "Bull+2016 Table 1"

# ❌ 不应该出现的字段:
# - model: CPL
# - parameters: [H0, Omega_m, w0, wa]
# - reference: {H0: 67.36, Omega_m: 0.3153, w0: -1.0, wa: 0.0}
# - measurements: {...}
```

### P0: 类接口重构方案

**原则**: 通过依赖注入实现硬件配置与宇宙学模型的解耦

#### 新的类结构
```python
# 1. IntensityMappingSurvey只包含硬件/策略
class IntensityMappingSurvey:
    instrument: InstrumentConfig        # 硬件参数
    observing: ObservingStrategy        # 观测策略
    noise: NoiseModel                   # 噪声模型
    tracers: HIMetadata                 # HI物理参数
    redshift_bins: Sequence[RedshiftBin]  # 红移分bin

# 2. IntensityMappingFisher接受独立的cosmology
class IntensityMappingFisher:
    def __init__(
        self,
        survey: IntensityMappingSurvey,   # 硬件配置 (外部注入)
        cosmology: CosmologyBase,         # 宇宙学模型 (外部注入) ✅
        *,
        growth_gamma: float = 0.55
    ):
        """
        Fisher矩阵预测类 - 解耦设计

        Parameters
        ----------
        survey : IntensityMappingSurvey
            巡天硬件配置（只包含硬件信息）
        cosmology : CosmologyBase
            宇宙学模型实例（LCDM, wCDM, CPL等）
        """
        self.survey = survey
        self.cosmology = cosmology  # ✅ 从外部传入，不从survey读取
        ...
```

### P0: 调用层参数入口

**原则**: Fiducial cosmology和待估参数在**脚本层面**指定，不在配置文件中

```python
from hicosmo.models import LCDM, CPL
from hicosmo.forecasts import load_survey, IntensityMappingFisher

# 1. 加载巡天配置（只有硬件信息）
survey = load_survey('ska1_mid_band2')

# 2. 定义多个宇宙学模型用于对比
fiducials = {
    'LCDM': LCDM(H0=67.36, Omega_m=0.3153),
    'CPL': CPL(H0=67.36, Omega_m=0.3153, w0=-1.0, wa=0.0)
}

# 3. 指定要约束的参数
targets = ['H0', 'Omega_m', 'w0', 'wa']

# 4. 对每个模型运行Fisher预测 (同一巡天，不同模型) ✅
results = {}
for tag, cosmo in fiducials.items():
    fisher = IntensityMappingFisher(survey, cosmo)
    results[tag] = fisher.parameter_forecast(targets)

# 5. 对比不同模型的约束能力
plot_constraints_comparison(results, reference='Bull2016')
```

**优势**:
- ✅ 同一巡天可测试多个模型
- ✅ 参数来源清晰可追溯
- ✅ 代码灵活，易于扩展

### P1: 可扩展性建议

#### 配置预设与注册表
```python
# 引入 SurveyRegistry / CosmologyRegistry
survey = SurveyRegistry.get('ska1_mid_band2')

# YAML中可引用预设，避免拷贝粘贴
noise:
  preset: "ska1_mid.band2_standard"  # 引用预设噪声模型
```

---

## 4. 实施路线图

### Phase 1 (P0) - 紧急修复 (1-2天)

**目标**: 停止污染，建立新标准

#### 1.1 立即停止污染配置
- [ ] 创建新的YAML schema定义 (instrument/observing/noise/tracers)
- [ ] 重写 `ska1_mid_band2.yaml` 和 `ska1_wide_band1.yaml`
  - 移除 `model`, `parameters`, `reference` 字段
  - 只保留硬件和观测策略
- [ ] 添加JSON Schema或pydantic验证器

#### 1.2 修改示例脚本
- [ ] 更新 `examples/run_ska1_forecasts.py`
  - 显式创建cosmology实例
  - 通过构造函数传入Fisher类
- [ ] 添加多模型对比示例

#### 1.3 建立回归测试
- [ ] 在 `tests/forecasts/` 添加确定性fixture
- [ ] 验证与Bull 2016 Table 2-4结果一致

**验收标准**:
- ✅ 至少2个巡天配置文件完全清理
- ✅ 示例脚本能够运行多模型分析
- ✅ 回归测试通过

---

### Phase 2 (P0) - 完整重构 (3-5天)

**目标**: 彻底修复架构问题

#### 2.1 重构核心类
- [ ] 修改 `IntensityMappingSurvey` 数据类
  - 移除 `model` 和 `reference` 字段
  - 只包含硬件相关属性
- [ ] 重构 `IntensityMappingFisher.__init__`
  - 新增 `cosmology: CosmologyBase` 参数
  - 删除从 `survey.reference` 读取的代码
- [ ] 重构 `run_forecast` 方法
  - 删除复制survey的hack
  - 通过参数传递cosmology

#### 2.2 更新所有配置文件
- [ ] 清理所有8个巡天配置:
  - bingo.yaml
  - chime.yaml
  - meerkat.yaml
  - ska_mid.yaml
  - tianlai.yaml
  - ska1.yaml (完全重写)
  - ska1_mid_band2.yaml (已在Phase 1完成)
  - ska1_wide_band1.yaml (已在Phase 1完成)

#### 2.3 兼容性处理
- [ ] 为旧API提供deprecation警告
- [ ] 提供迁移指南文档
- [ ] 保留临时兼容层 (标记为deprecated)

**验收标准**:
- ✅ 所有配置文件符合新schema
- ✅ 核心类接口完全解耦
- ✅ 所有单元测试通过

---

### Phase 3 (P1) - 功能增强 (1周)

**目标**: 基于正确架构扩展功能

#### 3.1 多模型批量分析
- [ ] 实现 `ForecastComparison` 类
- [ ] 支持批量运行不同模型
- [ ] 自动生成对比报告

#### 3.2 结果存档系统
- [ ] 设计结果存储格式 (JSON/ASDF/NetCDF)
- [ ] 实现结果序列化/反序列化
- [ ] 添加元数据追踪 (survey + cosmology + timestamp)

#### 3.3 文档和验证
- [ ] 添加Bull 2016回归校验
- [ ] 更新用户文档
- [ ] 创建CLI/Notebook教程

**验收标准**:
- ✅ 能复现Bull 2016所有表格
- ✅ 文档完整，用户友好
- ✅ 性能无退化

---

## 5. 风险和注意事项

### P0 风险（必须应对）

#### 风险1: 破坏现有脚本
**问题**: 外部脚本可能依赖现有YAML字段结构
**应对**:
- 发布详细的迁移指南
- 提供deprecation警告 (至少保留1个版本)
- 在CHANGELOG中突出说明breaking changes

#### 风险2: 验证不充分导致结果退化
**问题**: 清理配置后Fisher输出可能改变
**应对**:
- 在 `tests/forecasts/` 添加deterministic fixtures
- 重跑Bull 2016所有基准测试
- 对比重构前后的数值输出（误差应<0.1%）

### P1 风险（需要注意）

#### 风险3: 历史结果丢失
**问题**: 去掉 `measurements` 字段后，用户遗失参考数据
**应对**:
- 把历史forecast结果迁移到 `docs/references/` 或 `results/historical/`
- 在README中明确说明数据来源（论文表格）
- 保留原始论文引用

#### 风险4: Schema管控不严格
**问题**: 未来可能再次混合概念
**应对**:
- 引入JSON Schema或pydantic强制验证
- 在CI/CD中添加配置文件lint检查
- 编写配置文件编写指南

---

## 6. 长期架构建议

### P1 建议（提升架构质量）

#### 建议1: 独立的cosmology配置
```python
# 引入 hicosmo/configs/cosmology/*.yaml
fiducials = load_cosmology_preset('planck2018_lcdm')

# 或Python registry
from hicosmo.cosmology import CosmologyPresets
fiducials = CosmologyPresets.PLANCK2018_LCDM
```

**优势**:
- Fiducial参数套餐化
- 与survey完全解耦
- 便于标准化对比

#### 建议2: 版本化配置管理
```yaml
metadata:
  version: "1.0"
  source: "Bull 2016 Table 1"
  date: "2025-10-28"
  schema_version: "2.0"
```

**优势**:
- 追踪Red Book vs Bull 2016差异
- 支持配置演化历史
- 便于回滚和验证

### P2 建议（长期规划）

#### 建议3: SurveyBuilder DSL
```python
# 提供DSL/CLI组合实验
survey = SurveyBuilder()\
    .instrument('ska1_mid_band2')\
    .observing(area=5000, time=10000)\
    .with_cosmology(LCDM(...))\
    .build()
```

**优势**:
- 更灵活的组合方式
- 降低配置文件复杂度
- 提高代码复用性

#### 建议4: 标准化结果格式
```python
# 序列化Fisher输出
result.save('results/ska1_lcdm_forecast.asdf')

# 支持多格式对比
compare_forecasts([
    'results/ska1_lcdm_forecast.asdf',
    'results/ska1_cpl_forecast.asdf',
])
```

**优势**:
- 便于对比多模型、多巡天
- 与后续MCMC流水线衔接
- 支持数据存档和共享

---

## 总结与建议

### 核心问题
1. **P0**: 配置文件混淆硬件配置和宇宙学模型 → 违反关注点分离
2. **P0**: 类接口强制耦合cosmology → 限制灵活性
3. **P0**: 偏离Bull 2016论文方法 → 无法正确复现

### 紧急行动
1. **Phase 1**: 停止污染配置，建立新标准（1-2天）
2. **Phase 2**: 重构核心类和所有配置（3-5天）
3. **Phase 3**: 验证、文档、功能增强（1周）

### 长期方向
- 引入cosmology配置/registry
- 版本化配置管理
- 标准化结果存档

**下一步**: 立即开始Phase 1，优先修复2个SKA配置文件和示例脚本。

---

**审查完成时间**: 2025-10-28
**Token使用**: 50,996
**建议审查周期**: 每季度一次架构健康检查
