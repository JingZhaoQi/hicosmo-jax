# HIcosmo Project Architecture & Development Log

> **Vision**: Create the most advanced JAX-based cosmological parameter estimation framework, surpassing qcosmc with modern high-performance computing

## Project Philosophy

### ⚠️ API设计第一原则 - 项目生命线

**API简洁性直接决定项目成败**：

```
API简洁性 → 降低学习成本 → 提高用户选择性 → 增强用户粘度 → 项目成功
```

**核心逻辑**：
- 用户调用越简单 → 用户选择性越高 → 用户粘度越高
- 10行代码的API意味着用户每次都要翻文档，容易出错，降低信心
- 对比竞品时，复杂的API会导致用户直接放弃我们的项目

**设计标准**：
- ✅ **3行代码完成核心功能** - 这是我们的API设计目标（已在fisher_forecast中实现）
- ✅ **智能默认参数** - 90%的使用场景不需要额外配置
- ✅ **渐进式复杂度** - 简单任务简单做，复杂任务才暴露复杂性
- ❌ **绝不为了"完整性"增加必选参数** - 每个新参数都是用户流失的风险

**反面案例（已修复）**：
```python
# ❌ 旧API - 10+行，用户直接放弃
from hicosmo.models import CPL
from hicosmo.fisher import load_survey, IntensityMappingFisher
survey = load_survey('ska1_mid_band2')
fiducial = CPL(H0=67.36, Omega_m=0.3153, w0=-1.0, wa=0.0)
fisher = IntensityMappingFisher(survey, fiducial)
result = fisher.parameter_forecast(['w0', 'wa'])
# ... 还要手动提取结果和绘图

# ✅ 新API - 3行，用户愉快使用
from hicosmo.api import fisher_forecast
forecast = fisher_forecast(survey='ska1_mid_band2', model='cpl', target=['w0', 'wa'])
forecast.plot('w0', 'wa', filename='forecast.png')
```

**未来所有新功能必须遵守此原则**：
- 实现新功能前，先设计最简API
- 代码审查时，API简洁性是第一检查点
- 如果无法用3-5行代码展示核心功能，说明设计有问题

---

### Core Design Principles
- **奥卡姆剃刀原则**: 如无必要，勿增实体 - 只实现必要功能，避免过度设计
- **基类简洁**: 绝不在基类中添加if判断处理特殊情况，保持接口清晰
- **继承重写**: 通过继承和方法重写实现特殊化，而非条件堆积
- **单一责任**: 每个类只负责一件事，每个方法只做一种计算
- **数据理解优先**: 实现前必须完全理解数据真实含义和物理背景
- **遵循成熟模式**: 参考Cobaya等成熟实现的设计模式

### JAX-First架构
- **纯函数优先**: 所有宇宙学计算为无副作用的纯函数
- **自动微分**: 利用JAX的autograd进行精确梯度计算
- **JIT编译**: 关键计算路径JIT编译获得4-7倍性能提升
- **GPU就绪**: 透明GPU加速支持
- **向量化**: vmap实现高效批量计算

## Current Architecture Status

### ✅ Completed Core Infrastructure

#### 1. Production-Ready Cosmology Base (`hicosmo/core/cosmology.py`)
**CosmologyBase** - 完整的抽象基类，包含：
- **背景演化**: E_z(), H_z(), w_z(), rho_DE_z(), Omega_m_z(), Omega_DE_z()
- **距离计算**: 共动距离、横向共动距离、光度距离、角直径距离、距离模数
- **时间演化**: 回望时间、宇宙年龄 (lookback_time, age_universe)
- **体积元素**: 微分共动体积、体积元素计算
- **增长理论**: 增长因子、增长率、fσ8参数
- **宇宙学参数**: 减速参数q(z)、jerk参数j(z)
- **工具方法**: 派生参数计算、模型摘要生成

**关键特性**:
- 高精度数值积分 (2000步梯形积分)
- 曲率支持 (平直/开放/封闭宇宙)
- 完整的类型注解和文档
- JAX JIT编译优化

#### 2. Advanced LCDM Implementation (`hicosmo/models/lcdm.py`)
**LCDM** - 生产级标准宇宙学模型，超越qcosmc功能：
- **完整参数处理**: H0, Omega_m, Omega_b, Omega_k, Omega_r, sigma8, n_s
- **物理一致性**: 严格参数验证和闭合关系检查
- **辐射分量**: 精确计算包括中微子贡献
- **声音视界**: 拖拽时刻拟合公式 (Eisenstein & Hu 1998)
- **增长函数**: 解析近似 (Carroll, Press & Turner 1992)
- **特殊距离**: 时间延迟距离、临界密度计算
- **红移漂移**: Sandage测试的漂移率计算
- **预设参数集**: Planck2018和WMAP9最优参数

**验证特性**:
- 物理参数范围检查
- 闭合关系验证 (Ω_total = 1 ± 1e-6)
- 负密度参数检测

#### 3. Enhanced Constants & Utilities (`hicosmo/utils/constants.py`)
完整的物理和天文学常数库：
- **基础常数**: 光速、引力常数、普朗克常数等
- **宇宙学常数**: CMB温度、临界密度因子、声学尺度
- **转换函数**: 单位转换、红移-尺度因子互换
- **默认参数**: Planck 2018基准宇宙学参数

### 📊 Current Capabilities Assessment

相比qcosmc的功能覆盖：
- ✅ **Background Evolution** - 完全实现并优化
- ✅ **Distance Calculations** - 超越原实现（包含曲率）
- ✅ **Growth Functions** - 解析公式实现
- ⭕ **Cosmological Models** - 仅LCDM (目标25+模型)
- ❌ **Likelihood Functions** - 待实现
- ❌ **MCMC Sampling** - 基础框架存在，需完善
- ❌ **Fisher Matrix** - 待实现
- ❌ **21cm Module** - 待实现

### 🎯 Performance Targets

基于qcosmc分析的性能目标：
- **距离计算**: 目标10x加速 (JAX JIT vs scipy.quad)
- **MCMC采样**: 目标5x加速 (NUTS vs emcee)
- **Fisher矩阵**: 目标20x加速 (自动微分 vs 数值微分)
- **内存使用**: 减少50% (高效JAX数组操作)

## Development Roadmap

### Phase 1: Core Models (Priority 1) 🚧
**Target**: 实现qcosmc中所有25+宇宙学模型

#### 1.1 Dark Energy Models
- [ ] **wCDM** - 常数暗能量状态方程
- [ ] **CPL** - Chevallier-Polarski-Linder参数化 
- [ ] **JBP** - Jassal-Bagla-Padmanabhan参数化
- [ ] **Geos** - 广义指数振荡暗能量

#### 1.2 Interacting Dark Energy
- [ ] **IwCDM1/IwCDM2** - 不同相互作用项
- [ ] **ILCDM1/ILCDM2** - 相互作用ΛCDM变种

#### 1.3 Modified Gravity
- [ ] **DGP** - Dvali-Gabadadze-Porrati braneworld
- [ ] **fR_power/fR_power2** - f(R)修正引力
- [ ] **FT_law/FT_exp/FT_tanh** - f(T)修正引力

#### 1.4 Exotic Models
- [ ] **Holographic Dark Energy** - HDE_CA/HDE_CT/HDE_EH
- [ ] **Chaplygin Gas** - GCG/CGG/MCG variants
- [ ] **Running Vacuum** - RVM model
- [ ] **Generalized Early Dark Energy** - GEDE

### Phase 2: Observational Data (Priority 1) 🚧
**Target**: 超越qcosmc的数据处理能力

#### 2.1 Supernovae
- [ ] **Pantheon+** - 最新1701个SNe Ia样本
- [ ] **Union3** - 备选SNe数据集
- [ ] **完整nuisance参数** - α, β, M_B, ΔM处理

#### 2.2 Baryon Acoustic Oscillations
- [ ] **DESI DR1** - 最新BAO测量
- [ ] **BOSS/eBOSS** - 历史BAO数据
- [ ] **6dFGS + MGS** - 低红移约束
- [ ] **声学尺度提取** - D_M/r_d, H*r_d测量

#### 2.3 Cosmic Microwave Background
- [ ] **Planck 2018压缩似然** - l_A, R, z_star
- [ ] **重组历史** - 声学视界r_s(z_d)计算
- [ ] **声学峰值** - 完整CMB功率谱处理

#### 2.4 Advanced Probes
- [ ] **H0 Measurements** - SH0ES距离阶梯
- [ ] **Strong Lensing** - H0LiCOW时间延迟
- [ ] **Gravitational Waves** - 标准汽笛距离
- [ ] **Fast Radio Bursts** - DM(z)宇宙学约束
- [ ] **21cm Intensity Mapping** - BINGO/SKA/MeerKAT预测

### Phase 3: Advanced MCMC (Priority 2) 🚧
**Target**: 企业级MCMC框架

#### 3.1 Sampling Algorithms
- [x] **NUTS Sampler** - 基础NumPyro封装
- [ ] **智能初始化** - 多策略参数初始化
- [ ] **自适应调参** - 动态步长和质量矩阵
- [ ] **多链并行** - 高效CPU/GPU并行

#### 3.2 Convergence Diagnostics
- [ ] **实时R̂监控** - Gelman-Rubin统计量
- [ ] **有效样本数** - ESS计算和报告
- [ ] **链健康检查** - 自动异常检测
- [ ] **收敛警告系统** - 智能诊断建议

#### 3.3 Production Features
- [ ] **检查点系统** - 断点续跑机制
- [ ] **实时可视化** - Rich库进度条和诊断表
- [ ] **内存管理** - 大规模采样优化
- [ ] **结果导出** - GetDist兼容格式

### Phase 4: Fisher Matrix Analysis (Priority 2) 🚧
**Target**: 超越qcosmc的预测能力

#### 4.1 Core Functionality
- [ ] **自动微分** - JAX梯度计算Fisher矩阵
- [ ] **参数变换** - 不同参数空间转换
- [ ] **先验整合** - 外部约束添加
- [ ] **边际化** - 参数投影和约束预测

#### 4.2 Multi-Probe Analysis
- [ ] **Fisher矩阵合并** - 多探针联合约束
- [ ] **相关性分析** - 参数简并性研究
- [ ] **暗能量FoM** - Figure of Merit计算
- [ ] **巡天优化** - 观测策略优化

### Phase 5: 21cm Cosmology (Priority 3) 🚧
**Target**: 专门的21cm强度映射模块

#### 5.1 Signal Modeling
- [ ] **HI功率谱** - 中性氢功率谱计算
- [ ] **BAO特征提取** - 从21cm信号提取BAO
- [ ] **RSD建模** - 红移空间畸变效应
- [ ] **前景去除** - 银河系前景处理

#### 5.2 Survey Configurations
- [ ] **BINGO配置** - 巴西射电望远镜
- [ ] **SKA预测** - Square Kilometre Array
- [ ] **MeerKAT分析** - 南非射电望远镜
- [ ] **天籁配置** - 中国21cm实验

### Phase 6: Visualization & Results (Priority 3) 🚧
**Target**: 专业级科学可视化

#### 6.1 Statistical Plotting
- [ ] **Corner图** - 参数约束椭圆
- [ ] **Chain诊断** - MCMC轨迹可视化
- [ ] **置信区间** - 1σ/2σ/3σ等高线
- [ ] **GetDist集成** - 无缝兼容性

#### 6.2 Scientific Visualization
- [ ] **宇宙学函数图** - E(z), w(z), f(z)等
- [ ] **距离-红移关系** - Hubble图生成
- [ ] **增长历史** - D(z), f(z)演化图
- [ ] **巡天比较** - 多实验约束对比

## Technical Implementation Notes

### JAX Best Practices
```python
# ✅ 正确：纯函数 + JIT
@jit
def E_z(z, params):
    return jnp.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)

# ❌ 错误：副作用
def E_z(z, params):
    self.last_z = z  # 副作用！
    return jnp.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)
```

### Error Handling Strategy
```python
# 物理约束验证
if not 0.01 < params['Omega_m'] < 1.0:
    raise ValueError(f"Omega_m = {params['Omega_m']} outside physical range")

# 数值稳定性检查
if jnp.any(jnp.isnan(result)):
    raise RuntimeError("Numerical instability detected in calculation")
```

### Performance Optimization
- **内存预分配**: 使用jnp.empty预分配大数组
- **向量化**: vmap替代Python循环
- **JIT编译**: @jit装饰器用于热点函数
- **静态参数**: static_argnums用于非数组参数

## Quality Assurance

### Testing Strategy
- **单元测试**: 每个函数单独测试
- **积分测试**: 端到端流程测试
- **基准测试**: 与qcosmc结果对比
- **性能测试**: 执行时间和内存使用监控

### Documentation Requirements
- **NumPy风格docstrings**: 完整参数和返回值说明
- **类型注解**: 所有函数必须有类型提示
- **示例代码**: 每个主要功能包含使用示例
- **架构文档**: 设计决策和权衡说明

## Git & GitHub Integration

### Repository Structure
```
hicosmo/
├── .github/workflows/    # CI/CD配置
├── docs/                # 文档
├── examples/            # 示例脚本
├── hicosmo/             # 主包
├── tests/               # 测试套件
├── benchmarks/          # 性能基准
└── data/               # 示例数据
```

### Development Workflow
1. **Feature Branches**: 每个新功能独立分支
2. **Pull Requests**: 代码审查流程
3. **Continuous Integration**: 自动测试和质量检查
4. **Release Management**: 语义化版本控制

### Collaboration Guidelines
- **Issue Tracking**: GitHub Issues管理任务和bug
- **Code Review**: 至少一人审查所有PR
- **Documentation**: 更新文档伴随代码更改
- **Testing**: 新代码必须包含测试

## Current Status & Next Steps

### Immediate Priorities (本周)
1. **Git仓库初始化** - 设置GitHub仓库和CI/CD
2. **wCDM模型实现** - 第一个动态暗能量模型
3. **基础似然函数** - Pantheon+ SNe数据处理
4. **MCMC完善** - 智能初始化和诊断

### Success Metrics
- **性能**: 距离计算 > 5x加速
- **精度**: 与qcosmc结果误差 < 0.1%
- **覆盖**: 实现 > 80%的qcosmc功能
- **可用性**: 完整文档和示例

---

**最后更新**: 2024年1月8日  
**开发状态**: 架构完成，核心功能实现中  
**下一里程碑**: wCDM模型 + GitHub集成