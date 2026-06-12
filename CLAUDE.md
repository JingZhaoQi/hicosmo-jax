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

## Performance: Shared Distance Grid Optimization (2026-03)

### 问题
`CombinedLikelihood` 的 for 循环让 SN+BAO+CMB 各自独立调用 `compute_grid_traced`，距离积分做了三遍。d=4 (wCDM joint) ESS/s 从 170 (d=2 SN) 降到 10.8。

### 根因
- SN: `compute_grid_traced(z_grid_sn, params)` — 独立第1次（N=2761）
- BAO: `compute_grid_traced(z_grid_bao, params)` — 独立第2次（N=5660）
- CMB: `compute_grid_traced(z_star * z_base, params)` — 独立第3次（N=4096, z~1090）
- 三次 E(z) 计算 + 三次积分 + 三条 AD 反向路径

### 优化方案（分支 `optimize/shared-distance-grid`，已 merge 到 `codex`）

**迭代1：SN+BAO 共享距离网格**
- `CombinedLikelihood.__init__` 检测共享机会（继承链感知的 cosmology class 检测）
- `__call__` 调用 `compute_grid_traced` 一次，通过 `_loglike_from_grid` 分发
- SN 和 BAO 各自提供 `_loglike_from_grid` JIT 闭包
- DESI BAO 的 `_loglike_from_grid` 必须尊重 `omega_b_mode='h0rd'` 的 rd 路径

**迭代2：CMB 轻量路径（已回退删除，2026-06-11）**
- 曾新增 `compute_DM_at_z(z_target, params, n_grid=1024)`，commit e71cfac 回退后成为零调用死代码，
  且其 1024 点均匀网格对 $\int_0^{1090} dz/E(z)$ 有约 1.4% 端点误差（诱导性 API），已删除。
- CMB 现走 `compute_grid_traced(z_star * z_base)` 4096 点路径（梯度成本仅 ~0.24ms，占比小）。

**修改的文件：**
- `hicosmo/models/base.py` — `compute_distances_core`（`make_compute_shared_grid` 已删除）
- `hicosmo/likelihoods/combined.py` — 共享网格编排
- `hicosmo/likelihoods/sn/pantheonplus.py` — `_loglike_from_grid`
- `hicosmo/likelihoods/bao/base.py` — `_loglike_from_grid`（基类）
- `hicosmo/likelihoods/bao/datasets.py` — `_loglike_from_grid`（DESI override）

### 结果

| 配置 | d | 优化前 ESS/s | 优化后 ESS/s | 提升 |
|------|---|------------|------------|------|
| LCDM+SN | 2 | 170 | 282 | +66% |
| LCDM Joint | 3 | 29.5 | 43.1 | +46% |
| wCDM Joint | 4 | 10.8 | 14.2 | +31% |
| CPL Joint | 5 | — | 7.5 | 新基线 |

**⚠️ 归因修正（2026-06-11 实测）**：上表提升主要来自同期其他改动，不是共享网格本身——
单独实测共享网格在前向只省 8-11%、梯度只省 3%（LCDM+SN 单似然无共享可言却也 +66% 即为旁证）。
真正的梯度成本主体是 SN 似然内部：`jnp.interp` 的 AD 反向（约一半）+ 1580² 协方差 matvec。
2026-06-11 用等距网格 O(1) 索引插值（`utils.jax_tools.interp_linspace`）替换全部似然热路径的
`jnp.interp` 后，联合梯度 3.86 → 2.32 ms（−40%），数值与 `jnp.interp` 逐位一致。
**注意：`interp_linspace` 要求等距网格，所有 `_z_grid` 与共享网格必须保持 `jnp.linspace` 构造。**

### ⚠️ 关键教训（硬性规则）

**网格密度对梯度采样器的 τ 有决定性影响，不可压缩。**

实验证据：将共享网格从 5660 点减到 2048 点，logL 精度看似够（diff < 1.4e-03），但 τ(d=4) 从 3.5 暴涨到 14.7。梯度精度比似然值精度要求高得多。

**规则：对于 NUTS 等梯度采样器，插值网格密度的标准不是 logL 精度，而是 ∇logL 精度。永远保持子似然原始的网格密度。**

### 新似然如何参与共享网格优化

**当前状态**：新似然（如 GW）默认走 fallback 路径（独立 `compute_grid_traced`），不影响正确性，但无法获得共享优化。

**参与共享优化需要的条件**：
1. 有 `_z_grid` 属性（`jnp.linspace`，低红移）
2. 有 `_cosmology_class` 属性
3. 实现 `_loglike_from_grid(cosmo_grid, z_grid, params_jax)` JIT 闭包

**模板**（以 GW 引力波标准汽笛为例）：
```python
class GWLikelihood(Likelihood):
    def __init__(self, cosmology_class, ...):
        self._cosmology_class = cosmology_class
        self._z_grid = jnp.linspace(0, z_max, n_grid)
        # ... 加载数据、构建 JIT 闭包 ...
        self._build_grid_accepting_loglike()

    def _build_grid_accepting_loglike(self):
        """构造接受预计算网格的 loglike 版本。"""
        z_obs = self._z_obs
        d_L_obs = self._d_L_obs
        sigma = self._sigma

        @jit
        def _loglike_from_grid_impl(cosmo_grid, grid_z, params):
            # 从共享网格插值距离
            d_L = jnp.interp(z_obs, grid_z, cosmo_grid["d_L"])
            # 计算 chi2（与 _loglike_fast 逻辑一致）
            diff = d_L - d_L_obs
            chi2 = jnp.sum((diff / sigma) ** 2)
            return -0.5 * chi2

        self._loglike_from_grid = _loglike_from_grid_impl
```

**关键规则**：
- `_loglike_from_grid` 的结果必须与 `_loglike_fast` / `__call__` 完全一致
- 从 `cosmo_grid` 中只取需要的量（`d_L`, `D_M`, `D_H`, `E_z`）
- 如果有特殊的 rd/nuisance 计算逻辑，必须在 `_loglike_from_grid` 中复制
- **不要**假设 `grid_z` 与 `self._z_grid` 相同（共享网格可能更密/更广）

**TODO：未来改进**：在基类中提供 `_auto_build_loglike_from_grid()` 方法，
通过分析子类的距离使用模式自动生成 grid-accepting 版本，
实现真正的零代码扩展。当前需手动实现（约 15 行 JIT 闭包）。

### 废弃的方案

**Monkey-patch `compute_grid_traced`**：在 `CombinedLikelihood.__call__` 中临时替换类方法。导致 JIT 重编译，比原始慢 50-70×。

**统一 z_grid 后重初始化似然**：修改 `lik._z_grid` 后调用 `_initialize_fast_likelihood()`，但 SN 的 JIT 闭包捕获了旧 z_grid 的引用不会更新，导致精度下降。

---

## Current Status

### 已完成
- ✅ 完整的 LCDM/wCDM/CPL/ILCDM 模型
- ✅ 7 种 SN/BAO 数据集（Pantheon+ / Union3 / DESY5；DESI DR1+DR2 / SDSS / 6dFGS）+ Planck CMB + H0LiCOW + SH0ES
- ✅ NumPyro NUTS 采样 + Fisher 矩阵预测
- ✅ 与 Cobaya 交叉验证（chi2 精度 < 10^{-6}）
- ✅ GPU 透明部署（JAX 设备透明性）
- ✅ JCAP 论文撰写中

### 性能实测
- 距离计算 137× 加速（scipy → JAX JIT）
- SN 似然 14× 加速
- GPU 加速 30-35×（RTX A6000）
- ESS 吞吐量 ~8× 优于 Cobaya（LCDM+SN 基准）
- 联合似然梯度 3.86 → 2.32 ms（interp_linspace 优化，2026-06-11）

## 2026-06-11 架构审核修复（详见 ARCHITECTURE_REVIEW_2026-06-11.md）

本轮按审核报告完成的关键修复，未来开发必须维持的不变量：

1. **x64 由包级显式启用**（`hicosmo/__init__.py`），不再依赖 GW 模块 import 副作用；
   `HICOSMO_DISABLE_X64=1` 可关闭。
2. **NUTS trace 期异常必须传播**：`log_probability` 不得 try/except 返回常数，
   否则 NUTS 静默采样先验（曾是 P0 bug）。
3. **num_warmup 是 per-chain 语义**（不可按链数均摊）；num_samples 仍是 total。
4. **BBN prior 只存在于 BAO 的 JIT 闭包内**（`_maybe_add_bbn_prior`），
   任何外层 wrapper 不得再加（曾出现 0/1/2 次三路径不一致）。
5. **固定 M_B/M wrapper 必须自带 `_loglike_from_grid`**（注入固定值后委托 base）。
6. **似然热路径插值统一用 `interp_linspace`**（等距网格 O(1) 索引）；
   所有 `_z_grid` 与共享网格必须是 `jnp.linspace`。
7. **未知宇宙学参数名会触发 UserWarning**（did-you-mean 提示），
   wCDM 的参数名是 `w0` 不是 `w`。
8. **共享网格一致性测试矩阵**（test_combined_likelihood.py）：新增似然/模式时
   必须补充对应的 logL + 梯度一致性用例。
9. 已删除死代码：`compute_DM_at_z`、`make_compute_shared_grid`、
   `_precomputed_grid` 旧协议、`parameters/collector.py`。
10. 用户标准 API 契约固化于 `tests/test_user_api_contract.py`，任何重构不得破坏。

---

**最后更新**: 2026-06-11
**开发状态**: 核心功能完成，论文修订中
**当前分支**: codex（含共享距离网格优化 + 架构审核修复）