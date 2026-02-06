# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HIcosmo** (**H**igh-performance **I**nference for **Cosmo**logy) is a modern JAX-based cosmological parameter estimation framework targeting 5-10x performance improvements over traditional scipy implementations. Built with modern software engineering practices and designed for both CPU and GPU acceleration.

**Code Scale**: ~31,000 lines across 77 Python files (hicosmo/ directory)
**Tech Stack**: JAX + NumPyro + GetDist + Astropy
**Performance Goal**: Single calculation < 0.01ms, 5-10x faster than qcosmc

## Essential Commands

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=hicosmo

# Run specific test file
pytest tests/test_lcdm.py -v

# Run performance benchmarks
pytest tests/test_*_benchmark.py -v

# Skip slow tests
pytest -m "not slow" -v
```

### Development
```bash
# Install in development mode
pip install -e ".[dev]"

# Format code (auto-fixes)
black hicosmo/ tests/

# Sort imports
isort hicosmo/ tests/

# Type checking
mypy hicosmo/

# Linting
flake8 hicosmo/ tests/
```

### Running Examples
```bash
# Cosmology MCMC example
python examples/example_cosmology_mcmc.py

# SKA1 Fisher forecasts
python examples/run_ska1_forecasts.py

# H0LiCOW constraints
python example_h0licow_mcmc.py

# SH0ES analysis
python example_sh0es_mcmc.py
```

## Architecture Overview

### 6-Layer Hierarchical Design

#### Layer 1: Core Foundation (`hicosmo/core/`)
- **CosmologyBase** (base.py:449 lines): Abstract base class + shared distance calculations
  - `compute_distances_from_E_z()`: JIT-compiled distance engine (D_M, D_H, d_L, d_C, dVc_dz)
  - `make_compute_grid_traced()`: Factory for model-specific traced functions
  - Supports flat/open/closed universes via Omega_k
- **FastIntegration** (fast_integration.py:230 lines): Performance integration utilities
- **CosmologicalParameters** (unified_parameters.py): Centralized parameter management
  - Single source of truth for all parameters
  - Built-in validation and default values

#### Layer 2: Cosmological Models (`hicosmo/models/`)
- **LCDM** (lcdm.py:1,355 lines): Reference implementation, fully featured
  - Supports non-flat universes (Omega_k)
  - Sound horizon calculation (Eisenstein & Hu 1998)
  - Growth functions (Carroll, Press & Turner 1992)
  - Derived parameters: rd, rd_h, H0_rd
- **wCDM** (wcdm.py:4,495 lines): Constant dark energy equation of state
- **CPL** (cpl.py:4,431 lines): Chevallier-Polarski-Linder parameterization (w0, wa)
- **ILCDM** (ilcdm.py:10,205 lines): Interacting dark energy models

**Pattern**: Each model implements `_E_z_static()` and uses `make_compute_grid_traced()` factory

#### Layer 3: Likelihood System (`hicosmo/likelihoods/`)
- **PantheonPlus** (pantheonplus.py:22,113 lines): 1,701 SNe Ia with full covariance
- **BAO** (bao_datasets.py:22,341 lines): DESI 2024, SDSS, 6dFGS datasets
- **Strong Lensing**: H0LiCOW (h0licow.py:18,875 lines), TDCOSMO (tdcosmo.py:20,325 lines)
- **CMB**: Planck 2018 distance priors (planck_distance.py:5,533 lines)
- **H0**: SH0ES distance ladder (sh0es.py:1,630 lines)
- **Gravitational Waves**: Standard siren (gw_standard_siren.py:63,897 lines)

**⚠️ CRITICAL: Likelihoods must NOT implement cosmological calculations!** See Module Responsibility Rule below.

#### Layer 4: MCMC Sampling (`hicosmo/samplers/`, 5,616 lines total)
- **MCMC** (inference.py:1,416 lines): High-level dict-driven interface
- **ParameterConfig** (config.py): Parameter setup and validation
- **NumPyro Backend** (numpyro_backend.py): NUTS sampler wrapper
- **Persistence** (init.py): Checkpoint save/restore system

#### Layer 5: Fisher Matrix (`hicosmo/fisher/`)
- **IntensityMapping** (intensity_mapping.py:54,777 lines): 21cm IM Fisher forecasting
- **FisherMatrix** (fisher_matrix.py:23,784 lines): Autodiff-based exact Fisher matrix

#### Layer 6: Visualization (`hicosmo/visualization/`, 1,879 lines total)
- **Function Interface**: `plot_corner()`, `plot_chains()`, `plot_traces()`
- **GetDist Backend**: Professional publication-quality plots

---

## 🚨 Module Responsibility Rule - ABSOLUTE PRINCIPLE

**Each module has ONE job. Never cross boundaries!**

### 判断标准：计算的物理语义

| 模块 | 计算本质 | 输入 → 输出 |
|------|---------|------------|
| `models/` | **宇宙学预测**：从宇宙学参数推导宇宙的物理量 | (H0, Ω_m, w0, z) → (d_L, D_M, D_H, r_d, E(z)) |
| `likelihoods/` | **统计推断**：比较理论预测与观测数据 | (theory, data, covariance) → χ² |
| `samplers/` | **参数探索**：MCMC采样、链管理 | (prior, likelihood) → posterior samples |
| `visualization/` | **结果展示**：绘图、调用model计算派生参数 | (samples) → plots |

### 如何判断一段代码属于哪里？

问自己：**这段代码在计算什么？**

- **"从宇宙学参数计算某个物理量"** → `models/`
  - 例：从(H0, Ω_m, Ω_k, z)计算横向共动距离D_M
  - 例：从(Ω_b, Ω_m)计算声视界r_d
  - 例：从(H0, Ω_m, z)计算E(z) = H(z)/H0

- **"比较理论值和观测值"** → `likelihoods/`
  - 例：χ² = (data - theory)ᵀ C⁻¹ (data - theory)
  - 例：从观测的D_M/r_d和理论D_M/r_d计算残差

### ❌ VIOLATION EXAMPLE
```python
# In likelihoods/bao_datasets.py - WRONG!
class DESI2024BAO:
    def _transverse_distance(self, d_c, H0, Omega_k):
        # ❌ 这是"从宇宙学参数计算物理量"，属于models/！
        return D_H / sqrt_ok * sinh(sqrt_ok * delta)
```

### ✅ CORRECT PATTERN
```python
# In core/base.py - 宇宙学计算属于这里
def compute_distances_from_E_z(z_grid, E_z_grid, H0, Omega_k):
    # ✅ 从宇宙学参数计算物理量
    return {'D_M': D_M_grid, 'D_H': D_H_grid, ...}

# In likelihoods/bao_datasets.py - 只做"比较"
class DESI2024BAO:
    def theory(self, cosmology):
        grid = self._cosmology_class.compute_grid_traced(z, params)
        DM_grid = grid['D_M']  # ✅ 直接使用model的输出
        DH_grid = grid['D_H']  # ✅ 不重复实现！
```

### Why This Matters
1. **单一真相源** - 物理公式只在一个地方定义
2. **易于维护** - 修改公式只需改一处
3. **JIT优化** - model层统一优化
4. **可测试性** - 物理计算和统计推断分开测试

---

## ⚠️ API Design First Principle - Project Lifeline

**API Simplicity = Project Success**

```
API Simplicity → Lower Learning Curve → Higher User Adoption → Stronger User Retention → Project Success
```

**Design Standards**:
- ✅ **3 lines to core functionality** - Our API design target
- ✅ **Smart defaults** - 90% use cases need no extra configuration
- ✅ **Progressive complexity** - Simple tasks stay simple, complexity only for advanced use
- ❌ **Never add required parameters for "completeness"** - Each new parameter is a risk of user churn

---

## Critical Development Rules

### 🚨 Testing Rules (Most Important!)
- **NO REIMPLEMENTATION IN TESTS**: Tests must use existing modules, never reimplement!
- **TEST EXISTING CODE**: Purpose is to verify existing code works, not implement tested features!
- **USE EXISTING IMPORTS**: Import from hicosmo.models, hicosmo.likelihoods, hicosmo.samplers
- **NO TEST ADAPTERS**: Test production code directly, no wrapper layers!

### 🚨 Example Scripts Rules (ABSOLUTE!)
**Example的目的是测试HIcosmo，不是得到结果！绝对不能绕过HIcosmo另写一套！**

#### 1. 必须使用HIcosmo的并行系统
```python
# ❌ WRONG: 自己写ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(run_mcmc) for _ in range(4)]

# ✅ CORRECT: 使用HIcosmo的Config.init()
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
from hicosmo.samplers import Config, MCMC
Config.init(cpu_cores='auto', num_devices=8)
mcmc.run(num_samples=2000, num_chains=4)  # HIcosmo内部处理并行
```

#### 2. 必须使用HIcosmo的Plotter
```python
# ❌ WRONG: 手写matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(samples['H0'])

# ✅ CORRECT: 使用HIcosmo的Plotter
from hicosmo.visualization import Plotter
plotter = Plotter(chain_names, labels=['A', 'B'])
plotter.corner(['H0', 'Omega_m'], filename='corner.pdf')
```

#### 3. MCMC自动收集nuisance参数
```python
# ❌ WRONG: 手动检查和添加nuisance参数
nuisance = likelihood.nuisance_parameters()
for p in nuisance:
    params[p.name] = (p.value, p.prior['min'], p.prior['max'])

# ✅ CORRECT: MCMC自动处理，无需手动干预
mcmc = MCMC(params, likelihood)  # MCMC内部调用_collect_nuisance_from_likelihood_object()
mcmc.run()
```

#### 4. 核心原则
- **Example测试HIcosmo功能** - 不是得到论文的数值结果
- **绝不重复造轮子** - HIcosmo已有的功能必须使用
- **发现缺失功能** - 如果HIcosmo没有某功能，先实现到HIcosmo，再在example中使用

### Performance Standards
- **NO SLOW CODE**: Any calculation > 1ms must be optimized or rewritten
- **NO DIFFRAX**: Verified too slow, use FastIntegration or native JAX
- **NO NUMPY IN HOT PATHS**: Hot paths must use JAX for JIT compilation
- **BENCHMARK EVERYTHING**: New features require performance comparison tests

### Architecture Standards
- **NO DUPLICATE PARAMETERS**: Use unified CosmologicalParameters system
- **NO REDUNDANT MODULES**: Check for existing implementations before creating new ones
- **NO BLOATED BASE CLASSES**: Keep base classes minimal, specialize via inheritance
- **NO MIXED IMPORTS**: Import core components consistently from hicosmo.core

### Code Quality Standards
- **NO PARTIAL IMPLEMENTATION**: Either fully implement or don't do it
- **NO CODE DUPLICATION**: Check existing implementations first
- **NO DEAD CODE**: Remove unused code immediately
- **NO CHEATER TESTS**: Tests must reflect real usage scenarios

### 🚨 代码简洁性规则 - ABSOLUTE（2026-01-08 教训总结）

**背景**：用户写了一个15行的示例代码，Claude改写成了50+行的冗余版本，违反了HIcosmo的API设计哲学。

#### 核心原则：使用最简API，不要画蛇添足

```python
# ❌ WRONG: Claude的冗余写法
plotter = Plotter(samples, labels={'a': r'$a$', 'b': r'$b$', 'c': r'$c$'})
plotter.corner(['a', 'b', 'c'], filename='polynomial_fit_corner.pdf')
plotter.report(filename='polynomial_fit_report.md')

# ✅ CORRECT: 用户的简洁写法
plotter = Plotter('polynomial_fit')  # 直接用chain_name加载
plotter.corner()                      # 默认参数已经够用
plotter.report()                      # 不需要指定filename
```

```python
# ❌ WRONG: 使用不存在或复杂的API
mcmc = MCMC.from_simple_list(params, log_likelihood, chain_name='...')
mcmc.run(num_warmup=500, num_samples=2000, num_chains=8)
mcmc.print_summary()

# ✅ CORRECT: 使用标准API
mcmc = MCMC(params, log_likelihood, chain_name='polynomial_fit')
mcmc.run(num_samples=20000)  # 其他参数用默认值
```

#### 具体规则

| 场景 | ❌ 错误 | ✅ 正确 |
|------|--------|--------|
| 加载链 | `Plotter(samples, labels={...})` | `Plotter('chain_name')` |
| Corner图 | `plotter.corner(['a','b'], filename='x.pdf')` | `plotter.corner()` |
| 报告 | `plotter.report(filename='x.md')` | `plotter.report()` |
| 参数格式 | 列表 `[name, init, min, max, latex]` | 字典 `{name: (init, min, max, latex)}` |
| MCMC运行 | 指定所有参数 | 只指定 `num_samples`，其他用默认 |

#### 检查清单

1. **使用默认参数** - 如果默认值已经合理，不要显式指定
2. **使用最短API** - `Plotter('name')` 而不是 `Plotter(data, labels=...)`
3. **参考用户代码** - 用户写的代码就是最佳实践，严格遵循
4. **不要发明API** - 使用现有的API，不要假设存在 `from_simple_list` 等方法
5. **代码行数是质量指标** - 能用10行解决的问题，绝不写50行

---

## 🔬 Scientific Computing Rules - MANDATORY (从TDCOSMO教训总结)

**背景**：Claude写的TDCOSMO likelihood因违反以下规则导致完全错误的结果（H0精度18% vs 论文6%），Codex重写后才成功复现。

### 规则1：数值稳定≠加任意大的数

```python
# ❌ FATAL ERROR: "数值稳定"加了物理上荒谬的值
cov_total += jnp.eye(n) * (1e-4 * c_km_s**2)  # = 9×10^6 km²/s²，相当于3000 km/s误差！

# ✅ CORRECT: 真正的数值稳定级别
cov_total += jnp.eye(n) * 1e-6  # 对sigma_v~200 km/s可忽略
```

**规则**：添加任何"稳定项"前，必须计算其物理量级，确保远小于物理量的典型值。

### 规则2：必须对齐参考实现

```python
# ❌ WRONG: 读论文 → 自己发明实现
def my_marginalization():
    # 5×5×N Gauss-Hermite三重循环（自创）
    for lam in hermite_nodes:
        for ani in hermite_nodes:
            for kappa in kappa_bins:
                ...

# ✅ CORRECT: 读论文 → 找官方代码 → 严格对齐
def aligned_marginalization():
    # Match hierarc: num_distribution_draws=200, Latin-Hypercube抽样
    n = 200
    base = (jnp.arange(n) + 0.5) / n
    # 代码中注释清楚对齐点
    # hierarc uses `num_distribution_draws=200` for LOS sampling
```

**规则**：
1. 实现前必须找到参考实现（hierarc, cobaya, CosmoMC等）
2. 代码中用`# Match {reference}: ...`注释标注对齐点
3. 测试时必须与参考结果数值一致，不是"差不多"

### 规则3：JAX向量化是必须的，不是优化

```python
# ❌ FATAL: Python循环 → 10秒/iteration，MCMC无法运行
total = 0.0
for lens in self.lens_data.values():
    total += self._single_lens_loglike(lens)

# ✅ CORRECT: vmap向量化 → 毫秒级
loglike = jax.vmap(self._lens_loglike)(all_lens_data)
```

**规则**：任何`for item in collection`循环都是红旗，必须问"能不能用vmap？"

### 规则4：理解物理量级

在写代码前，必须列出所有物理量的典型值：

| 物理量 | 典型值 | 单位 |
|--------|--------|------|
| sigma_v | 200-300 | km/s |
| D_dt | 1000-5000 | Mpc |
| kappa_ext | 0.0-0.1 | - |
| lambda_int | 0.8-1.2 | - |

**规则**：任何数值操作（加减乘除、截断、稳定项）都要检查是否在物理合理范围内。

### 规则5：单函数完整流程 > 优雅抽象

```python
# ❌ BAD: 过度分层，每层都可能有bug
def _marginalize_lambda(): ...
def _marginalize_ani(): ...
def _marginalize_kappa(): ...
def _integrated_loglike():
    return _marginalize_lambda(_marginalize_ani(_marginalize_kappa(...)))

# ✅ GOOD: 一个函数看到完整流程
def _integrated_lens_loglike_draws(self, lens, ddt, dd, ...):
    # 1. 生成所有draws
    lambda_draws = self._normal_draws(loc, scale, q_lam)
    ani_draws = self._truncated_normal_draws(...)
    kappa_draws = self._kappa_draws(lens, q_kappa)

    # 2. 计算中间量
    lambda_tot = lambda_draws * (1.0 - kappa_draws)
    ...

    # 3. 计算likelihood
    loglike = ddt_log + kin_log

    # 4. 聚合
    return logsumexp(loglike) - jnp.log(n_draws)
```

**规则**：科学计算中，"一眼能看到完整流程"比"优雅的抽象"更重要。

### 规则6：测试必须验证数值结果

```python
# ❌ BAD: 只测试"能跑"
def test_tdcosmo():
    result = tdcosmo(H0=70, Omega_m=0.3)
    assert result is not None  # 毫无意义

# ✅ GOOD: 测试数值结果与参考一致
def test_tdcosmo_matches_hierarc():
    result = tdcosmo(H0=73.8, Omega_m=0.3, ...)
    # 与hierarc官方链对比
    assert abs(result - hierarc_loglike) < 0.1

def test_h0_precision():
    # MCMC结果必须与论文一致
    h0_std / h0_mean < 0.07  # 论文: ~6%
```

### 检查清单（每次提交前）

- [ ] 所有数值常数都有物理意义注释
- [ ] 没有Python循环处理数组（用vmap）
- [ ] 与参考实现对齐点有`# Match {ref}: ...`注释
- [ ] 测试验证数值结果，不只是"能跑"
- [ ] 性能：单次likelihood < 10ms

---

### 🔪 AI Code Bloat Removal (Critical!)

#### 1. Useless Comments (废话文学)
```python
# ❌ Bad: Explaining the obvious
score = score + 1  # Increment the score by 1

# ✅ Good: Only comment non-obvious physics/algorithms
# Eisenstein & Hu 1998 fitting formula for sound horizon
r_s = 44.5 * log(9.83 / omega_m) / sqrt(1 + 10 * omega_b**0.75)
```

#### 2. Paranoid Defensive Checks (被害妄想症)
```python
# ❌ Bad: Redundant validation in internal functions
def _compute_E_z(z, params):
    if params is None:  # Caller already validated!
        raise ValueError("params cannot be None")

# ✅ Good: Trust internal contracts, validate only at boundaries
def _compute_E_z(z, params):
    Omega_m = params['Omega_m']  # Trust the caller
```

#### 3. Type Escape Hatches (类型逃逸)
```python
# ❌ Bad: Silencing type errors with Any/cast
result: Any = complex_function()

# ✅ Good: Proper type definitions
result: CosmologyResult = compute_grid_traced(z, params)
```

#### 4. Style Inconsistency (画风突变)
```python
# ❌ Bad: Mixed naming conventions
def calculate_luminosity_distance(z):
    dL = compute_dL(z)  # camelCase variable in snake_case function

# ✅ Good: Consistent style
def luminosity_distance(z):
    d_L = self._compute_distance(z)
```

---

## Performance Benchmarks

| Operation | qcosmc (scipy) | HIcosmo (JAX) | Speedup |
|-----------|----------------|---------------|---------|
| Distance calculation (1000 pts) | 0.15s | 0.02s | **7.5x** |
| MCMC sampling (10k samples) | 180s | 45s | **4.0x** |
| Fisher matrix | 2.1s | 0.5s | **4.2x** |
| BAO likelihood (JIT) | 21.8ms | 1.24ms | **17.6x** |

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `hicosmo/core/base.py` | 449 | CosmologyBase + compute_distances_from_E_z |
| `hicosmo/models/lcdm.py` | 1,355 | Reference LCDM implementation |
| `hicosmo/samplers/inference.py` | 1,416 | MCMC high-level API |
| `hicosmo/core/unified_parameters.py` | - | Parameter management system |
| `hicosmo/likelihoods/bao_datasets.py` | 22,341 | BAO likelihood (DESI, SDSS, 6dFGS) |

## Common Pitfalls

1. **Don't mix NumPy and JAX**
   ```python
   # ❌ Bad: Mixed NumPy/JAX
   result = jnp.sqrt(np.array([1, 2, 3]))

   # ✅ Good: Pure JAX
   result = jnp.sqrt(jnp.array([1, 2, 3]))
   ```

2. **Don't modify arrays in-place**
   ```python
   # ❌ Bad: JAX arrays are immutable
   arr[0] = 5

   # ✅ Good: Create new array
   arr = arr.at[0].set(5)
   ```

3. **Don't use Python loops for array operations**
   ```python
   # ❌ Bad: Python loop
   results = [model.E_z(z, params) for z in z_array]

   # ✅ Good: Vectorized
   results = jax.vmap(lambda z: model.E_z(z, params))(z_array)
   ```

## Module Status

### ✅ Production-Ready
- Core cosmology (CosmologyBase, LCDM, wCDM, CPL, ILCDM)
- Distance calculations with curvature support (D_M, D_H)
- MCMC sampling framework (NumPyro NUTS)
- Visualization system (GetDist backend)
- Likelihoods: PantheonPlus, BAO (DESI 2024), H0LiCOW, SH0ES

### 🚧 In Development
- Fisher matrix forecasting (21cm intensity mapping)
- Gravitational wave standard siren likelihoods
- Advanced MCMC diagnostics

## Success Criteria for New Code

- ✅ Performance test passes, beats competitors
- ✅ Unified architecture test passes
- ✅ Code is concise, no duplication
- ✅ Clean import structure
- ✅ Complete test coverage
- ✅ Type annotations on all functions
- ✅ NumPy-style docstrings

## Notes

- This is an active research codebase with frequent updates
- Performance is non-negotiable - benchmark everything
- When in doubt, check LCDM implementation as reference
