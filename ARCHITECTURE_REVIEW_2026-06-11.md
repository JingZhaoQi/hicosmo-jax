# HIcosmo 架构审核报告

> **修复状态（2026-06-11 当日完成）**：审核当日已按本报告完成第一、二、三批全部修复——
> 5 个 P0（异常吞噬、gw 导入链与 x64 副作用、CLI 示例、工作区风险中的 CI 部分）、
> 11 个 P1（Omega_b 注入、rd 分叉、装饰器防护、BBN prior、M_B wrapper、未知参数检测、
> emcee 先验、Fisher 死安全网、warmup 均摊、progress_bar 透传、一致性测试缺失）、
> 死代码清理与文档同步（CLAUDE.md、规则文档、docs_zh/docs_en）。
> 性能优化实测：联合似然梯度 3.86 → 2.32 ms（−40%，interp_linspace）。
> 验证：112 个测试全过 + 四路径冒烟（用户标准 API / hicosmo() / YAML runner / 性能基准）全过 +
> 模拟新克隆（无 data、无 gw）import 与 CI 子集通过 + 中英文档 Sphinx 构建 0 错误。
> **残留**（需要决策或分批进行）：git 提交与推送（用户决策）、数据分发通道（P0-5，需 Zenodo）、
> 第四批架构收敛（SN 基类提取、入口统一、MCMC god class 拆分）。

**日期**：2026-06-11
**审核对象**：`hicosmo/` 包（约 34,200 行、76 个 Python 文件），分支 `codex`（领先 `main` 6 个提交，另有 28 个文件未提交修改、12 个未跟踪新文件）
**审核方法**：六个维度的静态代码审查（核心模型层、似然层、采样与 Fisher 层、API 层、测试与工程化），叠加在本机（Apple Silicon，JAX 0.4.33 CPU，x64）的运行时实测：单点似然与梯度基准、梯度成本分解、优化方案验证、端到端 MCMC 吞吐。所有发现均带 `file:line` 引用；标注"实测"的数字来自真实运行，标注"估算"的为解析推导。

---

## 一、执行摘要

HIcosmo 的**核心架构是对的，而且执行得很好**："子类只写 $E(z)$ 物理 + 装饰器自动生成 traced 接口"让新模型只需 60–110 行；NUTS 调用链从 NumPyro model 到 `CombinedLikelihood` 再到各 JIT 闭包，全部编排都发生在 trace 期，编译后采样循环是纯 XLA，没有逐步 Python 开销。这两点是框架性能声明的真实根基，值得保持。96 个现有测试全部通过（14.4 秒）。

但本次审核同时确认了 **5 个 P0、约 14 个 P1** 问题。最危险的三类：

1. **静默错误结果**：似然异常被吞后 NUTS 会"成功地"采样先验而无任何提示；`bbn_prior` 模式下 BBN 先验在三条求值路径中分别被计入 0/1/2 次；固定 `M_B` 的设置在共享网格路径下被静默丢弃；拼错宇宙学参数名（如对 wCDM 传 `w` 而非 `w0`）被静默忽略且项目自己的规则文档教的就是错误拼法。
2. **仓库不可用**：新克隆 `import hicosmo.likelihoods` 必然 `ImportError`（gw 模块被 gitignore 却被已跟踪文件导入）；CI 已被删除；6.4 GB 数据无任何下载通道；83/96 个测试函数未提交。**当前整个 DESY5/Union3/共享网格功能只存在于这台机器的工作区里。**
3. **性能优化方向偏移**：实测共享距离网格在梯度路径只节省 3%。真正的瓶颈是 Pantheon+ 似然（占联合梯度成本 84%），其中 `jnp.interp` 的 AD 反向传播一项就占约一半。本报告验证了一个 10 行的替代实现，SN 梯度立省 28% 且数值逐位一致（见 §3.4）。

各维度评分（1–5）：

| 维度 | 评分 | 一句话评价 |
|------|------|-----------|
| 运算速度（热路径设计） | 4.5 | trace 期编排纪律贯彻彻底，实测单点梯度 3–4 ms 量级 |
| 运算速度（可挖掘空间） | — | interp/matvec/warmup 三项合计还有约 1.5–2 倍 ESS/s 空间 |
| 架构与模块化 | 4.0 | 模型层优秀；似然层与采样层有系统性重复 |
| 可扩展性：新模型 | 5.0 | 60–110 行，零距离代码 |
| 可扩展性：新数据集 | 2.5 | 新 SN 数据集需复制约 600 行模板、改 4 个文件 |
| 可扩展性：新采样后端 | 2.0 | 名义上"继承+注册"，实际有 3 处隐藏耦合 |
| 易用性 | 3.0 | 单条路径达成"3 行"，但五条入口并行、官方示例不可运行 |
| 正确性保障（测试） | 2.5 | 新测试质量高（A 级），但 2/3 子系统零测试 |
| 工程化（CI/打包/分发） | 1.5 | 无 CI、克隆即崩、数据不可获取 |

---

## 二、P0 问题（立即处理）

### P0-1 似然异常被吞，NUTS 静默采样先验

`samplers/inference.py:867-941`。`log_probability` 包装器的 `try/except Exception` 在 JAX trace 期执行：似然函数抛出任何异常（shape 错误、`ConcretizationTypeError`、数据缺失、用户 bug）都被捕获并返回常数 `-1e10`，`numpyro.factor` 收到常数后**后验恒等于先验**。初始化不会失败，采样"成功"完成，`verbose=False` 时零提示。这同时架空了 `numpyro_backend.py:301-354` 写得很好的三类错误诊断（那些分支永远不可达）。

**修复**：删除 trace 期的 try/except，让异常自然传播到 backend 的错误分类器；或在 `_setup_unified_backend` 时先用 ref 值做一次 eager 探测调用，失败立即抛错。约 20 行改动。

### P0-2 新克隆仓库必然 ImportError

导入链：`likelihoods/__init__.py:73` → `lensing/__init__.py:28` → `hierarchical_helper.py:8`（`from ..gw.standard_siren import ...`）。而 `.gitignore:34` 忽略整个 `hicosmo/likelihoods/gw/`，`git ls-files` 确认 0 个 gw 文件被跟踪（`main` 分支最新提交 5e7171b 就是 "Remove GW module from repo"）。任何人克隆仓库后 `import hicosmo.likelihoods` 直接崩溃。

**连锁后果（P1-1）**：全库唯一的 float64 开关藏在 `gw/population_rate.py:13`（`jax_config.update("jax_enable_x64", True)` 模块级副作用）。实测确认：`import jax` 后 x64 为 False，`import hicosmo.likelihoods` 后变 True。一旦 gw 模块缺失或改为惰性导入，整个项目静默退回 float32，代码中所有 `dtype=jnp.float64` 请求被静默截断（仅发 UserWarning），宣称的与 Cobaya $\chi^2 < 10^{-6}$ 交叉验证不再成立。`fisher/fisher_matrix.py:30` 还在 import 时把 dtype 固化成常量，使 Fisher 的精度取决于 import 顺序。

**修复**：在 `hicosmo/__init__.py` 顶部显式 `jax.config.update("jax_enable_x64", True)`（在任何 JAX 操作前）；`hierarchical_helper` 改为延迟导入或移出 `lensing/__init__.py`；fisher 的 dtype 改为读取时求值。

### P0-3 仓库状态：功能资产大面积游离于版本控制之外

实测统计：`codex` 领先 `main` 6 个提交；工作区另有 28 个文件未提交（+958/−492 行）；12 个未跟踪文件包含 `api_registry.py`、`desy5.py`、`union3.py`、`parameters/setup.py`、`samplers/derived.py`、`samplers/options.py` 及 6 个测试文件（83/96 个测试函数）。`.github/` 目录不存在（提交 29f4d25 证明 CI 曾存在后被删）。一次误操作（`git checkout .`、磁盘故障）即可丢失全部近期工作。

**修复**：立即按功能拆分提交全部工作区内容；恢复最小 CI（lint + 干净环境 import 冒烟 + 不依赖数据的测试子集）。

### P0-4 CLI 官方 Quick Start 示例不可运行

`cli.py:77-84` 的 banner 教用户执行 `inference.summary()` 和 `inference.corner_plot(...)`，但 `hicosmo()` 返回的 `MCMC` 类（`samplers/inference.py:75`）只有 `print_summary`，没有这两个方法（grep 全文件确认）。新用户的前四行代码必然 `AttributeError`。这是把 `InferenceRunner` 的方法面误当成了 `MCMC` 的。

**修复**：让 `hicosmo()` 返回 `InferenceRunner`，或给 `MCMC` 加 `summary`/`corner_plot` 委托方法；把 banner 示例纳入测试。

### P0-5 数据无分发通道，可复现性为零

`runner/datasets.py:16-77` 注册表中所有数据集 `url=None`、`sha256=None`，`ensure_dataset` 的下载分支是永远走不到的空架子；6.4 GB 的 `hicosmo/data` 不在版本控制。论文发表后读者无法复现任何结果。

**修复**：KB 级小数据（DESI 13 个点、Union3 22 bin、Planck 距离先验）直接入库；大协方差矩阵传 Zenodo 并填入 URL + sha256。

---

## 三、运算速度专项（重点）

### 3.1 实测基准

环境：Apple Silicon CPU、JAX 0.4.33、float64。所有 per-call 均为 JIT 预热后均值。

**单点似然与梯度**（CPL，$d=4$；LCDM/wCDM 数值相近）：

| 组件 | 前向 | 梯度（`value_and_grad`） | 编译时间 |
|------|------:|------:|------:|
| SN（Pantheon+，1580 SNe） | 0.68 ms | 3.24 ms | 110–250 ms |
| BAO（DESI DR2） | 0.18 ms | 0.63 ms | ~105 ms |
| CMB（Planck distance priors） | 0.16 ms | 0.24 ms | ~60 ms |
| **联合（共享网格路径）** | **0.84 ms** | **3.86 ms** | ~210 ms（前向）/ 660–805 ms（梯度） |
| 联合（独立网格 naive sum） | 0.94 ms | 3.99 ms | — |

**端到端 MCMC**（`hicosmo()` 顶层 API，2000 总样本、2 链、默认配置，含编译与 warmup 的总墙钟）：

| 配置 | 维度 | 墙钟 | min ESS | min ESS/s |
|------|---|------:|------:|------:|
| LCDM + SN | 2 | 48.2 s | 1528 | 31.7 |
| wCDM + SN+BAO+CMB | 3 | 209.1 s | 811 | 3.9 |

说明：CLAUDE.md 记录的基线（LCDM+SN 282、wCDM joint 14.2 ESS/s）与本测口径不同（链数、样本量摊销、硬件、warmup 配置均不同），两组数字不能直接比较；本表的价值在于内部相对关系与下文的成本分解。

### 3.2 瓶颈分解：钱花在哪里

对 SN 似然梯度（LCDM，$d=2$）逐段累加实测：

| 阶段 | 梯度耗时（累计） | 增量 |
|------|------:|------:|
| 距离积分（2761 点网格，cumtrapz） | 0.09 ms | 0.09 ms |
| + `jnp.interp` 到 1580 个 SN 红移 | 1.64 ms | **+1.55 ms** |
| + 协方差 matvec（$r^\top C^{-1} r$） | 2.85 ms | +1.21 ms |
| （实际 SN 似然梯度，交叉验证） | 2.78 ms | 吻合 |

三个结论：

1. **距离积分网格根本不是成本**（0.09 ms，占 3%）。CLAUDE.md 中"网格密度不可压缩"的教训依然成立（那是精度约束），但据此推断"积分是性能瓶颈"不成立。
2. **`jnp.interp` 的 AD 反向传播是最大单项**（约 1.55 ms，占 SN 梯度的一半以上）。原因：`jnp.interp` 反向是对网格数组的 scatter-add，且前向的 `searchsorted` 不知道 `z_grid` 是等距 linspace。
3. **协方差 matvec 是第二大项**（约 1.2 ms）。$1580 \times 1580$ 的 float64 矩阵 20 MB，前向一次读取 + 反向两次，CPU 内存带宽决定下限。这解释了为什么 GPU 加速能到 30 倍：matvec 与 scatter 恰是 GPU 最擅长的。

### 3.3 共享距离网格优化的再评估

实测（CPL joint，共享网格确认已激活，`_shared_grid_enabled=True`，共享网格 5660 点）：

- 前向：共享 0.955 ms vs 独立 1.070 ms，**节省 11%**；
- 梯度：共享 3.864 ms vs 独立 3.994 ms，**节省 3%**；
- 代价：共享路径与独立路径的 $\log L$ 相差 $1.25 \times 10^{-3}$（SN 从 5660 点共享网格插值而非自己的 2761 点网格）。

这与成本分解一致：共享网格消除的是重复的距离积分（三次变一次），但积分只占总成本的百分之几。CLAUDE.md 把 +31–66% 的 ESS/s 提升归因于共享网格，但其中 LCDM+SN（单似然，无共享可言）也提升了 66%，说明那批收益主要来自同期的其他改动。**共享网格机制本身换来的约 3%，却引入了三条求值路径的一致性维护负担，而这正是 §四 中两个 P1 数值 bug 的温床。** 建议：保留机制但停止向其投入，新似然不必强制实现 `_loglike_from_grid`；优化资源转向下述方向。

### 3.4 优化建议（按投入产出排序）

**(1) 预计算插值索引替代 `jnp.interp`（已验证，建议立即做）**

`z_grid` 与 SN 红移在构造期都已固定，插值索引和权重可一次性预计算，热路径只剩两次 gather：

```python
# 构造期（likelihood __init__）
idx = np.clip(np.searchsorted(z_grid, z_cmb) - 1, 0, len(z_grid) - 2)
w = (z_cmb - z_grid[idx]) / (z_grid[idx + 1] - z_grid[idx])
# 热路径
d_L = dl_grid[idx] * (1.0 - w) + dl_grid[idx + 1] * w
```

本机实测：SN 梯度 2.74 → 1.99 ms（**−28%**），$\chi^2$ 与梯度和 `jnp.interp` 逐位一致。推广到联合似然估算梯度 3.86 → 约 3.1 ms，ESS/s 约 +25%。适用位置：`pantheonplus.py`、`union3.py`、`desy5.py`、`bao/base.py` 的所有 `jnp.interp` 调用点（共享网格路径下 `grid_z` 由 `CombinedLikelihood` 统一构造，索引可在首次构造共享网格时缓存）。每处约 10 行。

**(2) 修复 warmup 按链均摊（正确性兼性能，P1）**

`samplers/inference.py:762-799` 把 `num_warmup` 当总预算除以链数：默认 2000 样本 4 链时每链仅 100 步 warmup，远低于 NumPyro 默认的 1000。warmup 是每条链独立的 mass-matrix/step-size 适配，统计上不可摊薄；不足的 warmup 直接恶化 $\tau$，浪费全部后续采样。框架精心选择的 dense mass matrix（$d \le 20$，`numpyro_backend.py:236-243`，好设计）在 100 步窗口里根本完不成适配。**修复后 d≥3 联合分析的 ESS/s 预期改善显著**（本测 wCDM joint 的 3.9 ESS/s 部分就吃了这个亏）。改为 per-chain 语义、默认 ≥500/链。

**(3) 非交互场景默认关闭进度条**

NumPyro 的进度条路径每步触发一次 host callback（NumPyro 自己注明 `progbar=False will be faster`，对 ~1 ms/步的似然估 5–20% 开销）。实测发现顶层 `hicosmo(..., progress_bar=False)` 传入后**并未生效**，进度条照常输出（kwargs 透传链路断裂，属易用性 bug 兼性能问题）。建议：修复透传；`sys.stdout` 非 TTY 时自动关闭。

**(4) 协方差路径换 Cholesky 白化 + 评估 float32 存储（候选，需验证 $\tau$）**

当前 `pantheonplus.py:268` 预计算 `jnp.linalg.inv`。建议改为预计算 Cholesky 因子 $L^{-1}$（白化矩阵 $W$），热路径 $\chi^2 = \lVert W r \rVert^2$：数值上更稳健（对称正定保证），成本相同。进一步可试验 $W$ 用 float32 存储（20 MB → 10 MB，matvec 带宽减半，SN 梯度估再省 ~0.5 ms）；$\chi^2 \sim 700$ 下 float32 的相对误差 $\sim 10^{-4}$，对 NUTS 大概率无害，但**必须按项目既有规矩用 $\tau$ 实测验证后再合入**。

**(5) `run_chunked` 的 chunk 尺寸量化（长跑场景）**

`numpyro_backend.py:556-569` 按吞吐动态估算 chunk 尺寸并直接改 `mcmc.num_samples`，形状变化导致整个 `fori_collect` 循环重编译，每 chunk 付 10–30 s。默认 checkpoint 间隔 600 s 下，超过 10 分钟的 run 每个 chunk 都可能重编译。修复：chunk 尺寸量化到固定档位（2 的幂），保证编译缓存命中。

**(6) 合并模型无关的重复 JIT 单元（启动时间）**

`sound_horizon_traced` 等 3 个函数物理上与模型无关，却按模型类各生成一份（4 模型 × 3 = 12 个相同编译单元，`models/base.py:870-874`）；LCDM 实例还各自携带 2 个逐实例 jit（`lcdm.py:124-128`）。改为模块级单例可削减启动编译时间。

**优化路线预期合计**：(1)+(2)+(3) 三项低风险改动后，d=3–5 联合分析的 ESS/s 预期提升 1.5–2 倍；(4) 验证通过再加 15–20%。GPU 部署收益（已实测 30–35×）与上述正交。

---

## 四、正确性风险（P1，按危害排序）

以下问题不影响"默认配置的主路径"（LCDM/wCDM/CPL + Pantheon+ marginalized + DESI h0rd + CMB 的联合采样经核查是正确的），但每一条都会在特定配置下静默产出错误科学结果。

| # | 问题 | 位置 | 危害 |
|---|------|------|------|
| 1 | **BBN prior 三路径计数 0/1/2 次**：`__call__` 双重计入、`log_likelihood_from_params` 计 1 次、共享网格闭包计 0 次 | `likelihoods/bao/datasets.py:639-646, 704-744, 406-447` | `omega_b_mode='bbn_prior'` 时单独采样后验过紧、联合采样丢失 BBN 约束 |
| 2 | **固定 M_B 被共享网格绕过**：`_FixedMBLikelihood` 只在 `__call__` 注入固定值，`CombinedLikelihood` 经 `__getattr__` 直接拿到底层闭包，回落到硬编码默认 −19.3 | `likelihoods/sn/factory.py:19-67` + `combined.py:181-191` + `pantheonplus.py:541` | 用户设定的 `M_B=-19.5` 被静默丢弃 |
| 3 | **拼错参数名静默忽略**：实测 `sn(H0=70, Omega_m=0.3, w=-0.5)` 与不传 `w` 返回完全相同的 $\log L$，无警告；而 `.claude/rules/cosmology-model-architecture.md:19` 的示例教的就是 `w=-1.0`（实现只认 `w0`）。顶层 MCMC 路径有 registry 拦截（报 `KeyError: w` 但不提示 `w0`），直接调用似然/模型时完全静默 | `models/wcdm.py` + 规则文档 | 模型对比、调试、外部脚本中极易拿到错误结果 |
| 4 | **Omega_b 注入破坏 CMB 参数基**：`LCDM(h=0.70, omega_b_h2=0.0224, ...)` 时默认 `Omega_b=0.0493` 被当作用户值写入，反向推导被跳过（正确值 0.0457，错 8%），下游 $r_d$ 与 $\omega_b h^2$ 观测量用错值。Planck 默认值数值巧合掩盖了它 | `models/lcdm.py:101` | 任何非默认 $h$ 或 $\omega_b h^2$ 的 CMB 分析 |
| 5 | **声视界 $r_d$ 双实现已分叉 2.7%**：MCMC 路径用带 `CAMB_CALIBRATION=0.9736` 的版本，`derived_parameters()` 在 CAMB 缺席时回退到无校准版，后处理报告的 `rd/rd_h/H0_rd` 与采样实际使用值系统性差约 4 Mpc | `models/lcdm.py:286-314` vs `utils/jax_tools.py:539-579` | 派生参数表静默偏差 |
| 6 | **装饰器两类静默失效**：子类 `E_z` 忘写 `@staticmethod`，或忘加 `@register_cosmology_model`，都会静默继承 **LCDM 的物理**做 MCMC，无任何报错 | `models/base.py:830-847, 866-869` | 新模型开发最常见的笔误直接产出错误科学结果 |
| 7 | **emcee 后端忽略非均匀先验**：`_log_prior` 对所有分布只做边界检查返回 0.0，normal/truncnorm 先验密度从不计入；同一配置两个 backend 给出不同后验 | `samplers/emcee_backend.py:141-159` | backend 可互换契约被破坏 |
| 8 | **Fisher 模块安全网全为死代码**：JAX 无 `jnp.linalg.LinAlgError` 属性，5 处 except 永不触发，奇异矩阵静默产出 `inf` 误差；`get_fisher_summary` 因 `float(complex)` 对任何输入必崩（说明从未被端到端跑过） | `fisher/fisher_matrix.py:282, 388, 411, 512, 524, 556-563` | Fisher 预测不可信 |
| 9 | **`update_parameters` 后实例距离方法返回陈旧结果**：逐实例 jit 闭包把参数烘焙为编译常量 | `models/lcdm.py:124-128` + `base.py:208-215` | 休眠地雷（库内暂无调用方） |
| 10 | **`resume`/`continue_sampling` 语义冲突**：续跑请求被 checkpoint 的 remaining 覆盖可致 no-op；非 resume 实例上旧样本被直接丢弃且固定 `seed=42` 使"新链"与旧链完全相同 | `samplers/inference.py:1308-1329, 1886-1952` | 检查点功能不可靠 |
| 11 | **共享网格协议无系统性一致性测试**：全仓唯一的数值一致性断言是 union3 单参数点 1e-4 容差；`marginalize_M_B=False`、`include_shoes`、`bbn_prior`、固定 M_B 等分支全部未覆盖，上表 #1、#2 正是漏网之鱼；CLAUDE.md 列为硬规则的梯度一致性（$\nabla \log L$）没有任何测试 | `tests/test_combined_likelihood.py`（全 mock） | 协议靠自觉维护，必然再次漂移 |
| 12 | **`compute_DM_at_z` 死代码带精度陷阱**：全库零调用（CLAUDE.md 却记录为已上线的"CMB 轻量路径"），其默认 1024 点均匀网格对 $\int_0^{1090} dz/E(z)$ 的端点误差致 $D_M(z_*)$ 高估约 1.4%（估算），任何人按 docstring 使用即中招 | `models/base.py:461-499` | 文档与代码漂移 + 诱导性 API |

其余值得知晓的 P2：DESY5 对**逆**协方差按 z-cut 掩码切片在非默认 cut 下统计错误（条件化≠边际化，`desy5.py:227`）；H0LiCOW 的 $D_{ds}$ 公式仅平直宇宙成立但接口接受 `Omega_k`（`h0licow.py:426`）；三个 SN 似然的归一化常数口径互不一致（影响跨数据集 AIC/BIC）；`age` 近似公式缺 $\mathrm{arcsinh}$ 因子（`summary()` 展示 11.7 Gyr，真值 13.8）；增长率 $\gamma$ 系数 0.0055 疑为 Linder 2005 的 0.05 笔误（`lcdm.py:566`）；`print_summary` 的 R̂ 过滤键名不匹配导致收敛状态从不显示（`inference.py:1132-1144`）；R̂ 在扁平化样本上计算丢失多链信息；TDCOSMO 行为依赖项目根的未跟踪目录存在与否（`likelihoods/__init__.py:244-252`）。

---

## 五、架构与模块化

### 做对了的（应当保持）

1. **E_z-only 子类 + 工厂闭包**（`models/base.py:393-503, 802-876`）：物理与框架彻底分离，wCDM 全文件 62 行、物理核 18 行。这是整个框架最有价值的设计资产。
2. **trace 期编排纪律**：从 numpyro model（prior 循环、dict 组装）到 `CombinedLikelihood` 分发再到 `_loglike_from_grid` 闭包，运行期零 Python 回调（进度条除外）。
3. **`compute_omega_r` 拒绝 $\Omega_r = 0$ 默认**（`base.py:24-59`）：注释明确 $z_*$ 处辐射占 $E^2$ 约 24%，强制从 $T_{\rm cmb}/N_{\rm eff}$ 计算且对 $H_0$ 可微，避免了一类 CMB 距离灾难。
4. **教训制度化**："2048 点网格使 $\tau$ 从 3.5 涨到 14.7"被钉在 `combined.py:117-119` 的注释里，revert 留痕。
5. **BAO 观测量向量化**：`obs_codes` + `jnp.where` 链让 5 种异构观测量在单一 JIT 闭包内零分支处理（`bao/base.py:312-318`）；DESI `h0rd` 参数化直接采样 $H_0 r_d$，物理干净。
6. **链文件元数据闭环**：labels/ranges 随 HDF5 落盘，`Plotter('chain_name')` 一行回填；run manifest 记录版本与设备（`utils/manifest.py`）。

### 需要收敛的（系统性重复）

| 重复 | 位置 | 规模 |
|------|------|------|
| 三个 SN 似然的模板复制（数据加载、网格、双闭包、`__call__` 骨架） | pantheonplus ↔ desy5 ↔ union3 | ~300 行逐行同构 + ~120 行 |
| tdcosmo 两套并行统计工具（normal/truncnorm draws、kappa、积分） | `TDCOSMOLikelihood` ↔ `ExternalLensLikelihood` | 500–800 行可提取 |
| 曲率 $D_M$ 变换 | `base.py` 4 处 + `lcdm.py` 2 处 | 6 份同一公式 |
| $H_0 \leftrightarrow h$、$\Omega \leftrightarrow \omega h^2$ 换算 | `normalize_params` / `ParameterRelations` / `_compute_additional_derived` | 3 套（P1-4 的 Omega_b bug 即源于此） |
| BAO 数据集注册 | factory、`get_available_datasets`、两个 dataset_map | 4 处已失同步（`BAOCollection` 漏了 `desi_dr2`） |
| nuisance 收集 | `inference.py:385-451` vs `453-543` | ~80 行近重复 |
| 距离模数公式 $5\log_{10}[(1+z_{\rm cmb})(1+z_{\rm hel}) d_A] + 25$ | SN 文件内 | 6 处 |
| 旧共享网格协议（`_precomputed_grid` kwargs）残留 | 4 个文件 + `make_compute_shared_grid` 79 行 | 全部死代码，与新协议并存误导扩展者 |

`samplers/inference.py`（2022 行）是确认的 god class：参数配置、双路 nuisance 收集、优化初始化、checkpoint、resume、保存、诊断展示十余项职责集于一身，170 行的 `BackendWrapper` 定义在方法体内。新拆出的 `derived.py`/`options.py` 方向正确，应继续把 persistence 编排和诊断展示拆出去。`models/unified_parameters.py` 与 `hicosmo/parameters/` 构成两套参数系统并存（外加 `samplers/config.py` 的第三套表达），其中 `parameters/collector.py`（473 行）全库零引用，而 `lcdm.py:688` 的 docstring 还在教用户 import 它（执行即 ImportError）。

---

## 六、可扩展性

**新宇宙学模型：优秀**。继承 LCDM、写 `@staticmethod E_z(z, params)`、加装饰器，60–110 行完成，距离/积分代码零行。两处需要修补：装饰器的静默失效模式（§四 #6，建议 `cls.__dict__` 检查 + `__init_subclass__` 告警）；LCDM 自己绕过装饰器手动接线（`lcdm.py:1416-1434`），两处必须同步演化，应统一走装饰器。

**新数据集：差**。新增一个 SN 数据集要复制约 600 行模板、改 4 个文件；BAO 有 4 处注册点且已失同步。建议提取 `SNDistanceModulusLikelihood` 中间基类（参数化 marginalize/z_hel/双模式闭包工厂），新 SN 数据集可缩至约 80 行纯数据描述；BAO 注册收敛到 `factory._BAO_DATASETS` 单一真相源。

**新似然探针：中等**。`Likelihood` 基类 + `nuisance_parameters()` 协议清晰，fallback 安全（不实现共享网格协议只降速不出错）。但 `_loglike_from_grid` 要求手工复制 `__call__` 全部逻辑且无一致性保障，CLAUDE.md 设想的 `_auto_build_loglike_from_grid()` 落地前，至少先建参数化一致性测试矩阵。

**新采样后端：名实不符**。`SamplerBackend` 抽象存在，但实际有三处隐藏耦合：`run_chunked` 靠 `hasattr` 探测（`inference.py:1058-1060`）、checkpoint 状态收集硬编码 backend 名（`:1259-1275`）、`nested.py` 完全绕开抽象由 executor 单独分支调用。`SamplerResults.samples` 的形状契约三方不一致（基类文档承诺分链二维、numpyro 返回扁平、emcee 返回二维），下游靠 `flatten()` 兜底——这正是 burn-in 切错位置、R̂ 丢失链结构两个 P2 的根源。建议把可选接口纳入基类、统一分链形状并用断言固化。

---

## 七、易用性

**达成的**：`hicosmo('LCDM', ['sn','bao'], free_params=['H0','Omega_m'])` 三行采样成立且有测试背书，nuisance 自动收集（`H0_rd` 自动入列）真实可用；`Plotter('chain_name')` 一行加载并自动回填 LaTeX 标签，超出多数竞品。

**断裂的**：

1. **五条入口并行**（`hicosmo()`、`InferenceRunner`、`MCMC` 直构、YAML `run_from_config`、CLI REPL），同一概念拼写不一（`cosmology` vs `cosmology_class`、`likelihood` vs `likelihoods`）、默认值不同（`hicosmo()` 默认 8000 样本，`InferenceRunner`/YAML 落到后端默认 1000——同一配置两条路径跑出长度差 8 倍的链）、能力不对等（`hicosmo()` 返回的对象出不了图，§P0-4；YAML 的 `derived` 字段被静默忽略而 `MCMC` 直构支持它）。
2. **数据集名字空间三套互不知晓**：`hicosmo('lcdm', 'desi_dr2', ...)` 抛裸 KeyError，错误信息列出的是 probe 级名（`bao, cmb, ...`），不提示正确写法 `{'name':'bao','dataset':'desi_dr2'}`；`parameters/validation.py:157-193` 已有 difflib did-you-mean 基建但注册表全都没用它。
3. **YAML 最 Cobaya 风格的写法必崩**：`params` 标量 + `prior` 映射 + `free` 列表的组合中，用户的 prior 在 `Parameter.from_simple_config`（`parameters/parameter.py:417-425`）被丢弃，随后报 "no prior distribution defined"。
4. **可调用模块 hack**（`hicosmo.py:196-203`，`sys.modules[__name__].__class__ = ...`）：IDE/mypy 报 "module is not callable"、`help()` 失效、需要两个专门回归测试守护。包→模块→函数三层同名是混淆根源，长期建议函数改名 `run()`/`infer()` 并在包级 re-export。
5. **import 副作用**：`import hicosmo.visualization` 就在 cwd 创建 `results/` 目录（`plotting.py:102-103`）；`_get_results_dir` 用 `inspect` 栈帧猜调用方目录。
6. `hicosmo()` 无法自定义 prior（想给 $H_0$ 加 SH0ES normal prior 必须跳到 registry 手工模式，渐进复杂度有断崖）；`progress_bar=False` 传入不生效（实测）；自动链名落到 `mcmc_unknown_func_<时间戳>`。

---

## 八、测试与工程化

**测试现状**：8 个文件、96 个函数、全部通过（14.4 s）。新增测试质量高：DESY5 与官方 Dovekie 公式对齐到 $10^{-8}$、Union3 锚定 Rubin et al. 2023 的 $\Omega_m \approx 0.344$、DESI DR2 用断言子类强制走快路径——没有发现"测试重新实现被测逻辑"的违规。

**结构性缺口**：约 2/3 子系统零测试。models 物理（$d_L$、$D_M$、$r_d$ 与 astropy/Cobaya 的数值对比为零，"与 Cobaya $\chi^2<10^{-6}$"的声明没有测试固化）、CMB、强透镜（与 CLAUDE.md 自己的"TDCOSMO 教训：测试必须验证数值结果"直接矛盾）、Fisher（`get_fisher_summary` 必崩即为证）、可视化、persistence、emcee/nested 后端全部裸奔。无 conftest.py、无 slow 标记、x64 启用与否取决于测试文件收集顺序、所有真实数据测试无 skipif 防护（无数据环境 96 个几乎全挂）。

**工程化**：依赖全部仅 `>=` 下界且 `jax` 无上界（0.5/0.6 有破坏性变更）；README 与 requirements.txt 的 numpyro 版本声明矛盾；pyproject 缺 license/classifiers/urls；Sphinx 构建产物（含全套字体）占 616 个跟踪文件中的约 350 个；裸 `except:` 10 处、`except Exception` 58 处（与 fail-fast 原则相悖，P0-1 是其极端案例）；union3/desy5/desi_dr2 三个新数据集在 29 篇 rst 文档中出现 0 次。

---

## 九、优先级路线图

**第一批：止血（1–2 天，全部是小改动）**

1. 提交全部工作区资产，按功能拆分（P0-3）。
2. `hicosmo/__init__.py` 显式启用 x64；修复 gw 导入链；fisher dtype 改为读取时求值（P0-2/P1-1）。
3. 删除 NUTS 路径的异常吞噬（P0-1）。
4. 修 CLI banner 示例（P0-4）。
5. 恢复最小 CI：lint + 干净环境 import 冒烟 + 无数据测试子集（P0-3）。

**第二批：正确性（3–5 天）**

6. BBN prior 三路径统一（~10 行）+ 固定 M_B wrapper 委托（~5 行）。
7. 建立共享网格协议参数化一致性测试矩阵：每个似然 × 每种构造模式，断言 $|{\rm combined} - \sum {\rm lik}| < 10^{-10}$ 且 `jax.grad` 一致。
8. `normalize_params` 加未知参数名检测（修 §四 #3 的静默忽略）；修正规则文档的 `w` 示例。
9. 修 Omega_b 注入守卫；删除 lcdm 本地 EH98 副本统一走带校准版本；装饰器加 `cls.__dict__` 检查与 `__init_subclass__` 告警。
10. emcee 真实 prior 密度；Fisher 改 `eigvalsh` + 有限性前置检测。
11. 删除死代码：`compute_DM_at_z`、`make_compute_shared_grid`、`_precomputed_grid` 旧协议、`parameters/collector.py`，并同步更新 CLAUDE.md（共享网格归因、CMB 轻量路径、测试命令等多处与代码已漂移）。

**第三批：性能（1 周，按 §3.4 顺序）**

12. gather 替代 `jnp.interp`（已验证 −28% SN 梯度）。
13. warmup 改 per-chain 语义。
14. 修 `progress_bar` 透传 + 非 TTY 默认关闭。
15. Cholesky 白化；float32 协方差实验（必须 $\tau$ 验证）。
16. `run_chunked` chunk 档位化。

**第四批：架构收敛（2–3 周，随论文节奏）**

17. SN 中间基类提取（新数据集 600 行 → 80 行）；BAO 注册单源化。
18. 入口收敛：`hicosmo()` 返回带出图能力的对象、统一默认值、dataset 级字符串直达、did-you-mean 错误提示。
19. `MCMC` god class 继续拆分；backend 契约固化（分链形状、可选接口入基类）。
20. KB 级数据入库 + Zenodo 通道（P0-5）；新数据集文档。

---

## 附：实测原始数据

- 测试套件：`pytest tests/` → 96 passed, 14.38 s。
- x64 验证：`import jax` 后 `jax_enable_x64=False`；`import hicosmo.likelihoods` 后 `=True`（副作用源 `gw/population_rate.py:13`）。
- 参数静默忽略验证：`sn(H0=70, Omega_m=0.3, w=-0.5)` → −694.8585（与缺省相同）；`sn(..., w0=-0.5)` → −751.7677。
- 共享网格 introspection：`_shared_grid_enabled=True`，共享低红移网格 5660 点；共享 vs 独立 $\log L$：987.038392 vs 987.037141。
- 梯度分解与 gather 验证脚本：`/tmp/hicosmo_bench/`（bench_likelihood.py、bench_decompose.py、bench_interp_fix.py、bench_verify.py、bench_mcmc.py）。
- 仓库状态：`main..codex` 6 提交；28 文件未提交（+958/−492）；12 个未跟踪文件；`git ls-files hicosmo/likelihoods/gw/` → 0；tests/ 已跟踪 2/8 文件。

*报告由 Claude Code 生成（静态审查 6 个并行维度 + 本机运行时实测交叉验证）。*
