# HIcosmo APJS 论文撰写方案（建议稿）

> 目标期刊：**The Astrophysical Journal Supplement Series (ApJS)**
>
> 目标定位：方法/软件型论文（cosmology inference software + JAX/HPC）
>
> 本文档是**写作思路 + 结构大纲 + 结果呈现方案**的整合草案，便于团队协作与逐段填充。

---

## 1. 写作思路（面向 ApJS 的组织原则）

1. **工具类论文核心叙事**：强调 *可复现、可扩展、高性能* 的宇宙学推断框架，说明为什么 JAX/JIT 对宇宙学推断是“结构性改进”。
2. **从问题到工程**：
   - 观测数据多样化与联合分析需求上升
   - 传统采样框架在速度、可微性、硬件扩展上的瓶颈
   - HIcosmo 采用 JAX/JIT 和模块化的似然设计解决这些瓶颈
3. **以“可用性 + 可信度 + 性能”为三条主线**：
   - **可用性**：简洁 API、模型扩展和配置能力
   - **可信度**：与公开结果的一致性验证（Planck/BAO/SN/透镜/GW 等）
   - **性能**：JIT 加速、多核/多设备、对比基线
4. **结果展示聚焦“验证 + 速度 + 组合应用”**：
   - 不是只给一个科学结果，而是多组“基准对比 + 工具能力演示”
5. **强调开放性与可复现**：
   - 代码、配置、脚本、数据版本
   - 固定随机种子、版本号、CPU/GPU 指标

---

## 2. 论文结构大纲（建议版本）

### Title（建议方向）
- **HIcosmo: A JAX-native High-performance Cosmological Inference Framework for Multi-probe Analyses**

### Abstract（摘要要点）
- 说明 HIcosmo 解决了哪些痛点（性能、可微性、联合推断、扩展性）
- 简述关键技术（JAX/JIT、NumPyro + emcee、多探针似然、Fisher 预报）
- 给出核心验证结果（与公开标准结果一致 + 性能提升数量级）
- 说明代码开放与可复现方案

### 1. Introduction
- 现状：多探针 cosmology 需要快速、可组合的推断框架
- 传统工具（CLASS/CAMB + Cobaya/MontePython）在某些工作流中存在：
  - GPU 加速/自动微分/批量并行的限制
  - 多探针联合与新模型扩展成本高
- HIcosmo 设计目标：**可微、可扩展、可复用、高性能**

### 2. Design Overview & Architecture
- 架构层次：模型层、似然层、采样层、可视化/后处理层
- JAX 原生设计原则：
  - JIT 编译路径
  - 纯函数接口
  - 批量化/向量化
- 关键工程选择：
  - NumPyro NUTS（HMC）为主后端
  - emcee 作为兼容备选
  - 统一参数系统（支持多种参数基）

### 3. Cosmological Models
- LCDM / wCDM / CPL / ILCDM 的接口与差异
- 参数标准化与派生参数
- 关键物理量计算（E(z)、距离、增长因子、声学尺度等）

### 4. Likelihood Modules
- **SNe Ia**（Pantheon+）
- **BAO**（如 DESI/SDSS 组合）
- **CMB distance priors**（Planck 2018）
- **Strong Lensing**（H0LiCOW / TDCOSMO）
- **GW Standard Sirens**
- 似然组合机制：`likelihood = L1 + L2 + ...`

### 5. Inference Engines
- NumPyro NUTS：
  - 自适应步长、并行 chains、梯度自动计算
- emcee：
  - 作为传统 MCMC 对比
- 多核/多设备并行策略

### 6. Validation & Reproducibility
- 与公开基准一致性（Planck/SN/BAO/H0LiCOW/GW 等）
- 复现实验配置（配置文件、数据版本、随机种子）
- 误差控制与数值稳定性（如高 z 积分、声视界计算）

### 7. Performance Benchmarks
- CPU 单核 vs 多核
- GPU 加速（如有）
- JIT path vs non-JIT path
- 与传统框架（Cobaya/MontePython/CLASS+MCMC）的对比

### 8. Example Science Cases
- SN + BAO + Planck 的联合约束
- 21cm Fisher forecast 示例
- GW 与传统探针联合的示例

### 9. Limitations & Future Work
- 当前假设（如近似声视界、特定模型限制）
- 未来扩展（完整 CMB likelihood、更多探针、更精细的重子物理等）

### 10. Conclusion
- 总结核心贡献
- 强调开放与可复现

### Appendices
- 模块 API 列表
- 关键配置与命令
- 数据来源与版本说明

---

## 3. 结果呈现（Figures & Tables 设计）

### 表格（Table）建议

1. **核心模块总览表**
   - 模型 / 似然 / 采样器 / 数据集 / 输出

2. **默认数据集与版本表**
   - Pantheon+、Planck 2018 distance priors、DESI BAO、H0LiCOW/TDCOSMO

3. **性能对比表**
   - 任务：LCDM + SN/BAO/Planck
   - 指标：每秒样本数、wall-time、ESS/s
   - 对比：HIcosmo (JAX) vs baseline

4. **复现性清单**
   - 代码版本、硬件、JAX/NumPyro 版本、数据版本、随机种子

### 图像（Figure）建议

1. **架构示意图**
   - Model → Likelihood → Sampler → Posterior/Plots

2. **Corner plot（示例科学结果）**
   - SN+BAO+Planck 的 H0/Ωm 约束

3. **对比性能曲线**
   - ESS/s vs chain length / wall-time

4. **拟合一致性对比图**
   - 与 Planck/LCDM published constraints 的对比

5. **21cm Fisher forecast**
   - 预报误差椭圆

---

## 4. 建议在文中明确的数据和指标（避免“模糊声明”）

> 这部分建议在最终写作时填入具体数值（可从代码运行或已有报告中提取）。

- **验证指标**：
  - Planck distance prior：R, l_a, ω_b h^2 与公开值一致性
  - BAO 与 SN 的 posterior 与标准结果一致性

- **性能指标**：
  - 单核 vs 多核加速比
  - JIT 编译 amortized cost
  - ESS/s 提升

- **准确性指标**：
  - 对照 CAMB/CLASS 或公开链的偏差
  - 数值积分误差控制（z>1000）

---

## 5. 可复现与发布准备

建议在论文中包含下列“可复现条目”：

- 代码版本（commit hash）
- 数据版本（Pantheon+、Planck distance prior、DESI BAO 数据说明）
- 硬件说明（CPU/GPU 型号与核心数）
- JAX/NumPyro 版本号
- 随机种子（尤其是 MCMC）
- 运行命令与配置文件

附加建议：
- 提供一个 `scripts/reproduce_apjs/` 目录，放入：
  - 生成图表的脚本
  - 生成数值对比的脚本
  - 生成表格与参数输出

---

## 6. 论文写作节奏建议（里程碑）

1. **M0：立项准备**
   - 确定投 ApJS
   - 收集验证结果（与 Planck/BAO/SN 的一致性）

2. **M1：框架描述完成**
   - 完成架构与核心模块描述

3. **M2：验证 + 性能结果完成**
   - 所有 benchmark 完整输出

4. **M3：科学示例案例完成**
   - multi-probe 联合示例

5. **M4：整理可复现材料**
   - 一键复现脚本

---

## 7. 建议的“APJS 写作风格”注意事项

- 强调“工具贡献”：设计思想与工程可复现性
- 避免“单一科学结果”为主导
- 所有数值、图表必须可复现
- 代码与数据引用应清晰

---

## 8. 附：具体写作内容清单（可直接作为 TODO）

- [ ] Abstract（300–500 字）
- [ ] Introduction（问题动机 + 现有工具对比）
- [ ] Framework & Architecture（模型层/似然层/采样层）
- [ ] Likelihood implementations（列出支持观测）
- [ ] Validation section（对齐公开结果）
- [ ] Performance benchmarks（对比图+表）
- [ ] Example science cases（multi-probe）
- [ ] Limitations/Future
- [ ] Reproducibility checklist
- [ ] Software and Data availability section

---

## 9. 可选：建议引用的参考方向（占位符）

- JAX / NumPyro 相关官方论文或技术报告
- Pantheon+ / DESI / Planck 2018 / H0LiCOW / TDCOSMO / LIGO-Virgo GW 相关文献
- Cobaya / MontePython / CLASS / CAMB 等对比工具

---

**备注**：本文档是结构化草案，建议在实际写作中逐段替换为可复现数据与实际对比结果。
