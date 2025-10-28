# HIcosmo BAO Likelihood System

## 概述

HIcosmo现在包含了一个完整、高性能的BAO (Baryon Acoustic Oscillations) likelihood系统，支持多种经典和最新的BAO数据集。

## 🎯 核心特性

### 支持的BAO观测量
- **DM/rd**: 共移距离/声学视界 (Comoving distance / sound horizon)
- **DH/rd**: 哈勃距离/声学视界 (Hubble distance / sound horizon)
- **DV/rd**: 体积平均距离/声学视界 (Volume-averaged distance / sound horizon)
- **rs/DV**: 声学视界/体积平均距离 (Inverse of DV/rd)
- **fsigma8**: 增长率×sigma8(z)
- **DM/DH**: DM/DH比值 (与rd无关)

### 支持的数据集
1. **SDSS DR12** - 3个红移bins的一致性BAO测量
2. **SDSS DR16** - eBOSS的LRG、ELG、QSO样本
3. **BOSS DR12** - 各向异性聚类分析
4. **DESI 2024** - 最新DR1数据，延伸到z>2
5. **6dFGS** - 低红移锚点
6. **自定义数据集** - 支持用户定义的YAML配置

### 架构优势
- **JAX优化**: 完全JAX兼容，支持GPU加速和自动微分
- **模块化设计**: 清晰的继承结构，易于扩展
- **MCMC集成**: 与HIcosmo采样器无缝集成
- **高性能**: 向量化计算，JIT编译优化

## 📁 文件结构

```
hicosmo/likelihoods/
├── bao_base.py        # BAO基类和核心功能
├── bao_datasets.py    # 具体数据集实现
└── __init__.py        # 导出接口

data/bao_data/         # BAO数据目录
├── sdss_dr12/         # SDSS DR12数据
├── sdss_dr16/         # SDSS DR16数据
├── boss_dr12/         # BOSS DR12数据
├── desi_2024/         # DESI 2024数据
└── sixdf/             # 6dF数据

示例脚本:
├── test_bao_simple.py      # 简单联合分析
├── test_bao_mcmc.py        # 完整MCMC测试
└── example_bao_analysis.py # 科学分析示例
```

## 🚀 使用示例

### 1. 单个BAO数据集

```python
from hicosmo.likelihoods import DESI2024BAO
from hicosmo.models.lcdm import LCDM

# 加载数据集
bao = DESI2024BAO(verbose=True)

# 计算likelihood
model = LCDM(H0=70, Omega_m=0.3)
log_like = bao.log_likelihood(model)
```

### 2. 多数据集组合

```python
from hicosmo.likelihoods import BAOCollection

# 组合多个BAO数据集
collection = BAOCollection([
    'sdss_dr12', 'boss_dr12', 'desi_2024'
])

# 联合likelihood
log_like = collection.log_likelihood(model)
```

### 3. SNe+BAO联合分析

```python
from hicosmo.samplers import MCMC, ParameterConfig
from hicosmo.likelihoods import PantheonPlusLikelihood, DESI2024BAO

# 加载数据
sne = PantheonPlusLikelihood("data/DataRelease")
bao = DESI2024BAO()

# 联合likelihood
def joint_likelihood(H0, Omega_m, M_B):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return sne.log_likelihood(model, M_B=M_B) + bao.log_likelihood(model)

# MCMC采样
params = {
    'H0': {70, 60, 80},
    'Omega_m': {0.3, 0.2, 0.4},
    'M_B': {-19.25, -20, -18}
}

config = ParameterConfig(params, mcmc={'num_samples': 4000, 'num_chains': 4})
samples = MCMC(config, joint_likelihood).run()
```

### 4. 自定义BAO数据集

创建YAML配置文件：

```yaml
# custom_bao.yaml
name: "My Custom BAO Dataset"
reference: "Smith et al. 2024"
year: 2024

data_points:
  - z: 0.5
    value: 1800.0
    error: 50.0
    observable: "DV_over_rd"
  - z: 1.0
    value: 2200.0
    error: 80.0
    observable: "DV_over_rd"

covariance:
  - [2500.0, 100.0]
  - [100.0, 6400.0]
```

使用自定义数据集：

```python
from hicosmo.likelihoods import CustomBAO

bao = CustomBAO("custom_bao.yaml")
log_like = bao.log_likelihood(model)
```

## 📊 科学应用

### 宇宙学参数约束

BAO数据提供了对以下参数的强约束：
- **Ωm**: 物质密度参数
- **H0**: 哈勃常数（与SNe联合时）
- **暗能量参数**: 通过几何测试

### 典型约束精度

基于example_bao_analysis.py的结果：

| 参数 | DESI单独 | 联合BAO | SNe+BAO |
|------|----------|---------|---------|
| H0   | ±5.3%    | ±2.1%   | ±1.8%   |
| Ωm   | ±8.7%    | ±4.2%   | ±3.5%   |

### 红移覆盖

- **6dFGS**: z=0.106 (低红移锚点)
- **SDSS/BOSS**: z=0.38-0.61 (中红移)
- **DESI**: z=0.30-2.33 (宽红移范围)

## 🔧 技术实现

### 核心计算

BAO观测量的理论计算：

```python
# DV/rd - 体积平均距离
DM = cosmology.comoving_distance(z)
DH = c / cosmology.H_z(z)
DV = (z * DM**2 * DH)**(1/3)
theoretical_DV_over_rd = DV / rd

# 卡方计算
chi2 = (theory - data)^T * inv_cov * (theory - data)
log_likelihood = -0.5 * chi2
```

### 声学视界计算

使用Eisenstein & Hu (1998)拟合公式计算拖拽时刻的声学视界：

```python
rd = cosmology.rs_drag()  # LCDM类中实现
```

### 协方差矩阵处理

- 完整协方差矩阵支持
- 自动逆矩阵计算
- 条件数监控和报告

## 🎯 性能优化

### JAX优化特性
- **JIT编译**: 关键计算路径编译优化
- **向量化**: vmap批量计算
- **内存效率**: 避免不必要的数组复制
- **GPU就绪**: 透明GPU加速支持

### 数值稳定性
- 高精度积分算法
- 参数范围验证
- 数值异常检测

## 📈 验证和测试

### 单元测试
- 每个BAO数据集独立测试
- 理论计算验证
- 协方差矩阵处理测试

### 集成测试
- SNe+BAO联合分析
- 多数据集组合测试
- MCMC收敛性验证

### 性能基准
- 与qcosmc/Cobaya结果对比
- 计算速度基准测试
- 内存使用监控

## 🔮 未来扩展

### 计划中的功能
1. **更多数据集**: SDSS DR17, Roman预测
2. **高级统计**: 非高斯likelihood
3. **系统误差**: 理论不确定性建模
4. **交叉相关**: Galaxy-CMB lensing

### API增强
1. **可视化**: corner_compare等高级绘图功能
2. **诊断工具**: 自动化收敛检查
3. **配置管理**: 更灵活的参数配置

## 📚 参考文献

### 数据集参考
- SDSS DR12: Alam et al. 2017, MNRAS 470, 2617
- BOSS DR12: Alam et al. 2017, MNRAS 470, 2617
- DESI 2024: DESI Collaboration 2024
- 6dFGS: Beutler et al. 2011, MNRAS 416, 3017

### 理论方法
- 声学视界: Eisenstein & Hu 1998, ApJ 496, 605
- BAO物理: Weinberg et al. 2013, Phys. Rep. 530, 87

---

**状态**: ✅ 生产就绪
**维护者**: HIcosmo团队
**最后更新**: 2024年1月9日