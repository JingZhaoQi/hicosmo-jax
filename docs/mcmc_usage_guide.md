# HIcosmo MCMC 使用指南

**HIcosmo** - 中性氢宇宙学计算库，专为21cm巡天观测和模拟设计  
**作者**: Jingzhao Qi

## 概述

HIcosmo MCMC 是一个高性能的通用贝叶斯推断系统，支持各类宇宙学分析，并对中性氢宇宙学和21cm巡天有特殊优化。它提供了简洁的API、智能的默认配置、强大的多核支持和完善的检查点系统。

## 快速开始

### 1. 基础设置

```python
from hicosmo.samplers import init_hicosmo, MCMC
import numpy as np

# 一行初始化多核环境
init_hicosmo()  # 自动检测CPU核心数并优化配置
```

### 2. 最简单的使用方式

```python
# 定义似然函数
def chi2_likelihood(H0, Omega_m, data):
    """简单的卡方似然函数"""
    theory_predictions = your_model(H0, Omega_m)
    chi2 = np.sum((data['obs'] - theory_predictions)**2 / data['sigma']**2)
    return -0.5 * chi2

# 定义参数（简单格式）
params = [
    ['H0', 70, 50, 100],         # [名称, 初值, 最小值, 最大值]
    ['Omega_m', 0.3, 0.1, 0.5]
]

# 运行MCMC
mcmc = MCMC.from_simple_list(params, chi2_likelihood, data=your_data)
results = mcmc.run()

# 获取结果
print(f"H0 = {results['H0']['value']:.2f} ± {results['H0']['error']:.2f}")
print(f"Omega_m = {results['Omega_m']['value']:.3f} ± {results['Omega_m']['error']:.3f}")
```

## 详细参数配置

### 1. 完整配置格式

```python
config = {
    'parameters': {
        'H0': {
            'prior': {'dist': 'uniform', 'min': 50, 'max': 100},
            'ref': 70.0,                    # 初始值
            'latex': r'H_0'                # LaTeX 标签
        },
        'Omega_m': {
            'prior': {'dist': 'normal', 'loc': 0.3, 'scale': 0.05},
            'ref': 0.3,
            'latex': r'\Omega_m'
        },
        'Omega_b': {
            'prior': {'dist': 'truncnorm', 'loc': 0.05, 'scale': 0.01, 'min': 0.01, 'max': 0.1},
            'ref': 0.05,
            'latex': r'\Omega_b'
        }
    },
    'mcmc': {
        'num_samples': 4000,        # 总样本数（所有链合计）
        'num_chains': 4,            # 链数（默认匹配CPU核心数）
        'num_warmup': None,         # 预热步数（默认为总样本数的20%）
        'optimize_init': False,     # 是否使用JAX优化初始化
        'enable_checkpoints': True  # 是否启用检查点
    }
}

mcmc = MCMC(config, chi2_likelihood, data=your_data)
results = mcmc.run()
```

### 2. 支持的先验分布

```python
# 均匀分布
'prior': {'dist': 'uniform', 'min': 50, 'max': 100}

# 正态分布
'prior': {'dist': 'normal', 'loc': 70, 'scale': 5}

# 截断正态分布
'prior': {'dist': 'truncnorm', 'loc': 70, 'scale': 5, 'min': 60, 'max': 80}

# 对数正态分布
'prior': {'dist': 'lognormal', 'loc': 0, 'scale': 1}

# Beta分布
'prior': {'dist': 'beta', 'alpha': 2, 'beta': 3}

# Gamma分布
'prior': {'dist': 'gamma', 'alpha': 2, 'beta': 1}

# 半正态分布
'prior': {'dist': 'halfnormal', 'scale': 1}

# 半柯西分布
'prior': {'dist': 'halfcauchy', 'scale': 1}

# 指数分布
'prior': {'dist': 'exponential', 'rate': 1}
```

## MCMC 配置详解

### 1. 智能默认配置

HIcosmo 会自动设置最优的默认值：

```python
# 系统会自动设置：
# - num_chains: 匹配CPU核心数（1-8之间）
# - num_warmup: 总样本数的20%（最少100步）
# - num_samples: 默认2000（理解为所有链的总样本数）
# - optimize_init: False（使用标准预热，推荐）
# - enable_checkpoints: True（自动保存检查点）

mcmc = MCMC(config, likelihood)  # 使用所有智能默认值
```

### 2. 采样模式选择

```python
# 标准预热（推荐，适用于大多数问题）
mcmc = MCMC(config, likelihood, optimize_init=False)

# JAX优化预热（适用于复杂问题）
mcmc = MCMC(config, likelihood, optimize_init=True)
```

**什么时候使用JAX优化？**
- 似然函数计算时间 > 10ms
- 参数维度 > 20
- 多峰或复杂后验分布
- 初始化困难的问题

### 3. 样本数配置策略

```python
# 用户输入被理解为总样本数
config = {
    'mcmc': {
        'num_samples': 4000,  # 总样本数
        'num_chains': 4       # 4条链
    }
}
# 实际结果：每条链1000个样本，总共4000个样本
```

## 多核优化

### 1. 系统初始化

```python
from hicosmo.samplers import init_hicosmo

# 自动配置（推荐）
init_hicosmo()

# 指定核心数
init_hicosmo(cpu_cores=4)

# 静默初始化
init_hicosmo(verbose=False)
```

### 2. 执行模式

系统会根据硬件配置自动选择最佳执行模式：

- **parallel**: 多设备并行执行（推荐）
- **sequential**: 单设备多链执行
- **vectorized**: 向量化执行

```python
# 手动指定执行模式（一般不需要）
mcmc = MCMCSampler(
    likelihood, 
    parameters, 
    chain_method='parallel'
)
```

## 高级功能

### 1. 检查点系统

```python
# 启用检查点（默认开启）
mcmc = MCMC(config, likelihood, enable_checkpoints=True)

# 自定义检查点间隔
mcmc = MCMC(config, likelihood, checkpoint_interval=500)

# 从检查点恢复
mcmc = MCMC.resume('checkpoint.h5', likelihood_func)

# 继续采样
additional_samples = mcmc.continue_sampling(1000)
```

### 2. 配置文件支持

```python
# 从YAML文件加载配置
mcmc = MCMC.from_yaml('mcmc_config.yaml', likelihood, data=your_data)

# 从JSON文件加载配置
mcmc = MCMC.from_json('mcmc_config.json', likelihood, data=your_data)
```

### 3. 快速MCMC

```python
from hicosmo.samplers import quick_mcmc

# 一行运行MCMC
results = quick_mcmc(
    params=['H0', 70, 50, 100],
    likelihood=chi2_likelihood,
    num_samples=2000,
    data=your_data
)
```

## 结果分析

### 1. 基本结果获取

```python
# 运行MCMC
results = mcmc.run()

# 获取参数估计值和不确定度
for param_name in ['H0', 'Omega_m']:
    value = results[param_name]['value']
    error = results[param_name]['error']
    print(f"{param_name} = {value:.3f} ± {error:.3f}")

# 获取原始样本
samples = mcmc.get_samples()  # Dict[str, ndarray]
```

### 2. 诊断工具

```python
# 获取诊断信息
diagnostics = mcmc.get_diagnostics()

print(f"R-hat: {diagnostics['r_hat']}")
print(f"Effective sample size: {diagnostics['n_eff']}")

# 收敛性检查
if all(r < 1.1 for r in diagnostics['r_hat'].values()):
    print("✅ 所有参数都已收敛")
else:
    print("⚠️ 部分参数可能未收敛，需要更多样本")
```

### 3. 高级诊断

```python
from hicosmo.samplers import DiagnosticsTools

# 自相关分析
autocorr = DiagnosticsTools.autocorrelation(samples['H0'])

# 积分自相关时间
tau = DiagnosticsTools.integrated_autocorrelation_time(samples['H0'])

# 可视化
DiagnosticsTools.plot_trace(samples)
DiagnosticsTools.plot_corner(samples)
```

## 似然函数编写指南

### 1. 基本要求

```python
def your_likelihood(param1, param2, data):
    """
    似然函数必须：
    1. 参数名与配置文件中的参数名匹配
    2. 返回对数似然值（浮点数）
    3. 接收 data 参数（可选）
    """
    # 计算理论预测
    theory = your_model(param1, param2)
    
    # 计算似然
    chi2 = np.sum((data['obs'] - theory)**2 / data['sigma']**2)
    log_likelihood = -0.5 * chi2
    
    return log_likelihood
```

### 2. 参数映射

系统支持灵活的参数名匹配：

```python
# 以下参数名会被自动识别为同一参数
variations = {
    'H0': ['hubble', 'h_0', 'hubble_constant'],
    'Omega_m': ['om', 'omegam', 'omega_matter'],
    'Omega_b': ['ob', 'omegab', 'omega_baryon']
}
```

### 3. 复杂似然函数

```python
def complex_likelihood(H0, Omega_m, Omega_b, sigma8, data):
    """包含多个数据集的复杂似然函数"""
    total_log_likelihood = 0
    
    # SNe Ia数据
    if 'sne' in data:
        mu_theory = distance_modulus(data['sne']['z'], H0, Omega_m)
        chi2_sne = np.sum((data['sne']['mu_obs'] - mu_theory)**2 / data['sne']['sigma']**2)
        total_log_likelihood += -0.5 * chi2_sne
    
    # CMB数据
    if 'cmb' in data:
        Cl_theory = compute_cmb_power_spectrum(H0, Omega_m, Omega_b)
        chi2_cmb = compute_cmb_chi2(Cl_theory, data['cmb'])
        total_log_likelihood += -0.5 * chi2_cmb
    
    # BAO数据
    if 'bao' in data:
        dA_theory = angular_diameter_distance(data['bao']['z'], H0, Omega_m)
        chi2_bao = np.sum((data['bao']['dA_obs'] - dA_theory)**2 / data['bao']['sigma']**2)
        total_log_likelihood += -0.5 * chi2_bao
    
    return total_log_likelihood
```

## 性能优化指南

### 1. 采样配置优化

```python
# 对于快速似然函数（<1ms）
config = {
    'mcmc': {
        'num_chains': 8,          # 更多链
        'num_samples': 8000,      # 更多样本
        'optimize_init': False    # 标准预热
    }
}

# 对于昂贵似然函数（>10ms）
config = {
    'mcmc': {
        'num_chains': 4,          # 适中链数
        'num_samples': 2000,      # 适中样本数
        'optimize_init': True     # JAX优化
    }
}
```

### 2. 内存管理

```python
# 大规模采样时的内存优化
mcmc = MCMC(
    config, 
    likelihood, 
    save_samples_in_memory=False,  # 不在内存中保存所有样本
    checkpoint_interval=1000        # 更频繁的检查点
)
```

### 3. 并行性能监控

```python
# 查看并行配置信息
status = mcmc.get_performance_status()
print(f"CPU cores: {status['cpu_cores']}")
print(f"JAX devices: {status['jax_devices']}")
print(f"Chain method: {status['chain_method']}")
```

## 常见问题和解决方案

### 1. 收敛性问题

**问题**: R-hat > 1.1，链未收敛

**解决方案**:
```python
# 增加预热步数
config['mcmc']['num_warmup'] = int(config['mcmc']['num_samples'] * 0.5)

# 增加链数
config['mcmc']['num_chains'] = 8

# 使用JAX优化
config['mcmc']['optimize_init'] = True
```

### 2. 性能问题

**问题**: 采样速度慢

**解决方案**:
```python
# 检查似然函数效率
import time
start = time.time()
likelihood(*test_params, data=test_data)
print(f"似然函数计算时间: {time.time() - start:.4f}s")

# 优化似然函数（使用JAX JIT编译）
from jax import jit
likelihood_jit = jit(likelihood)
```

### 3. 内存问题

**问题**: 内存不足

**解决方案**:
```python
# 减少内存使用
config = {
    'mcmc': {
        'num_samples': 2000,           # 减少样本数
        'num_chains': 4,               # 减少链数
        'checkpoint_interval': 500,    # 更频繁保存
        'save_samples_in_memory': False # 不在内存中保存
    }
}
```

## 完整示例

### 1. 超新星数据拟合

```python
import numpy as np
from hicosmo.samplers import init_hicosmo, MCMC

# 初始化
init_hicosmo()

# 模拟超新星数据
def load_sne_data():
    z = np.linspace(0.01, 2.0, 50)
    mu_obs = 5 * np.log10(luminosity_distance(z, 70, 0.3) * 1e6 / 10) + np.random.normal(0, 0.1, 50)
    sigma = np.full_like(mu_obs, 0.1)
    return {'z': z, 'mu_obs': mu_obs, 'sigma': sigma}

# 似然函数
def sne_likelihood(H0, Omega_m, data):
    mu_theory = 5 * np.log10(luminosity_distance(data['z'], H0, Omega_m) * 1e6 / 10)
    chi2 = np.sum((data['mu_obs'] - mu_theory)**2 / data['sigma']**2)
    return -0.5 * chi2

# 配置
config = {
    'parameters': {
        'H0': {'prior': {'dist': 'uniform', 'min': 50, 'max': 100}, 'ref': 70},
        'Omega_m': {'prior': {'dist': 'uniform', 'min': 0.1, 'max': 0.5}, 'ref': 0.3}
    },
    'mcmc': {
        'num_samples': 4000,
        'num_chains': 4
    }
}

# 运行MCMC
sne_data = load_sne_data()
mcmc = MCMC(config, sne_likelihood, data=sne_data)
results = mcmc.run()

# 结果
print("超新星数据拟合结果：")
print(f"H0 = {results['H0']['value']:.2f} ± {results['H0']['error']:.2f} km/s/Mpc")
print(f"Ω_m = {results['Omega_m']['value']:.3f} ± {results['Omega_m']['error']:.3f}")
```

### 2. 多数据集联合分析

```python
# 复合数据似然函数
def joint_likelihood(H0, Omega_m, Omega_b, data):
    total_log_likelihood = 0
    
    # SNe数据贡献
    mu_theory = 5 * np.log10(luminosity_distance(data['sne']['z'], H0, Omega_m) * 1e6 / 10)
    chi2_sne = np.sum((data['sne']['mu_obs'] - mu_theory)**2 / data['sne']['sigma']**2)
    total_log_likelihood += -0.5 * chi2_sne
    
    # BAO数据贡献（示例）
    if 'bao' in data:
        # BAO标准尺度计算
        rs = sound_horizon(H0, Omega_m, Omega_b)
        DV_theory = dilation_scale(data['bao']['z'], H0, Omega_m)
        ratio_theory = DV_theory / rs
        
        chi2_bao = ((data['bao']['ratio_obs'] - ratio_theory) / data['bao']['sigma'])**2
        total_log_likelihood += -0.5 * chi2_bao
    
    return total_log_likelihood

# 扩展配置
config = {
    'parameters': {
        'H0': {'prior': {'dist': 'uniform', 'min': 60, 'max': 80}, 'ref': 70},
        'Omega_m': {'prior': {'dist': 'uniform', 'min': 0.2, 'max': 0.4}, 'ref': 0.3},
        'Omega_b': {'prior': {'dist': 'uniform', 'min': 0.04, 'max': 0.06}, 'ref': 0.05}
    },
    'mcmc': {
        'num_samples': 6000,
        'num_chains': 6,
        'optimize_init': True  # 复杂问题使用优化
    }
}

# 运行联合分析
joint_data = {
    'sne': sne_data,
    'bao': bao_data  # 你的BAO数据
}

mcmc = MCMC(config, joint_likelihood, data=joint_data)
results = mcmc.run()
```

## verbose 参数的作用

`verbose` 参数控制MCMC运行过程中的诊断输出：

```python
# 详细输出（默认）
mcmc = MCMC(config, likelihood, verbose=True)

# 静默运行
mcmc = MCMC(config, likelihood, verbose=False)
```

**verbose=True 时显示的信息**：
- MCMC配置摘要（参数数量、链数、样本数）
- 实际的采样策略（并行/顺序执行）
- CPU核心使用情况
- 采样进度条和速度
- 实时诊断信息（R-hat、有效样本数）
- 运行时间统计
- 收敛性警告和建议
- 检查点保存状态

**verbose=False 时**：
- 只显示关键错误和警告信息
- 适合批量处理或生产环境

## 总结

HIcosmo MCMC 提供了一个功能完整、性能优化的贝叶斯推断解决方案：

- **易用性**: 智能默认配置，支持多种输入格式
- **性能**: 自动多核优化，智能采样策略选择
- **可靠性**: 完善的检查点系统，强大的错误处理
- **灵活性**: 支持各种先验分布和复杂似然函数
- **扩展性**: 模块化设计，易于定制和扩展

立即开始使用：`init_hicosmo()` → 定义参数 → `MCMC.run()` → 获取结果！