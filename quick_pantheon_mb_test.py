#!/usr/bin/env python3
"""
快速测试PantheonPlus with M_B as free parameter
"""
import os
import numpy as np

# 数据路径
data_path = os.path.join(os.path.dirname(__file__), "data", "DataRelease")

print("测试HIcosmo MCMC系统 - M_B作为自由参数")
print("=" * 60)

# 首先初始化多核配置
from hicosmo.samplers import init_hicosmo
print("\n🚀 初始化多核配置...")
init_hicosmo(cpu_cores='auto', verbose=True)

# 使用已有的MCMC系统 - 正确的导入
from hicosmo.samplers import MCMC, ParameterConfig, AutoParameter
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood
from hicosmo.visualization import HIcosmoViz

# 1. 测试PantheonPlus数据 - M_B作为自由参数
print("\n1. PantheonPlus MCMC测试 (M_B作为自由参数)")
likelihood_plus_obj = PantheonPlusLikelihood(
    data_path=data_path,
    include_shoes=False,
    z_min=0.01,
    apply_z_cut=True,
    include_systematics=False,  # 关闭系统误差加速
    marginalize_M_B=False  # 不边际化M_B，作为自由参数
)

# 参数配置 - 包含M_B作为自由参数
H0_param = AutoParameter(
    name='H0',
    prior={'dist': 'uniform', 'min': 60.0, 'max': 80.0},
    ref=70.0,
    latex=r'H_0',
    description='Hubble constant [km/s/Mpc]'
)

Omega_m_param = AutoParameter(
    name='Omega_m',
    prior={'dist': 'uniform', 'min': 0.2, 'max': 0.4},
    ref=0.3,
    latex=r'\Omega_m',
    description='Matter density parameter'
)

M_B_param = AutoParameter(
    name='M_B',
    prior={'dist': 'uniform', 'min': -20.0, 'max': -18.0},
    ref=-19.25,
    latex=r'M_B',
    description='Absolute magnitude of SNe Ia'
)

param_config_plus = ParameterConfig(
    parameters={
        'H0': H0_param,
        'Omega_m': Omega_m_param,
        'M_B': M_B_param
    },
    mcmc={
        'num_samples': 2000,  # 减少样本数快速测试
        'num_chains': 4,      # 减少链数
        'chain_method': 'vectorized'  # 向量化方法
    }
)

# 包装似然函数 - M_B作为自由参数
def likelihood_plus_func(H0, Omega_m, M_B):
    """PantheonPlus似然函数包装（M_B作为自由参数）"""
    try:
        model = LCDM(H0=H0, Omega_m=Omega_m)
        return likelihood_plus_obj.log_likelihood(model, M_B=M_B)
    except Exception as e:
        print(f"Error in likelihood_plus_func: {e}")
        return -np.inf

# 正确的MCMC初始化
mcmc_plus = MCMC(
    config=param_config_plus,
    likelihood_func=likelihood_plus_func
)

print("运行PantheonPlus MCMC (M_B作为自由参数)...")
samples_plus = mcmc_plus.run()

# 输出统计信息 - 包含M_B
print("\n统计结果:")
print(f"PantheonPlus:      H0 = {np.mean(samples_plus['H0']):.1f} ± {np.std(samples_plus['H0']):.1f}")
print(f"                   Ωm = {np.mean(samples_plus['Omega_m']):.3f} ± {np.std(samples_plus['Omega_m']):.3f}")
print(f"                   M_B = {np.mean(samples_plus['M_B']):.3f} ± {np.std(samples_plus['M_B']):.3f}")

# 生成corner图
print("\n生成corner图...")
viz = HIcosmoViz()

# 准备参数信息 - 包含M_B
param_names = ['H0', 'Omega_m', 'M_B']
param_labels = [H0_param.latex, Omega_m_param.latex, M_B_param.latex]
param_ranges = {
    'H0': [60.0, 80.0],
    'Omega_m': [0.2, 0.4],
    'M_B': [-20.0, -18.0]
}

fig_plus = viz.corner(
    data=samples_plus,
    params=param_names,
    labels=param_labels,
    ranges=param_ranges,
    filename="pantheon_plus_mb_corner.pdf"
)

print("结果保存到: pantheon_plus_mb_corner.pdf")

print("\n✅ MCMC测试完成！")