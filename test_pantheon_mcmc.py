#!/usr/bin/env python3
"""
测试已有的MCMC系统对PantheonPlus和PantheonPlusSH0ES数据的参数约束
"""

import os
import numpy as np

# 数据路径
data_path = os.path.join(os.path.dirname(__file__), "data", "DataRelease")

print("测试HIcosmo MCMC系统")
print("=" * 60)

# 首先初始化多核配置
from hicosmo.samplers import init_hicosmo
print("\n🚀 初始化多核配置...")
init_hicosmo(cpu_cores='auto', verbose=True)

# 使用已有的MCMC系统 - 正确的导入
from hicosmo.samplers import MCMC, ParameterConfig, AutoParameter
from hicosmo.models.lcdm import LCDM
from hicosmo.models.wcdm import wCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood
from hicosmo.visualization import HIcosmoViz

# 1. 测试PantheonPlus数据 - 带M_B作为自由参数
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
        'num_samples': 10000,
        'num_chains': 8,  # 使用8条链进行并行计算
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

print("运行PantheonPlus MCMC...")
samples_plus = mcmc_plus.run()

# 2. 测试PantheonPlusSH0ES数据 - 带M_B作为自由参数
print("\n2. PantheonPlusSH0ES MCMC测试 (M_B作为自由参数)")
likelihood_shoes_obj = PantheonPlusLikelihood(
    data_path=data_path,
    include_shoes=True,
    z_min=0.01,
    apply_z_cut=True,
    include_systematics=False,  # 关闭系统误差加速
    marginalize_M_B=False  # 不边际化M_B，作为自由参数
)

# 使用相同的参数配置（包含M_B）
param_config_shoes = ParameterConfig(
    parameters={
        'H0': H0_param,
        'Omega_m': Omega_m_param,
        'M_B': M_B_param
    },
    mcmc={
        'num_samples': 10000,
        'num_chains': 8,  # 使用8条链进行并行计算
        'chain_method': 'vectorized'  # 向量化方法
    }
)

# 包装似然函数 - M_B作为自由参数
def likelihood_shoes_func(H0, Omega_m, M_B):
    """PantheonPlusSH0ES似然函数包装（M_B作为自由参数）"""
    try:
        model = LCDM(H0=H0, Omega_m=Omega_m)
        return likelihood_shoes_obj.log_likelihood(model, M_B=M_B)
    except Exception as e:
        print(f"Error in likelihood_shoes_func: {e}")
        return -np.inf

# 正确的MCMC初始化
mcmc_shoes = MCMC(
    config=param_config_shoes,
    likelihood_func=likelihood_shoes_func
)

print("运行PantheonPlusSH0ES MCMC...")
samples_shoes = mcmc_shoes.run()

# 3. 可视化对比
print("\n3. 生成对比图")
viz = HIcosmoViz()

# 准备参数信息 - 包含M_B
param_names = ['H0', 'Omega_m', 'M_B']
param_labels = [H0_param.latex, Omega_m_param.latex, M_B_param.latex]
param_ranges = {
    'H0': [60.0, 80.0],
    'Omega_m': [0.2, 0.4],
    'M_B': [-20.0, -18.0]
}

print("为PantheonPlus生成corner图...")
# 生成PantheonPlus corner图
fig_plus = viz.corner(
    data=samples_plus,
    params=param_names,
    labels=param_labels,
    ranges=param_ranges,
    filename="pantheon_plus_corner.pdf"
)

print("为PantheonPlusSH0ES生成corner图...")
# 生成PantheonPlusSH0ES corner图
fig_shoes = viz.corner(
    data=samples_shoes,
    params=param_names,
    labels=param_labels,
    ranges=param_ranges,
    filename="pantheon_shoes_corner.pdf"
)

print("结果保存到: pantheon_plus_corner.pdf 和 pantheon_shoes_corner.pdf")

# 输出统计信息 - 包含M_B
print("\n统计结果:")
print(f"PantheonPlus:      H0 = {np.mean(samples_plus['H0']):.1f} ± {np.std(samples_plus['H0']):.1f}")
print(f"                   Ωm = {np.mean(samples_plus['Omega_m']):.3f} ± {np.std(samples_plus['Omega_m']):.3f}")
print(f"                   M_B = {np.mean(samples_plus['M_B']):.3f} ± {np.std(samples_plus['M_B']):.3f}")
print(f"PantheonPlusSH0ES: H0 = {np.mean(samples_shoes['H0']):.1f} ± {np.std(samples_shoes['H0']):.1f}")
print(f"                   Ωm = {np.mean(samples_shoes['Omega_m']):.3f} ± {np.std(samples_shoes['Omega_m']):.3f}")
print(f"                   M_B = {np.mean(samples_shoes['M_B']):.3f} ± {np.std(samples_shoes['M_B']):.3f}")

# 3. 测试wCDM模型 - 带M_B作为自由参数
print("\n3. wCDM模型测试 (M_B作为自由参数)")

w_param = AutoParameter(
    name='w',
    prior={'dist': 'uniform', 'min': -2.0, 'max': -0.3},
    ref=-1.0,
    latex=r'w',
    description='Dark energy equation of state'
)

param_config_wcdm = ParameterConfig(
    parameters={
        'H0': H0_param,
        'Omega_m': Omega_m_param,
        'w': w_param,
        'M_B': M_B_param
    },
    mcmc={
        'num_samples': 3000,
        'num_chains': 4,
        'chain_method': 'vectorized'
    }
)

def likelihood_wcdm_func(H0, Omega_m, w, M_B):
    """wCDM似然函数包装（M_B作为自由参数）"""
    try:
        model = wCDM(H0=H0, Omega_m=Omega_m, w=w)
        return likelihood_plus_obj.log_likelihood(model, M_B=M_B)
    except Exception as e:
        print(f"Error in wCDM likelihood: {e}")
        return -np.inf

mcmc_wcdm = MCMC(
    config=param_config_wcdm,
    likelihood_func=likelihood_wcdm_func
)

print("运行wCDM MCMC...")
samples_wcdm = mcmc_wcdm.run()

print("为wCDM生成corner图...")
fig_wcdm = viz.corner(
    data=samples_wcdm,
    params=['H0', 'Omega_m', 'w', 'M_B'],
    labels=[H0_param.latex, Omega_m_param.latex, w_param.latex, M_B_param.latex],
    ranges={'H0': [60.0, 80.0], 'Omega_m': [0.2, 0.4], 'w': [-2.0, -0.3], 'M_B': [-20.0, -18.0]},
    filename="wcdm_pantheon_corner.pdf"
)

print(f"wCDM结果:")
print(f"                  H0 = {np.mean(samples_wcdm['H0']):.1f} ± {np.std(samples_wcdm['H0']):.1f}")
print(f"                  Ωm = {np.mean(samples_wcdm['Omega_m']):.3f} ± {np.std(samples_wcdm['Omega_m']):.3f}")
print(f"                  w  = {np.mean(samples_wcdm['w']):.2f} ± {np.std(samples_wcdm['w']):.2f}")
print(f"                  M_B = {np.mean(samples_wcdm['M_B']):.3f} ± {np.std(samples_wcdm['M_B']):.3f}")

w_lambda_diff = abs(np.mean(samples_wcdm['w']) - (-1.0)) / np.std(samples_wcdm['w'])
print(f"与ΛCDM偏离: {w_lambda_diff:.1f}σ {'✅' if w_lambda_diff < 2 else '⚠️'}")

print("\n✅ MCMC测试完成！")