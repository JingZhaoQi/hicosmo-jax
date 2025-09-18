#!/usr/bin/env python3
"""
使用HIcosmo已有的MCMC系统测试PantheonPlus vs PantheonPlusSH0ES
"""

import os
import numpy as np
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood
from hicosmo.samplers import MCMC, ParameterConfig, AutoParameter
from hicosmo.visualization import MCplot

# 数据路径
data_path = "/Users/qijingzhao/Programs/hicosmo_new1/data/DataRelease"

print("🧪 PantheonPlus vs PantheonPlusSH0ES MCMC对比测试")
print("=" * 60)

# 1. PantheonPlus (纯超新星)
print("\n1️⃣ PantheonPlus (纯超新星)")
likelihood_plus = PantheonPlusLikelihood(
    data_path=data_path,
    include_shoes=False,
    z_min=0.01
)

# 参数设置
config_plus = ParameterConfig({
    'H0': AutoParameter(
        name='H0',
        prior={'dist': 'uniform', 'min': 60.0, 'max': 80.0},
        ref=70.0,
        latex=r'H_0'
    ),
    'Omega_m': AutoParameter(
        name='Omega_m',
        prior={'dist': 'uniform', 'min': 0.2, 'max': 0.4},
        ref=0.3,
        latex=r'\Omega_m'
    )
})

# 运行MCMC
mcmc_plus = MCMC(
    model=LCDM,
    config=config_plus,
    likelihood_func=likelihood_plus.log_likelihood,
    num_chains=2,
    num_warmup=200,
    num_samples=500
)

samples_plus = mcmc_plus.run()

# 2. PantheonPlusSH0ES (包含造父变星)
print("\n2️⃣ PantheonPlusSH0ES (包含造父变星)")
likelihood_shoes = PantheonPlusLikelihood(
    data_path=data_path,
    include_shoes=True,
    z_min=0.01
)

# 相同参数设置
config_shoes = ParameterConfig({
    'H0': AutoParameter(
        name='H0',
        prior={'dist': 'uniform', 'min': 60.0, 'max': 80.0},
        ref=70.0,
        latex=r'H_0'
    ),
    'Omega_m': AutoParameter(
        name='Omega_m',
        prior={'dist': 'uniform', 'min': 0.2, 'max': 0.4},
        ref=0.3,
        latex=r'\Omega_m'
    )
})

# 运行MCMC
mcmc_shoes = MCMC(
    model=LCDM,
    config=config_shoes,
    likelihood_func=likelihood_shoes.log_likelihood,
    num_chains=2,
    num_warmup=200,
    num_samples=500
)

samples_shoes = mcmc_shoes.run()

# 3. 结果对比
print("\n📊 结果对比")
print("-" * 40)

# Plus结果
H0_plus_mean = np.mean(samples_plus['H0'])
H0_plus_std = np.std(samples_plus['H0'])
Om_plus_mean = np.mean(samples_plus['Omega_m'])
Om_plus_std = np.std(samples_plus['Omega_m'])

# SH0ES结果
H0_shoes_mean = np.mean(samples_shoes['H0'])
H0_shoes_std = np.std(samples_shoes['H0'])
Om_shoes_mean = np.mean(samples_shoes['Omega_m'])
Om_shoes_std = np.std(samples_shoes['Omega_m'])

print(f"PantheonPlus:     H0 = {H0_plus_mean:.1f} ± {H0_plus_std:.1f} km/s/Mpc")
print(f"                  Ωm = {Om_plus_mean:.3f} ± {Om_plus_std:.3f}")
print(f"PantheonPlusSH0ES: H0 = {H0_shoes_mean:.1f} ± {H0_shoes_std:.1f} km/s/Mpc")
print(f"                  Ωm = {Om_shoes_mean:.3f} ± {Om_shoes_std:.3f}")

# 4. 可视化
print("\n🎨 生成对比图")
plot = MCplot()

# Corner图对比
fig = plot.corner(
    samples_plus['samples_array'],
    labels=['H_0', r'\Omega_m'],
    ranges={'H0': [60, 80], 'Omega_m': [0.2, 0.4]},
    color='blue',
    alpha=0.5,
    title='PantheonPlus (蓝) vs PantheonPlusSH0ES (红)'
)

# 叠加SH0ES结果
plot.corner(
    samples_shoes['samples_array'],
    labels=['H_0', r'\Omega_m'],
    ranges={'H0': [60, 80], 'Omega_m': [0.2, 0.4]},
    color='red',
    alpha=0.5,
    fig=fig
)

# 保存
output_file = 'results/pantheon_mcmc_comparison.pdf'
fig.savefig(output_file)
print(f"✅ 结果保存到: {output_file}")

print("\n🎯 主要发现:")
H0_diff = H0_shoes_mean - H0_plus_mean
print(f"   • H0差异: {H0_diff:.1f} km/s/Mpc")
print(f"   • SH0ES精度改善: {(H0_plus_std - H0_shoes_std)/H0_plus_std * 100:.1f}%")
print(f"   • 造父变星影响显著性: {abs(H0_diff)/H0_plus_std:.1f}σ")