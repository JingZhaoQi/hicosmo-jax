#!/usr/bin/env python3
"""
简单测试：运行两个数据集的MCMC并对比结果
基于test_existing_mcmc.py的成功版本
"""

import numpy as np
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood
from hicosmo.samplers import MCMC, ParameterConfig, AutoParameter

print("测试PantheonPlus vs PantheonPlusSH0ES")
print("=" * 60)

data_path = "/Users/qijingzhao/Programs/hicosmo_new1/data/DataRelease"

# ===== 运行PantheonPlus MCMC =====
print("\n1. PantheonPlus (纯超新星)")

# 创建似然对象
likelihood_plus = PantheonPlusLikelihood(
    data_path=data_path,
    include_shoes=False,
    include_systematics=False
)

# 定义似然函数（参考test_existing_mcmc.py第101行）
def cosmology_likelihood_plus(H0, Omega_m):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return likelihood_plus.log_likelihood(model)

# 参数配置（参考test_existing_mcmc.py第65行）
H0_param = AutoParameter(
    name='H0',
    prior={'dist': 'uniform', 'min': 60.0, 'max': 80.0},
    ref=70.0,
    latex=r'H_0'
)

Omega_m_param = AutoParameter(
    name='Omega_m',
    prior={'dist': 'uniform', 'min': 0.2, 'max': 0.4},
    ref=0.3,
    latex=r'\Omega_m'
)

param_config = ParameterConfig(
    parameters={'H0': H0_param, 'Omega_m': Omega_m_param},
    mcmc={'num_samples': 200, 'num_warmup': 100, 'num_chains': 1}
)

# 创建MCMC
mcmc = MCMC(model=LCDM, config=param_config, likelihood_func=cosmology_likelihood_plus)
samples_plus = mcmc.run()

H0_plus = np.mean(samples_plus['H0'])
H0_plus_std = np.std(samples_plus['H0'])
Om_plus = np.mean(samples_plus['Omega_m'])
Om_plus_std = np.std(samples_plus['Omega_m'])

print(f"结果: H0 = {H0_plus:.1f} ± {H0_plus_std:.1f}")
print(f"      Ωm = {Om_plus:.3f} ± {Om_plus_std:.3f}")

# ===== 运行PantheonPlusSH0ES MCMC =====
print("\n2. PantheonPlusSH0ES (含造父变星)")

# 创建似然对象
likelihood_shoes = PantheonPlusLikelihood(
    data_path=data_path,
    include_shoes=True,
    include_systematics=False
)

# 定义似然函数
def cosmology_likelihood_shoes(H0, Omega_m):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return likelihood_shoes.log_likelihood(model)

# 重新创建MCMC
mcmc2 = MCMC(model=LCDM, config=param_config, likelihood_func=cosmology_likelihood_shoes)
samples_shoes = mcmc2.run()

H0_shoes = np.mean(samples_shoes['H0'])
H0_shoes_std = np.std(samples_shoes['H0'])
Om_shoes = np.mean(samples_shoes['Omega_m'])
Om_shoes_std = np.std(samples_shoes['Omega_m'])

print(f"结果: H0 = {H0_shoes:.1f} ± {H0_shoes_std:.1f}")
print(f"      Ωm = {Om_shoes:.3f} ± {Om_shoes_std:.3f}")

# ===== 对比 =====
print("\n对比结果:")
print("-" * 40)
print(f"PantheonPlus:     H0 = {H0_plus:.1f} ± {H0_plus_std:.1f}")
print(f"PantheonPlusSH0ES: H0 = {H0_shoes:.1f} ± {H0_shoes_std:.1f}")
print(f"差异: ΔH0 = {H0_shoes - H0_plus:.1f} km/s/Mpc")

print("\n✅ 测试完成")