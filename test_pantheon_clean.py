#!/usr/bin/env python3
"""
使用HIcosmo现有系统对比PantheonPlus和PantheonPlusSH0ES
完全基于test_existing_mcmc.py的成功模式
"""

import numpy as np
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood
from hicosmo.samplers import MCMC, ParameterConfig, AutoParameter
from hicosmo.visualization import HIcosmoViz

print("🧪 PantheonPlus vs PantheonPlusSH0ES MCMC对比")
print("=" * 60)

# 数据路径
data_path = "/Users/qijingzhao/Programs/hicosmo_new1/data/DataRelease"

# ===== Part 1: PantheonPlus =====
print("\n1️⃣  PantheonPlus (纯超新星)")

# 创建似然函数对象（关闭系统误差加速）
likelihood_plus_obj = PantheonPlusLikelihood(
    data_path=data_path,
    include_shoes=False,
    include_systematics=False,
    z_min=0.01
)

# 包装似然函数（参考test_existing_mcmc.py）
def likelihood_plus(H0, Omega_m):
    """PantheonPlus似然函数包装"""
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return likelihood_plus_obj.log_likelihood(model)

# 参数配置
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

config_plus = ParameterConfig(
    parameters={
        'H0': H0_param,
        'Omega_m': Omega_m_param
    },
    mcmc={
        'num_samples': 200,
        'num_warmup': 100,
        'num_chains': 1  # 单链加速
    }
)

# 创建MCMC
mcmc_plus = MCMC(
    model=LCDM,
    config=config_plus,
    likelihood_func=likelihood_plus
)

# 运行MCMC
print("运行MCMC...")
samples_plus = mcmc_plus.run()

# 结果
H0_plus = np.mean(samples_plus['H0'])
H0_plus_std = np.std(samples_plus['H0'])
Om_plus = np.mean(samples_plus['Omega_m'])
Om_plus_std = np.std(samples_plus['Omega_m'])

print(f"结果: H0 = {H0_plus:.1f} ± {H0_plus_std:.1f}")
print(f"      Ωm = {Om_plus:.3f} ± {Om_plus_std:.3f}")

# ===== Part 2: PantheonPlusSH0ES =====
print("\n2️⃣  PantheonPlusSH0ES (含造父变星)")

# 创建似然函数对象
likelihood_shoes_obj = PantheonPlusLikelihood(
    data_path=data_path,
    include_shoes=True,
    include_systematics=False,
    z_min=0.01
)

# 包装似然函数
def likelihood_shoes(H0, Omega_m):
    """PantheonPlusSH0ES似然函数包装"""
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return likelihood_shoes_obj.log_likelihood(model)

# 使用相同配置
config_shoes = ParameterConfig(
    parameters={
        'H0': H0_param,
        'Omega_m': Omega_m_param
    },
    mcmc={
        'num_samples': 200,
        'num_warmup': 100,
        'num_chains': 1
    }
)

# 创建MCMC
mcmc_shoes = MCMC(
    model=LCDM,
    config=config_shoes,
    likelihood_func=likelihood_shoes
)

# 运行MCMC
print("运行MCMC...")
samples_shoes = mcmc_shoes.run()

# 结果
H0_shoes = np.mean(samples_shoes['H0'])
H0_shoes_std = np.std(samples_shoes['H0'])
Om_shoes = np.mean(samples_shoes['Omega_m'])
Om_shoes_std = np.std(samples_shoes['Omega_m'])

print(f"结果: H0 = {H0_shoes:.1f} ± {H0_shoes_std:.1f}")
print(f"      Ωm = {Om_shoes:.3f} ± {Om_shoes_std:.3f}")

# ===== Part 3: 对比结果 =====
print("\n📊 对比结果")
print("-" * 50)
print(f"PantheonPlus:     H0 = {H0_plus:.1f} ± {H0_plus_std:.1f}")
print(f"                  Ωm = {Om_plus:.3f} ± {Om_plus_std:.3f}")
print(f"PantheonPlusSH0ES: H0 = {H0_shoes:.1f} ± {H0_shoes_std:.1f}")
print(f"                  Ωm = {Om_shoes:.3f} ± {Om_shoes_std:.3f}")
print("-" * 50)
print(f"差异: ΔH0 = {H0_shoes - H0_plus:.1f} km/s/Mpc")
print(f"      ΔΩm = {Om_shoes - Om_plus:.3f}")

# ===== Part 4: 可视化 =====
print("\n🎨 生成对比图")

# 使用HIcosmoViz
viz = HIcosmoViz()

# 生成corner图
fig = viz.corner(
    samples_plus['samples_array'],
    labels=[r'H_0', r'\Omega_m'],
    ranges={'H0': [60, 80], 'Omega_m': [0.2, 0.4]},
    title='PantheonPlus vs PantheonPlusSH0ES'
)

# 保存
output_file = 'results/pantheon_comparison.pdf'
fig.savefig(output_file)
print(f"✅ 图片保存到: {output_file}")

print(f"\n✅ 测试完成！")
print(f"数据点: Plus={likelihood_plus_obj.n_sne}, SH0ES={likelihood_shoes_obj.n_sne}")
print(f"造父变星数量: {likelihood_shoes_obj.n_sne - likelihood_plus_obj.n_sne}")