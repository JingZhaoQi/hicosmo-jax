#!/usr/bin/env python3
"""
使用HIcosmo已有的MCMC系统测试PantheonPlus vs PantheonPlusSH0ES
"""

import numpy as np
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood
from hicosmo.samplers import MCMC, ParameterConfig, AutoParameter
from hicosmo.visualization import HIcosmoViz

print("🧪 使用已有MCMC系统测试PantheonPlus vs PantheonPlusSH0ES")
print("=" * 70)

# 数据路径
data_path = "/Users/qijingzhao/Programs/hicosmo_new1/data/DataRelease"

# ========== 1. PantheonPlus (纯超新星) ==========
print("\n1️⃣  测试PantheonPlus (纯超新星)")
print("-" * 40)

# 创建似然函数（关闭系统误差以加速）
likelihood_plus = PantheonPlusLikelihood(
    data_path=data_path,
    include_shoes=False,
    include_systematics=False,  # 关闭系统误差加速计算
    z_min=0.01
)

# 参数配置（参考test_existing_mcmc.py）
H0_param = AutoParameter(
    name='H0',
    prior={'dist': 'uniform', 'min': 60.0, 'max': 80.0},
    ref=70.0,
    latex=r'H_0'
)

Om_param = AutoParameter(
    name='Omega_m',
    prior={'dist': 'uniform', 'min': 0.2, 'max': 0.4},
    ref=0.3,
    latex=r'\Omega_m'
)

config_plus = ParameterConfig({
    'H0': H0_param,
    'Omega_m': Om_param
})

# 创建MCMC实例
mcmc_plus = MCMC(
    model=LCDM,
    config=config_plus,
    likelihood_func=likelihood_plus.log_likelihood,
    num_chains=1,  # 单链加速
    num_warmup=100,  # 短warmup
    num_samples=200  # 短采样
)

print("运行MCMC...")
samples_plus = mcmc_plus.run()

# 提取结果
H0_plus_mean = np.mean(samples_plus['H0'])
H0_plus_std = np.std(samples_plus['H0'])
Om_plus_mean = np.mean(samples_plus['Omega_m'])
Om_plus_std = np.std(samples_plus['Omega_m'])

print(f"结果: H0 = {H0_plus_mean:.1f} ± {H0_plus_std:.1f} km/s/Mpc")
print(f"      Ωm = {Om_plus_mean:.3f} ± {Om_plus_std:.3f}")

# ========== 2. PantheonPlusSH0ES (包含造父变星) ==========
print("\n2️⃣  测试PantheonPlusSH0ES (包含造父变星)")
print("-" * 40)

# 创建似然函数
likelihood_shoes = PantheonPlusLikelihood(
    data_path=data_path,
    include_shoes=True,  # 包含SH0ES
    include_systematics=False,  # 关闭系统误差加速计算
    z_min=0.01
)

# 使用相同的参数配置
config_shoes = ParameterConfig({
    'H0': H0_param,
    'Omega_m': Om_param
})

# 创建MCMC实例
mcmc_shoes = MCMC(
    model=LCDM,
    config=config_shoes,
    likelihood_func=likelihood_shoes.log_likelihood,
    num_chains=1,  # 单链加速
    num_warmup=100,  # 短warmup
    num_samples=200  # 短采样
)

print("运行MCMC...")
samples_shoes = mcmc_shoes.run()

# 提取结果
H0_shoes_mean = np.mean(samples_shoes['H0'])
H0_shoes_std = np.std(samples_shoes['H0'])
Om_shoes_mean = np.mean(samples_shoes['Omega_m'])
Om_shoes_std = np.std(samples_shoes['Omega_m'])

print(f"结果: H0 = {H0_shoes_mean:.1f} ± {H0_shoes_std:.1f} km/s/Mpc")
print(f"      Ωm = {Om_shoes_mean:.3f} ± {Om_shoes_std:.3f}")

# ========== 3. 结果对比 ==========
print("\n📊 参数约束对比")
print("=" * 50)
print(f"{'数据集':<20} {'H0 [km/s/Mpc]':<20} {'Ωm':<15}")
print("-" * 50)
print(f"{'PantheonPlus':<20} {H0_plus_mean:.1f} ± {H0_plus_std:.1f}{'':>5} {Om_plus_mean:.3f} ± {Om_plus_std:.3f}")
print(f"{'PantheonPlusSH0ES':<20} {H0_shoes_mean:.1f} ± {H0_shoes_std:.1f}{'':>5} {Om_shoes_mean:.3f} ± {Om_shoes_std:.3f}")
print("-" * 50)
print(f"{'差异':<20} {H0_shoes_mean - H0_plus_mean:+.1f}{'':>10} {Om_shoes_mean - Om_plus_mean:+.3f}")

# ========== 4. 可视化对比 ==========
print("\n🎨 生成对比图")

# 使用HIcosmoViz
viz = HIcosmoViz()

# 准备数据
samples_plus_array = samples_plus['samples_array']
samples_shoes_array = samples_shoes['samples_array']

# 生成corner图
import matplotlib.pyplot as plt
from getdist import plots, MCSamples

# 创建GetDist样本
names = ['H0', 'Omega_m']
labels = [r'H_0', r'\Omega_m']
ranges = {'H0': [60, 80], 'Omega_m': [0.2, 0.4]}

# PantheonPlus样本
samples1 = MCSamples(samples=samples_plus_array, names=names, labels=labels, ranges=ranges)
# PantheonPlusSH0ES样本
samples2 = MCSamples(samples=samples_shoes_array, names=names, labels=labels, ranges=ranges)

# 创建对比图
g = plots.get_subplot_plotter()
g.triangle_plot([samples1, samples2],
                filled=True,
                legend_labels=['PantheonPlus', 'PantheonPlusSH0ES'],
                contour_colors=['blue', 'red'])

# 保存
output_file = 'results/pantheon_mcmc_comparison.pdf'
plt.savefig(output_file)
print(f"✅ 对比图保存到: {output_file}")

# ========== 5. 分析总结 ==========
print("\n🔍 分析总结")
print("-" * 50)

# H0张力
H0_diff = H0_shoes_mean - H0_plus_mean
H0_significance = abs(H0_diff) / np.sqrt(H0_plus_std**2 + H0_shoes_std**2)

print(f"1. H0差异: {H0_diff:.1f} km/s/Mpc ({H0_significance:.1f}σ)")

# 精度改善
precision_improvement = (H0_plus_std - H0_shoes_std) / H0_plus_std * 100
print(f"2. SH0ES精度改善: {precision_improvement:.1f}%")

# 数据集信息
print(f"3. 数据点数量:")
print(f"   - PantheonPlus: {likelihood_plus.n_sne} SNe")
print(f"   - PantheonPlusSH0ES: {likelihood_shoes.n_sne} objects")
print(f"   - 增加的造父变星: {likelihood_shoes.n_sne - likelihood_plus.n_sne}")

print("\n✅ MCMC测试完成！已使用HIcosmo现有系统成功完成参数约束对比。")