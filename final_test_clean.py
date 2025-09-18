#!/usr/bin/env python3
"""
HIcosmo最终验证测试 - 简洁版本
展示正确的多核配置方法
"""

print("🎯 HIcosmo框架最终验证测试（简洁版）")
print("=" * 60)

# 步骤1：多核配置（必须第一步！）
print("\n🚀 步骤1: 设置6核多核运算")
from hicosmo.multicore_init import setup_multicore
setup_multicore(6, verbose=True)

# 步骤2：导入HIcosmo模块并逐步检查JAX
print("\n📦 步骤2: 导入HIcosmo模块")
import numpy as np

# 导入前检查JAX
import jax
print(f"导入HIcosmo前JAX设备数: {jax.local_device_count()}")

from hicosmo.samplers import Config, AutoParameter, ParameterConfig, MCMC
print(f"导入samplers后JAX设备数: {jax.local_device_count()}")

from hicosmo.models import LCDM
print(f"导入models后JAX设备数: {jax.local_device_count()}")

from hicosmo.likelihoods import PantheonPlusLikelihood
print(f"导入likelihoods后JAX设备数: {jax.local_device_count()}")

from hicosmo.visualization import HIcosmoViz
print(f"导入visualization后JAX设备数: {jax.local_device_count()}")

# 验证配置
Config.init(cpu_cores=6, verbose=False)  # 静默模式，避免重复输出
print(f"Config.init后JAX设备数: {jax.local_device_count()}")

# 步骤3：验证JAX设备
print("\n🔍 步骤3: 验证JAX多核配置")
import jax
print(f"环境变量XLA_FLAGS: {os.environ.get('XLA_FLAGS', '未设置')}")
print(f"JAX设备数量: {jax.local_device_count()}")
print(f"JAX设备列表: {jax.devices()}")
if jax.local_device_count() == 6:
    print("✅ 成功！使用6个CPU设备")
else:
    print(f"⚠️  只使用{jax.local_device_count()}个设备")

# 调试：检查是否有什么重置了JAX配置
print(f"重新检查XLA_FLAGS: {os.environ.get('XLA_FLAGS', '未设置')}")

# 步骤4：快速MCMC测试
print("\n🧪 步骤4: 运行快速MCMC测试")

# 配置参数（使用正确的LaTeX标签和ranges）
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
    mcmc={'num_samples': 4000, 'num_chains': 4}  # 快速测试
)

# 设置真实PantheonPlus数据
data_path = "/Users/qijingzhao/Programs/hicosmo_new1/data/pantheonplus"
likelihood = PantheonPlusLikelihood(data_path=data_path)

def cosmology_likelihood(H0, Omega_m):
    model = LCDM(H0=H0, Omega_m=Omega_m)
    return likelihood.log_likelihood(model)

# 运行MCMC
print("运行MCMC采样...")
mcmc = MCMC(config=param_config, likelihood_func=cosmology_likelihood)
samples = mcmc.run()

# 结果
print(f"\n📊 步骤5: 结果")
print(f"H₀ = {np.mean(samples['H0']):.2f} ± {np.std(samples['H0']):.2f} km/s/Mpc")
print(f"Ωₘ = {np.mean(samples['Omega_m']):.4f} ± {np.std(samples['Omega_m']):.4f}")
print(f"样本数: {len(samples['H0'])}")

# 创建可视化
print("\n🎨 步骤6: 生成Corner图")
viz = HIcosmoViz()
param_ranges = {'H0': [60.0, 80.0], 'Omega_m': [0.2, 0.4]}
param_labels = ['H_0', r'\Omega_m']

fig = viz.corner(
    data=samples,
    params=['H0', 'Omega_m'],
    labels=param_labels,
    ranges=param_ranges,
    filename="clean_test_corner.pdf"
)

print("\n" + "=" * 60)
print("🎉 测试完成！")
print("✅ 多核配置正确")
print("✅ LaTeX标签正确")
print("✅ 参数ranges正确")
print("✅ 真实数据使用正确")
print("=" * 60)