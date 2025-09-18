#!/usr/bin/env python3
"""
测试真实多核配置 - 在导入任何JAX相关模块之前设置
"""

# 第一步：在导入任何东西之前设置环境变量
import os
print("🔧 在导入JAX之前设置环境变量...")
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=6'
print(f"✅ XLA_FLAGS设置为: {os.environ['XLA_FLAGS']}")

# 第二步：现在导入HIcosmo和JAX
print("\n🚀 导入模块...")
from hicosmo.samplers import Config
Config.init(cpu_cores=6, verbose=True)

# 第三步：检查JAX设备
print("\n🔍 检查JAX设备配置...")
import jax
print(f"JAX local device count: {jax.local_device_count()}")
print(f"JAX devices: {jax.devices()}")

if jax.local_device_count() == 6:
    print("✅ 成功！JAX使用6个CPU设备")
else:
    print(f"❌ 失败！JAX只使用{jax.local_device_count()}个设备")

# 第四步：运行简单的MCMC测试
print("\n🧪 运行简单MCMC测试...")
import numpy as np
from hicosmo.samplers import AutoParameter, ParameterConfig, MCMC

# 简单的似然函数
def simple_likelihood(x, y):
    return -0.5 * ((x - 1.0)**2 + (y - 2.0)**2)

# 配置参数
H0_param = AutoParameter(
    name='x',
    prior={'dist': 'uniform', 'min': 0.0, 'max': 2.0},
    ref=1.0,
    latex=r'x'
)

Omega_m_param = AutoParameter(
    name='y',
    prior={'dist': 'uniform', 'min': 1.0, 'max': 3.0},
    ref=2.0,
    latex=r'y'
)

param_config = ParameterConfig(
    parameters={
        'x': H0_param,
        'y': Omega_m_param
    },
    mcmc={'num_samples': 1000, 'num_chains': 4}
)

# 运行MCMC
mcmc = MCMC(config=param_config, likelihood_func=simple_likelihood)
samples = mcmc.run()

print(f"✅ MCMC完成，获得样本数: {len(samples['x'])}")
print(f"x均值: {np.mean(samples['x']):.3f}")
print(f"y均值: {np.mean(samples['y']):.3f}")