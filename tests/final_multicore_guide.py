#!/usr/bin/env python3
"""
最终的多核设置指南

这个脚本展示了在任何环境下都能工作的多核设置方法
"""

import os

print("🚀 HiCosmo 多核设置完整指南")
print("="*60)

# 🚀 方法1：环境变量 + NumPyro设置（最可靠）
print("\n📋 方法1：完整的多核设置")
print("-"*40)

# 设置环境变量（必须在导入JAX之前）
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
print("✅ 已设置 XLA_FLAGS 环境变量")

# 设置NumPyro
import numpyro
numpyro.set_host_device_count(4)
print("✅ 已设置 NumPyro 主机设备数")

# 验证JAX设备
import jax
print(f"JAX设备: {jax.devices()}")
print(f"设备数量: {jax.local_device_count()}")

if jax.local_device_count() >= 4:
    print("🎉 多核设置成功!")
    
    # 演示HiCosmo多核MCMC
    print("\n🔧 演示HiCosmo多核MCMC")
    print("-"*40)
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    import numpy as np
    import time
    from hicosmo.samplers import MCMC
    
    # 简单测试
    np.random.seed(42)
    x = np.linspace(0, 2, 20)
    y_true = 1.5 * x + 0.8
    y_err = 0.1 * np.ones_like(x)
    y_obs = y_true + np.random.normal(0, y_err)
    
    def likelihood_func(a, b):
        y_pred = a * x + b
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    config = {
        'parameters': {
            'a': (1.0, 0.0, 3.0),
            'b': (0.5, 0.0, 2.0)
        },
        'mcmc': {
            'num_samples': 2000,
            'num_warmup': 1000,
            'num_chains': 4  # 4条并行链
        }
    }
    
    start_time = time.time()
    mcmc = MCMC(config, likelihood_func, optimize_init=False, 
               chain_name="multicore_guide")
    samples = mcmc.run()
    execution_time = time.time() - start_time
    
    print(f"   执行时间: {execution_time:.2f}s")
    print(f"   参数估计:")
    for param in ['a', 'b']:
        if param in samples:
            mean_val = np.mean(samples[param])
            std_val = np.std(samples[param])
            print(f"     {param}: {mean_val:.3f} ± {std_val:.3f}")
    
    print("🎉 多核MCMC演示成功!")
    
else:
    print("❌ 多核设置失败")

print("\n📝 使用指南:")
print("="*60)
print("1. 在脚本开头设置环境变量:")
print("   os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'")
print()
print("2. 然后设置NumPyro:")
print("   import numpyro")
print("   numpyro.set_host_device_count(4)")
print()
print("3. 最后导入HiCosmo:")
print("   from hicosmo.samplers import MCMC")
print()
print("4. 使用多链配置:")
print("   config = {'mcmc': {'num_chains': 4}}")
print()
print("🔑 关键：环境变量 + NumPyro设置 + 正确导入顺序")