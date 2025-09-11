#!/usr/bin/env python3
"""
检查MCMC样本数量的实际情况
"""

import os
import sys
from pathlib import Path

# 设置多核
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
import numpyro
numpyro.set_host_device_count(4)

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax
from hicosmo.samplers import MCMC

print("🔍 检查MCMC样本数量")
print("="*50)
print(f"JAX设备数: {jax.local_device_count()}")

# 简单测试数据
np.random.seed(42)
x = np.linspace(0, 1, 10)
y_true = 2.0 * x + 1.0
y_err = 0.1 * np.ones_like(x)
y_obs = y_true + np.random.normal(0, y_err)

def likelihood_func(a, b):
    y_pred = a * x + b
    chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
    return -0.5 * chi2

config = {
    'parameters': {
        'a': (2.0, 0.0, 4.0),
        'b': (1.0, 0.0, 2.0)
    },
    'mcmc': {
        'num_samples': 1000,  # 明确设置1000样本
        'num_warmup': 500,
        'num_chains': 4       # 4条链
    }
}

print(f"配置: {config['mcmc']['num_chains']} 链, {config['mcmc']['num_samples']} 样本每链")

mcmc = MCMC(config, likelihood_func, optimize_init=False, chain_name="sample_count_test")
samples = mcmc.run()

print("\n📊 实际样本数量:")
for param, values in samples.items():
    print(f"参数 {param}:")
    print(f"  样本形状: {np.array(values).shape}")
    print(f"  总样本数: {len(values)}")
    
    # 如果是多维数组，打印详细信息
    if hasattr(values, 'shape') and len(np.array(values).shape) > 1:
        print(f"  详细形状: {np.array(values).shape}")
        print(f"  每链样本数: {np.array(values).shape[-1] if len(np.array(values).shape) > 1 else len(values)}")

print(f"\n💡 理论计算:")
print(f"  配置的每链样本数: {config['mcmc']['num_samples']}")
print(f"  链数: {config['mcmc']['num_chains']}")
print(f"  理论总样本数: {config['mcmc']['num_samples'] * config['mcmc']['num_chains']}")

print(f"\n🎯 结论:")
if len(samples.get('a', [])) == config['mcmc']['num_samples'] * config['mcmc']['num_chains']:
    print("  ✅ 总样本数 = 每链样本数 × 链数")
elif len(samples.get('a', [])) == config['mcmc']['num_samples']:
    print("  ✅ 返回的是每链样本数，多链被合并或只返回一条链")
else:
    print(f"  ❓ 实际样本数与预期不符: {len(samples.get('a', []))}")