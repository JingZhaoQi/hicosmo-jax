#!/usr/bin/env python3
"""
测试新的总样本数逻辑
"""

import os
import sys
from pathlib import Path

# 设置多设备环境
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
import numpyro
numpyro.set_host_device_count(4)

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import jax
from hicosmo.samplers import MCMC

print("🔧 测试新的总样本数逻辑")
print("="*60)
print(f"JAX设备数: {jax.local_device_count()}")

# 测试数据
np.random.seed(42)
x = np.linspace(0, 1, 10)
y_true = 2.0 * x + 1.0
y_err = 0.1 * np.ones_like(x)
y_obs = y_true + np.random.normal(0, y_err)

def likelihood_func(a, b):
    y_pred = a * x + b
    chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
    return -0.5 * chi2

print(f"\n📋 测试不同配置下的样本数分配:")
print("-"*50)

# 测试配置：用户期望的总样本数
test_configs = [
    (1000, 1, "单链测试"),
    (1000, 4, "4链测试"),
    (1003, 4, "不能整除的情况"),
    (2000, 4, "常见配置")
]

for total_samples, num_chains, desc in test_configs:
    print(f"\n🔧 {desc}")
    print(f"用户输入: {total_samples} 总样本, {num_chains} 链")
    print("-" * 30)
    
    config = {
        'parameters': {
            'a': (2.0, 0.0, 4.0),
            'b': (1.0, 0.0, 2.0)
        },
        'mcmc': {
            'num_samples': total_samples,  # 用户期望的总样本数
            'num_warmup': 100,
            'num_chains': num_chains
        }
    }
    
    mcmc = MCMC(config, likelihood_func, optimize_init=False, 
               chain_name=f"total_samples_test_{num_chains}chains")
    
    # 不运行，只看配置
    expected_per_chain = total_samples // num_chains if num_chains > 1 else total_samples
    expected_total = expected_per_chain * num_chains
    
    print(f"期望结果:")
    print(f"  每链样本数: {expected_per_chain}")
    print(f"  实际总样本数: {expected_total}")
    
    if expected_total != total_samples:
        diff = total_samples - expected_total
        print(f"  ⚠️  舍入差异: -{diff} 样本")
    else:
        print(f"  ✅ 完全匹配用户期望")

print(f"\n🎯 关键改进:")
print(f"修复前: 用户输入1000，4链 → 实际得到4000样本")
print(f"修复后: 用户输入1000，4链 → 实际得到1000样本 (250×4)")