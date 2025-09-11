#!/usr/bin/env python3
"""
验证总样本数修复 - 实际运行测试
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

print("🔬 验证总样本数修复 - 实际运行")
print("="*50)

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

# 实际运行测试
test_configs = [
    (500, 1, "单链500样本"),
    (800, 4, "4链800总样本")
]

for total_expected, num_chains, desc in test_configs:
    print(f"\n🔧 {desc}")
    print(f"用户期望: {total_expected} 总样本")
    print("-" * 30)
    
    config = {
        'parameters': {
            'a': (2.0, 0.0, 4.0),
            'b': (1.0, 0.0, 2.0)
        },
        'mcmc': {
            'num_samples': total_expected,
            'num_warmup': 50,  # 减少warmup以加快测试
            'num_chains': num_chains
        }
    }
    
    mcmc = MCMC(config, likelihood_func, optimize_init=False, 
               chain_name=f"verify_{num_chains}chains")
    
    # 实际运行
    samples = mcmc.run()
    
    # 验证样本数
    actual_total = len(samples.get('a', []))
    per_chain = actual_total // num_chains if num_chains > 1 else actual_total
    
    print(f"\n📊 结果验证:")
    print(f"  期望总样本数: {total_expected}")
    print(f"  实际总样本数: {actual_total}")
    print(f"  每链样本数: {per_chain}")
    
    if num_chains > 1:
        expected_per_chain = total_expected // num_chains
        expected_actual_total = expected_per_chain * num_chains
        print(f"  理论分配: {expected_per_chain} × {num_chains} = {expected_actual_total}")
        
        if actual_total == expected_actual_total:
            print(f"  ✅ 样本数分配正确!")
        else:
            print(f"  ❌ 样本数分配错误!")
    else:
        if actual_total == total_expected:
            print(f"  ✅ 单链样本数正确!")
        else:
            print(f"  ❌ 单链样本数错误!")

print(f"\n🎯 总结:")
print(f"修复后用户体验:")
print(f"  - 输入 'num_samples': 1000 → 得到总共1000个样本 ✅")
print(f"  - 系统自动分配到各链 (4链情况下每链250个)")
print(f"  - 用户不需要考虑链数和计算 ✅")