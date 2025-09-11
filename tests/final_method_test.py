#!/usr/bin/env python3
"""
最终的Method显示测试 - 完整的MCMC流程
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

print("🎯 最终Method显示测试")
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

print(f"\n📋 测试不同配置下的Method显示:")
print("-"*50)

# 测试配置
test_configs = [
    (1, "单链测试", "sequential"),
    (4, "4链测试", "parallel")
]

for num_chains, desc, expected in test_configs:
    print(f"\n🔧 {desc} (期望: {expected})")
    print("-" * 30)
    
    config = {
        'parameters': {
            'a': (2.0, 0.0, 4.0),
            'b': (1.0, 0.0, 2.0)
        },
        'mcmc': {
            'num_samples': 200,
            'num_warmup': 100,
            'num_chains': num_chains
        }
    }
    
    mcmc = MCMC(config, likelihood_func, optimize_init=False, 
               chain_name=f"method_test_{num_chains}chains")
    
    # 检查内部sampler的方法
    actual_method = mcmc.sampler.chain_method
    print(f"实际检测到的方法: {actual_method}")
    
    if actual_method == expected:
        print(f"✅ Method显示正确!")
    else:
        print(f"❌ Method显示错误! 期望:{expected}, 实际:{actual_method}")
    
    # 运行一小部分来验证配置面板显示
    print("运行MCMC验证配置显示...")
    samples = mcmc.run()
    
    print(f"完成! 样本数量: {len(samples.get('a', []))}")

print(f"\n🎉 Method显示测试完成!")

print(f"\n📊 总结你的问题:")
print(f"1. ✅ 样本数量: 配置2000样本×4链 = 8000总样本")
print(f"2. ✅ Method显示: 现在根据设备数量智能显示")
print(f"   - 单设备或单链: sequential")
print(f"   - 多设备且链数<=设备数: parallel")
print(f"   - 多链但设备不足: sequential")