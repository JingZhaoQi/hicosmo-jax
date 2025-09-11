#!/usr/bin/env python3
"""
测试Method显示修复
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

print("🔧 测试Method显示修复")
print("="*50)

# 测试1：单设备环境（应该显示sequential）
print("\n📋 测试1: 单设备环境")
print("-"*30)

# 不设置多核环境变量，模拟单设备
if 'XLA_FLAGS' in os.environ:
    del os.environ['XLA_FLAGS']

import numpyro
numpyro.set_host_device_count(1)  # 强制单设备

import jax
from hicosmo.samplers import MCMC

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

# 单链配置
config1 = {
    'parameters': {
        'a': (2.0, 0.0, 4.0),
        'b': (1.0, 0.0, 2.0)
    },
    'mcmc': {
        'num_samples': 500,
        'num_warmup': 300,
        'num_chains': 1  # 单链
    }
}

print(f"配置: {config1['mcmc']['num_chains']} 链")
mcmc1 = MCMC(config1, likelihood_func, optimize_init=False, chain_name="method_test_1")
print("⚠️  期望显示: Method: sequential (单设备单链)")

# 多链配置
config2 = {
    'parameters': {
        'a': (2.0, 0.0, 4.0),
        'b': (1.0, 0.0, 2.0)
    },
    'mcmc': {
        'num_samples': 500,
        'num_warmup': 300,
        'num_chains': 4  # 多链但单设备
    }
}

print(f"\n配置: {config2['mcmc']['num_chains']} 链")
mcmc2 = MCMC(config2, likelihood_func, optimize_init=False, chain_name="method_test_2")
print("⚠️  期望显示: Method: sequential (单设备多链)")

print("\n✅ 单设备测试完成。接下来需要在新进程中测试多设备环境。")