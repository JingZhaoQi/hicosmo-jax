#!/usr/bin/env python3
"""
测试修复后的Method显示
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

print("🔧 测试修复后的Method显示")
print("="*50)

# 测试1：单设备环境
print("\n📋 测试1: 单设备环境 (期望显示sequential)")
print("-"*40)

# 清除多核环境变量
if 'XLA_FLAGS' in os.environ:
    del os.environ['XLA_FLAGS']

import numpyro
numpyro.set_host_device_count(1)

import jax
print(f"JAX设备数: {jax.local_device_count()}")

from hicosmo.samplers.core import MCMCSampler

# 测试core.py的MCMCSampler直接使用
def simple_model():
    import numpyro
    import numpyro.distributions as dist
    a = numpyro.sample('a', dist.Normal(0, 1))
    b = numpyro.sample('b', dist.Normal(0, 1))

# 单链测试
print("\n单链配置:")
sampler1 = MCMCSampler(simple_model, num_chains=1, num_samples=100, num_warmup=50)
print(f"检测到的方法: {sampler1.chain_method}")

# 多链测试  
print("\n多链配置:")
sampler2 = MCMCSampler(simple_model, num_chains=4, num_samples=100, num_warmup=50)
print(f"检测到的方法: {sampler2.chain_method}")

print(f"\n✅ 单设备测试完成")
print(f"预期: 单链和多链都应该显示 'sequential'")
print(f"实际: 单链={sampler1.chain_method}, 多链={sampler2.chain_method}")

# 验证逻辑
if sampler1.chain_method == 'sequential' and sampler2.chain_method == 'sequential':
    print("🎉 单设备检测逻辑正确!")
else:
    print("❌ 单设备检测逻辑有问题!")