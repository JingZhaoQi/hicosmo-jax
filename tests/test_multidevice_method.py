#!/usr/bin/env python3
"""
测试多设备环境的Method显示
"""

import os
import sys
from pathlib import Path

# 设置多设备环境
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import numpyro
numpyro.set_host_device_count(4)

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
print("🔧 测试多设备环境的Method显示")
print("="*50)
print(f"JAX设备数: {jax.local_device_count()}")
print(f"JAX设备: {jax.devices()}")

from hicosmo.samplers.core import MCMCSampler

def simple_model():
    import numpyro
    import numpyro.distributions as dist
    a = numpyro.sample('a', dist.Normal(0, 1))
    b = numpyro.sample('b', dist.Normal(0, 1))

# 测试不同的链配置
configs = [
    (1, "单链"),
    (2, "2链"),
    (4, "4链"),
    (8, "8链")
]

print(f"\n📋 测试多设备环境下不同链配置:")
print("-"*40)

for num_chains, desc in configs:
    sampler = MCMCSampler(simple_model, num_chains=num_chains, num_samples=100, num_warmup=50)
    print(f"{desc:>6}: {sampler.chain_method}")

print(f"\n💡 逻辑说明:")
print(f"  设备数: {jax.local_device_count()}")
print(f"  预期:")
print(f"    1链: sequential (单链总是sequential)")
print(f"    2链: parallel (2链 <= 4设备)")
print(f"    4链: parallel (4链 = 4设备)")
print(f"    8链: sequential (8链 > 4设备，设备不足)")