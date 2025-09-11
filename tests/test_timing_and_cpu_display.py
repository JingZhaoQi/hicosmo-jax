#!/usr/bin/env python3
"""
测试运行时长和CPU核数显示功能
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 优雅的多核初始化
from hicosmo.samplers import Config
Config.init(cpu_cores=4)

import numpy as np
import jax
from hicosmo.samplers import MCMC

print("🔬 测试运行时长和CPU核数显示功能")
print("="*60)

# 简单测试数据
np.random.seed(42)
x = np.linspace(0, 1, 50)
y_true = 2.0 * x + 1.0
y_err = 0.1 * np.ones_like(x)
y_obs = y_true + np.random.normal(0, y_err)

def likelihood_func(a, b):
    y_pred = a * x + b
    chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
    return -0.5 * chi2

print(f"当前JAX设备数: {jax.local_device_count()}")

# 测试配置
config = {
    'parameters': {
        'a': (2.0, 0.0, 4.0),
        'b': (1.0, 0.0, 2.0)
    },
    'mcmc': {
        'num_samples': 20000,
        'num_warmup': 100,
        'num_chains': 4
    }
}

print(f"\n🔧 运行MCMC测试...")
print(f"期望看到:")
print(f"1. MCMC Configuration 面板应显示 'CPU Cores: 4 CPU cores'")
print(f"2. 应显示开始时间")
print(f"3. 应显示结束时间和详细时长信息")
print("-"*60)

# 运行MCMC
mcmc = MCMC(config, likelihood_func, optimize_init=False, 
           chain_name="timing_test")
samples = mcmc.run()

print(f"\n✅ 测试完成!")
print(f"检查上方输出是否包含:")
print(f"  ✓ CPU Cores 信息在配置面板中")
print(f"  ✓ 开始时间显示")
print(f"  ✓ 结束时间和详细时长显示")