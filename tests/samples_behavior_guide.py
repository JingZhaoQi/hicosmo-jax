#!/usr/bin/env python3
"""
HiCosmo 样本数行为指南

展示新的"总样本数"逻辑，让用户理解直观的样本配置
"""

import os
import sys
from pathlib import Path

# 设置多设备环境
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
import numpyro
numpyro.set_host_device_count(4)

sys.path.insert(0, str(Path(__file__).parent.parent))

print("📚 HiCosmo 样本数行为指南")
print("="*60)
print("新的直观样本配置：用户输入 = 总样本数")
print("="*60)

# 示例配置
configurations = [
    {
        'title': "单链配置",
        'config': {
            'parameters': {'a': (1.0, 0.0, 2.0)},
            'mcmc': {
                'num_samples': 1000,  # 用户想要1000个样本
                'num_chains': 1
            }
        },
        'expected': "得到1000个样本"
    },
    {
        'title': "4链配置",
        'config': {
            'parameters': {'a': (1.0, 0.0, 2.0)},
            'mcmc': {
                'num_samples': 1000,  # 用户想要1000个样本
                'num_chains': 4
            }
        },
        'expected': "得到1000个样本 (每链250个)"
    },
    {
        'title': "大样本配置",
        'config': {
            'parameters': {'a': (1.0, 0.0, 2.0)},
            'mcmc': {
                'num_samples': 10000,  # 用户想要10000个样本
                'num_chains': 4
            }
        },
        'expected': "得到10000个样本 (每链2500个)"
    },
    {
        'title': "不能整除的情况",
        'config': {
            'parameters': {'a': (1.0, 0.0, 2.0)},
            'mcmc': {
                'num_samples': 1003,  # 用户想要1003个样本
                'num_chains': 4
            }
        },
        'expected': "得到1000个样本 (每链250个，舍入-3)"
    }
]

from hicosmo.samplers import MCMC
import numpy as np

# 简单测试数据
np.random.seed(42)
x = np.array([1, 2, 3])
y = np.array([2, 4, 6])

def simple_likelihood(a):
    return -0.5 * np.sum((y - a * x)**2)

print(f"\n📋 配置行为演示:")
print("-"*60)

for i, example in enumerate(configurations, 1):
    print(f"\n{i}. {example['title']}")
    print(f"   用户配置: {example['config']['mcmc']}")
    print(f"   期望结果: {example['expected']}")
    print("   " + "-"*30)
    
    # 创建MCMC对象（不运行，只看配置输出）
    mcmc = MCMC(example['config'], simple_likelihood, 
               optimize_init=False, 
               chain_name=f"guide_example_{i}")
    
    print()

print(f"\n💡 关键要点:")
print("="*60)
print("1. ✅ 直观配置：用户想要多少样本就写多少")
print("   'num_samples': 2000  → 总共得到2000个样本")
print()
print("2. ✅ 自动分配：系统自动分配到各链")
print("   4链配置下：2000样本 → 每链500个")
print()
print("3. ✅ 智能处理：不能整除时向下舍入")
print("   1003样本，4链 → 1000样本 (每链250个)")
print()
print("4. ✅ 性能优化：多链并行执行")
print("   4链在4核CPU上真正并行运行")

print(f"\n🚀 用户体验改进:")
print("="*60)
print("修复前的问题:")
print("  ❌ 用户输入2000，实际得到8000 (2000×4链)")
print("  ❌ 用户需要心算：想要2000样本 → 输入500")
print()
print("修复后的体验:")
print("  ✅ 用户输入2000，实际得到2000")
print("  ✅ 无需考虑链数，系统自动分配")
print("  ✅ 配置直观，符合用户预期")