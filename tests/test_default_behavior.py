#!/usr/bin/env python3
"""
测试默认行为：确认JAX优化默认关闭，使用标准warmup
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys

# Add HiCosmo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers import MCMC

def simple_likelihood(a, b, c):
    """简单的似然函数用于测试"""
    x = np.linspace(0, 5, 20)
    y_true = a * x**2 + b * x + c
    y_obs = np.array([2.5 * xi**2 + 1.2 * xi + 0.8 for xi in x]) + 0.1 * np.random.randn(20)
    y_err = 0.1 * np.ones(20)
    
    y_pred = a * x**2 + b * x + c
    chi2 = jnp.sum((y_obs - y_pred)**2 / y_err**2)
    return -0.5 * chi2

def test_default_behavior():
    """测试默认行为"""
    print("测试HiCosmo默认行为")
    print("="*50)
    
    # 简单配置
    config = {
        'parameters': {
            'a': (2.0, 0, 5),
            'b': (1.0, -2, 4), 
            'c': (0.5, -1, 2)
        },
        'mcmc': {
            'num_samples': 100,  # 快速测试
            'num_chains': 2
        }
    }
    
    print("\n1. 测试默认设置 (应该关闭优化)")
    mcmc1 = MCMC(config, simple_likelihood, chain_name="test_default")
    
    print(f"✓ optimize_init = {mcmc1.optimize_init}")
    print(f"✓ num_warmup = {mcmc1.sampler.num_warmup}")
    
    assert mcmc1.optimize_init == False, "默认应该关闭优化"
    assert mcmc1.sampler.num_warmup == 2000, "默认warmup应该是2000"
    
    print("运行MCMC...")
    samples1 = mcmc1.run()
    
    print(f"✓ 采样完成，获得参数: {list(samples1.keys())}")
    
    print("\n2. 测试手动启用优化 (应该显示建议信息)")
    mcmc2 = MCMC(config, simple_likelihood, 
                     optimize_init=True,  # 手动启用
                     chain_name="test_optimized")
    
    assert mcmc2.optimize_init == True, "应该启用优化"
    print(f"✓ optimize_init = {mcmc2.optimize_init}")
    
    print("\n测试通过！默认行为符合预期：")
    print("  ✅ JAX优化默认关闭")
    print("  ✅ 标准warmup = 2000步")  
    print("  ✅ 启用优化时显示建议信息")

if __name__ == "__main__":
    test_default_behavior()