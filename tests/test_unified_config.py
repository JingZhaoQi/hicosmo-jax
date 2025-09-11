#!/usr/bin/env python3
"""
测试统一配置接口：智能warmup设置
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys
import numpyro

# Add HiCosmo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers import MCMC

# Enable multi-core for testing
numpyro.set_host_device_count(2)

def simple_likelihood(a, b):
    """简单的似然函数用于测试"""
    x = np.linspace(0, 5, 20)
    y_obs = 2.0 * x + 1.0 + 0.1 * np.random.randn(20)
    y_pred = a * x + b
    chi2 = jnp.sum((y_obs - y_pred)**2 / 0.1**2)
    return -0.5 * chi2

def test_unified_config():
    """测试统一配置接口"""
    print("测试统一MCMC配置接口")
    print("="*50)
    
    # 基础配置（不指定warmup）
    base_config = {
        'parameters': {
            'a': (1.8, 0, 4),
            'b': (0.8, -1, 3), 
        },
        'mcmc': {
            'num_samples': 500,  # 快速测试
            'num_chains': 2
        }
    }
    
    print("\n🧪 测试1: 默认模式 (无优化)")
    print("-" * 40)
    mcmc1 = MCMC(base_config, simple_likelihood, chain_name="test_no_opt")
    print(f"✓ optimize_init = {mcmc1.optimize_init}")
    print(f"✓ num_warmup = {mcmc1.sampler.num_warmup} (应该是2000)")
    assert mcmc1.optimize_init == False
    assert mcmc1.sampler.num_warmup == 2000
    
    print("\n🧪 测试2: 启用优化模式")
    print("-" * 40)  
    mcmc2 = MCMC(base_config, simple_likelihood, 
                     optimize_init=True, chain_name="test_opt")
    print(f"✓ optimize_init = {mcmc2.optimize_init}")
    print(f"✓ num_warmup = {mcmc2.sampler.num_warmup} (应该是300)")
    assert mcmc2.optimize_init == True
    assert mcmc2.sampler.num_warmup == 300
    
    print("\n🧪 测试3: 用户指定warmup（优先级最高）")
    print("-" * 40)
    custom_config = base_config.copy()
    custom_config['mcmc']['num_warmup'] = 1500  # 用户指定
    
    mcmc3 = MCMC(custom_config, simple_likelihood, 
                     optimize_init=True, chain_name="test_custom")
    print(f"✓ optimize_init = {mcmc3.optimize_init}")
    print(f"✓ num_warmup = {mcmc3.sampler.num_warmup} (应该是1500，用户指定)")
    assert mcmc3.optimize_init == True
    assert mcmc3.sampler.num_warmup == 1500
    
    print("\n🧪 测试4: 用户指定warmup，无优化")
    print("-" * 40)
    mcmc4 = MCMC(custom_config, simple_likelihood, 
                     optimize_init=False, chain_name="test_custom_no_opt")
    print(f"✓ optimize_init = {mcmc4.optimize_init}")
    print(f"✓ num_warmup = {mcmc4.sampler.num_warmup} (应该是1500，用户指定)")
    assert mcmc4.optimize_init == False
    assert mcmc4.sampler.num_warmup == 1500
    
    print("\n" + "="*50)
    print("🎉 统一配置接口测试通过！")
    print("✅ 策略正确实现：")
    print("   • 无优化 + 无warmup指定 → 2000")
    print("   • 有优化 + 无warmup指定 → 300")
    print("   • 用户指定warmup → 用户值（优先级最高）")
    print("   • 所有逻辑集中在MCMC._apply_intelligent_defaults()")

if __name__ == "__main__":
    test_unified_config()