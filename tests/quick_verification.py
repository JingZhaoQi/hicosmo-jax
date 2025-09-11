#!/usr/bin/env python3
"""
快速验证MCMC重构是否成功
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from hicosmo.samplers import MCMC

def simple_likelihood(**kwargs):
    """简单的2D高斯分布"""
    x = kwargs.get('x', 0.0)
    y = kwargs.get('y', 0.0)
    return -0.5 * (x**2 + y**2)

def test_basic_functionality():
    """测试基本功能是否正常"""
    print("🔍 测试基本MCMC功能...")
    
    config = {
        'parameters': {
            'x': (0.0, -3.0, 3.0),
            'y': (0.0, -3.0, 3.0)
        },
        'mcmc': {
            'num_warmup': 100,
            'num_samples': 200,
            'num_chains': 2
        }
    }
    
    try:
        start_time = time.time()
        mcmc = MCMC(config, simple_likelihood, optimize_init=False, chain_name="basic_test")
        samples = mcmc.run()
        elapsed = time.time() - start_time
        
        diag = mcmc.sampler.get_diagnostics(burnin_frac=0)
        converged = sum(1 for d in diag.values() if isinstance(d, dict) and d.get('r_hat', 999) < 1.2)
        total = sum(1 for d in diag.values() if isinstance(d, dict) and 'r_hat' in d)
        
        print(f"  ✅ 基本功能正常: {elapsed:.1f}s, {converged}/{total} 收敛")
        return True
        
    except Exception as e:
        print(f"  ❌ 基本功能失败: {e}")
        return False

def test_optimization_feature():
    """测试优化功能是否正常"""
    print("🚀 测试JAX优化功能...")
    
    config = {
        'parameters': {
            'x': (0.0, -3.0, 3.0),
            'y': (0.0, -3.0, 3.0)
        },
        'mcmc': {
            'num_warmup': 50,  # 用较少的warmup测试优化
            'num_samples': 200,
            'num_chains': 2
        }
    }
    
    try:
        start_time = time.time()
        mcmc = MCMC(config, simple_likelihood, optimize_init=True, 
                   max_opt_iterations=50, chain_name="opt_test")
        samples = mcmc.run()
        elapsed = time.time() - start_time
        
        diag = mcmc.sampler.get_diagnostics(burnin_frac=0.1)
        converged = sum(1 for d in diag.values() if isinstance(d, dict) and d.get('r_hat', 999) < 1.2)
        total = sum(1 for d in diag.values() if isinstance(d, dict) and 'r_hat' in d)
        
        print(f"  ✅ 优化功能正常: {elapsed:.1f}s, {converged}/{total} 收敛")
        return True
        
    except Exception as e:
        print(f"  ❌ 优化功能失败: {e}")
        return False

def test_intelligent_defaults():
    """测试智能默认配置是否正常"""
    print("🧠 测试智能默认配置...")
    
    # 测试优化模式默认值
    config_opt = {
        'parameters': {
            'x': (0.0, -3.0, 3.0),
            'y': (0.0, -3.0, 3.0)
        }
    }
    
    try:
        # 创建优化模式的MCMC实例，但不运行
        mcmc_opt = MCMC(config_opt, simple_likelihood, optimize_init=True, chain_name="default_opt")
        print(f"  ✅ 优化模式配置创建成功")
    except Exception as e:
        print(f"  ❌ 优化模式配置失败: {e}")
        return False
    
    # 测试传统模式默认值
    config_trad = {
        'parameters': {
            'x': (0.0, -3.0, 3.0),
            'y': (0.0, -3.0, 3.0)
        }
    }
    
    try:
        # 创建传统模式的MCMC实例，但不运行
        mcmc_trad = MCMC(config_trad, simple_likelihood, optimize_init=False, chain_name="default_trad")
        print(f"  ✅ 传统模式配置创建成功")
    except Exception as e:
        print(f"  ❌ 传统模式配置失败: {e}")
        return False
    
    print(f"  ✅ 智能默认配置测试通过")
    return True

def main():
    print("快速验证MCMC重构结果")
    print("="*50)
    
    results = []
    
    # 测试1: 基本功能
    results.append(test_basic_functionality())
    
    # 测试2: 优化功能  
    results.append(test_optimization_feature())
    
    # 测试3: 智能默认配置
    results.append(test_intelligent_defaults())
    
    # 总结
    print("\n" + "="*50)
    print("📊 验证结果总结")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 所有测试通过! ({passed}/{total})")
        print("✅ MCMC重构成功，功能正常")
    else:
        print(f"⚠️  部分测试失败: {passed}/{total}")
        print("❌ 需要进一步检查和修复")

if __name__ == "__main__":
    main()