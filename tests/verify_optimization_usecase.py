#!/usr/bin/env python3
"""
验证JAX优化在什么情况下才真正有用的测试

测试两种极端情况：
1. 计算昂贵的似然函数 - 模拟复杂的物理计算
2. 收敛困难的多模态问题 - 传统MCMC容易失败的情况
"""

import numpy as np
import jax.numpy as jnp
import jax
from pathlib import Path
import sys
import time
import numpyro

sys.path.insert(0, str(Path(__file__).parent.parent))
from hicosmo.samplers import MCMC

numpyro.set_host_device_count(4)


def create_expensive_likelihood():
    """创建计算昂贵的似然函数 - 模拟复杂物理计算"""
    np.random.seed(42)
    n_data = 50
    true_params = [2.0, 1.5, 0.8]
    
    x = np.linspace(0, 10, n_data)
    y_true = true_params[0] * np.sin(true_params[1] * x) * np.exp(-true_params[2] * x)
    y_err = 0.1 * np.ones_like(x)
    y_obs = y_true + np.random.normal(0, y_err)
    
    def expensive_likelihood(a, b, c):
        """模拟昂贵的计算 - 每次调用耗时显著"""
        # 模拟复杂的数值计算（例如积分、求解微分方程等）
        expensive_calc = 0
        for i in range(1000):  # 昂贵的循环计算
            expensive_calc += jnp.sin(i * a) * jnp.cos(i * b) * jnp.exp(-i * c / 1000)
        
        # 真正的似然计算
        y_pred = a * jnp.sin(b * x) * jnp.exp(-c * x)
        chi2 = jnp.sum((y_obs - y_pred)**2 / y_err**2)
        
        return -0.5 * chi2 + expensive_calc * 1e-6  # 添加昂贵计算的微小贡献
    
    config = {
        'parameters': {
            'a': (1.0, 0.5, 4.0),
            'b': (1.0, 0.5, 3.0), 
            'c': (0.5, 0.1, 2.0)
        }
    }
    
    return expensive_likelihood, config, true_params


def create_multimodal_likelihood():
    """创建多模态似然函数 - 收敛困难的问题"""
    def multimodal_likelihood(x, y):
        """双峰分布 - 传统MCMC容易卡在一个峰上"""
        # 两个峰：一个在(-2, -2)，一个在(2, 2)
        peak1 = -0.5 * ((x + 2)**2 + (y + 2)**2) / 0.5**2
        peak2 = -0.5 * ((x - 2)**2 + (y - 2)**2) / 0.5**2
        
        # log-sum-exp技巧避免数值不稳定
        max_peak = jnp.maximum(peak1, peak2)
        log_likelihood = max_peak + jnp.log(jnp.exp(peak1 - max_peak) + jnp.exp(peak2 - max_peak))
        
        return log_likelihood
    
    config = {
        'parameters': {
            'x': (0.0, -5.0, 5.0),
            'y': (0.0, -5.0, 5.0)
        }
    }
    
    return multimodal_likelihood, config


def test_expensive_likelihood():
    """测试计算昂贵的似然函数"""
    print("\n" + "="*60)
    print("测试1: 计算昂贵的似然函数")
    print("模拟每次似然计算耗时显著的情况")
    print("="*60)
    
    likelihood, config, true_params = create_expensive_likelihood()
    
    # 传统方法
    print("\n🐌 传统方法 (更多warmup)")
    config1 = config.copy()
    config1['mcmc'] = {'num_warmup': 1000, 'num_samples': 1000, 'num_chains': 2}
    
    start_time = time.time()
    mcmc1 = MCMC(config1, likelihood, optimize_init=False, chain_name="expensive_trad")
    samples1 = mcmc1.run()
    time1 = time.time() - start_time
    
    diag1 = mcmc1.sampler.get_diagnostics(burnin_frac=0)
    converged1 = sum(1 for d in diag1.values() if isinstance(d, dict) and d.get('r_hat', 999) < 1.1)
    total1 = sum(1 for d in diag1.values() if isinstance(d, dict) and 'r_hat' in d)
    
    # 优化方法
    print("\n🚀 优化方法 (减少似然计算)")  
    config2 = config.copy()
    config2['mcmc'] = {'num_warmup': 200, 'num_samples': 1000, 'num_chains': 2}
    
    start_time = time.time()
    mcmc2 = MCMC(config2, likelihood, optimize_init=True, 
                     max_opt_iterations=100, chain_name="expensive_opt")
    samples2 = mcmc2.run()
    time2 = time.time() - start_time
    
    diag2 = mcmc2.sampler.get_diagnostics(burnin_frac=0.1)
    converged2 = sum(1 for d in diag2.values() if isinstance(d, dict) and d.get('r_hat', 999) < 1.1)
    total2 = sum(1 for d in diag2.values() if isinstance(d, dict) and 'r_hat' in d)
    
    print(f"\n📊 昂贵似然函数结果:")
    print(f"  传统方法: {time1:.1f}s, {converged1}/{total1} 收敛")
    print(f"  优化方法: {time2:.1f}s, {converged2}/{total2} 收敛") 
    print(f"  加速比: {time1/time2:.2f}x")
    
    return time1/time2


def test_multimodal_problem():
    """测试多模态收敛困难问题"""
    print("\n" + "="*60)
    print("测试2: 多模态收敛困难问题")
    print("模拟传统MCMC容易卡在局部最优的情况")
    print("="*60)
    
    likelihood, config = create_multimodal_likelihood()
    
    # 传统方法 - 可能卡在局部最优
    print("\n😰 传统方法 (随机初始化)")
    config1 = config.copy()
    config1['mcmc'] = {'num_warmup': 1000, 'num_samples': 2000, 'num_chains': 4}
    
    try:
        start_time = time.time()
        mcmc1 = MCMC(config1, likelihood, optimize_init=False, chain_name="multimodal_trad")
        samples1 = mcmc1.run()
        time1 = time.time() - start_time
        
        diag1 = mcmc1.sampler.get_diagnostics(burnin_frac=0)
        converged1 = sum(1 for d in diag1.values() if isinstance(d, dict) and d.get('r_hat', 999) < 1.1)
        total1 = sum(1 for d in diag1.values() if isinstance(d, dict) and 'r_hat' in d)
        
        # 检查是否找到了两个模式
        x_mean = np.mean(samples1['x'])
        y_mean = np.mean(samples1['y'])
        print(f"  找到的解: x={x_mean:.2f}, y={y_mean:.2f}")
        
    except Exception as e:
        print(f"  ❌ 传统方法失败: {e}")
        converged1, total1, time1 = 0, 2, float('inf')
    
    # 优化方法 - 从全局最优开始
    print("\n🎯 优化方法 (全局最优初始化)")
    config2 = config.copy()
    config2['mcmc'] = {'num_warmup': 200, 'num_samples': 2000, 'num_chains': 4}
    
    try:
        start_time = time.time()
        mcmc2 = MCMC(config2, likelihood, optimize_init=True,
                        max_opt_iterations=300, chain_name="multimodal_opt")
        samples2 = mcmc2.run()
        time2 = time.time() - start_time
        
        diag2 = mcmc2.sampler.get_diagnostics(burnin_frac=0.1)
        converged2 = sum(1 for d in diag2.values() if isinstance(d, dict) and d.get('r_hat', 999) < 1.1)
        total2 = sum(1 for d in diag2.values() if isinstance(d, dict) and 'r_hat' in d)
        
        x_mean = np.mean(samples2['x'])
        y_mean = np.mean(samples2['y'])
        print(f"  找到的解: x={x_mean:.2f}, y={y_mean:.2f}")
        
    except Exception as e:
        print(f"  ❌ 优化方法失败: {e}")
        converged2, total2, time2 = 0, 2, float('inf')
    
    print(f"\n📊 多模态问题结果:")
    print(f"  传统方法: {time1:.1f}s, {converged1}/{total1} 收敛")
    print(f"  优化方法: {time2:.1f}s, {converged2}/{total2} 收敛")
    if time1 != float('inf') and time2 != float('inf'):
        print(f"  加速比: {time1/time2:.2f}x")
    
    return converged1/total1, converged2/total2


def main():
    print("验证JAX优化的真正适用场景")
    print("="*70)
    
    # 测试1: 计算昂贵的似然函数
    speedup1 = test_expensive_likelihood()
    
    # 测试2: 多模态收敛困难问题  
    conv1, conv2 = test_multimodal_problem()
    
    # 总结
    print("\n" + "="*70)
    print("🏆 最终结论")
    print("="*70)
    
    print(f"\n💰 计算昂贵的似然函数:")
    if speedup1 > 1.2:
        print(f"  ✅ JAX优化有效! {speedup1:.2f}x 加速")
        print(f"  📝 原因: 减少昂贵的似然计算次数")
    else:
        print(f"  ❌ JAX优化无效: {speedup1:.2f}x")
        print(f"  📝 原因: 似然计算不够昂贵，优化开销占主导")
    
    print(f"\n🎭 多模态困难问题:")
    print(f"  传统方法收敛率: {conv1:.1%}")
    print(f"  优化方法收敛率: {conv2:.1%}")
    
    if conv2 > conv1:
        print(f"  ✅ JAX优化提高了收敛成功率!")
        print(f"  📝 原因: 好的初始点避免了局部最优陷阱")
    else:
        print(f"  ❌ JAX优化没有明显改善收敛")
        
    print(f"\n🎯 实用建议:")
    print(f"  • 常规统计推断: 直接用NumPyro，不需要优化")
    print(f"  • 昂贵似然函数: JAX优化可能有用")
    print(f"  • 收敛困难问题: JAX优化可能提高成功率")
    print(f"  • 真正的优化: JIT编译似然函数，使用更多并行链")


if __name__ == "__main__":
    main()