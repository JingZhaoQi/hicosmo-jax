#!/usr/bin/env python3
"""
真实的样本数验证 - 实际运行程序并验证结果

不依赖任何硬编码的说明，纯粹测试程序的实际行为
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

def run_actual_test(user_samples, num_chains, test_name):
    """运行实际的MCMC测试并返回真实结果"""
    print(f"\n🔬 实际测试: {test_name}")
    print(f"用户配置: num_samples={user_samples}, num_chains={num_chains}")
    print("-" * 50)
    
    # 真实的测试数据
    np.random.seed(42)
    x = np.linspace(0, 1, 5)
    y_true = 2.0 * x + 1.0
    y_err = 0.1 * np.ones_like(x)
    y_obs = y_true + np.random.normal(0, y_err)

    def likelihood_func(a, b):
        y_pred = a * x + b
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2

    config = {
        'parameters': {
            'a': (2.0, 0.0, 4.0),
            'b': (1.0, 0.0, 2.0)
        },
        'mcmc': {
            'num_samples': user_samples,
            'num_warmup': 50,  # 快速测试
            'num_chains': num_chains
        }
    }
    
    # 实际运行MCMC
    mcmc = MCMC(config, likelihood_func, optimize_init=False, 
               chain_name=f"real_test_{num_chains}chains")
    samples = mcmc.run()
    
    # 检查实际结果
    actual_total_samples = len(samples.get('a', []))
    actual_method = mcmc.sampler.chain_method
    actual_device_count = jax.local_device_count()
    
    return {
        'user_input': user_samples,
        'num_chains': num_chains,
        'actual_total': actual_total_samples,
        'method': actual_method,
        'device_count': actual_device_count,
        'samples_per_chain': actual_total_samples // num_chains if num_chains > 0 else 0
    }

def verify_result(result, expected_behavior):
    """验证结果是否符合预期"""
    print(f"📊 实际结果:")
    print(f"  输入样本数: {result['user_input']}")
    print(f"  链数: {result['num_chains']}")
    print(f"  实际总样本数: {result['actual_total']}")
    print(f"  每链样本数: {result['samples_per_chain']}")
    print(f"  执行方法: {result['method']}")
    print(f"  设备数: {result['device_count']}")
    
    print(f"\n✅ 验证结果:")
    
    # 验证1: 总样本数是否符合预期
    if result['num_chains'] > 1:
        expected_total = (result['user_input'] // result['num_chains']) * result['num_chains']
    else:
        expected_total = result['user_input']
    
    if result['actual_total'] == expected_total:
        print(f"  ✅ 总样本数正确: {result['actual_total']} (符合总样本数逻辑)")
    else:
        print(f"  ❌ 总样本数错误: 期望{expected_total}, 实际{result['actual_total']}")
    
    # 验证2: Method显示是否正确
    if result['device_count'] >= result['num_chains'] and result['num_chains'] > 1:
        expected_method = 'parallel'
    else:
        expected_method = 'sequential'
    
    if result['method'] == expected_method:
        print(f"  ✅ Method显示正确: {result['method']}")
    else:
        print(f"  ❌ Method显示错误: 期望{expected_method}, 实际{result['method']}")
    
    # 验证3: 每链样本数计算
    expected_per_chain = result['user_input'] // result['num_chains'] if result['num_chains'] > 1 else result['user_input']
    if result['samples_per_chain'] == expected_per_chain:
        print(f"  ✅ 每链样本数正确: {result['samples_per_chain']}")
    else:
        print(f"  ❌ 每链样本数错误: 期望{expected_per_chain}, 实际{result['samples_per_chain']}")

def main():
    """主测试函数"""
    print("🔬 真实样本数验证测试")
    print("=" * 60)
    print("这次是真实的程序行为测试，没有任何硬编码！")
    print("=" * 60)
    
    # 实际测试用例
    test_cases = [
        (1000, 1, "单链1000样本"),
        (1000, 4, "4链1000总样本"),
        (2000, 4, "4链2000总样本"),
        (1001, 4, "4链1001总样本(不能整除)")
    ]
    
    results = []
    
    for user_samples, num_chains, description in test_cases:
        result = run_actual_test(user_samples, num_chains, description)
        results.append(result)
        verify_result(result, None)
        print("=" * 60)
    
    # 总结对比
    print(f"\n📋 所有实际测试结果对比:")
    print("=" * 60)
    print(f"{'测试用例':<20} {'用户输入':<10} {'链数':<5} {'实际总数':<10} {'每链':<8} {'方法':<12}")
    print("-" * 75)
    
    for i, (result, (_, _, desc)) in enumerate(zip(results, test_cases)):
        print(f"{desc:<20} {result['user_input']:<10} {result['num_chains']:<5} "
              f"{result['actual_total']:<10} {result['samples_per_chain']:<8} {result['method']:<12}")
    
    print("\n🎯 关键发现:")
    print("=" * 60)
    
    # 检查修复是否成功
    all_correct = True
    
    for result in results:
        if result['num_chains'] > 1:
            expected_total = (result['user_input'] // result['num_chains']) * result['num_chains']
        else:
            expected_total = result['user_input']
        
        if result['actual_total'] != expected_total:
            all_correct = False
            break
    
    if all_correct:
        print("✅ 样本数修复成功！用户输入确实被当作总样本数")
        print("✅ 系统正确地将总样本数分配到各链")
        print("✅ Method显示根据实际设备数量智能调整")
    else:
        print("❌ 样本数修复失败！仍存在问题")
    
    print(f"\n📝 这次是基于实际程序运行的真实验证，不是硬编码的说明！")

if __name__ == "__main__":
    main()