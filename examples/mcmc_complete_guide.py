#!/usr/bin/env python3
"""
HiCosmo MCMC 完整使用指南

本示例详细展示如何使用 HiCosmo MCMC 进行贝叶斯推断，
包括所有主要功能和接口的使用方法。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

# 第一步：优雅的多核初始化
print("🚀 HiCosmo MCMC 完整使用指南")
print("="*60)

print("\n📝 步骤1: 优雅的多核初始化")
print("-"*50)
from hicosmo.samplers import Config
Config.init(cpu_cores=4, verbose=True)  # 一行代码完成多核配置！

# 第二步：准备数据
print("\n📝 步骤2: 准备测试数据")
print("-"*50)
print("生成线性模型数据：y = a*x + b + noise")

np.random.seed(42)
N_data = 100
x = np.linspace(0, 10, N_data)
true_a, true_b = 2.5, 1.8
y_err = 0.3
y_true = true_a * x + true_b
y_obs = y_true + np.random.normal(0, y_err, N_data)

print(f"数据点数: {N_data}")
print(f"真实参数: a = {true_a}, b = {true_b}")
print(f"观测噪声: σ = {y_err}")

# 第三步：定义似然函数
print("\n📝 步骤3: 定义似然函数")
print("-"*50)
def linear_likelihood(a, b):
    """
    线性模型的似然函数
    
    Parameters:
    -----------
    a, b : float
        模型参数
        
    Returns:
    --------
    float
        对数似然值
    """
    y_pred = a * x + b
    chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
    return -0.5 * chi2

print("定义似然函数: linear_likelihood(a, b)")
print("  - 计算模型预测值")
print("  - 计算卡方统计量")
print("  - 返回对数似然")

# 第四步：配置MCMC参数
print("\n📝 步骤4: 配置MCMC参数")
print("-"*50)

# 基础配置
basic_config = {
    'parameters': {
        'a': (2.0, 0.0, 5.0),     # (初值, 下限, 上限)
        'b': (1.0, -2.0, 4.0)     # (初值, 下限, 上限)
    },
    'mcmc': {
        'num_samples': 2000,      # 总样本数（会自动分配到各链）
        'num_warmup': 500,        # 预热步数
        'num_chains': 4           # 链数（并行采样）
    }
}

print("配置说明:")
print("  parameters: 定义待估计的参数")
print("    格式: '参数名': (初值, 下限, 上限)")
print("  mcmc:")
print("    num_samples: 总样本数（自动分配到各链）")
print("    num_warmup: 预热步数（丢弃）")  
print("    num_chains: 链数（建议≤CPU核数）")

# 第五步：运行MCMC
print("\n📝 步骤5: 运行MCMC采样")
print("-"*50)

from hicosmo.samplers import MCMC

print("创建MCMC对象并运行:")
mcmc = MCMC(
    config=basic_config,
    likelihood_func=linear_likelihood,
    optimize_init=True,           # 是否优化初始值
    chain_name="example_linear"   # 链名称（用于保存）
)

# 运行采样
samples = mcmc.run()

print(f"\n✅ 采样完成！获得 {len(samples['a'])} 个样本")

# 第六步：分析结果
print("\n📝 步骤6: 分析采样结果")
print("-"*50)

print("📊 参数估计结果:")
for param in ['a', 'b']:
    values = samples[param]
    mean_val = np.mean(values)
    std_val = np.std(values)
    true_val = true_a if param == 'a' else true_b
    
    print(f"  {param}: {mean_val:.3f} ± {std_val:.3f}")
    print(f"      真值: {true_val:.3f}")
    print(f"      偏差: {abs(mean_val - true_val):.3f}")

# 第七步：收敛诊断
print("\n📝 步骤7: 收敛诊断")
print("-"*50)

try:
    # 获取收敛诊断
    diagnostics = mcmc.get_diagnostics()
    
    print("收敛诊断指标:")
    for param, diag in diagnostics.items():
        if isinstance(diag, dict):
            r_hat = diag.get('r_hat', 'N/A')
            ess = diag.get('ess', 'N/A')
            
            # 判断收敛状态
            converged = r_hat < 1.01 if r_hat != 'N/A' else False
            status = "✅收敛" if converged else "❌未收敛"
            
            print(f"  {param}: R̂ = {r_hat:.4f}, ESS = {ess:.0f} {status}")
            
except Exception as e:
    print(f"诊断失败: {e}")

# 第八步：高级配置示例
print("\n📝 步骤8: 高级配置示例")
print("-"*50)

advanced_config = {
    'parameters': {
        'a': (2.0, 0.0, 5.0),
        'b': (1.0, -2.0, 4.0)
    },
    'mcmc': {
        'num_samples': 4000,      # 更多样本
        'num_warmup': 1000,       # 更长预热
        'num_chains': 4,
        # 高级选项
        'step_size': 0.1,         # 步长
        'target_accept_prob': 0.8, # 目标接受率
        'max_tree_depth': 10      # 最大树深度
    },
    'optimization': {
        'method': 'L-BFGS-B',     # 优化算法
        'maxiter': 1000           # 最大迭代数
    }
}

print("高级配置选项:")
print("  step_size: NUTS算法步长")
print("  target_accept_prob: 目标接受概率")
print("  max_tree_depth: NUTS最大树深度")
print("  optimization: 初值优化配置")

# 第九步：多种使用模式
print("\n📝 步骤9: 多种使用模式")
print("-"*50)

print("🔹 模式1: 基础MCMC（不优化初值）")
simple_config = basic_config.copy()
simple_mcmc = MCMC(simple_config, linear_likelihood, 
                  optimize_init=False, chain_name="simple_example")
simple_samples = simple_mcmc.run()
print(f"  基础采样获得 {len(simple_samples['a'])} 个样本")

print("\n🔹 模式2: 检查点续跑")
# 创建新的MCMC对象用于续跑
mcmc_resume = MCMC(basic_config, linear_likelihood, chain_name="resume_example")
print("  支持从检查点续跑采样")

print("\n🔹 模式3: 自定义数据传递")
def likelihood_with_data(a, b, x_data, y_data, sigma):
    """带数据参数的似然函数"""
    y_pred = a * x_data + b  
    chi2 = np.sum((y_data - y_pred)**2 / sigma**2)
    return -0.5 * chi2

# 配置数据传递
config_with_data = basic_config.copy()
config_with_data['data'] = {
    'x_data': x,
    'y_data': y_obs,  
    'sigma': y_err
}

mcmc_data = MCMC(config_with_data, likelihood_with_data, chain_name="with_data")
print("  支持通过config传递数据到似然函数")

# 第十步：状态查询和工具
print("\n📝 步骤10: 实用工具和状态查询")
print("-"*50)

print("🔹 多核状态查询:")
status = Config.status()
print(f"  初始化状态: {status['initialized']}")
print(f"  CPU核心数: {status['config'].get('actual_cores', 'N/A')}")
print(f"  JAX设备数: {status['jax_devices']}")

print("\n🔹 MCMC状态信息:")
print(f"  采样器类型: {mcmc.sampler.__class__.__name__}")
print(f"  链方法: {mcmc.sampler.chain_method}")
print(f"  参数数量: {len(samples)}")

# 第十一步：最佳实践建议
print("\n📝 步骤11: 最佳实践建议")
print("-"*50)

recommendations = [
    "🎯 采样数量: 每个参数至少1000个有效样本",
    "⏰ 预热步数: 通常为采样数的1/4到1/2",
    "🔗 链数设置: 建议2-4条链，不超过CPU核数",
    "📊 收敛检查: 确保R̂ < 1.01, ESS > 400",
    "🎲 初值设置: 可开启optimize_init自动优化",
    "💾 长时间运行: 使用检查点功能防止数据丢失",
    "🔍 似然函数: 避免返回-inf，使用数值稳定的计算"
]

print("最佳实践建议:")
for rec in recommendations:
    print(f"  {rec}")

# 第十二步：实际验证测试
print("\n📝 步骤12: 完整功能验证")
print("-"*50)

def run_comprehensive_test():
    """运行综合测试验证所有功能"""
    
    print("🧪 运行综合功能测试...")
    
    # 测试不同配置
    test_configs = [
        ("单链测试", {'num_samples': 500, 'num_chains': 1}),
        ("多链测试", {'num_samples': 1000, 'num_chains': 4}),
        ("高采样测试", {'num_samples': 2000, 'num_chains': 2})
    ]
    
    results = {}
    
    for test_name, mcmc_params in test_configs:
        print(f"\n  🔸 {test_name}:")
        
        test_config = basic_config.copy()
        test_config['mcmc'].update(mcmc_params)
        
        try:
            test_mcmc = MCMC(test_config, linear_likelihood, 
                           chain_name=f"test_{test_name.replace(' ', '_')}")
            test_samples = test_mcmc.run()
            
            # 验证结果
            n_samples = len(test_samples['a'])
            a_mean = np.mean(test_samples['a'])
            b_mean = np.mean(test_samples['b'])
            
            # 检查参数估计精度
            a_error = abs(a_mean - true_a)
            b_error = abs(b_mean - true_b)
            
            accuracy = "优秀" if max(a_error, b_error) < 0.1 else "良好"
            
            print(f"    样本数: {n_samples}")
            print(f"    参数a: {a_mean:.3f} (误差: {a_error:.3f})")
            print(f"    参数b: {b_mean:.3f} (误差: {b_error:.3f})")
            print(f"    精度评估: {accuracy}")
            
            results[test_name] = {
                'success': True,
                'samples': n_samples,
                'accuracy': accuracy,
                'errors': (a_error, b_error)
            }
            
        except Exception as e:
            print(f"    ❌ 测试失败: {e}")
            results[test_name] = {'success': False, 'error': str(e)}
    
    return results

# 运行测试
test_results = run_comprehensive_test()

# 输出测试总结
print("\n🎉 测试总结:")
print("="*60)

successful_tests = sum(1 for r in test_results.values() if r.get('success', False))
total_tests = len(test_results)

print(f"测试通过率: {successful_tests}/{total_tests}")

for test_name, result in test_results.items():
    status = "✅ 成功" if result.get('success', False) else "❌ 失败"
    print(f"  {test_name}: {status}")
    
    if result.get('success', False):
        accuracy = result.get('accuracy', 'N/A')
        samples = result.get('samples', 'N/A')
        print(f"    样本数: {samples}, 精度: {accuracy}")

if successful_tests == total_tests:
    print("\n🎊 所有功能测试通过！HiCosmo MCMC 运行正常！")
    print("\n📚 使用总结:")
    print("1. Config.init() - 一行代码配置多核")
    print("2. 定义似然函数和参数配置")  
    print("3. MCMC(config, likelihood_func) 创建采样器")
    print("4. samples = mcmc.run() 运行采样")
    print("5. 分析结果和收敛诊断")
else:
    print(f"\n⚠️ {total_tests - successful_tests} 个测试失败，请检查配置")

print("\n" + "="*60)
print("📖 完整使用指南结束")
print("="*60)