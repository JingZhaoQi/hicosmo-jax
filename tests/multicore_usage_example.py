#!/usr/bin/env python3
"""
HiCosmo 多核优化功能使用示例

展示如何正确使用多核并行功能
"""

# 🚀 STEP 1: 在任何其他导入之前设置多核
# 这是最关键的一步！

# 添加路径以便导入hicosmo
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers.multicore import setup_multicore_execution, print_multicore_info

# 自动检测并设置最优的多核配置
success = setup_multicore_execution(auto_detect=True)
if success:
    print("✅ 多核设置成功!")
else:
    print("⚠️  多核设置失败，将使用单核模式")

# 打印多核配置信息
print_multicore_info()

# 🚀 STEP 2: 现在可以安全地导入其他模块
import numpy as np

from hicosmo.samplers import MCMC, get_optimal_chain_count

def main():
    """主要的MCMC测试"""
    print("\n" + "="*60)
    print("🚀 HiCosmo 多核优化示例")
    print("="*60)
    
    # 生成测试数据
    np.random.seed(42)
    x = np.linspace(0, 3, 25)
    y_true = 2.5 * x**2 + 1.2 * x + 0.8
    y_err = 0.15 * np.ones_like(x)
    y_obs = y_true + np.random.normal(0, y_err)
    
    print(f"📊 测试数据: {len(x)} 个数据点")
    
    # 定义似然函数
    def likelihood_func(a, b, c):
        """多项式拟合的似然函数"""
        y_pred = a * x**2 + b * x + c
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    # 🚀 STEP 3: 使用智能链数配置
    optimal_chains = get_optimal_chain_count()
    print(f"🔗 推荐链数: {optimal_chains}")
    
    # MCMC配置
    config = {
        'parameters': {
            'a': (2.0, 0.0, 5.0),  # (initial, min, max)
            'b': (1.0, 0.0, 3.0),
            'c': (1.0, 0.0, 2.0)
        },
        'mcmc': {
            'num_samples': 2000,
            'num_warmup': 1000,
            'num_chains': optimal_chains  # 使用推荐的链数
        }
    }
    
    print(f"\n🔧 配置:")
    print(f"   参数数量: {len(config['parameters'])}")
    print(f"   样本数量: {config['mcmc']['num_samples']}")
    print(f"   预热步数: {config['mcmc']['num_warmup']}")
    print(f"   并行链数: {config['mcmc']['num_chains']}")
    
    # 运行MCMC
    import time
    start_time = time.time()
    
    mcmc = MCMC(config, likelihood_func, optimize_init=False, 
               chain_name="multicore_example")
    samples = mcmc.run()
    
    execution_time = time.time() - start_time
    
    # 分析结果
    print(f"\n📈 结果:")
    print(f"   执行时间: {execution_time:.2f}s")
    print(f"   样本形状: {len(samples.get('a', []))} per chain")
    
    # 获取参数估计
    for param in ['a', 'b', 'c']:
        if param in samples:
            mean_val = np.mean(samples[param])
            std_val = np.std(samples[param])
            print(f"   {param}: {mean_val:.3f} ± {std_val:.3f}")
    
    # 检查收敛性
    try:
        diagnostics = mcmc.sampler.get_diagnostics(burnin_frac=0.2)
        converged = sum(1 for d in diagnostics.values() 
                       if isinstance(d, dict) and d.get('r_hat', 999) < 1.1)
        total = sum(1 for d in diagnostics.values() 
                   if isinstance(d, dict) and 'r_hat' in d)
        print(f"   收敛情况: {converged}/{total} 参数收敛")
    except Exception as e:
        print(f"   收敛检查失败: {e}")
    
    print("\n✅ 多核MCMC示例完成!")
    
    # 性能提示
    if optimal_chains > 1:
        estimated_serial_time = execution_time * optimal_chains
        speedup = estimated_serial_time / execution_time
        print(f"\n🚀 性能估算:")
        print(f"   估计串行时间: {estimated_serial_time:.2f}s")
        print(f"   实际并行时间: {execution_time:.2f}s")
        print(f"   加速比: {speedup:.2f}x")
    else:
        print(f"\n⚠️  单核模式运行，考虑启用多核以获得更好性能")

if __name__ == "__main__":
    main()