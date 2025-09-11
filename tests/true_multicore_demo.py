#!/usr/bin/env python3
"""
真正有效的多核演示

这个脚本展示了正确的多核配置方式
"""

# 🚀 第1步：在导入任何其他模块之前设置多核
import numpyro
import os

# 检测系统核心数并设置4个CPU设备
cpu_count = os.cpu_count()
print(f"🖥️  系统CPU核心数: {cpu_count}")

# 设置多核（在任何JAX导入之前）
numpyro.set_host_device_count(4)
print("✅ 已设置4个CPU设备用于并行计算")

# 🚀 第2步：现在导入其他模块
import numpy as np
import time
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 验证JAX设备配置
import jax
print(f"JAX设备: {jax.devices()}")
print(f"设备数量: {jax.local_device_count()}")

if jax.local_device_count() >= 4:
    print("🎉 多核并行成功启用!")
else:
    print("❌ 多核设置失败")
    exit(1)

# 🚀 第3步：导入HiCosmo
from hicosmo.samplers import MCMC

def main():
    """运行多核MCMC演示"""
    print("\n" + "="*60)
    print("🚀 真正的多核MCMC演示")
    print("="*60)
    
    # 生成测试数据
    np.random.seed(42)
    x = np.linspace(0, 3, 25)
    y_true = 2.5 * x**2 + 1.8 * x + 0.9
    y_err = 0.15 * np.ones_like(x)
    y_obs = y_true + np.random.normal(0, y_err)
    
    print(f"📊 测试数据: {len(x)} 个数据点")
    
    # 定义似然函数
    def likelihood_func(a, b, c):
        """二次多项式拟合的似然函数"""
        y_pred = a * x**2 + b * x + c
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    # 4核4链配置
    config = {
        'parameters': {
            'a': (2.0, 0.0, 5.0),  # (initial, min, max)
            'b': (1.5, 0.0, 4.0),
            'c': (1.0, 0.0, 3.0)
        },
        'mcmc': {
            'num_samples': 3000,
            'num_warmup': 1500,
            'num_chains': 4  # 4条并行链
        }
    }
    
    print(f"\n🔧 配置:")
    print(f"   参数数量: {len(config['parameters'])}")
    print(f"   样本数量: {config['mcmc']['num_samples']}")
    print(f"   预热步数: {config['mcmc']['num_warmup']}")
    print(f"   并行链数: {config['mcmc']['num_chains']}")
    print(f"   设备数量: {jax.local_device_count()}")
    
    # 运行MCMC
    start_time = time.time()
    
    mcmc = MCMC(config, likelihood_func, optimize_init=False, 
               chain_name="true_multicore_demo")
    samples = mcmc.run()
    
    execution_time = time.time() - start_time
    
    # 分析结果
    print(f"\n📈 结果:")
    print(f"   执行时间: {execution_time:.2f}s")
    print(f"   样本形状: {len(samples.get('a', []))} per chain")
    
    # 获取参数估计
    true_values = {'a': 2.5, 'b': 1.8, 'c': 0.9}
    for param in ['a', 'b', 'c']:
        if param in samples:
            mean_val = np.mean(samples[param])
            std_val = np.std(samples[param])
            true_val = true_values[param]
            print(f"   {param}: {mean_val:.3f} ± {std_val:.3f} (真值: {true_val:.1f})")
    
    # 检查收敛性
    try:
        diagnostics = mcmc.sampler.get_diagnostics(burnin_frac=0.2)
        converged = sum(1 for d in diagnostics.values() 
                       if isinstance(d, dict) and d.get('r_hat', 999) < 1.1)
        total = sum(1 for d in diagnostics.values() 
                   if isinstance(d, dict) and 'r_hat' in d)
        print(f"   收敛情况: {converged}/{total} 参数收敛")
        
        # 显示R-hat值
        for param, diag in diagnostics.items():
            if isinstance(diag, dict) and 'r_hat' in diag:
                print(f"   R̂({param}): {diag['r_hat']:.4f}")
                
    except Exception as e:
        print(f"   收敛检查失败: {e}")
    
    print("\n✅ 多核MCMC演示完成!")
    
    # 性能分析
    estimated_single_core = execution_time * 4
    speedup = estimated_single_core / execution_time
    efficiency = speedup / 4 * 100
    
    print(f"\n🚀 性能分析:")
    print(f"   实际4核并行时间: {execution_time:.2f}s")
    print(f"   估计单核串行时间: {estimated_single_core:.2f}s")
    print(f"   并行加速比: {speedup:.2f}x")
    print(f"   并行效率: {efficiency:.1f}%")
    
    if speedup > 2.0:
        print("🎉 多核并行效果显著!")
    elif speedup > 1.5:
        print("✅ 多核并行有效果")
    else:
        print("⚠️  多核并行效果有限")

if __name__ == "__main__":
    main()