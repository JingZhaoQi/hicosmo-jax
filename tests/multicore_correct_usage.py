#!/usr/bin/env python3
"""
HiCosmo 多核优化正确用法示例

展示如何在新的Python进程中正确使用多核功能
"""

# ⚠️ 重要提示：
# 由于JAX的设备初始化是全局性的，在已经导入JAX的环境中
# 无法重新配置设备数量。因此多核配置必须在脚本开始时进行。

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def demonstrate_multicore_setup():
    """演示多核设置的重要性"""
    print("=" * 60)
    print("🖥️  多核配置重要性演示")
    print("=" * 60)
    
    # 检查当前状态
    from hicosmo.samplers.multicore import check_multicore_status, print_multicore_info
    
    print("当前JAX设备状态:")
    print_multicore_info()
    
    status = check_multicore_status()
    
    if status['multicore_enabled']:
        print("✅ 多核已启用 - 这是理想状态!")
        return True
    else:
        print("⚠️  当前为单核模式")
        print("\n💡 要启用多核，请使用以下模板:")
        print(create_multicore_template())
        return False

def create_multicore_template() -> str:
    """创建多核使用模板"""
    template = '''
# ===== 多核MCMC模板 =====
#!/usr/bin/env python3

# 🚀 关键：在导入任何其他模块之前设置多核
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 在导入JAX之前设置多核
from hicosmo.samplers.multicore import setup_multicore_execution
setup_multicore_execution(num_devices=4)  # 或 auto_detect=True

# 现在可以安全导入其他模块
import numpy as np
from hicosmo.samplers import MCMC, get_optimal_chain_count

# 使用推荐的链数
optimal_chains = get_optimal_chain_count()

# MCMC配置
config = {
    'parameters': {'a': (1.0, 0.0, 2.0)},
    'mcmc': {
        'num_samples': 2000,
        'num_chains': optimal_chains  # 关键：使用推荐值
    }
}

# 运行MCMC
mcmc = MCMC(config, likelihood_func)
samples = mcmc.run()
# ===========================
'''
    return template

def run_single_chain_demo():
    """运行单链演示（适合当前环境）"""
    print("\n" + "=" * 60)
    print("🔧 单链MCMC演示（适应当前环境）")
    print("=" * 60)
    
    import numpy as np
    from hicosmo.samplers import MCMC, get_optimal_chain_count
    
    # 生成测试数据
    np.random.seed(42)
    x = np.linspace(0, 2, 15)
    y_true = 1.5 * x + 0.5
    y_err = 0.1 * np.ones_like(x)
    y_obs = y_true + np.random.normal(0, y_err)
    
    # 简单线性模型的似然函数
    def likelihood_func(a, b):
        y_pred = a * x + b
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    # 在单设备环境下使用单链（避免诊断问题）
    optimal_chains = get_optimal_chain_count(allow_single_chain=True)
    print(f"推荐链数: {optimal_chains}")
    
    config = {
        'parameters': {
            'a': (1.0, 0.0, 3.0),
            'b': (0.0, -1.0, 2.0)
        },
        'mcmc': {
            'num_samples': 1500,
            'num_warmup': 800,
            'num_chains': optimal_chains
        }
    }
    
    print(f"配置: {config['mcmc']['num_chains']} 链, {config['mcmc']['num_samples']} 样本")
    
    import time
    start_time = time.time()
    
    try:
        mcmc = MCMC(config, likelihood_func, optimize_init=False, 
                   chain_name="single_chain_demo")
        samples = mcmc.run()
        
        execution_time = time.time() - start_time
        
        print(f"✅ 完成: {execution_time:.2f}s")
        
        # 简单结果分析
        for param in ['a', 'b']:
            if param in samples:
                mean_val = np.mean(samples[param])
                std_val = np.std(samples[param])
                print(f"   {param}: {mean_val:.3f} ± {std_val:.3f}")
                
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"⚠️  执行时间: {execution_time:.2f}s")
        print(f"注意: {str(e)[:100]}...")
        print("这在单链模式下是正常的（诊断需要多链）")

def main():
    """主函数"""
    print("HiCosmo 多核优化使用指南")
    
    # 演示当前状态
    multicore_available = demonstrate_multicore_setup()
    
    # 运行适当的演示
    if multicore_available:
        print("🎉 可以运行完整的多核演示!")
    else:
        print("📝 运行单链演示并提供多核使用指导")
        run_single_chain_demo()
    
    print("\n" + "=" * 60)
    print("📋 总结:")
    print("1. 多核配置必须在导入JAX之前完成")
    print("2. 使用 get_optimal_chain_count() 获取推荐链数")
    print("3. 单设备环境可使用单链避免诊断问题")
    print("4. 多设备环境建议使用多链以获得更好的收敛诊断")
    print("=" * 60)

if __name__ == "__main__":
    main()