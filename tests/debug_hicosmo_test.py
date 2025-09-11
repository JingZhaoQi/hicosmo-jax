#!/usr/bin/env python3
"""
调试HiCosmo诊断处理失败的问题
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hicosmo.samplers import MCMC

def debug_hicosmo_test():
    """调试HiCosmo测试，找出具体失败原因"""
    print("🔍 调试HiCosmo测试...")
    
    # 加载数据
    data_path = Path(__file__).parent / "data" / "sim_data.txt"
    try:
        x, y_obs, y_err = np.loadtxt(data_path, unpack=True)
        print(f"✅ 数据加载: {len(x)} 个数据点")
    except:
        # 使用模拟数据
        np.random.seed(42)
        x = np.linspace(0, 3, 20)
        y_true = 3.0 * x**2 + 1.5 * x + 1.0
        y_err = 0.1 * np.ones_like(x)
        y_obs = y_true + np.random.normal(0, y_err)
        print(f"📊 使用模拟数据: {len(x)} 个数据点")
    
    def likelihood_func(a, b, c):
        """二次多项式拟合的似然函数"""
        y_pred = a * x**2 + b * x + c
        chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
        return -0.5 * chi2
    
    config = {
        'parameters': {
            'a': (3.5, 0.0, 10.0),
            'b': (2.0, 0.0, 4.0),
            'c': (1.0, 0.0, 2.0)
        },
        'mcmc': {
            'num_samples': 1000,  # 减少样本数以便调试
            'num_warmup': 500,
            'num_chains': 1
        }
    }
    
    try:
        print("\n1️⃣ 创建MCMC对象...")
        mcmc = MCMC(config, likelihood_func, optimize_init=False, 
                   chain_name="debug_test")
        print("✅ MCMC对象创建成功")
        
        print("\n2️⃣ 运行MCMC...")
        start_time = time.time()
        samples = mcmc.run()
        execution_time = time.time() - start_time
        print(f"✅ MCMC运行完成: {execution_time:.2f}s")
        
        print(f"\n3️⃣ 检查样本结果...")
        print(f"样本类型: {type(samples)}")
        if isinstance(samples, dict):
            print("样本键:", list(samples.keys()))
            for key, value in samples.items():
                print(f"  {key}: shape={getattr(value, 'shape', 'No shape')}, type={type(value)}")
        else:
            print(f"样本不是字典: {samples}")
        
        print(f"\n4️⃣ 获取诊断信息...")
        try:
            diagnostics = mcmc.sampler.get_diagnostics(burnin_frac=0.1)
            print(f"✅ 诊断信息获取成功")
            print(f"诊断键: {list(diagnostics.keys()) if isinstance(diagnostics, dict) else type(diagnostics)}")
            
            # 检查收敛
            converged = 0
            total = 0
            for key, diag in diagnostics.items():
                if isinstance(diag, dict) and 'r_hat' in diag:
                    total += 1
                    if diag.get('r_hat', 999) < 1.1:
                        converged += 1
                    print(f"  {key}: r_hat = {diag.get('r_hat', 'N/A')}")
            
            print(f"✅ 收敛检查完成: {converged}/{total}")
            
        except Exception as diag_e:
            print(f"❌ 诊断处理失败: {diag_e}")
            print(f"错误类型: {type(diag_e)}")
            import traceback
            traceback.print_exc()
            return False
            
        print(f"\n5️⃣ 计算样本统计...")
        try:
            sample_count = len(samples.get('a', [])) if isinstance(samples, dict) and 'a' in samples else 0
            print(f"✅ 样本统计: {sample_count} 个样本")
        except Exception as stat_e:
            print(f"❌ 样本统计失败: {stat_e}")
        
        print(f"\n🎉 全部步骤完成!")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("HiCosmo诊断失败调试")
    print("=" * 50)
    success = debug_hicosmo_test()
    print(f"\n最终结果: {'✅ 成功' if success else '❌ 失败'}")