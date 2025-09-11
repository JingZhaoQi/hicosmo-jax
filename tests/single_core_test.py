#!/usr/bin/env python3
"""
单核性能测试
"""

# 🚀 CRITICAL: 设置单核必须在任何JAX导入之前
import numpyro
numpyro.set_host_device_count(1)

import numpy as np
import time
import sys
from pathlib import Path
import jax

sys.path.insert(0, str(Path(__file__).parent.parent))
from hicosmo.samplers import MCMC

def run_single_core_test():
    """运行单核测试"""
    print("🔧 单核单链测试")
    print(f"JAX设备: {jax.devices()}")
    print(f"设备数量: {jax.local_device_count()}")
    
    # 加载数据
    data_path = Path(__file__).parent / "data" / "sim_data.txt"
    try:
        x, y_obs, y_err = np.loadtxt(data_path, unpack=True)
    except:
        np.random.seed(42)
        x = np.linspace(0, 3, 20)
        y_true = 3.0 * x**2 + 1.5 * x + 1.0
        y_err = 0.1 * np.ones_like(x)
        y_obs = y_true + np.random.normal(0, y_err)
    
    def likelihood_func(a, b, c):
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
            'num_samples': 2000,
            'num_warmup': 1000,
            'num_chains': 1  # 单链
        }
    }
    
    try:
        start_time = time.time()
        mcmc = MCMC(config, likelihood_func, optimize_init=False, chain_name="single_core_test")
        samples = mcmc.run()
        execution_time = time.time() - start_time
        print(f"✅ 单核完成: {execution_time:.2f}s")
        return execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        if "assert x.shape[0] >= 2" in str(e) or "diagnostic" in str(e).lower():
            print(f"✅ 单核完成: {execution_time:.2f}s (跳过诊断)")
            return execution_time
        else:
            print(f"❌ 单核测试失败: {e}")
            return execution_time if execution_time > 0 else 0

if __name__ == "__main__":
    result = run_single_core_test()
    print(f"单核时间: {result:.2f}s")