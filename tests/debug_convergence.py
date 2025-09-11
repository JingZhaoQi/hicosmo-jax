#!/usr/bin/env python3
"""
调试收敛问题
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

print("🔍 调试收敛问题")
print("="*60)

# 使用与测试相同的数据和设置
np.random.seed(42)
x = np.linspace(0, 1, 5)
y_true = 2.0 * x + 1.0
y_err = 0.1 * np.ones_like(x)
y_obs = y_true + np.random.normal(0, y_err)

print(f"测试数据:")
print(f"x = {x}")
print(f"y_true = {y_true}")
print(f"y_obs = {y_obs}")
print(f"y_err = {y_err}")

def likelihood_func(a, b):
    y_pred = a * x + b
    chi2 = np.sum((y_obs - y_pred)**2 / y_err**2)
    return -0.5 * chi2

# 测试不同的配置
configs_to_test = [
    {
        'name': '原始配置 (125样本/链)',
        'config': {
            'parameters': {
                'a': (2.0, 0.0, 4.0),
                'b': (1.0, 0.0, 2.0)
            },
            'mcmc': {
                'num_samples': 500,   # 总500样本，4链每链125
                'num_warmup': 20000,  # 用户设置的大量warmup
                'num_chains': 4
            }
        }
    },
    {
        'name': '增加样本数 (500样本/链)',
        'config': {
            'parameters': {
                'a': (2.0, 0.0, 4.0),
                'b': (1.0, 0.0, 2.0)
            },
            'mcmc': {
                'num_samples': 2000,  # 总2000样本，4链每链500
                'num_warmup': 1000,   # 合理的warmup
                'num_chains': 4
            }
        }
    },
    {
        'name': '单链测试',
        'config': {
            'parameters': {
                'a': (2.0, 0.0, 4.0),
                'b': (1.0, 0.0, 2.0)
            },
            'mcmc': {
                'num_samples': 2000,  # 单链2000样本
                'num_warmup': 1000,
                'num_chains': 1
            }
        }
    }
]

for test_case in configs_to_test:
    print(f"\n" + "="*60)
    print(f"🧪 测试: {test_case['name']}")
    print("="*60)
    
    config = test_case['config']
    
    try:
        mcmc = MCMC(config, likelihood_func, optimize_init=False, 
                   chain_name=f"convergence_test")
        samples = mcmc.run()
        
        # 分析样本
        print(f"\n📊 样本分析:")
        for param in ['a', 'b']:
            if param in samples:
                values = samples[param]
                mean_val = np.mean(values)
                std_val = np.std(values)
                true_val = 2.0 if param == 'a' else 1.0
                
                print(f"  {param}: {mean_val:.3f} ± {std_val:.3f} (真值: {true_val:.1f})")
        
        # 获取详细的诊断信息
        try:
            diagnostics = mcmc.sampler.get_diagnostics(burnin_frac=0.2)
            print(f"\n🔍 收敛诊断:")
            for param, diag in diagnostics.items():
                if isinstance(diag, dict):
                    r_hat = diag.get('r_hat', 'N/A')
                    ess = diag.get('ess', 'N/A')
                    print(f"  {param}: R̂ = {r_hat:.4f}, ESS = {ess:.0f}" if r_hat != 'N/A' else f"  {param}: 诊断不可用")
        except Exception as e:
            print(f"  诊断失败: {e}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

print(f"\n" + "="*60)
print("🤔 可能的收敛问题原因:")
print("1. 样本数太少 (125样本/链可能不够)")
print("2. 数据点太少 (只有5个数据点)")
print("3. 参数范围设置问题")
print("4. 链间没有足够混合")
print("="*60)