#!/usr/bin/env python3
"""
Debug配置流程
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.samplers import MCMC
from hicosmo.samplers.config import ParameterConfig

def debug_config_flow():
    """Debug配置流程"""
    
    config = {
        'parameters': {
            'a': (2.0, 0, 5),
        },
        'mcmc': {
            'num_samples': 100,
            'num_chains': 2
            # 注意：这里没有num_warmup！
        }
    }
    
    print("1. 原始配置：")
    print(f"  mcmc config: {config['mcmc']}")
    print(f"  'num_warmup' in config['mcmc']: {'num_warmup' in config['mcmc']}")
    
    print("\n2. 解析为ParameterConfig：")
    param_config = ParameterConfig.from_dict(config)
    print(f"  param_config.mcmc: {param_config.mcmc}")
    print(f"  'num_warmup' in param_config.mcmc: {'num_warmup' in param_config.mcmc}")
    
    print("\n3. MCMC处理：")
    
    def dummy_likelihood(a):
        return -0.5 * a**2
    
    mcmc = MCMC(config, dummy_likelihood, chain_name="debug")
    print(f"  final num_warmup: {mcmc.sampler.num_warmup}")

if __name__ == "__main__":
    debug_config_flow()