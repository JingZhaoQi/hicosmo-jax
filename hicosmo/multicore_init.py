#!/usr/bin/env python3
"""
HIcosmo多核初始化模块

这个模块必须在导入任何其他HIcosmo模块之前导入！
提供最简单的一行式多核配置。
"""

import os
from typing import Union

def setup_multicore(cpu_cores: Union[int, str] = 'auto', verbose: bool = True) -> bool:
    """
    设置HIcosmo多核环境 - 必须在导入其他HIcosmo模块之前调用！

    Parameters
    ----------
    cpu_cores : int or 'auto'
        CPU核心数，'auto'表示自动检测
    verbose : bool
        是否显示详细信息

    Returns
    -------
    bool
        是否设置成功

    Examples
    --------
    >>> # 在所有HIcosmo导入之前！
    >>> from hicosmo.multicore_init import setup_multicore
    >>> setup_multicore(6)  # 设置6核
    >>>
    >>> # 现在可以安全导入其他HIcosmo模块
    >>> from hicosmo.samplers import MCMC, AutoParameter
    """

    try:
        # 确定核心数
        if cpu_cores == 'auto':
            system_cores = os.cpu_count() or 4
            num_cores = min(system_cores, 8)  # 最多8核
        else:
            num_cores = int(cpu_cores)

        if num_cores < 1:
            num_cores = 1

        # 设置JAX环境变量
        if num_cores > 1:
            os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_cores}'
            if verbose:
                print(f"🚀 HIcosmo多核配置: {num_cores} CPU核心")
                print(f"✅ XLA_FLAGS = {os.environ['XLA_FLAGS']}")
        else:
            if verbose:
                print("🔧 HIcosmo单核模式")

        return True

    except Exception as e:
        if verbose:
            print(f"❌ 多核设置失败: {e}")
        return False

# 便捷别名
init_multicore = setup_multicore
enable_multicore = setup_multicore