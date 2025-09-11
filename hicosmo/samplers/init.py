#!/usr/bin/env python3
"""
HiCosmo 优雅的多核初始化模块

提供简洁、优美的多核环境配置方案
"""

import os
import warnings
from typing import Optional, Union


class Config:
    """
    HiCosmo 全局配置管理器
    
    提供优雅的一行初始化方案，自动处理多核、环境变量等配置
    """
    
    _initialized = False
    _config = {}
    
    @classmethod
    def init(cls, 
             cpu_cores: Union[int, str, None] = 'auto',
             verbose: bool = True) -> bool:
        """
        优雅的一行初始化方案
        
        Parameters
        ----------
        cpu_cores : int, 'auto', or None
            CPU核心数配置:
            - int: 指定核心数 (如 4)
            - 'auto': 自动检测并使用最优配置
            - None: 不设置多核，使用默认配置
        verbose : bool
            是否显示初始化信息
            
        Returns
        -------
        bool
            True if successful, False otherwise
            
        Examples
        --------
        >>> # 最简单的用法：自动配置
        >>> Config.init()
        
        >>> # 指定核心数
        >>> Config.init(cpu_cores=4)
        
        >>> # 静默初始化
        >>> Config.init(cpu_cores='auto', verbose=False)
        """
        if cls._initialized:
            if verbose:
                print("✅ HiCosmo already initialized")
            return True
            
        try:
            # 设置CPU核心数
            if cpu_cores is not None:
                success = cls._setup_multicore(cpu_cores, verbose)
                if not success and verbose:
                    print("⚠️  Multi-core setup had issues, continuing with single-core")
            
            # 标记为已初始化
            cls._initialized = True
            cls._config['cpu_cores'] = cpu_cores
            cls._config['verbose'] = verbose
            
            if verbose:
                cls._print_initialization_summary()
                
            return True
            
        except Exception as e:
            if verbose:
                print(f"❌ HiCosmo initialization failed: {e}")
            return False
    
    @classmethod
    def _setup_multicore(cls, cpu_cores: Union[int, str], verbose: bool) -> bool:
        """内部方法：设置多核配置"""
        try:
            # 确定核心数
            if cpu_cores == 'auto':
                system_cores = os.cpu_count() or 4
                # 使用系统核心数，但最多不超过8核（避免过度并行）
                num_cores = min(system_cores, 8)
                # 至少使用4核（如果系统支持）
                num_cores = max(min(num_cores, system_cores), 4) if system_cores >= 4 else system_cores
            else:
                num_cores = int(cpu_cores)
            
            # 验证核心数
            if num_cores < 1:
                if verbose:
                    print("⚠️  Invalid core count, using 1")
                num_cores = 1
            
            # 设置环境变量（必须在NumPyro导入之前）
            if num_cores > 1:
                os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_cores}'
            
            # 导入并配置NumPyro
            import numpyro
            numpyro.set_host_device_count(num_cores)
            
            cls._config['actual_cores'] = num_cores
            
            if verbose:
                print(f"🚀 Multi-core configured: {num_cores} CPU cores")
                
            return True
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Multi-core setup failed: {e}")
            return False
    
    @classmethod
    def _print_initialization_summary(cls):
        """打印初始化摘要"""
        print("\n" + "="*50)
        print("🎉 HiCosmo Initialization Complete")
        print("="*50)
        
        # 显示配置信息
        cpu_config = cls._config.get('cpu_cores', 'default')
        actual_cores = cls._config.get('actual_cores', 1)
        
        print(f"CPU Configuration: {cpu_config}")
        print(f"Active CPU Cores: {actual_cores}")
        
        # 验证JAX设备
        try:
            import jax
            devices = jax.devices()
            device_count = len(devices)
            print(f"JAX Devices: {device_count} ({devices[0].platform})")
            
            if device_count != actual_cores:
                print(f"⚠️  Device count mismatch: expected {actual_cores}, got {device_count}")
            else:
                print("✅ Multi-core setup verified")
                
        except ImportError:
            print("JAX: Not available")
        
        print(f"Ready for high-performance MCMC! 🚀")
        print("="*50)
    
    @classmethod
    def status(cls) -> dict:
        """
        获取当前配置状态
        
        Returns
        -------
        dict
            配置状态信息
        """
        status_info = {
            'initialized': cls._initialized,
            'config': cls._config.copy(),
            'system_cores': os.cpu_count(),
        }
        
        # JAX信息
        try:
            import jax
            status_info['jax_devices'] = len(jax.devices())
            status_info['jax_device_list'] = [str(d) for d in jax.devices()]
        except ImportError:
            status_info['jax_devices'] = 0
            status_info['jax_device_list'] = []
        
        return status_info
    
    @classmethod
    def reset(cls):
        """重置配置（主要用于测试）"""
        cls._initialized = False
        cls._config = {}


# 提供便捷的导入接口
def init_hicosmo(cpu_cores: Union[int, str, None] = 'auto', verbose: bool = True) -> bool:
    """
    HiCosmo 一行初始化函数
    
    这是最简洁的使用方式：
    
    Examples
    --------
    >>> from hicosmo.samplers import init_hicosmo
    >>> init_hicosmo()  # 自动配置，就这么简单！
    """
    return Config.init(cpu_cores=cpu_cores, verbose=verbose)


# 向后兼容的多核设置函数（基于新的优雅方案）
def setup_multicore_execution(num_devices: Optional[int] = None, 
                             auto_detect: bool = True,
                             force_override: bool = False) -> bool:
    """向后兼容的多核设置函数"""
    if num_devices is None:
        cpu_cores = 'auto' if auto_detect else None
    else:
        cpu_cores = num_devices
        
    return Config.init(cpu_cores=cpu_cores, verbose=True)