#!/usr/bin/env python3
"""
HIcosmo Elegant Multi-core Initialization Module

Provides clean and elegant multi-core environment configuration
for neutral hydrogen cosmology and 21cm surveys.

Author: Jingzhao Qi
"""

import os
import warnings
from typing import Optional, Union


class Config:
    """
    HIcosmo Global Configuration Manager
    
    Provides elegant one-line initialization with automatic multi-core and environment setup
    for neutral hydrogen cosmology and 21cm surveys
    """
    
    _initialized = False
    _config = {}
    
    @classmethod
    def init(cls, 
             cpu_cores: Union[int, str, None] = 'auto',
             verbose: bool = True) -> bool:
        """
        Elegant one-line initialization
        
        Parameters
        ----------
        cpu_cores : int, 'auto', or None
            CPU cores configuration:
            - int: Specify number of cores (e.g., 4)
            - 'auto': Auto-detect and use optimal configuration
            - None: Don't set multi-core, use default
        verbose : bool
            Whether to show initialization messages
            
        Returns
        -------
        bool
            True if successful, False otherwise
            
        Examples
        --------
        >>> # Simplest usage: auto configuration
        >>> Config.init()
        
        >>> # Specify core count
        >>> Config.init(cpu_cores=4)
        
        >>> # Silent initialization
        >>> Config.init(cpu_cores='auto', verbose=False)
        """
        if cls._initialized:
            if verbose:
                print("✅ HIcosmo already initialized")
            return True
            
        try:
            # Set CPU cores
            if cpu_cores is not None:
                success = cls._setup_multicore(cpu_cores, verbose)
                if not success and verbose:
                    print("⚠️  Multi-core setup had issues, continuing with single-core")
            
            # Mark as initialized
            cls._initialized = True
            cls._config['cpu_cores'] = cpu_cores
            cls._config['verbose'] = verbose
            
            if verbose:
                cls._print_initialization_summary()
                
            return True
            
        except Exception as e:
            if verbose:
                print(f"❌ HIcosmo initialization failed: {e}")
            return False
    
    @classmethod
    def _setup_multicore(cls, cpu_cores: Union[int, str], verbose: bool) -> bool:
        """Internal method: setup multi-core configuration"""
        try:
            # Determine core count
            if cpu_cores == 'auto':
                system_cores = os.cpu_count() or 4
                # Use system cores, but cap at 8 (avoid excessive parallelization)
                num_cores = min(system_cores, 8)
                # Use at least 4 cores (if system supports)
                num_cores = max(min(num_cores, system_cores), 4) if system_cores >= 4 else system_cores
            else:
                num_cores = int(cpu_cores)
            
            # Validate core count
            if num_cores < 1:
                if verbose:
                    print("⚠️  Invalid core count, using 1")
                num_cores = 1
            
            # Set environment variables (must be before NumPyro import)
            if num_cores > 1:
                os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={num_cores}'
            
            # Import and configure NumPyro
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
        """Print initialization summary"""
        print("\n" + "="*50)
        print("🎉 HIcosmo Initialization Complete")
        print("="*50)
        
        # Display configuration info
        cpu_config = cls._config.get('cpu_cores', 'default')
        actual_cores = cls._config.get('actual_cores', 1)
        
        print(f"CPU Configuration: {cpu_config}")
        print(f"Active CPU Cores: {actual_cores}")
        
        # Verify JAX devices
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
        Get current configuration status
        
        Returns
        -------
        dict
            Configuration status information
        """
        status_info = {
            'initialized': cls._initialized,
            'config': cls._config.copy(),
            'system_cores': os.cpu_count(),
        }
        
        # JAX information
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
        """Reset configuration (mainly for testing)"""
        cls._initialized = False
        cls._config = {}


# Provide convenient import interface
def init_hicosmo(cpu_cores: Union[int, str, None] = 'auto', verbose: bool = True) -> bool:
    """
    HIcosmo one-line initialization function
    
    This is the most concise usage for neutral hydrogen cosmology:
    
    Examples
    --------
    >>> from hicosmo.samplers import init_hicosmo
    >>> init_hicosmo()  # Auto configuration for 21cm surveys!
    """
    return Config.init(cpu_cores=cpu_cores, verbose=verbose)


# Backward compatible multi-core setup function (based on new elegant solution)
def setup_multicore_execution(num_devices: Optional[int] = None, 
                             auto_detect: bool = True,
                             force_override: bool = False) -> bool:
    """Backward compatible multi-core setup function"""
    if num_devices is None:
        cpu_cores = 'auto' if auto_detect else None
    else:
        cpu_cores = num_devices
        
    return Config.init(cpu_cores=cpu_cores, verbose=True)