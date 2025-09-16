#!/usr/bin/env python3
"""
Multi-core Optimization Utilities for HIcosmo

Provides utilities for configuring and optimizing multi-core parallel execution.
"""

import os
import warnings
from typing import Optional

from .constants import (
    DEFAULT_NUM_CPU_DEVICES,
    AUTO_DETECT_CPU_CORES,
    DEVICE_MISMATCH_WARNING_THRESHOLD
)

def setup_multicore_execution(
    num_devices: Optional[int] = None,
    auto_detect: bool = AUTO_DETECT_CPU_CORES,
    force_override: bool = False
) -> bool:
    """
    Setup multi-core parallel execution for JAX/NumPyro.
    
    CRITICAL: This function must be called BEFORE any JAX imports!
    
    Parameters:
    -----------
    num_devices : int, optional
        Number of CPU devices to use. If None, will use default or auto-detect.
    auto_detect : bool
        Whether to auto-detect available CPU cores
    force_override : bool
        Whether to force override even if JAX is already initialized
        
    Returns:
    --------
    bool
        True if setup was successful, False otherwise
        
    Examples:
    ---------
    # Before any JAX imports:
    >>> from hicosmo.samplers.multicore import setup_multicore_execution
    >>> setup_multicore_execution(num_devices=4)
    True
    
    # Auto-detect cores:
    >>> setup_multicore_execution(auto_detect=True)
    True
    """
    try:
        import numpyro
        
        # Check if JAX is already initialized
        jax_initialized = False
        try:
            import jax
            if jax.devices():  # This will work if JAX is already initialized
                jax_initialized = True
        except:
            pass
            
        if jax_initialized and not force_override:
            warnings.warn(
                "JAX is already initialized. Multi-core setup should be called "
                "before any JAX imports for best results.",
                UserWarning
            )
            return False
        
        # Determine number of devices
        if num_devices is None:
            if auto_detect:
                cpu_count = os.cpu_count()
                # Use min(cpu_count, DEFAULT_NUM_CPU_DEVICES) to avoid excessive parallelism
                num_devices = min(cpu_count, DEFAULT_NUM_CPU_DEVICES) if cpu_count else DEFAULT_NUM_CPU_DEVICES
            else:
                num_devices = DEFAULT_NUM_CPU_DEVICES
        
        # Validate device count
        if num_devices < 1:
            warnings.warn(f"Invalid device count {num_devices}, using 1", UserWarning)
            num_devices = 1
            
        # Check against system capabilities
        system_cores = os.cpu_count()
        if system_cores and num_devices > system_cores:
            warnings.warn(
                f"Requested {num_devices} devices but system only has {system_cores} cores. "
                f"Performance may be suboptimal.",
                UserWarning
            )
        
        # Set up NumPyro devices
        numpyro.set_host_device_count(num_devices)
        
        print(f"🚀 Multi-core setup: {num_devices} CPU devices configured")
        
        # Verify setup if JAX is available
        try:
            import jax
            actual_devices = jax.local_device_count()
            if actual_devices != num_devices:
                if abs(actual_devices - num_devices) >= DEVICE_MISMATCH_WARNING_THRESHOLD:
                    warnings.warn(
                        f"Device count mismatch: requested {num_devices}, got {actual_devices}",
                        UserWarning
                    )
            print(f"✅ JAX detected {actual_devices} devices: {jax.devices()}")
        except ImportError:
            pass
        
        return True
        
    except Exception as e:
        warnings.warn(f"Multi-core setup failed: {e}", UserWarning)
        return False

def get_optimal_chain_count(
    num_devices: Optional[int] = None, 
    allow_single_chain: bool = True
) -> int:
    """
    Get optimal number of chains for current device configuration.
    
    Parameters:
    -----------
    num_devices : int, optional
        Number of available devices. If None, will try to detect.
    allow_single_chain : bool
        Whether to allow single chain (useful for single device setups)
        
    Returns:
    --------
    int
        Recommended number of chains
    """
    if num_devices is None:
        try:
            import jax
            num_devices = jax.local_device_count()
        except:
            num_devices = 1
    
    # For single device
    if num_devices == 1:
        if allow_single_chain:
            return 1  # Single chain for single device
        else:
            return 2  # Minimum 2 chains for convergence diagnostics
    
    # For multiple devices, use equal number of chains
    # But ensure we have at least 2 chains for convergence diagnostics
    return max(2, num_devices)

def check_multicore_status() -> dict:
    """
    Check current multi-core configuration status.
    
    Returns:
    --------
    dict
        Status information including devices, chains recommendation, etc.
    """
    status = {
        'jax_available': False,
        'num_devices': 1,
        'devices': [],
        'system_cores': os.cpu_count(),
        'recommended_chains': 1,
        'multicore_enabled': False
    }
    
    try:
        import jax
        status['jax_available'] = True
        status['num_devices'] = jax.local_device_count()
        status['devices'] = [str(d) for d in jax.devices()]
        status['multicore_enabled'] = status['num_devices'] > 1
        status['recommended_chains'] = get_optimal_chain_count(status['num_devices'])
    except ImportError:
        pass
    
    return status

def print_multicore_info():
    """Print detailed multi-core configuration information."""
    status = check_multicore_status()
    
    print("🖥️  Multi-core Configuration Status")
    print("=" * 50)
    print(f"System CPU cores: {status['system_cores']}")
    print(f"JAX available: {status['jax_available']}")
    
    if status['jax_available']:
        print(f"JAX devices: {status['num_devices']}")
        print(f"Device list: {status['devices']}")
        print(f"Multi-core enabled: {status['multicore_enabled']}")
        print(f"Recommended chains: {status['recommended_chains']}")
        
        if status['multicore_enabled']:
            print("✅ Multi-core parallel execution is enabled")
        else:
            print("⚠️  Running in single-core mode")
            print("💡 To enable multi-core, call setup_multicore_execution() before importing JAX")
    else:
        print("❌ JAX not available")
    
    print("=" * 50)