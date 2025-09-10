#!/usr/bin/env python3
"""
MCMC Data Persistence and Recovery System.

This module provides comprehensive MCMC checkpoint management including
real-time saving, automatic recovery, and complete state preservation.
"""

import json
import inspect
import platform
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
import warnings
import gzip
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import h5py
import numpyro

from .config import ParameterConfig, AutoParameter


# =============================================================================
# MCMC State Management
# =============================================================================

@dataclass
class MCMCState:
    """
    Complete MCMC state for checkpointing and recovery.
    
    This class captures everything needed to fully restore an MCMC run,
    including internal NumPyro state, samples, and configuration.
    """
    
    # Core MCMC data
    samples: Dict[str, np.ndarray] = field(default_factory=dict)
    warmup_samples: Optional[Dict[str, np.ndarray]] = None
    mcmc_state: Optional[Dict] = None
    
    # Progress tracking
    current_step: int = 0
    total_steps: int = 0
    warmup_steps: int = 0
    
    # Reproducibility
    random_state: Optional[Dict] = None
    
    # Timing
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    last_save_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Configuration
    parameter_config: Optional[ParameterConfig] = None
    likelihood_info: Dict[str, Any] = field(default_factory=dict)
    data_info: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis data
    diagnostics_history: Dict[str, List] = field(default_factory=dict)
    
    # Metadata
    run_metadata: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields."""
        if not self.environment_info:
            self.environment_info = self._collect_environment_info()
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect system and package environment information."""
        import sys
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_executable': sys.executable,
            'hostname': platform.node(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Collect package versions
        packages = ['jax', 'numpyro', 'numpy', 'h5py']
        for pkg in packages:
            try:
                module = __import__(pkg)
                env_info[f'{pkg}_version'] = getattr(module, '__version__', 'unknown')
            except ImportError:
                env_info[f'{pkg}_version'] = 'not_installed'
        
        # JAX backend info
        try:
            env_info['jax_backend'] = jax.default_backend()
            env_info['jax_devices'] = str(jax.devices())
        except:
            pass
            
        return env_info
    
    def update_progress(self, current_step: int, samples: Dict[str, np.ndarray]):
        """Update progress and samples."""
        self.current_step = current_step
        self.samples = samples
        self.last_save_time = datetime.now().isoformat()
    
    def add_diagnostic(self, name: str, value: Any):
        """Add a diagnostic value to history."""
        if name not in self.diagnostics_history:
            self.diagnostics_history[name] = []
        self.diagnostics_history[name].append(value)
    
    def get_completion_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_steps <= 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100.0
    
    def get_estimated_time_remaining(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if self.current_step <= 0 or self.total_steps <= self.current_step:
            return None
            
        try:
            start = datetime.fromisoformat(self.start_time)
            now = datetime.now()
            elapsed = (now - start).total_seconds()
            
            progress = self.current_step / self.total_steps
            if progress <= 0:
                return None
                
            total_estimated = elapsed / progress
            remaining = total_estimated - elapsed
            return max(0, remaining)
        except:
            return None
    
    def create_signature(self) -> str:
        """Create a unique signature for this MCMC configuration."""
        sig_data = {
            'parameter_names': list(self.parameter_config.parameters.keys()) if self.parameter_config else [],
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'likelihood_hash': self.likelihood_info.get('source_hash', ''),
            'data_hash': self.data_info.get('data_hash', '')
        }
        
        sig_str = json.dumps(sig_data, sort_keys=True)
        return hashlib.md5(sig_str.encode()).hexdigest()[:12]


# =============================================================================
# Checkpoint Management
# =============================================================================

class CheckpointManager:
    """
    Manages automatic checkpointing and backup of MCMC runs.
    
    This class handles real-time saving of MCMC state with configurable
    intervals, backup retention, and compression options.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "./mcmc_checkpoints",
        checkpoint_interval: int = 1000,
        backup_count: int = 5,
        compression: str = "gzip",
        save_warmup: bool = True,
        auto_cleanup: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.backup_count = backup_count
        self.compression = compression
        self.save_warmup = save_warmup
        self.auto_cleanup = auto_cleanup
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal state
        self._last_save_step = 0
        self._checkpoint_files = []
    
    def should_save(self, current_step: int) -> bool:
        """Check if we should save at current step."""
        if self.checkpoint_interval <= 0:
            return False
            
        return (current_step - self._last_save_step) >= self.checkpoint_interval
    
    def save_checkpoint(
        self, 
        state: MCMCState, 
        run_name: str,
        force: bool = False
    ) -> Optional[Path]:
        """
        Save MCMC state as checkpoint.
        
        Parameters
        ----------
        state : MCMCState
            Complete MCMC state to save
        run_name : str
            Name identifier for this run
        force : bool
            Force save even if interval not reached
            
        Returns
        -------
        Path or None
            Path to saved checkpoint file
        """
        if not force and not self.should_save(state.current_step):
            return None
        
        # Use simple run_name as filename
        filename = f"{run_name}.h5"
        filepath = self.checkpoint_dir / filename
        
        try:
            self._save_hdf5_checkpoint(state, filepath)
            
            # Update tracking
            self._last_save_step = state.current_step
            self._checkpoint_files.append(filepath)
            
            # Cleanup old checkpoints if needed
            if self.auto_cleanup:
                self._cleanup_old_checkpoints(run_name)
            
            return filepath
            
        except Exception as e:
            warnings.warn(f"Failed to save checkpoint: {e}")
            return None
    
    def _save_hdf5_checkpoint(self, state: MCMCState, filepath: Path):
        """Save checkpoint in comprehensive HDF5 format."""
        with h5py.File(filepath, 'w') as f:
            # Set compression
            compression = self.compression if self.compression != 'none' else None
            
            # Metadata group
            meta_group = f.create_group('metadata')
            meta_group.attrs['hicosmo_version'] = '2.0'
            meta_group.attrs['save_format_version'] = '2.0'
            meta_group.attrs['creation_time'] = datetime.now().isoformat()
            
            # Configuration
            if state.parameter_config:
                config_str = json.dumps(state.parameter_config.to_dict(), indent=2)
                meta_group.create_dataset('parameter_config', data=config_str)
            
            # Likelihood and environment info
            for info_name, info_data in [
                ('likelihood_info', state.likelihood_info),
                ('environment_info', state.environment_info),
                ('run_metadata', state.run_metadata)
            ]:
                if info_data:
                    info_str = json.dumps(info_data, indent=2, default=str)
                    meta_group.create_dataset(info_name, data=info_str)
            
            # Progress info
            progress_group = f.create_group('progress')
            progress_attrs = {
                'current_step': state.current_step,
                'total_steps': state.total_steps,
                'warmup_steps': state.warmup_steps,
                'start_time': state.start_time,
                'last_save_time': state.last_save_time,
                'completion_percentage': state.get_completion_percentage()
            }
            for key, value in progress_attrs.items():
                progress_group.attrs[key] = value
            
            # Samples group
            if state.samples:
                samples_group = f.create_group('samples')
                for param_name, samples in state.samples.items():
                    samples_group.create_dataset(
                        param_name, 
                        data=samples,
                        compression=compression
                    )
            
            # Warmup samples group
            if state.warmup_samples and self.save_warmup:
                warmup_group = f.create_group('warmup_samples')
                for param_name, samples in state.warmup_samples.items():
                    warmup_group.create_dataset(
                        param_name,
                        data=samples, 
                        compression=compression
                    )
            
            # Data info
            if state.data_info:
                data_group = f.create_group('data')
                data_str = json.dumps(state.data_info, indent=2, default=str)
                data_group.create_dataset('data_info', data=data_str)
                
                # Save actual data arrays if available
                for key, value in state.data_info.items():
                    if key.endswith('_data') and isinstance(value, np.ndarray):
                        data_group.create_dataset(key, data=value, compression=compression)
            
            # Diagnostics history
            if state.diagnostics_history:
                diag_group = f.create_group('diagnostics')
                for name, values in state.diagnostics_history.items():
                    try:
                        diag_group.create_dataset(name, data=np.array(values))
                    except Exception as e:
                        warnings.warn(f"Could not save diagnostic {name}: {e}")
            
            # Random and MCMC states
            for state_name, state_data in [
                ('random_state', state.random_state),
                ('mcmc_state', state.mcmc_state)
            ]:
                if state_data:
                    state_group = f.create_group(state_name)
                    try:
                        state_str = json.dumps(state_data, indent=2, default=str)
                        state_group.create_dataset('state', data=state_str)
                    except:
                        # Fallback to pickle for complex objects
                        state_pickle = pickle.dumps(state_data)
                        state_group.create_dataset('state_pickle', data=np.void(state_pickle))
    
    def _cleanup_old_checkpoints(self, run_name: str):
        """Remove old checkpoint files beyond backup_count."""
        if self.backup_count <= 0:
            return
            
        # Find all checkpoints for this run
        pattern = f"{run_name}_step_*.h5"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Remove files beyond backup count
        files_to_remove = checkpoint_files[self.backup_count:]
        for filepath in files_to_remove:
            try:
                filepath.unlink()
            except Exception as e:
                warnings.warn(f"Could not remove old checkpoint {filepath}: {e}")
    
    def list_checkpoints(self, run_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints.
        
        Parameters
        ----------
        run_name : str, optional
            Filter by run name
            
        Returns
        -------
        List[Dict]
            List of checkpoint information
        """
        pattern = f"{run_name}_step_*.h5" if run_name else "*_step_*.h5"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        
        checkpoints = []
        for filepath in checkpoint_files:
            try:
                with h5py.File(filepath, 'r') as f:
                    progress = f['progress']
                    
                    checkpoint_info = {
                        'filepath': str(filepath),
                        'filename': filepath.name,
                        'current_step': progress.attrs['current_step'],
                        'total_steps': progress.attrs['total_steps'],
                        'completion_percentage': progress.attrs['completion_percentage'],
                        'last_save_time': progress.attrs['last_save_time'],
                        'file_size_mb': filepath.stat().st_size / (1024 * 1024)
                    }
                    
                    if run_name:
                        checkpoint_info['run_name'] = run_name
                    
                    checkpoints.append(checkpoint_info)
                    
            except Exception as e:
                warnings.warn(f"Could not read checkpoint {filepath}: {e}")
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x['current_step'])
        return checkpoints
    
    def find_latest_checkpoint(self, run_name: str) -> Optional[Path]:
        """Find the latest checkpoint for a run."""
        checkpoints = self.list_checkpoints(run_name)
        if not checkpoints:
            return None
        
        # Return path to latest checkpoint
        latest = max(checkpoints, key=lambda x: x['current_step'])
        return Path(latest['filepath'])


# =============================================================================
# Resume Management
# =============================================================================

class ResumeManager:
    """
    Manages MCMC run resumption and compatibility validation.
    
    This class provides intelligent recovery from checkpoints with careful
    validation of compatibility between saved state and current configuration.
    """
    
    def __init__(
        self,
        strict_validation: bool = True,
        auto_detect: bool = True,
        merge_chains: bool = True
    ):
        self.strict_validation = strict_validation
        self.auto_detect = auto_detect
        self.merge_chains = merge_chains
        
        # Internal state
        self._compatibility_cache = {}
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> MCMCState:
        """
        Load MCMC state from checkpoint file.
        
        Parameters
        ----------
        checkpoint_path : str or Path
            Path to checkpoint file
            
        Returns
        -------
        MCMCState
            Loaded MCMC state
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        return self._load_hdf5_checkpoint(checkpoint_path)
    
    def _load_hdf5_checkpoint(self, filepath: Path) -> MCMCState:
        """Load checkpoint from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            # Validate file format
            if 'metadata' not in f:
                raise ValueError("Invalid checkpoint file: missing metadata")
            
            # Load progress information
            progress = f['progress']
            current_step = progress.attrs['current_step']
            total_steps = progress.attrs['total_steps']
            warmup_steps = progress.attrs.get('warmup_steps', 0)
            start_time = progress.attrs['start_time']
            last_save_time = progress.attrs['last_save_time']
            
            # Load configuration
            parameter_config = None
            if 'parameter_config' in f['metadata']:
                config_str = f['metadata']['parameter_config'][()].decode()
                config_dict = json.loads(config_str)
                parameter_config = ParameterConfig.from_dict(config_dict)
            
            # Load samples
            samples = {}
            if 'samples' in f:
                samples_group = f['samples']
                for param_name in samples_group.keys():
                    samples[param_name] = np.array(samples_group[param_name])
            
            # Load warmup samples
            warmup_samples = None
            if 'warmup_samples' in f:
                warmup_samples = {}
                warmup_group = f['warmup_samples']
                for param_name in warmup_group.keys():
                    warmup_samples[param_name] = np.array(warmup_group[param_name])
            
            # Load metadata
            metadata_fields = ['likelihood_info', 'data_info', 'environment_info', 'run_metadata']
            loaded_metadata = {}
            
            for field in metadata_fields:
                if field in f['metadata']:
                    try:
                        field_str = f['metadata'][field][()].decode()
                        loaded_metadata[field] = json.loads(field_str)
                    except:
                        loaded_metadata[field] = {}
                else:
                    loaded_metadata[field] = {}
            
            # Load diagnostics history
            diagnostics_history = {}
            if 'diagnostics' in f:
                diag_group = f['diagnostics']
                for name in diag_group.keys():
                    try:
                        diagnostics_history[name] = np.array(diag_group[name]).tolist()
                    except:
                        pass
            
            # Load states
            random_state = None
            mcmc_state = None
            
            for state_name in ['random_state', 'mcmc_state']:
                if state_name in f:
                    try:
                        if 'state' in f[state_name]:
                            state_str = f[state_name]['state'][()].decode()
                            state_data = json.loads(state_str)
                        elif 'state_pickle' in f[state_name]:
                            state_pickle = f[state_name]['state_pickle'][()].tobytes()
                            state_data = pickle.loads(state_pickle)
                        else:
                            state_data = None
                        
                        if state_name == 'random_state':
                            random_state = state_data
                        else:
                            mcmc_state = state_data
                    except Exception as e:
                        warnings.warn(f"Could not load {state_name}: {e}")
            
            # Create MCMCState object
            state = MCMCState(
                samples=samples,
                warmup_samples=warmup_samples,
                mcmc_state=mcmc_state,
                current_step=current_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                random_state=random_state,
                start_time=start_time,
                last_save_time=last_save_time,
                parameter_config=parameter_config,
                likelihood_info=loaded_metadata['likelihood_info'],
                data_info=loaded_metadata['data_info'],
                diagnostics_history=diagnostics_history,
                run_metadata=loaded_metadata['run_metadata'],
                environment_info=loaded_metadata['environment_info']
            )
            
            return state
    
    def validate_compatibility(
        self, 
        saved_state: MCMCState,
        current_config: ParameterConfig,
        current_likelihood: Optional[Callable] = None,
        current_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate compatibility between saved state and current configuration.
        
        Parameters
        ----------
        saved_state : MCMCState
            Loaded checkpoint state
        current_config : ParameterConfig
            Current parameter configuration
        current_likelihood : Callable, optional
            Current likelihood function
        current_data : Dict[str, Any], optional
            Current data
            
        Returns
        -------
        compatible : bool
            Whether configurations are compatible
        issues : List[str]
            List of compatibility issues found
        """
        issues = []
        
        # Check parameter compatibility
        if saved_state.parameter_config:
            param_issues = self._check_parameter_compatibility(
                saved_state.parameter_config, current_config
            )
            issues.extend(param_issues)
        
        # Check likelihood compatibility
        if current_likelihood and saved_state.likelihood_info:
            likelihood_issues = self._check_likelihood_compatibility(
                saved_state.likelihood_info, current_likelihood
            )
            issues.extend(likelihood_issues)
        
        # Check data compatibility
        if current_data and saved_state.data_info:
            data_issues = self._check_data_compatibility(
                saved_state.data_info, current_data
            )
            issues.extend(data_issues)
        
        # Determine overall compatibility
        critical_issues = [issue for issue in issues if issue.startswith("CRITICAL:")]
        
        if self.strict_validation:
            compatible = len(issues) == 0
        else:
            compatible = len(critical_issues) == 0
        
        return compatible, issues
    
    def _check_parameter_compatibility(
        self, 
        saved_config: ParameterConfig, 
        current_config: ParameterConfig
    ) -> List[str]:
        """Check parameter configuration compatibility."""
        issues = []
        
        saved_params = set(saved_config.get_parameter_names())
        current_params = set(current_config.get_parameter_names())
        
        # Check for missing parameters
        missing_params = saved_params - current_params
        if missing_params:
            issues.append(f"CRITICAL: Missing parameters in current config: {missing_params}")
        
        # Check for extra parameters
        extra_params = current_params - saved_params  
        if extra_params:
            issues.append(f"WARNING: Extra parameters in current config: {extra_params}")
        
        # Check parameter specifications for common parameters
        common_params = saved_params & current_params
        for param_name in common_params:
            saved_param = saved_config.parameters[param_name]
            current_param = current_config.parameters[param_name]
            
            # Check prior distributions
            if saved_param.prior != current_param.prior:
                if self.strict_validation:
                    issues.append(f"CRITICAL: Prior changed for {param_name}: "
                                f"{saved_param.prior} -> {current_param.prior}")
                else:
                    issues.append(f"WARNING: Prior changed for {param_name}")
            
            # Check bounds
            if saved_param.bounds != current_param.bounds:
                issues.append(f"WARNING: Bounds changed for {param_name}")
        
        return issues
    
    def _check_likelihood_compatibility(
        self, 
        saved_info: Dict[str, Any], 
        current_likelihood: Callable
    ) -> List[str]:
        """Check likelihood function compatibility."""
        issues = []
        
        current_info = create_likelihood_info(current_likelihood)
        
        # Check function name
        if saved_info.get('function_name') != current_info.get('function_name'):
            issues.append(f"WARNING: Likelihood function name changed")
        
        # Check source code hash
        saved_hash = saved_info.get('source_hash')
        current_hash = current_info.get('source_hash')
        if saved_hash and current_hash and saved_hash != current_hash:
            if self.strict_validation:
                issues.append("CRITICAL: Likelihood function source code changed")
            else:
                issues.append("WARNING: Likelihood function source code may have changed")
        
        return issues
    
    def _check_data_compatibility(
        self, 
        saved_info: Dict[str, Any], 
        current_data: Dict[str, Any]
    ) -> List[str]:
        """Check data compatibility."""
        issues = []
        
        current_info = create_data_info(current_data)
        
        # Check data keys
        saved_keys = set(saved_info.get('data_keys', []))
        current_keys = set(current_info.get('data_keys', []))
        
        if saved_keys != current_keys:
            missing_keys = saved_keys - current_keys
            extra_keys = current_keys - saved_keys
            
            if missing_keys:
                issues.append(f"CRITICAL: Missing data keys: {missing_keys}")
            if extra_keys:
                issues.append(f"WARNING: Extra data keys: {extra_keys}")
        
        return issues


# =============================================================================
# Utility Functions
# =============================================================================

def create_likelihood_info(likelihood_func: Callable) -> Dict[str, Any]:
    """Extract comprehensive information about likelihood function."""
    info = {
        'function_name': getattr(likelihood_func, '__name__', 'unknown'),
        'module': getattr(likelihood_func, '__module__', 'unknown'),
        'timestamp': datetime.now().isoformat()
    }
    
    # Get source code if available
    try:
        source = inspect.getsource(likelihood_func)
        info['source_code'] = source
        info['source_hash'] = hashlib.md5(source.encode()).hexdigest()
    except:
        info['source_code'] = 'source unavailable'
        info['source_hash'] = 'unknown'
    
    # Get signature
    try:
        sig = inspect.signature(likelihood_func)
        info['signature'] = str(sig)
        info['parameters'] = list(sig.parameters.keys())
    except:
        info['signature'] = 'unknown'
        info['parameters'] = []
    
    # Get docstring
    info['docstring'] = getattr(likelihood_func, '__doc__', None)
    
    return info


def create_data_info(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract information about input data."""
    info = {
        'data_keys': list(data_dict.keys()),
        'timestamp': datetime.now().isoformat()
    }
    
    # Analyze each data array
    for key, value in data_dict.items():
        if isinstance(value, (np.ndarray, list, tuple)):
            try:
                arr = np.asarray(value)
                info[f'{key}_shape'] = arr.shape
                info[f'{key}_dtype'] = str(arr.dtype)
                info[f'{key}_size'] = arr.size
                
                if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                    info[f'{key}_min'] = float(np.min(arr))
                    info[f'{key}_max'] = float(np.max(arr))
                    info[f'{key}_mean'] = float(np.mean(arr))
                    info[f'{key}_std'] = float(np.std(arr))
                
                # Create hash for signature
                info[f'{key}_hash'] = hashlib.md5(arr.tobytes()).hexdigest()[:8]
                
            except Exception as e:
                info[f'{key}_error'] = str(e)
        else:
            info[f'{key}_type'] = type(value).__name__
            info[f'{key}_value'] = str(value)[:100]
    
    # Create overall data hash
    data_str = json.dumps({k: v for k, v in info.items() if not k.endswith('_hash')})
    info['data_hash'] = hashlib.md5(data_str.encode()).hexdigest()[:12]
    
    return info