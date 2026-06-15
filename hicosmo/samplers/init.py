#!/usr/bin/env python3
"""
HIcosmo Elegant Multi-core Initialization Module

Provides clean and elegant multi-core environment configuration
for high-performance cosmological inference.

Author: Jingzhao Qi
"""

import os
import sys
import logging
from typing import Optional, Union
from ..utils.logging import get_logger, configure_logging

# Suppress irrelevant warnings from third-party libraries
logging.getLogger("jax._src.xla_bridge").setLevel(
    logging.WARNING
)  # rocm, tpu backend warnings
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)  # NumExpr thread info
logging.getLogger("getdist").setLevel(logging.WARNING)  # getdist verbose messages

logger = get_logger(__name__)


class Config:
    """
    HIcosmo Global Configuration Manager

    Provides elegant one-line initialization with automatic multi-core and environment setup
    for high-performance cosmological inference
    """

    _initialized = False
    _config = {}

    @classmethod
    def init(
        cls,
        num_devices: Union[int, str, None] = "auto",
        *,
        device: Union[str, None] = None,
        verbose: bool = True,
        force: bool = True,
    ) -> bool:
        """
        Elegant one-line initialization for parallel MCMC chains.

        Parameters
        ----------
        num_devices : int, 'auto', or None
            Number of parallel devices (= number of parallel chains):
            - int: Use N devices for N parallel chains (e.g., init(8) for 8 chains)
            - 'auto': Auto-detect optimal device count (default)
            - None: Use single device (vectorized mode)
            - 'GPU' / 'cuda': GPU auto-detect and configure
        device : str or None
            Explicit device selection (overrides num_devices if 'GPU'):
            - None (default): CPU mode
            - 'GPU' / 'cuda': GPU auto-detect and configure
        verbose : bool
            Whether to show initialization messages
        force : bool
            Force override existing XLA_FLAGS (default True for user convenience)

        Returns
        -------
        bool
            True if successful, False otherwise

        Examples
        --------
        >>> # 8 parallel chains (most common usage)
        >>> Config.init(8)

        >>> # Auto configuration
        >>> Config.init()

        >>> # Silent initialization with 4 chains
        >>> Config.init(4, verbose=False)

        >>> # GPU auto configuration
        >>> Config.init("GPU")
        """
        if cls._initialized:
            if verbose:
                logger.info("HIcosmo already initialized")
            return True

        try:
            configure_logging(verbosity=1 if verbose else 0)
            os.environ.setdefault("JAX_ENABLE_X64", "True")

            # Determine device type and normalize num_devices
            device_type = "cpu"
            normalized_devices = None
            preferred_num_chains = None

            # Check if num_devices is actually a device type string
            if isinstance(num_devices, str):
                lowered = num_devices.strip().lower()
                if lowered in {"gpu", "cuda"}:
                    device_type = "gpu"
                    num_devices = None  # Will auto-detect GPU count
                elif lowered == "auto":
                    # Auto-detect: use system cores up to 8
                    system_cores = os.cpu_count() or 4
                    num_devices = min(system_cores, 8)

            # Override with explicit device parameter
            if device is not None:
                lowered = device.strip().lower()
                if lowered in {"gpu", "cuda"}:
                    device_type = "gpu"

            # Handle GPU mode
            if device_type == "gpu":
                gpu_count = cls._detect_gpu_count(verbose=verbose)
                if gpu_count < 1:
                    if verbose:
                        logger.warning(
                            "No GPU devices detected; falling back to CPU mode."
                        )
                    device_type = "cpu"
                    normalized_devices = 1
                else:
                    if num_devices is None:
                        normalized_devices = gpu_count
                    else:
                        normalized_devices = int(num_devices)
                        if normalized_devices > gpu_count:
                            if verbose:
                                logger.warning(
                                    "Requested %s devices exceeds available GPUs (%s); using %s.",
                                    normalized_devices,
                                    gpu_count,
                                    gpu_count,
                                )
                            normalized_devices = gpu_count
                    preferred_num_chains = (
                        normalized_devices if normalized_devices > 1 else 4
                    )

            # Handle CPU mode
            if device_type == "cpu":
                if num_devices is None:
                    normalized_devices = 1
                else:
                    normalized_devices = int(num_devices)
                    if normalized_devices < 1:
                        normalized_devices = 1
                if normalized_devices > 1:
                    preferred_num_chains = normalized_devices

            # The CPU device count is fixed only when JAX initializes its
            # backend (the first computation or jax.devices() call), not merely
            # when jax is imported. Importing hicosmo imports jax to enable x64
            # but runs no computation, so hc.init() can still set the requested
            # CPU device count -- provided XLA_FLAGS is set before the backend is
            # touched (verified after setup, below). For GPU the host-platform
            # device-count flag does not apply, so the count is left to JAX.
            allow_xla_flags = True
            if device_type == "gpu" and "jax" in sys.modules:
                allow_xla_flags = False

            # Set up devices
            success = cls._setup_multicore(
                cpu_cores="auto",
                verbose=verbose,
                num_devices=normalized_devices,
                allow_xla_flags=allow_xla_flags,
                force=force,
                device_type=device_type,
            )
            if not success and verbose:
                logger.warning("Multi-core setup had issues, continuing with defaults")

            # Verify the requested CPU device count took effect. This is the
            # first jax.devices() call, which initializes the backend with the
            # XLA_FLAGS set above. If a JAX computation already ran before
            # hc.init(), the backend is fixed and the count cannot change.
            if device_type == "cpu":
                try:
                    import jax

                    actual_devices = len(jax.devices())
                    if actual_devices != normalized_devices and verbose:
                        logger.warning(
                            "Requested %s CPU devices but JAX is already "
                            "initialized with %s; call hc.init() before any JAX "
                            "computation to use all requested devices.",
                            normalized_devices,
                            actual_devices,
                        )
                    normalized_devices = actual_devices
                except Exception:
                    pass

            # Mark as initialized
            cls._initialized = True
            cls._config["num_devices"] = normalized_devices
            cls._config["verbose"] = verbose
            cls._config["device_type"] = device_type
            cls._config["preferred_num_chains"] = preferred_num_chains

            if verbose:
                cls._print_initialization_summary()

            return True

        except Exception as e:
            if verbose:
                logger.error(f"HIcosmo initialization failed: {e}")
            return False

    @classmethod
    def _setup_multicore(
        cls,
        cpu_cores: Union[int, str, None],
        verbose: bool,
        num_devices: int = 1,
        allow_xla_flags: bool = True,
        force: bool = False,
        device_type: str = "cpu",
    ) -> bool:
        """Internal method: setup multi-core configuration

        Parameters
        ----------
        cpu_cores : int or 'auto'
            Number of CPU cores for thread pool
        verbose : bool
            Print configuration info
        num_devices : int
            Number of JAX logical devices. Default 1 for vectorized mode.
            Set to num_chains (e.g., 4) for parallel mode with independent progress bars.
        allow_xla_flags : bool
            Whether to set XLA_FLAGS (should be False if JAX already imported)
        force : bool
            Override existing device count in XLA_FLAGS when allow_xla_flags is True
        device_type : str
            'cpu' or 'gpu'
        """
        try:
            # Normalize device count
            num_devices = int(num_devices)
            if num_devices < 1:
                if verbose:
                    logger.warning("Invalid device count, using 1")
                num_devices = 1

            # Determine core count (thread pool)
            num_cores = None
            threads_per_device = None
            if cpu_cores is not None:
                if cpu_cores == "auto":
                    system_cores = os.cpu_count() or 4
                    # Use system cores, but cap at 8 (avoid excessive parallelization)
                    num_cores = min(system_cores, 8)
                    # Use at least 4 cores (if system supports)
                    num_cores = (
                        max(min(num_cores, system_cores), 4)
                        if system_cores >= 4
                        else system_cores
                    )
                else:
                    num_cores = int(cpu_cores)

                # Validate core count
                if num_cores < 1:
                    if verbose:
                        logger.warning("Invalid core count, using 1")
                    num_cores = 1

                # When using multiple devices, divide threads among devices
                threads_per_device = max(1, num_cores // num_devices)
                os.environ.setdefault("JAX_NUM_THREADS", str(threads_per_device))

            # Set XLA flags for multi-device CPU support
            if device_type == "cpu":
                if allow_xla_flags:
                    cls._update_xla_flags(
                        num_devices=num_devices, force=force, verbose=verbose
                    )
                elif verbose and num_devices > 1:
                    logger.warning("JAX already imported; XLA_FLAGS not modified")

                # Import and configure NumPyro after environment setup
                import numpyro

                numpyro.set_host_device_count(num_devices)
            else:
                if verbose and num_devices > 1:
                    logger.info(
                        "GPU mode: using %s device(s); skipping XLA_FLAGS host device setup.",
                        num_devices,
                    )

            cls._config["actual_cores"] = num_cores
            cls._config["num_devices"] = num_devices
            cls._config["threads_per_device"] = threads_per_device

            if verbose:
                if num_devices == 1:
                    if num_cores is None:
                        logger.info("Configured 1 JAX device (vectorized mode)")
                    else:
                        logger.info(
                            f"Configured {num_cores} CPU cores via thread pool (vectorized mode)"
                        )
                else:
                    if threads_per_device is None:
                        logger.info(
                            f"Configured {num_devices} JAX devices (parallel mode)"
                        )
                    else:
                        logger.info(
                            f"Configured {num_devices} JAX devices × {threads_per_device} threads (parallel mode)"
                        )

            return True

        except Exception as e:
            if verbose:
                logger.warning(f"Multi-core setup failed: {e}")
            return False

    @staticmethod
    def _normalize_num_devices(num_devices: Union[int, str, None]) -> int:
        """Normalize num_devices input to an integer >= 1."""
        if num_devices is None:
            return 1
        if isinstance(num_devices, str):
            if num_devices.lower() == "auto":
                system_cores = os.cpu_count() or 1
                return max(1, min(system_cores, 8))
            return int(num_devices)
        return int(num_devices)

    @staticmethod
    def _normalize_device(
        cpu_cores: Union[int, str, None], device: Union[str, None]
    ) -> tuple[Union[int, str, None], str]:
        """Normalize device selection and handle positional GPU shorthand."""
        if device is not None:
            device_type = device.strip().lower()
        elif isinstance(cpu_cores, str):
            normalized = cpu_cores.strip().lower()
            if normalized in {"gpu", "cuda"}:
                device_type = "gpu"
                cpu_cores = None
            elif normalized in {"cpu"}:
                device_type = "cpu"
                cpu_cores = "auto"
            else:
                device_type = "cpu"
        else:
            device_type = "cpu"

        if device_type not in {"cpu", "gpu"}:
            device_type = "cpu"
        return cpu_cores, device_type

    @staticmethod
    def _detect_gpu_count(verbose: bool = False) -> int:
        """Detect available GPU devices."""
        try:
            import jax

            gpus = jax.devices("gpu")
            return len(gpus)
        except Exception:
            if verbose:
                logger.warning("Unable to detect GPU devices; defaulting to CPU.")
            return 0

    @staticmethod
    def _update_xla_flags(num_devices: int, force: bool, verbose: bool) -> None:
        """Update XLA_FLAGS safely for multi-device CPU."""
        existing = os.environ.get("XLA_FLAGS", "").strip()
        parts = existing.split() if existing else []
        has_device_flag = any(
            p.startswith("--xla_force_host_platform_device_count=") for p in parts
        )

        if has_device_flag and not force:
            if verbose:
                logger.warning(
                    "XLA_FLAGS already sets device count; pass force=True to override"
                )
            return

        # Remove previous device-related flags
        cleaned = [
            p
            for p in parts
            if not p.startswith("--xla_force_host_platform_device_count=")
            and not p.startswith("--xla_cpu_multi_thread_eigen=")
        ]
        cleaned.append(f"--xla_force_host_platform_device_count={num_devices}")
        cleaned.append("--xla_cpu_multi_thread_eigen=true")
        os.environ["XLA_FLAGS"] = " ".join(cleaned).strip()

    @classmethod
    def _print_initialization_summary(cls):
        """Print initialization summary."""
        num_devices = cls._config.get("num_devices", 1)
        device_type = cls._config.get("device_type", "cpu")

        try:
            import jax

            device_count = len(jax.devices())

            if device_type == "gpu":
                logger.info(
                    f"✅ HIcosmo: {device_count} GPU(s) ready for parallel MCMC"
                )
            elif device_count == 1:
                logger.info("✅ HIcosmo: 1 device (vectorized mode)")
            else:
                logger.info(
                    f"✅ HIcosmo: {device_count} devices ready for parallel MCMC"
                )

        except ImportError:
            logger.warning("JAX: Not available")

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
            "initialized": cls._initialized,
            "config": cls._config.copy(),
            "system_cores": os.cpu_count(),
        }

        # JAX information
        try:
            import jax

            status_info["jax_devices"] = len(jax.devices())
            status_info["jax_device_list"] = [str(d) for d in jax.devices()]
        except ImportError:
            status_info["jax_devices"] = 0
            status_info["jax_device_list"] = []

        return status_info

    @classmethod
    def reset(cls):
        """Reset configuration (mainly for testing)"""
        cls._initialized = False
        cls._config = {}


# Provide convenient import interface
def init_hicosmo(
    num_devices: Union[int, str, None] = "auto",
    *,
    device: Union[str, None] = None,
    verbose: bool = True,
    force: bool = True,
) -> bool:
    """
    HIcosmo one-line initialization function.

    Examples
    --------
    >>> from hicosmo.samplers import init_hicosmo
    >>> init_hicosmo(8)   # 8 parallel chains
    >>> init_hicosmo()    # Auto configuration
    >>> init_hicosmo("GPU")  # GPU mode
    """
    return Config.init(
        num_devices=num_devices,
        device=device,
        verbose=verbose,
        force=force,
    )


# Backward compatible multi-core setup function
def setup_multicore_execution(
    num_devices: Optional[int] = None,
    auto_detect: bool = True,
    force_override: bool = True,
) -> bool:
    """Backward compatible multi-core setup function."""
    devices = num_devices if num_devices else ("auto" if auto_detect else None)
    return Config.init(
        num_devices=devices,
        verbose=True,
        force=force_override,
    )


def init(
    num_devices: Union[int, str, None] = "auto",
    *,
    device: Union[str, None] = None,
    verbose: bool = True,
    force: bool = True,
) -> bool:
    """
    Top-level initialization helper (alias of Config.init).

    Examples
    --------
    >>> import hicosmo as hc
    >>> hc.init(8)      # 8 parallel chains (most common)
    >>> hc.init()       # Auto configuration
    >>> hc.init("GPU")  # GPU mode
    """
    return Config.init(
        num_devices=num_devices,
        device=device,
        verbose=verbose,
        force=force,
    )
