#!/usr/bin/env python3
"""
Configuration Constants for HiCosmo Samplers

Centralizes all configuration constants to eliminate hardcoding.
"""

from pathlib import Path

# ========== MCMC Sampling Defaults ==========

# Default number of MCMC samples
DEFAULT_NUM_SAMPLES = 2000

# Default number of parallel chains
DEFAULT_NUM_CHAINS = 4

# Warmup steps for different modes
DEFAULT_WARMUP_STANDARD = 2000      # Without optimization
DEFAULT_WARMUP_OPTIMIZED = 300      # With optimization

# ========== Optimization Defaults ==========

# Maximum iterations for optimization
DEFAULT_MAX_OPTIMIZATION_ITERATIONS = 1000

# Report progress every N iterations
OPTIMIZATION_PROGRESS_INTERVAL = 100

# Penalty factor for constraints
OPTIMIZATION_PENALTY_FACTOR = 1000.0

# ========== Checkpointing ==========

# Save checkpoint interval
DEFAULT_CHECKPOINT_INTERVAL = 1000

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = Path("mcmc_chains")

# File extension
CHECKPOINT_FILE_EXTENSION = ".h5"

# File patterns
CHECKPOINT_PATTERN_TEMPLATE = "{run_name}_step_*.h5"
FINAL_RESULT_PATTERN = "{run_name}.h5"

# ========== Random Number Generation ==========

# RNG seed modulo for 32-bit compatibility
RNG_SEED_MODULO = 2**32

# ========== Numerical Tolerances ==========

# R-hat convergence threshold
RHAT_CONVERGENCE_THRESHOLD = 1.1

# Minimum effective sample size
MIN_EFFECTIVE_SAMPLE_SIZE = 100

# ========== Multi-core Optimization ==========

# Default number of CPU devices for parallel execution
DEFAULT_NUM_CPU_DEVICES = 4

# Auto-detect CPU cores
AUTO_DETECT_CPU_CORES = True

# Warning threshold for device count mismatch
DEVICE_MISMATCH_WARNING_THRESHOLD = 2