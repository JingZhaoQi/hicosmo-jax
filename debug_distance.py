#!/usr/bin/env python3
"""Debug distance calculation error"""

import sys
import os
import traceback

# Add HiCosmo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from hicosmo.background.distances import CosmologicalDistances
    from hicosmo.background.background import BackgroundEvolution
    from hicosmo.models.lcdm import LCDMModel
    
    model = LCDMModel()
    bg = BackgroundEvolution(model)
    distances = CosmologicalDistances(bg)
    
    test_params = {
        'H0': 67.36,
        'Omega_m': 0.3153,
        'Omega_k': 0.0,
        'Omega_Lambda': 0.6847,
        'w0': -1.0,
        'wa': 0.0
    }
    
    print("Testing comoving distance...")
    d_c = distances.comoving_distance(1.0, test_params)
    print(f"d_c = {d_c}")
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()