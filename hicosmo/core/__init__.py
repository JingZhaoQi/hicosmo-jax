"""
HIcosmo Core - Clean and Unified Architecture
=============================================

This module provides the clean, unified foundation for all HIcosmo functionality.

Key components:
- base: Abstract base classes with minimal interfaces
- fast_integration: Ultra-high performance integration engine  
- unified_parameters: Single parameter management system
"""

from .base import CosmologyBase, DistanceCalculator
from .fast_integration import FastIntegration
from .unified_parameters import CosmologicalParameters, PLANCK_2018, PLANCK_2015, WMAP9

__all__ = [
    'CosmologyBase',
    'DistanceCalculator', 
    'FastIntegration',
    'CosmologicalParameters',
    'PLANCK_2018',
    'PLANCK_2015', 
    'WMAP9'
]