#!/usr/bin/env python3
"""
HIcosmo Visualization Styles - Minimalist Version

Removed 5 nearly identical style configurations, kept only core professional style.

Author: Jingzhao Qi
Total Lines: ~50 (vs original 563 lines)
"""

import matplotlib.pyplot as plt

# Professional style configuration - based on FigStyle.py
PROFESSIONAL_STYLE = {
    'axes.linewidth': 1.2,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 14,
    'legend.frameon': False,  # Frameless legend requirement
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.size': 12,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
}

def apply_style():
    """Apply professional style - only style function needed"""
    plt.rcParams.update(PROFESSIONAL_STYLE)

def qstyle(**kwargs):
    """Backward compatible style function"""
    config = PROFESSIONAL_STYLE.copy()
    config.update(kwargs)
    plt.rcParams.update(config)
    return ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#0F7173', '#7B2D26']

__all__ = ['apply_style', 'qstyle', 'PROFESSIONAL_STYLE']