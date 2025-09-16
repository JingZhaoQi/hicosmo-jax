#!/usr/bin/env python3
"""
Test tick optimization to verify improved readability
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt

print("Testing Tick Optimization")
print("=" * 30)

# Generate test data
np.random.seed(42)
chain_data = {
    'H0': np.random.normal(70.0, 2.0, 1000),
    'Omega_m': np.random.normal(0.30, 0.02, 1000)
}

print("Creating test plot to check tick intervals...")

try:
    from hicosmo.visualization import plot_corner

    # Create corner plot and check tick optimization
    fig = plot_corner(chain_data,
                     params=['H0', 'Omega_m'],
                     filename='test_tick_intervals.pdf')

    print("\n📊 Checking tick intervals on axes:")

    for i, ax in enumerate(fig.get_axes()):
        if ax.get_xlabel() or ax.get_ylabel():
            x_ticks = ax.get_xticks()
            y_ticks = ax.get_yticks()

            x_intervals = np.diff(x_ticks[x_ticks != 0])  # Remove 0 ticks
            y_intervals = np.diff(y_ticks[y_ticks != 0])

            print(f"  Axis {i}:")
            if len(x_intervals) > 0:
                print(f"    X-axis ticks: {len(x_ticks)} total")
                print(f"    X-axis intervals: {x_intervals[:3]}... (avg: {np.mean(x_intervals):.2f})")
            if len(y_intervals) > 0:
                print(f"    Y-axis ticks: {len(y_ticks)} total")
                print(f"    Y-axis intervals: {y_intervals[:3]}... (avg: {np.mean(y_intervals):.2f})")

    plt.close(fig)
    print("\n✅ Tick optimization test completed")
    print("📁 Check results/test_tick_intervals.pdf for visual verification")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 30)
print("✅ Tick Optimization Test Complete!")