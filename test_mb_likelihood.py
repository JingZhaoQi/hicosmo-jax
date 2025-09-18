#!/usr/bin/env python3
"""
测试M_B作为自由参数的likelihood计算
"""
import os
import numpy as np
from pathlib import Path

# 初始化多核配置
from hicosmo.samplers import init_hicosmo
init_hicosmo(cpu_cores=8, verbose=False)

from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood

def test_mb_likelihood():
    """测试M_B作为自由参数"""
    print("=== 测试M_B作为自由参数的likelihood ===")

    data_path = Path("data/DataRelease")

    # 测试PantheonPlus with M_B as free parameter
    likelihood = PantheonPlusLikelihood(
        data_path=str(data_path),
        include_shoes=False,
        z_min=0.01,
        marginalize_M_B=False  # 不边际化M_B
    )

    print(f"数据点数量: {likelihood.n_sne}")
    print(f"M_B处理: {'自由参数' if not likelihood.marginalize_M_B else '边际化'}")

    # 测试参数点（参考官方结果）
    test_points = [
        {"name": "Planck2018", "H0": 67.4, "Om": 0.315, "M_B": -19.25},
        {"name": "PantheonPlus官方", "H0": 73.4, "Om": 0.334, "M_B": -19.25},
        {"name": "Test Point 1", "H0": 70.0, "Om": 0.30, "M_B": -19.30},
        {"name": "Test Point 2", "H0": 72.0, "Om": 0.32, "M_B": -19.20},
        {"name": "Test Point 3", "H0": 74.0, "Om": 0.35, "M_B": -19.15},
    ]

    print("\n参数点测试 (M_B作为自由参数):")
    best_chi2 = np.inf
    best_point = None

    for point in test_points:
        model = LCDM(H0=point["H0"], Omega_m=point["Om"])

        # M_B作为参数传入
        log_like = likelihood.log_likelihood(model, M_B=point["M_B"])
        chi2 = -2 * log_like
        chi2_dof = chi2 / (likelihood.n_sne - 3)  # 3个自由参数

        print(f"{point['name']:15s}: H0={point['H0']:5.1f}, Ωm={point['Om']:5.3f}, M_B={point['M_B']:6.3f} → "
              f"χ²={chi2:7.1f}, χ²/dof={chi2_dof:.3f}")

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_point = point.copy()
            best_point['chi2'] = chi2

    print(f"\n当前测试的最优点:")
    print(f"  {best_point['name']}: H0={best_point['H0']}, Ωm={best_point['Om']}, M_B={best_point['M_B']}")
    print(f"  χ²_min = {best_point['chi2']:.1f}")

    # 测试M_B范围的影响
    print(f"\n测试M_B范围的影响 (固定H0=73.4, Ωm=0.334):")
    model = LCDM(H0=73.4, Omega_m=0.334)

    M_B_values = np.linspace(-19.5, -19.0, 11)
    chi2_values = []

    for M_B in M_B_values:
        log_like = likelihood.log_likelihood(model, M_B=M_B)
        chi2 = -2 * log_like
        chi2_values.append(chi2)
        print(f"  M_B = {M_B:.2f} → χ² = {chi2:.1f}")

    chi2_values = np.array(chi2_values)
    best_idx = np.argmin(chi2_values)
    best_M_B = M_B_values[best_idx]

    print(f"\n最优M_B: {best_M_B:.3f} (χ²_min = {chi2_values[best_idx]:.1f})")

if __name__ == "__main__":
    test_mb_likelihood()