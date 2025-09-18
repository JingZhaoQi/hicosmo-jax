#!/usr/bin/env python3
"""
快速测试PantheonPlus likelihood的关键参数点
"""
import numpy as np
from pathlib import Path

# 初始化多核配置
from hicosmo.samplers import init_hicosmo
init_hicosmo(cpu_cores=8, verbose=False)

from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood

def test_key_parameters():
    """测试关键参数点"""
    print("=== 快速PantheonPlus测试 ===")

    data_path = Path("data/DataRelease")

    # 测试PantheonPlus
    likelihood = PantheonPlusLikelihood(
        data_path=str(data_path),
        include_shoes=False,
        z_min=0.01,
        marginalize_M_B=True  # 使用边际化M_B
    )

    print(f"数据点数量: {likelihood.n_sne}")

    # 已知的文献约束值
    test_points = [
        {"name": "Planck2018", "H0": 67.4, "Om": 0.315},
        {"name": "PantheonPlus官方", "H0": 73.4, "Om": 0.334},  # Brout et al. 2022
        {"name": "SH0ES-like", "H0": 73.0, "Om": 0.3},
        {"name": "最优搜索1", "H0": 72.0, "Om": 0.33},
        {"name": "最优搜索2", "H0": 71.0, "Om": 0.32},
    ]

    print("\n参数点测试:")
    best_chi2 = np.inf
    best_point = None

    for point in test_points:
        model = LCDM(H0=point["H0"], Omega_m=point["Om"])

        # 使用边际化likelihood
        log_like = likelihood.log_likelihood(model)
        chi2 = -2 * log_like
        best_MB = likelihood.best_fit_M_B(model)

        chi2_dof = chi2 / (likelihood.n_sne - 3)

        print(f"{point['name']:15s}: H0={point['H0']:5.1f}, Ωm={point['Om']:5.3f} → "
              f"χ²={chi2:7.1f}, χ²/dof={chi2_dof:.3f}, M_B={best_MB:.3f}")

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_point = point.copy()
            best_point['chi2'] = chi2
            best_point['M_B'] = best_MB

    print(f"\n当前测试的最优点:")
    print(f"  {best_point['name']}: H0={best_point['H0']}, Ωm={best_point['Om']}")
    print(f"  χ²_min = {best_point['chi2']:.1f}, M_B = {best_point['M_B']:.3f}")

    # 文献期望值
    print(f"\n文献对比:")
    print(f"  Brout et al. 2022 (Pantheon+): H0=73.4±1.4, Ωm=0.334±0.018")
    print(f"  我们当前最优: H0={best_point['H0']}, Ωm={best_point['Om']}")

    # 计算差异
    h0_diff = abs(best_point['H0'] - 73.4) / 1.4
    om_diff = abs(best_point['Om'] - 0.334) / 0.018

    print(f"  差异: ΔH0 = {h0_diff:.1f}σ, ΔΩm = {om_diff:.1f}σ")

    if h0_diff < 2 and om_diff < 2:
        print(f"  ✅ 在2σ内与文献一致")
    else:
        print(f"  ⚠️ 超过2σ差异，需要进一步检查")

if __name__ == "__main__":
    test_key_parameters()