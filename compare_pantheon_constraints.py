#!/usr/bin/env python3
"""
对比我们的PantheonPlus约束与文献已知结果
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 首先初始化多核配置
from hicosmo.samplers import init_hicosmo
init_hicosmo(cpu_cores=8, verbose=False)

from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood

def create_chi2_grid():
    """创建Chi2网格对比已知约束"""
    print("=== Chi2网格分析 ===")

    data_path = Path("data/DataRelease")

    # 测试PantheonPlus
    likelihood = PantheonPlusLikelihood(
        data_path=str(data_path),
        include_shoes=False,
        z_min=0.01,
        marginalize_M_B=True  # 使用边际化M_B
    )

    # 参数网格
    H0_range = np.linspace(65, 75, 21)  # 21个点
    Om_range = np.linspace(0.25, 0.35, 21)  # 21个点

    chi2_grid = np.zeros((len(H0_range), len(Om_range)))
    best_MB_grid = np.zeros((len(H0_range), len(Om_range)))

    print(f"计算{len(H0_range)}×{len(Om_range)}网格...")

    for i, H0 in enumerate(H0_range):
        for j, Om in enumerate(Om_range):
            model = LCDM(H0=H0, Omega_m=Om)

            # 使用边际化likelihood
            log_like = likelihood.log_likelihood(model)
            chi2 = -2 * log_like  # 转换为chi2

            # 同时记录最优M_B
            best_MB = likelihood.best_fit_M_B(model)

            chi2_grid[i, j] = chi2
            best_MB_grid[i, j] = best_MB

        if (i + 1) % 5 == 0:
            print(f"  进度: {i+1}/{len(H0_range)}")

    # 找到最小chi2
    min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    best_H0 = H0_range[min_idx[0]]
    best_Om = Om_range[min_idx[1]]
    min_chi2 = chi2_grid[min_idx]
    best_MB = best_MB_grid[min_idx]

    print(f"\n最优参数:")
    print(f"  H0 = {best_H0:.1f}")
    print(f"  Ωm = {best_Om:.3f}")
    print(f"  M_B = {best_MB:.3f}")
    print(f"  χ²_min = {min_chi2:.1f}")
    print(f"  χ²/dof = {min_chi2/(likelihood.n_sne-3):.3f}")

    # 计算置信区间
    delta_chi2_1sigma = 2.3  # 2参数的1σ
    delta_chi2_2sigma = 6.17  # 2参数的2σ

    # 保存结果
    np.savez('pantheon_chi2_grid.npz',
             H0_range=H0_range, Om_range=Om_range,
             chi2_grid=chi2_grid, best_MB_grid=best_MB_grid,
             min_chi2=min_chi2, best_H0=best_H0, best_Om=best_Om, best_MB=best_MB)

    # 绘制contour图
    plt.figure(figsize=(10, 8))
    H0_mesh, Om_mesh = np.meshgrid(H0_range, Om_range)

    # 相对于最小值的Delta chi2
    delta_chi2 = chi2_grid.T - min_chi2  # 转置以匹配meshgrid

    # 绘制contours
    contours = plt.contour(H0_mesh, Om_mesh, delta_chi2,
                          levels=[delta_chi2_1sigma, delta_chi2_2sigma],
                          colors=['red', 'blue'], linewidths=2)
    plt.clabel(contours, inline=True, fontsize=10, fmt='%g σ')

    # 最优点
    plt.plot(best_H0, best_Om, 'k*', markersize=15, label=f'Best fit\n(H₀={best_H0:.1f}, Ωₘ={best_Om:.3f})')

    # 文献值对比
    plt.plot(67.4, 0.315, 'ro', markersize=8, label='Planck 2018')
    plt.plot(73.0, 0.3, 'go', markersize=8, label='SH0ES 2019')

    plt.xlabel('H₀ [km/s/Mpc]')
    plt.ylabel('Ωₘ')
    plt.title('PantheonPlus Constraints (our implementation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('our_pantheon_constraints.pdf')
    plt.savefig('our_pantheon_constraints.png', dpi=150)
    print(f"\nContour图保存为: our_pantheon_constraints.pdf/png")

    return H0_range, Om_range, chi2_grid, best_H0, best_Om, min_chi2

def compare_with_literature():
    """对比文献已知结果"""
    print("\n=== 文献对比 ===")

    # 已知的Pantheon+约束（来自官方论文）
    literature_values = {
        "Pantheon+": {
            "H0": 73.4,
            "H0_err": 1.4,
            "Om": 0.334,
            "Om_err": 0.018,
            "source": "Brout et al. 2022"
        },
        "Planck2018": {
            "H0": 67.4,
            "H0_err": 0.5,
            "Om": 0.315,
            "Om_err": 0.007,
            "source": "Planck Collaboration 2020"
        }
    }

    # 我们的结果（从网格分析）
    try:
        data = np.load('pantheon_chi2_grid.npz')
        our_H0 = float(data['best_H0'])
        our_Om = float(data['best_Om'])
        our_MB = float(data['best_MB'])

        print("我们的约束:")
        print(f"  H0 = {our_H0:.1f} km/s/Mpc")
        print(f"  Ωm = {our_Om:.3f}")
        print(f"  M_B = {our_MB:.3f}")
        print()

        for name, values in literature_values.items():
            print(f"{values['source']} ({name}):")
            print(f"  H0 = {values['H0']:.1f} ± {values['H0_err']:.1f} km/s/Mpc")
            print(f"  Ωm = {values['Om']:.3f} ± {values['Om_err']:.3f}")

            # 计算差异（sigma单位）
            h0_diff = abs(our_H0 - values['H0']) / values['H0_err']
            om_diff = abs(our_Om - values['Om']) / values['Om_err']

            print(f"  与我们的差异: ΔH0 = {h0_diff:.1f}σ, ΔΩm = {om_diff:.1f}σ")

            if h0_diff < 2 and om_diff < 2:
                print(f"  ✅ 在2σ内一致")
            else:
                print(f"  ⚠️  超过2σ差异")
            print()

    except FileNotFoundError:
        print("⚠️  需要先运行Chi2网格分析")

def test_specific_points():
    """测试特定的宇宙学参数点"""
    print("=== 特定参数点测试 ===")

    data_path = Path("data/DataRelease")

    # 测试点
    test_points = [
        {"name": "Planck2018", "H0": 67.4, "Om": 0.315},
        {"name": "SH0ES-like", "H0": 73.0, "Om": 0.3},
        {"name": "Conservative", "H0": 70.0, "Om": 0.3},
    ]

    for include_shoes in [False, True]:
        mode_name = "PantheonPlusSH0ES" if include_shoes else "PantheonPlus"
        print(f"\n--- {mode_name} ---")

        likelihood = PantheonPlusLikelihood(
            data_path=str(data_path),
            include_shoes=include_shoes,
            z_min=0.01,
            marginalize_M_B=True
        )

        for point in test_points:
            model = LCDM(H0=point["H0"], Omega_m=point["Om"])
            log_like = likelihood.log_likelihood(model)
            chi2 = -2 * log_like
            best_MB = likelihood.best_fit_M_B(model)

            print(f"  {point['name']:12s}: χ² = {chi2:7.1f}, χ²/dof = {chi2/(likelihood.n_sne-3):.3f}, M_B = {best_MB:.3f}")

if __name__ == "__main__":
    print("PantheonPlus约束对比分析")
    print("=" * 50)

    # 1. 创建Chi2网格
    create_chi2_grid()

    # 2. 对比文献
    compare_with_literature()

    # 3. 测试特定点
    test_specific_points()

    print("\n" + "=" * 50)
    print("分析完成！请检查生成的contour图和数值对比。")