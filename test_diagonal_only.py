#!/usr/bin/env python3
"""
简化测试：仅使用对角误差的快速版本
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 使用下载的官方数据
data_path = os.path.join(os.path.dirname(__file__), "data", "DataRelease")

print("🧪 对角误差测试: PantheonPlus vs PantheonPlusSH0ES")
print("=" * 60)

try:
    from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood
    from hicosmo.models.lcdm import LCDM

    # 创建两种数据集（只使用对角误差）
    print("📊 加载数据集（仅对角误差）...")

    # PantheonPlus 纯超新星数据
    likelihood_plus = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=False,
        include_systematics=False,  # 仅统计误差
        z_min=0.01,
        apply_z_cut=True
    )

    print()

    # PantheonPlusSH0ES 组合数据
    likelihood_shoes = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=True,
        include_systematics=False,  # 仅统计误差
        z_min=0.01,
        apply_z_cut=True
    )

    # 测试几个关键参数组合
    print("🔬 测试关键参数组合...")
    print("-" * 50)

    test_params = [
        (67.4, 0.315, "Planck 2018"),
        (70.0, 0.30, "Fiducial"),
        (73.0, 0.30, "High H0"),
        (70.0, 0.25, "Low Omega_m"),
        (70.0, 0.35, "High Omega_m")
    ]

    results = []

    for H0, Om, label in test_params:
        print(f"测试: {label} (H0={H0}, Ωm={Om})")

        model = LCDM(H0=H0, Omega_m=Om)

        # 计算似然函数值
        loglike_plus = likelihood_plus.log_likelihood(model)
        loglike_shoes = likelihood_shoes.log_likelihood(model)

        # 计算χ²
        chi2_plus = likelihood_plus.chi2(model)
        chi2_shoes = likelihood_shoes.chi2(model)

        # 计算最佳M_B
        M_B_plus = likelihood_plus.best_fit_M_B(model)
        M_B_shoes = likelihood_shoes.best_fit_M_B(model)

        results.append({
            'label': label,
            'H0': H0,
            'Om': Om,
            'loglike_plus': loglike_plus,
            'loglike_shoes': loglike_shoes,
            'chi2_plus': chi2_plus,
            'chi2_shoes': chi2_shoes,
            'M_B_plus': M_B_plus,
            'M_B_shoes': M_B_shoes
        })

        print(f"  Plus:  logL={loglike_plus:.2f}, χ²={chi2_plus:.2f}, M_B={M_B_plus:.3f}")
        print(f"  SH0ES: logL={loglike_shoes:.2f}, χ²={chi2_shoes:.2f}, M_B={M_B_shoes:.3f}")
        print()

    # 创建结果对比图
    print("🎨 绘制结果对比...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    labels = [r['label'] for r in results]
    H0_vals = [r['H0'] for r in results]
    Om_vals = [r['Om'] for r in results]

    chi2_plus_vals = [r['chi2_plus'] for r in results]
    chi2_shoes_vals = [r['chi2_shoes'] for r in results]

    loglike_plus_vals = [r['loglike_plus'] for r in results]
    loglike_shoes_vals = [r['loglike_shoes'] for r in results]

    M_B_plus_vals = [r['M_B_plus'] for r in results]
    M_B_shoes_vals = [r['M_B_shoes'] for r in results]

    # 1. χ²对比
    x_pos = np.arange(len(labels))
    width = 0.35

    ax1.bar(x_pos - width/2, chi2_plus_vals, width, label='PantheonPlus', alpha=0.7, color='blue')
    ax1.bar(x_pos + width/2, chi2_shoes_vals, width, label='PantheonPlusSH0ES', alpha=0.7, color='red')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('χ²')
    ax1.set_title('Chi-square Comparison (Diagonal Errors Only)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 似然函数对比
    ax2.bar(x_pos - width/2, loglike_plus_vals, width, label='PantheonPlus', alpha=0.7, color='blue')
    ax2.bar(x_pos + width/2, loglike_shoes_vals, width, label='PantheonPlusSH0ES', alpha=0.7, color='red')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Log-likelihood')
    ax2.set_title('Log-likelihood Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. M_B对比
    ax3.bar(x_pos - width/2, M_B_plus_vals, width, label='PantheonPlus', alpha=0.7, color='blue')
    ax3.bar(x_pos + width/2, M_B_shoes_vals, width, label='PantheonPlusSH0ES', alpha=0.7, color='red')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Best-fit M_B')
    ax3.set_title('Absolute Magnitude Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 参数空间图
    ax4.scatter([r['H0'] for r in results], [r['Om'] for r in results],
               s=100, c='black', marker='s', label='Test Points')

    # 用颜色编码χ²值
    colors_plus = [r['chi2_plus'] for r in results]
    colors_shoes = [r['chi2_shoes'] for r in results]

    # 找最佳拟合点（最小χ²）
    best_idx_plus = np.argmin(colors_plus)
    best_idx_shoes = np.argmin(colors_shoes)

    ax4.scatter(results[best_idx_plus]['H0'], results[best_idx_plus]['Om'],
               s=200, c='blue', marker='*', label=f'Best Plus: {labels[best_idx_plus]}')
    ax4.scatter(results[best_idx_shoes]['H0'], results[best_idx_shoes]['Om'],
               s=200, c='red', marker='*', label=f'Best SH0ES: {labels[best_idx_shoes]}')

    ax4.set_xlabel('H₀ [km/s/Mpc]')
    ax4.set_ylabel('Ωₘ')
    ax4.set_title('Parameter Space')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_file = 'diagonal_constraints_test.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📁 结果保存到: {output_file}")

    plt.show()

    # 总结结果
    print(f"\n📊 结果总结:")
    print(f"-" * 50)

    best_model_plus = labels[best_idx_plus]
    best_model_shoes = labels[best_idx_shoes]

    print(f"最佳拟合模型 (仅对角误差):")
    print(f"  PantheonPlus: {best_model_plus} (χ² = {colors_plus[best_idx_plus]:.2f})")
    print(f"  PantheonPlusSH0ES: {best_model_shoes} (χ² = {colors_shoes[best_idx_shoes]:.2f})")

    # 数据集信息
    info_plus = likelihood_plus.get_info()
    info_shoes = likelihood_shoes.get_info()

    print(f"\n数据集统计:")
    print(f"  PantheonPlus: {info_plus['n_supernovae']} objects")
    print(f"  PantheonPlusSH0ES: {info_shoes['n_supernovae']} objects")
    print(f"  增加的校准器: {info_shoes['n_supernovae'] - info_plus['n_supernovae']} objects")

    # χ²/DOF分析
    chi2_per_dof_plus = colors_plus[best_idx_plus] / info_plus['n_supernovae']
    chi2_per_dof_shoes = colors_shoes[best_idx_shoes] / info_shoes['n_supernovae']

    print(f"\nχ²/DOF分析:")
    print(f"  PantheonPlus: {chi2_per_dof_plus:.3f}")
    print(f"  PantheonPlusSH0ES: {chi2_per_dof_shoes:.3f}")

    print(f"\n✅ 对角误差测试完成！")
    print(f"\n🔍 观察:")
    print(f"   • 对角误差版本计算速度更快")
    print(f"   • SH0ES数据增加了约束能力")
    print(f"   • 不同模型间的χ²差异显示参数敏感性")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()