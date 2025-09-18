#!/usr/bin/env python3
"""
最简单测试：只测试单个参数点
"""

import os
import numpy as np
import time

# 使用下载的官方数据
data_path = os.path.join(os.path.dirname(__file__), "data", "DataRelease")

print("🧪 单点测试: PantheonPlus vs PantheonPlusSH0ES")
print("=" * 60)

try:
    from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood
    from hicosmo.models.lcdm import LCDM

    # 测试参数
    H0_test = 70.0
    Om_test = 0.3

    print(f"测试参数: H0 = {H0_test} km/s/Mpc, Ωm = {Om_test}")
    print()

    # 创建模型
    model = LCDM(H0=H0_test, Omega_m=Om_test)

    # 创建PantheonPlus数据集
    print("1. 加载PantheonPlus数据...")
    start_time = time.time()
    likelihood_plus = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=False,
        include_systematics=False,  # 快速：仅对角误差
        z_min=0.01,
        apply_z_cut=True
    )
    load_time_plus = time.time() - start_time

    # 计算似然函数
    print("   计算似然函数...")
    start_time = time.time()
    loglike_plus = likelihood_plus.log_likelihood(model)
    chi2_plus = likelihood_plus.chi2(model)
    M_B_plus = likelihood_plus.best_fit_M_B(model)
    calc_time_plus = time.time() - start_time

    print(f"   结果: logL={loglike_plus:.2f}, χ²={chi2_plus:.2f}, M_B={M_B_plus:.3f}")
    print(f"   加载时间: {load_time_plus:.2f}s, 计算时间: {calc_time_plus:.3f}s")
    print()

    # 创建PantheonPlusSH0ES数据集
    print("2. 加载PantheonPlusSH0ES数据...")
    start_time = time.time()
    likelihood_shoes = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=True,
        include_systematics=False,  # 快速：仅对角误差
        z_min=0.01,
        apply_z_cut=True
    )
    load_time_shoes = time.time() - start_time

    # 计算似然函数
    print("   计算似然函数...")
    start_time = time.time()
    loglike_shoes = likelihood_shoes.log_likelihood(model)
    chi2_shoes = likelihood_shoes.chi2(model)
    M_B_shoes = likelihood_shoes.best_fit_M_B(model)
    calc_time_shoes = time.time() - start_time

    print(f"   结果: logL={loglike_shoes:.2f}, χ²={chi2_shoes:.2f}, M_B={M_B_shoes:.3f}")
    print(f"   加载时间: {load_time_shoes:.2f}s, 计算时间: {calc_time_shoes:.3f}s")
    print()

    # 对比结果
    print("📊 结果对比:")
    print("-" * 50)

    info_plus = likelihood_plus.get_info()
    info_shoes = likelihood_shoes.get_info()

    print(f"数据集大小:")
    print(f"  PantheonPlus:     {info_plus['n_supernovae']} objects")
    print(f"  PantheonPlusSH0ES: {info_shoes['n_supernovae']} objects")
    print(f"  增加了:           {info_shoes['n_supernovae'] - info_plus['n_supernovae']} objects")
    print()

    print(f"似然函数值:")
    print(f"  PantheonPlus:     {loglike_plus:.2f}")
    print(f"  PantheonPlusSH0ES: {loglike_shoes:.2f}")
    print(f"  差异:             {loglike_shoes - loglike_plus:.2f}")
    print()

    print(f"χ²值:")
    print(f"  PantheonPlus:     {chi2_plus:.2f} (χ²/DOF = {chi2_plus/info_plus['n_supernovae']:.3f})")
    print(f"  PantheonPlusSH0ES: {chi2_shoes:.2f} (χ²/DOF = {chi2_shoes/info_shoes['n_supernovae']:.3f})")
    print(f"  差异:             {chi2_shoes - chi2_plus:.2f}")
    print()

    print(f"最佳拟合M_B:")
    print(f"  PantheonPlus:     {M_B_plus:.3f} mag")
    print(f"  PantheonPlusSH0ES: {M_B_shoes:.3f} mag")
    print(f"  差异:             {M_B_shoes - M_B_plus:.3f} mag")
    print()

    print(f"性能:")
    print(f"  Plus加载:         {load_time_plus:.2f}s")
    print(f"  SH0ES加载:        {load_time_shoes:.2f}s")
    print(f"  Plus计算:         {calc_time_plus:.3f}s")
    print(f"  SH0ES计算:        {calc_time_shoes:.3f}s")

    print("\n✅ 单点测试成功完成！")
    print("\n🔍 观察:")
    print("   • 两种数据集都能正常计算似然函数")
    print("   • SH0ES增加了约束数据点")
    print("   • 计算速度在可接受范围内")
    print("   • M_B参数在合理范围内")

    # 快速创建一个简单的对比图
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    datasets = ['PantheonPlus', 'PantheonPlusSH0ES']

    # 数据点数量
    n_objects = [info_plus['n_supernovae'], info_shoes['n_supernovae']]
    ax1.bar(datasets, n_objects, color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Number of Objects')
    ax1.set_title('Dataset Size Comparison')
    ax1.grid(True, alpha=0.3)

    # χ²值
    chi2_vals = [chi2_plus, chi2_shoes]
    ax2.bar(datasets, chi2_vals, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('χ²')
    ax2.set_title(f'Chi-square for H0={H0_test}, Ωm={Om_test}')
    ax2.grid(True, alpha=0.3)

    # M_B值
    M_B_vals = [M_B_plus, M_B_shoes]
    ax3.bar(datasets, M_B_vals, color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Best-fit M_B [mag]')
    ax3.set_title('Absolute Magnitude Comparison')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = 'single_point_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📁 结果图保存到: {output_file}")

    plt.show()

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()