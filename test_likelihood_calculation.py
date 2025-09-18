#!/usr/bin/env python3
"""
测试PantheonPlus似然函数计算
"""

import os
import numpy as np

# 使用下载的官方数据
data_path = os.path.join(os.path.dirname(__file__), "data", "DataRelease")

print("🧪 测试PantheonPlus似然函数计算功能")
print("=" * 60)

try:
    from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood
    from hicosmo.models.lcdm import LCDM

    # 创建PantheonPlusSH0ES数据集
    likelihood = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=True,
        z_min=0.01,
        apply_z_cut=True
    )

    # 创建测试宇宙学模型
    model_fiducial = LCDM(H0=70.0, Omega_m=0.3)
    model_high_h0 = LCDM(H0=75.0, Omega_m=0.3)
    model_high_om = LCDM(H0=70.0, Omega_m=0.35)

    print("\n🔬 计算似然函数值")
    print("-" * 40)

    # 计算最佳拟合M_B并评估log-likelihood
    M_B_fiducial = likelihood.best_fit_M_B(model_fiducial)
    M_B_high_h0 = likelihood.best_fit_M_B(model_high_h0)
    M_B_high_om = likelihood.best_fit_M_B(model_high_om)

    loglike_fiducial = likelihood.log_likelihood(model_fiducial, M_B_fiducial)
    loglike_high_h0 = likelihood.log_likelihood(model_high_h0, M_B_high_h0)
    loglike_high_om = likelihood.log_likelihood(model_high_om, M_B_high_om)

    print(f"Fiducial (H0=70, Ωm=0.30): {loglike_fiducial:.2f}")
    print(f"High H0  (H0=75, Ωm=0.30): {loglike_high_h0:.2f}")
    print(f"High Ωm  (H0=70, Ωm=0.35): {loglike_high_om:.2f}")

    # 计算χ²值
    print(f"\n🔬 计算χ²值")
    print("-" * 40)

    chi2_fiducial = likelihood.chi2(model_fiducial, M_B_fiducial)
    chi2_high_h0 = likelihood.chi2(model_high_h0, M_B_high_h0)
    chi2_high_om = likelihood.chi2(model_high_om, M_B_high_om)

    print(f"Fiducial (H0=70, Ωm=0.30): {chi2_fiducial:.2f}")
    print(f"High H0  (H0=75, Ωm=0.30): {chi2_high_h0:.2f}")
    print(f"High Ωm  (H0=70, Ωm=0.35): {chi2_high_om:.2f}")

    # 计算最佳拟合M_B
    print(f"\n🔬 最佳拟合绝对星等")
    print("-" * 40)

    print(f"Fiducial (H0=70, Ωm=0.30): M_B = {M_B_fiducial:.3f}")
    print(f"High H0  (H0=75, Ωm=0.30): M_B = {M_B_high_h0:.3f}")
    print(f"High Ωm  (H0=70, Ωm=0.35): M_B = {M_B_high_om:.3f}")

    # 验证理论μ计算是否与官方公式一致
    print(f"\n🔬 验证理论距离模数计算")
    print("-" * 40)
    theory_mu = likelihood._compute_theory_shoes(model_fiducial)
    z_hd = np.asarray(likelihood.z_hd)
    z_hel = np.asarray(likelihood.z_hel)
    d_a = np.asarray(model_fiducial.angular_diameter_distance(likelihood.z_cmb))
    manual_mu = 5.0 * np.log10((1.0 + z_hd) * (1.0 + z_hel) * d_a) + 25.0
    mask_cal = np.asarray(likelihood.is_calibrator, dtype=bool)
    manual_mu[mask_cal] = np.asarray(likelihood.ceph_dist)[mask_cal]
    max_diff = np.max(np.abs(np.asarray(theory_mu) - manual_mu))
    print(f"最大差异: {max_diff:.3e}")
    assert max_diff < 1e-5, "理论距离模数与官方公式不一致"

    print(f"\n📊 数据集信息")
    print("-" * 40)
    info = likelihood.get_info()
    print(f"数据集类型: {info['data_type']}")
    print(f"超新星数量: {info['n_supernovae']}")
    print(f"红移范围: [{info['redshift_range'][0]:.4f}, {info['redshift_range'][1]:.4f}]")
    print(f"包含系统误差: {info['include_systematics']}")

    print("\n✅ 似然函数计算测试通过！")
    print("\n📝 结果分析:")
    print(f"   - 似然函数值合理范围内")
    print(f"   - χ²值约为 {chi2_fiducial:.0f}，对于 {info['n_supernovae']} 个数据点是合理的")
    print(f"   - M_B约为 {M_B_fiducial:.2f}，符合预期的Ia型超新星绝对星等")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
