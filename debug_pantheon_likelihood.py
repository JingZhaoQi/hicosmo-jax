#!/usr/bin/env python3
"""
PantheonPlus Likelihood诊断脚本
对比我们的实现与官方预期，找出差异
"""

import numpy as np
import pandas as pd
from pathlib import Path

# 首先初始化多核配置
from hicosmo.samplers import init_hicosmo
init_hicosmo(cpu_cores=8, verbose=False)

from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood

def compare_with_astropy():
    """对比我们的距离计算与astropy"""
    print("=== 1. 距离计算对比测试 ===")

    try:
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u

        # 测试参数
        H0_test = 70.0
        Om0_test = 0.3

        # 我们的模型
        our_model = LCDM(H0=H0_test, Omega_m=Om0_test)

        # Astropy模型
        astropy_cosmo = FlatLambdaCDM(H0=H0_test*u.km/u.s/u.Mpc, Om0=Om0_test)

        # 测试红移点
        z_test = np.array([0.01, 0.1, 0.5, 1.0, 1.5])

        print(f"测试参数: H0={H0_test}, Ωm={Om0_test}")
        print(f"红移点: {z_test}")
        print()

        for z in z_test:
            # 我们的计算
            our_dA = float(our_model.angular_diameter_distance(z))
            our_dL = float(our_model.luminosity_distance(z))
            our_mu = float(our_model.distance_modulus(z))

            # Astropy计算
            astropy_dA = float(astropy_cosmo.angular_diameter_distance(z).to(u.Mpc).value)
            astropy_dL = float(astropy_cosmo.luminosity_distance(z).to(u.Mpc).value)
            astropy_mu = float(astropy_cosmo.distmod(z).value)

            # 计算差异
            dA_diff = abs(our_dA - astropy_dA) / astropy_dA * 100
            dL_diff = abs(our_dL - astropy_dL) / astropy_dL * 100
            mu_diff = abs(our_mu - astropy_mu) / abs(astropy_mu) * 100

            print(f"z={z:.2f}:")
            print(f"  角直径距离: 我们={our_dA:.3f} vs Astropy={astropy_dA:.3f} (差异={dA_diff:.2f}%)")
            print(f"  光度距离:   我们={our_dL:.3f} vs Astropy={astropy_dL:.3f} (差异={dL_diff:.2f}%)")
            print(f"  距离模数:   我们={our_mu:.3f} vs Astropy={astropy_mu:.3f} (差异={mu_diff:.2f}%)")

            # 检查是否有显著差异
            if dA_diff > 0.1 or dL_diff > 0.1 or mu_diff > 0.1:
                print(f"  ⚠️  发现显著差异！")
            else:
                print(f"  ✅ 一致")
            print()

        return True

    except ImportError:
        print("⚠️  未安装astropy，跳过对比测试")
        return False

def diagnose_pantheon_likelihood():
    """诊断PantheonPlus likelihood实现"""
    print("=== 2. PantheonPlus Likelihood 诊断 ===")

    data_path = Path("data/DataRelease")

    # 测试两种模式
    for include_shoes in [False, True]:
        mode_name = "PantheonPlusSH0ES" if include_shoes else "PantheonPlus"
        print(f"\n--- {mode_name} 模式 ---")

        # 创建likelihood
        likelihood = PantheonPlusLikelihood(
            data_path=str(data_path),
            include_shoes=include_shoes,
            z_min=0.01,
            apply_z_cut=True,
            include_systematics=False,
            marginalize_M_B=False  # 用固定M_B测试
        )

        print(f"数据点数量: {likelihood.n_sne}")
        print(f"红移范围: {likelihood.redshifts.min():.4f} - {likelihood.redshifts.max():.4f}")

        # 测试标准宇宙学参数
        test_params = [
            {"H0": 70.0, "Omega_m": 0.3, "M_B": -19.3},
            {"H0": 67.4, "Omega_m": 0.315, "M_B": -19.3},  # Planck 2018
            {"H0": 73.0, "Omega_m": 0.3, "M_B": -19.3},    # SH0ES-like
        ]

        for i, params in enumerate(test_params):
            print(f"\n测试参数组 {i+1}: {params}")

            model = LCDM(H0=params["H0"], Omega_m=params["Omega_m"])

            # 计算chi2和log-likelihood
            chi2 = likelihood.chi2(model, params["M_B"])
            log_like = likelihood.log_likelihood(model, params["M_B"])

            # 计算最优M_B
            best_MB = likelihood.best_fit_M_B(model)

            print(f"  χ² = {chi2:.2f}")
            print(f"  χ²/dof = {chi2/(likelihood.n_sne-3):.3f}")
            print(f"  log-likelihood = {log_like:.2f}")
            print(f"  最优M_B = {best_MB:.3f}")

            # 计算一些基本统计
            if include_shoes:
                n_calibrators = np.sum(likelihood.is_calibrator)
                n_hubble = likelihood.n_sne - n_calibrators
                print(f"  Calibrators: {n_calibrators}, Hubble flow: {n_hubble}")

def check_data_consistency():
    """检查数据一致性"""
    print("\n=== 3. 数据一致性检查 ===")

    data_path = Path("data/DataRelease")
    data_file = data_path / "Pantheon+_Data" / "4_DISTANCES_AND_COVAR" / "Pantheon+SH0ES.dat"

    # 读取原始数据
    data = pd.read_csv(data_file, comment='#', sep=r'\s+')

    print(f"原始数据总数: {len(data)}")
    print(f"Calibrators数量: {np.sum(data['IS_CALIBRATOR'])}")
    print(f"z_HD > 0.01的数量: {np.sum(data['zHD'] > 0.01)}")
    print(f"z_HD > 0.01 且非calibrator: {np.sum((data['zHD'] > 0.01) & (~data['IS_CALIBRATOR']))}")
    print(f"z_HD > 0.01 或 calibrator: {np.sum((data['zHD'] > 0.01) | data['IS_CALIBRATOR'])}")

    # 检查数据范围
    print(f"\nzCMB范围: {data['zCMB'].min():.6f} - {data['zCMB'].max():.6f}")
    print(f"zHEL范围: {data['zHEL'].min():.6f} - {data['zHEL'].max():.6f}")
    print(f"zHD范围: {data['zHD'].min():.6f} - {data['zHD'].max():.6f}")
    print(f"m_b_corr范围: {data['m_b_corr'].min():.3f} - {data['m_b_corr'].max():.3f}")

    # 检查协方差矩阵
    cov_file = data_path / "Pantheon+_Data" / "4_DISTANCES_AND_COVAR" / "Pantheon+SH0ES_STAT+SYS.cov"
    if cov_file.exists():
        # 正确读取协方差矩阵（第一行是dimension）
        with open(cov_file, 'r') as f:
            first_line = f.readline().strip()
            n_dim = int(first_line)
            cov_data = np.loadtxt(f)

        print(f"\n协方差矩阵维度: {n_dim}")
        print(f"协方差数据大小: {cov_data.size}")
        print(f"预期大小: {n_dim * n_dim}")

        if cov_data.size == n_dim * n_dim:
            cov_matrix = cov_data.reshape(n_dim, n_dim)
            print(f"协方差矩阵条件数: {np.linalg.cond(cov_matrix):.2e}")
            print(f"对角线元素范围: {np.diag(cov_matrix).min():.6f} - {np.diag(cov_matrix).max():.6f}")
        else:
            print(f"⚠️  协方差矩阵数据大小不匹配")
            print(f"   数据大小: {cov_data.size}")
            print(f"   预期大小: {n_dim * n_dim}")
            print(f"   差异: {cov_data.size - n_dim * n_dim}")

def compare_distance_modulus_calculation():
    """对比距离模数计算方法"""
    print("\n=== 4. 距离模数计算对比 ===")

    # 测试参数
    H0, Om0 = 70.0, 0.3
    model = LCDM(H0=H0, Omega_m=Om0)

    # 测试红移和redshift corrections
    z_cmb = 0.1
    z_hel = 0.098

    # 我们的计算方法
    d_A = model.angular_diameter_distance(z_cmb)
    mu_our = 5.0 * np.log10((1.0 + z_cmb) * (1.0 + z_hel) * d_A) + 25.0

    # 标准距离模数（无redshift corrections）
    d_L = model.luminosity_distance(z_cmb)
    mu_standard = 5.0 * np.log10(d_L) + 25.0

    # 光度距离关系验证
    d_L_from_dA = d_A * (1 + z_cmb)**2

    print(f"测试: z_CMB={z_cmb}, z_HEL={z_hel}")
    print(f"角直径距离 d_A = {d_A:.3f} Mpc")
    print(f"光度距离 d_L = {d_L:.3f} Mpc")
    print(f"从d_A计算的d_L = {d_L_from_dA:.3f} Mpc (差异: {abs(d_L-d_L_from_dA)/d_L*100:.3f}%)")
    print()
    print(f"我们的距离模数 = {mu_our:.4f}")
    print(f"标准距离模数 = {mu_standard:.4f}")
    print(f"差异 = {mu_our - mu_standard:.4f}")
    print()

    # redshift correction的影响
    correction_factor = (1.0 + z_cmb) * (1.0 + z_hel)
    print(f"Redshift correction factor = {correction_factor:.6f}")
    print(f"Correction in magnitude = {5*np.log10(correction_factor):.4f}")

if __name__ == "__main__":
    print("PantheonPlus Likelihood 诊断开始")
    print("=" * 60)

    # 1. 对比距离计算
    astropy_available = compare_with_astropy()

    # 2. 诊断likelihood
    diagnose_pantheon_likelihood()

    # 3. 检查数据一致性
    check_data_consistency()

    # 4. 对比距离模数计算
    compare_distance_modulus_calculation()

    print("\n" + "=" * 60)
    print("诊断完成！请检查上述输出寻找可能的问题。")