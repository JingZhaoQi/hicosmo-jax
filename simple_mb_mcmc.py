#!/usr/bin/env python3
"""
简化的M_B自由参数MCMC测试
"""
import os
import numpy as np

# 初始化多核配置
from hicosmo.samplers import init_hicosmo
init_hicosmo(cpu_cores=8, verbose=False)

from hicosmo.samplers import MCMC, ParameterConfig, AutoParameter
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood

def main():
    print("=== 简化M_B自由参数MCMC测试 ===")

    # 数据路径
    data_path = os.path.join(os.path.dirname(__file__), "data", "DataRelease")

    # 初始化likelihood（M_B作为自由参数）
    likelihood_obj = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=False,
        z_min=0.01,
        apply_z_cut=True,
        include_systematics=False,
        marginalize_M_B=False  # 不边际化M_B
    )

    print(f"数据点数量: {likelihood_obj.n_sne}")

    # 定义参数
    H0_param = AutoParameter(
        name='H0',
        prior={'dist': 'uniform', 'min': 65.0, 'max': 80.0},
        ref=73.0,
        latex=r'H_0',
        description='Hubble constant [km/s/Mpc]'
    )

    Omega_m_param = AutoParameter(
        name='Omega_m',
        prior={'dist': 'uniform', 'min': 0.25, 'max': 0.40},
        ref=0.33,
        latex=r'\Omega_m',
        description='Matter density parameter'
    )

    M_B_param = AutoParameter(
        name='M_B',
        prior={'dist': 'uniform', 'min': -19.5, 'max': -19.0},
        ref=-19.25,
        latex=r'M_B',
        description='Absolute magnitude of SNe Ia'
    )

    # MCMC配置
    param_config = ParameterConfig(
        parameters={
            'H0': H0_param,
            'Omega_m': Omega_m_param,
            'M_B': M_B_param
        },
        mcmc={
            'num_samples': 1000,    # 更少样本数
            'num_chains': 4,        # 4条链
            'chain_method': 'vectorized'
        }
    )

    # 似然函数
    def likelihood_func(H0, Omega_m, M_B):
        """PantheonPlus似然函数（M_B作为自由参数）"""
        try:
            model = LCDM(H0=H0, Omega_m=Omega_m)
            return likelihood_obj.log_likelihood(model, M_B=M_B)
        except Exception as e:
            print(f"Error: {e}")
            return -np.inf

    # 运行MCMC
    mcmc = MCMC(config=param_config, likelihood_func=likelihood_func)

    print("开始MCMC...")
    samples = mcmc.run()

    # 输出结果
    print("\n🎯 MCMC结果:")
    print(f"H₀  = {np.mean(samples['H0']):.1f} ± {np.std(samples['H0']):.1f} km/s/Mpc")
    print(f"Ωₘ  = {np.mean(samples['Omega_m']):.3f} ± {np.std(samples['Omega_m']):.3f}")
    print(f"M_B = {np.mean(samples['M_B']):.3f} ± {np.std(samples['M_B']):.3f} mag")

    # 对比官方结果
    print("\n对比:")
    print("官方边际化M_B: H₀ = 73.4 ± 1.4, Ωₘ = 0.334 ± 0.018")
    print(f"我们自由M_B:   H₀ = {np.mean(samples['H0']):.1f} ± {np.std(samples['H0']):.1f}, Ωₘ = {np.mean(samples['Omega_m']):.3f} ± {np.std(samples['Omega_m']):.3f}")

    h0_diff = abs(np.mean(samples['H0']) - 73.4) / 1.4
    om_diff = abs(np.mean(samples['Omega_m']) - 0.334) / 0.018

    print(f"差异: ΔH₀ = {h0_diff:.1f}σ, ΔΩₘ = {om_diff:.1f}σ")

    if h0_diff < 2 and om_diff < 2:
        print("✅ 在2σ内与官方结果一致")
    else:
        print("⚠️ 超过2σ差异")

    print(f"\n最优M_B: {np.mean(samples['M_B']):.3f} ± {np.std(samples['M_B']):.3f}")

if __name__ == "__main__":
    main()