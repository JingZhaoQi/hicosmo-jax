#!/usr/bin/env python3
"""
对比优化前后的MCMC速度
"""
import time
import os

from hicosmo.samplers import init_hicosmo
init_hicosmo(cpu_cores=4, verbose=False)

from hicosmo.samplers import MCMC, ParameterConfig, AutoParameter
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood

def benchmark_mcmc_speed():
    """测试MCMC速度提升"""
    print("=== MCMC速度对比测试 ===")

    data_path = os.path.join(os.path.dirname(__file__), "data", "DataRelease")

    # 参数配置
    H0_param = AutoParameter(
        name='H0',
        prior={'dist': 'uniform', 'min': 70.0, 'max': 76.0},
        ref=73.0,
        latex=r'H_0'
    )

    Omega_m_param = AutoParameter(
        name='Omega_m',
        prior={'dist': 'uniform', 'min': 0.30, 'max': 0.36},
        ref=0.33,
        latex=r'\Omega_m'
    )

    M_B_param = AutoParameter(
        name='M_B',
        prior={'dist': 'uniform', 'min': -19.3, 'max': -19.15},
        ref=-19.25,
        latex=r'M_B'
    )

    # 测试自由M_B (优化后)
    print("\n🚀 测试M_B自由参数 (JAX优化后):")
    likelihood_free = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=False,
        z_min=0.01,
        marginalize_M_B=False
    )

    param_config_free = ParameterConfig(
        parameters={'H0': H0_param, 'Omega_m': Omega_m_param, 'M_B': M_B_param},
        mcmc={'num_samples': 500, 'num_chains': 2, 'chain_method': 'vectorized'}
    )

    def likelihood_free_func(H0, Omega_m, M_B):
        try:
            model = LCDM(H0=H0, Omega_m=Omega_m)
            return likelihood_free.log_likelihood(model, M_B=M_B)
        except:
            return -1e10

    mcmc_free = MCMC(config=param_config_free, likelihood_func=likelihood_free_func)

    start_time = time.time()
    samples_free = mcmc_free.run()
    time_free = time.time() - start_time

    print(f"⏱️ M_B自由参数时间: {time_free:.1f}s")
    print(f"📊 结果: H0={samples_free['H0'].mean():.1f}±{samples_free['H0'].std():.1f}, Ωm={samples_free['Omega_m'].mean():.3f}±{samples_free['Omega_m'].std():.3f}")

    # 测试边际化M_B
    print("\n🔄 测试边际化M_B:")
    likelihood_marg = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=False,
        z_min=0.01,
        marginalize_M_B=True
    )

    param_config_marg = ParameterConfig(
        parameters={'H0': H0_param, 'Omega_m': Omega_m_param},
        mcmc={'num_samples': 500, 'num_chains': 2, 'chain_method': 'vectorized'}
    )

    def likelihood_marg_func(H0, Omega_m):
        try:
            model = LCDM(H0=H0, Omega_m=Omega_m)
            return likelihood_marg.log_likelihood(model)
        except:
            return -1e10

    mcmc_marg = MCMC(config=param_config_marg, likelihood_func=likelihood_marg_func)

    start_time = time.time()
    samples_marg = mcmc_marg.run()
    time_marg = time.time() - start_time

    print(f"⏱️ 边际化M_B时间: {time_marg:.1f}s")
    print(f"📊 结果: H0={samples_marg['H0'].mean():.1f}±{samples_marg['H0'].std():.1f}, Ωm={samples_marg['Omega_m'].mean():.3f}±{samples_marg['Omega_m'].std():.3f}")

    # 性能对比
    print(f"\n✅ 性能对比:")
    print(f"M_B自由参数: {time_free:.1f}s")
    print(f"边际化M_B:  {time_marg:.1f}s")
    ratio = time_free / time_marg
    print(f"速度比值:    {ratio:.1f}x")

    if ratio < 2.0:
        print("🎉 优化成功！M_B自由参数性能已接近边际化水平")
    elif ratio < 5.0:
        print("✅ 显著改进，但还有优化空间")
    else:
        print("⚠️ 仍需进一步优化")

if __name__ == "__main__":
    benchmark_mcmc_speed()