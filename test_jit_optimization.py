#!/usr/bin/env python3
"""
测试JAX JIT优化效果
"""
import time
import numpy as np
from pathlib import Path

# 初始化多核配置
from hicosmo.samplers import init_hicosmo
init_hicosmo(cpu_cores=8, verbose=False)

from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood

def benchmark_likelihood_performance():
    """对比优化前后的性能"""
    print("=== JAX JIT优化效果测试 ===")

    data_path = Path("data/DataRelease")

    # 测试M_B作为自由参数
    likelihood = PantheonPlusLikelihood(
        data_path=str(data_path),
        include_shoes=False,
        z_min=0.01,
        marginalize_M_B=False  # M_B作为自由参数
    )

    print(f"数据点数量: {likelihood.n_sne}")

    # 测试参数
    model = LCDM(H0=73.4, Omega_m=0.334)
    M_B_values = [-19.3, -19.25, -19.2, -19.15, -19.1]

    print("\n🚀 第一次调用 (预期有JIT编译时间):")
    start_time = time.time()
    first_result = likelihood.log_likelihood(model, M_B=M_B_values[0])
    first_time = time.time() - start_time
    print(f"第一次: M_B={M_B_values[0]:.2f}, log_like={first_result:.1f}, 时间={first_time:.3f}s")

    print("\n⚡ 后续调用 (应该很快，无重复编译):")
    total_time = 0
    results = []

    for i, M_B in enumerate(M_B_values[1:], 1):
        start_time = time.time()
        result = likelihood.log_likelihood(model, M_B=M_B)
        call_time = time.time() - start_time
        total_time += call_time
        results.append((M_B, result, call_time))
        print(f"第{i+1}次: M_B={M_B:.2f}, log_like={result:.1f}, 时间={call_time:.3f}s")

    avg_time = total_time / len(M_B_values[1:])
    print(f"\n📊 性能统计:")
    print(f"首次调用时间: {first_time:.3f}s (包含JIT编译)")
    print(f"后续平均时间: {avg_time:.3f}s")
    print(f"加速比: {first_time/avg_time:.1f}x")

    # 测试边际化M_B的性能对比
    print("\n🔄 对比边际化M_B性能:")
    likelihood_marg = PantheonPlusLikelihood(
        data_path=str(data_path),
        include_shoes=False,
        z_min=0.01,
        marginalize_M_B=True  # 边际化M_B
    )

    start_time = time.time()
    marg_result = likelihood_marg.log_likelihood(model)
    marg_time = time.time() - start_time
    print(f"边际化M_B: log_like={marg_result:.1f}, 时间={marg_time:.3f}s")

    print(f"\n✅ 性能对比:")
    print(f"自由M_B (优化后): {avg_time:.3f}s")
    print(f"边际化M_B:        {marg_time:.3f}s")
    print(f"性能差异:         {avg_time/marg_time:.1f}x")

    if avg_time / marg_time < 5:
        print("🎉 优化成功！M_B自由参数性能已接近边际化水平")
    else:
        print("⚠️ 还需进一步优化")

if __name__ == "__main__":
    benchmark_likelihood_performance()