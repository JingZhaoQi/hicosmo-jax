#!/usr/bin/env python3
"""
测试专业绘图风格 - 基于analysis/core.py的实现

这个测试将展示：
1. 三种配色方案 (modern, sophisticated, classic)
2. 智能刻度优化防止重叠
3. 专业LaTeX标签处理
4. 单色专业风格contour
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 使用新的MCMCPlotter
from hicosmo.visualization.plotting.mcmc import MCMCPlotter
from hicosmo.visualization.chain_manager import ChainData

def create_test_chain():
    """创建测试链数据"""
    np.random.seed(42)
    n_samples = 2000

    # 创建相关的宇宙学参数
    H0 = np.random.normal(70.0, 5.0, n_samples)
    Omega_m = np.random.normal(0.3, 0.05, n_samples)
    sigma8 = np.random.normal(0.8, 0.05, n_samples)

    # 添加一些相关性
    H0 = H0 + 0.3 * (Omega_m - 0.3)
    sigma8 = sigma8 - 0.2 * (Omega_m - 0.3)

    # 组装为正确格式
    samples = np.column_stack([H0, Omega_m, sigma8])
    params = ['H0', 'Omega_m', 'sigma8']

    return ChainData(samples, params, label="Test Chain")

def test_color_schemes():
    """测试三种配色方案"""
    print("🎨 测试专业配色方案...")

    # 创建测试链
    chain = create_test_chain()

    # 测试三种配色方案
    schemes = ['modern', 'sophisticated', 'classic']

    for scheme in schemes:
        print(f"  → 测试 {scheme} 配色方案...")

        try:
            plotter = MCMCPlotter(color_scheme=scheme)
            fig = plotter.corner(chain, params=['H0', 'Omega_m', 'sigma8'])

            # 保存图片
            filename = f'professional_{scheme}_corner.pdf'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"    ✅ {scheme} 配色方案成功，保存为 {filename}")

        except Exception as e:
            print(f"    ❌ {scheme} 配色方案失败: {e}")

def test_latex_labels():
    """测试LaTeX标签处理"""
    print("📐 测试LaTeX标签处理...")

    chain = create_test_chain()

    try:
        plotter = MCMCPlotter(color_scheme='modern')

        # 测试H0参数的自动单位添加
        fig = plotter.corner(chain, params=['H0', 'Omega_m'])

        filename = 'professional_latex_test.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ LaTeX标签处理成功，H0应自动添加单位，保存为 {filename}")

    except Exception as e:
        print(f"  ❌ LaTeX标签处理失败: {e}")

def test_tick_optimization():
    """测试智能刻度优化"""
    print("📊 测试智能刻度优化...")

    # 创建极端值测试刻度优化
    np.random.seed(123)
    n_samples = 1000

    # 测试不同数量级的参数
    large_param = np.random.normal(1000, 100, n_samples)      # 大数值
    small_param = np.random.normal(0.001, 0.0001, n_samples) # 小数值
    normal_param = np.random.normal(1.0, 0.1, n_samples)     # 正常数值

    samples = np.column_stack([large_param, small_param, normal_param])
    params = ['large_param', 'small_param', 'normal_param']
    chain = ChainData(samples, params, label="Tick Test")

    try:
        plotter = MCMCPlotter(color_scheme='sophisticated')
        fig = plotter.corner(chain, params=['large_param', 'small_param', 'normal_param'])

        filename = 'professional_tick_optimization.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ 刻度优化成功，应该自动格式化不同数量级，保存为 {filename}")

    except Exception as e:
        print(f"  ❌ 刻度优化失败: {e}")

def test_style_comparison():
    """对比新旧风格"""
    print("🔄 对比新旧绘图风格...")

    chain = create_test_chain()

    # 新风格 (基于analysis/core.py)
    try:
        plotter_new = MCMCPlotter(color_scheme='modern')
        fig = plotter_new.corner(chain, params=['H0', 'Omega_m'])

        filename = 'professional_NEW_STYLE.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ 新风格成功: {filename}")
        print("     - 使用单色专业contour")
        print("     - 智能刻度优化防重叠")
        print("     - 专业配色方案")
        print("     - H0自动添加单位")

    except Exception as e:
        print(f"  ❌ 新风格失败: {e}")

if __name__ == "__main__":
    print("🚀 开始测试专业绘图风格 (基于analysis/core.py)")
    print("=" * 60)

    # 确保输出目录存在
    Path('.').mkdir(exist_ok=True)

    # 运行所有测试
    test_color_schemes()
    print()
    test_latex_labels()
    print()
    test_tick_optimization()
    print()
    test_style_comparison()

    print("\n" + "=" * 60)
    print("🎉 测试完成！检查生成的PDF文件查看效果")
    print("主要改进:")
    print("  • 三种专业配色方案 (modern/sophisticated/classic)")
    print("  • 智能刻度优化防止标签重叠")
    print("  • 专业LaTeX标签处理 (H0自动添加单位)")
    print("  • 单色专业contour风格")
    print("  • 基于analysis/core.py的成熟实现")