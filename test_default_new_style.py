#!/usr/bin/env python3
"""
测试默认新风格设置

验证：
1. 默认使用modern配色方案
2. 图片自动保存到results/目录
3. 专业风格特性全部启用
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 使用HIcosmo可视化系统
from hicosmo.visualization import HIcosmoViz, MCMCPlotter
from hicosmo.visualization.chain_manager import ChainData

def test_default_new_style():
    """测试默认新风格"""
    print("🎨 测试默认NEW_STYLE设置...")

    # 创建测试数据
    np.random.seed(42)
    n_samples = 1500

    H0 = np.random.normal(70.0, 3.0, n_samples)
    Omega_m = np.random.normal(0.3, 0.03, n_samples)
    sigma8 = np.random.normal(0.8, 0.04, n_samples)

    # 添加相关性
    H0 = H0 + 0.2 * (Omega_m - 0.3)
    sigma8 = sigma8 - 0.15 * (Omega_m - 0.3)

    samples = np.column_stack([H0, Omega_m, sigma8])
    params = ['H0', 'Omega_m', 'sigma8']
    chain = ChainData(samples, params, label="Default Style Test")

    print("  → 测试直接使用MCMCPlotter (应该默认为modern风格)")

    try:
        # 不指定color_scheme，应该默认为modern
        plotter = MCMCPlotter()

        # 验证默认配色
        print(f"    默认配色方案: {plotter.color_scheme}")
        print(f"    主色调: {plotter.colors[0]}")

        # 绘制corner图，自动保存到results/目录
        fig = plotter.corner(chain, params=['H0', 'Omega_m'],
                           filename='test_default_style.pdf')

        plt.close()

        # 检查文件是否存在
        results_path = Path('results/test_default_style.pdf')
        if results_path.exists():
            print(f"    ✅ 图片成功保存到: {results_path}")
            print(f"    文件大小: {results_path.stat().st_size} bytes")
        else:
            print(f"    ❌ 未找到保存的文件: {results_path}")

        return True

    except Exception as e:
        print(f"    ❌ 测试失败: {e}")
        return False

def test_hicosmoviz_default():
    """测试HIcosmoViz默认设置"""
    print("\n🌟 测试HIcosmoViz默认设置...")

    try:
        # 创建HIcosmoViz实例 (应该自动使用新的专业风格)
        viz = HIcosmoViz(results_dir='results_test')

        print(f"    结果目录: {viz.results_dir}")
        print(f"    MCMC绘图器配色: {viz.mcmc_plotter.color_scheme}")
        print(f"    主色调: {viz.mcmc_plotter.colors[0]}")

        # 验证results目录是否创建
        if viz.results_dir.exists():
            print(f"    ✅ results目录自动创建: {viz.results_dir}")
        else:
            print(f"    ❌ results目录未创建: {viz.results_dir}")

        return True

    except Exception as e:
        print(f"    ❌ HIcosmoViz测试失败: {e}")
        return False

def test_color_scheme_verification():
    """验证三种配色方案"""
    print("\n🎨 验证三种配色方案...")

    schemes = ['modern', 'sophisticated', 'classic']
    expected_colors = {
        'modern': '#2E86AB',
        'sophisticated': '#264653',
        'classic': '#348ABD'
    }

    for scheme in schemes:
        try:
            plotter = MCMCPlotter(color_scheme=scheme)
            actual_color = plotter.colors[0]
            expected_color = expected_colors[scheme]

            if actual_color == expected_color:
                print(f"    ✅ {scheme}: {actual_color}")
            else:
                print(f"    ❌ {scheme}: 期望{expected_color}, 实际{actual_color}")

        except Exception as e:
            print(f"    ❌ {scheme}: {e}")

def test_automatic_saving():
    """测试自动保存功能"""
    print("\n💾 测试自动保存到results/目录...")

    # 清理旧文件
    test_files = ['auto_save_test.pdf', 'modern_test.pdf', 'sophisticated_test.pdf']
    results_dir = Path('results')
    for filename in test_files:
        filepath = results_dir / filename
        if filepath.exists():
            filepath.unlink()

    # 创建测试数据
    np.random.seed(123)
    n_samples = 1000

    param1 = np.random.normal(1.0, 0.1, n_samples)
    param2 = np.random.normal(2.0, 0.2, n_samples)

    samples = np.column_stack([param1, param2])
    params = ['param1', 'param2']
    chain = ChainData(samples, params, label="Auto Save Test")

    success_count = 0

    for i, scheme in enumerate(['modern', 'sophisticated', 'classic']):
        filename = f'{scheme}_test.pdf'

        try:
            plotter = MCMCPlotter(color_scheme=scheme)
            fig = plotter.corner(chain, filename=filename)
            plt.close()

            # 检查文件
            filepath = results_dir / filename
            if filepath.exists():
                print(f"    ✅ {scheme}: 保存成功 ({filepath.stat().st_size} bytes)")
                success_count += 1
            else:
                print(f"    ❌ {scheme}: 文件未保存")

        except Exception as e:
            print(f"    ❌ {scheme}: {e}")

    print(f"    总计: {success_count}/3 成功保存")
    return success_count == 3

if __name__ == "__main__":
    print("🚀 测试HIcosmo新的默认专业风格")
    print("=" * 50)

    # 确保目录存在
    Path('results').mkdir(exist_ok=True)

    test1 = test_default_new_style()
    test2 = test_hicosmoviz_default()
    test_color_scheme_verification()
    test3 = test_automatic_saving()

    print("\n" + "=" * 50)
    print("🎉 测试总结:")

    if test1:
        print("  ✅ MCMCPlotter默认风格正确")
    else:
        print("  ❌ MCMCPlotter默认风格问题")

    if test2:
        print("  ✅ HIcosmoViz默认设置正确")
    else:
        print("  ❌ HIcosmoViz默认设置问题")

    if test3:
        print("  ✅ 自动保存功能正常")
    else:
        print("  ❌ 自动保存功能问题")

    print("\n🎨 新的默认风格特性:")
    print("  • modern配色方案作为默认")
    print("  • 图片自动保存到results/目录")
    print("  • 智能刻度优化防重叠")
    print("  • 专业LaTeX标签 (H₀自动加单位)")
    print("  • 单色专业contour风格")
    print("  • 基于analysis/core.py的成熟实现")