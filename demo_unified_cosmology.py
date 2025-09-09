#!/usr/bin/env python3
"""
统一宇宙学模型演示
=================

展示新设计的优势：
- 参数封装在对象中
- 一个类包含所有计算方法
- 简洁的API
- 符合面向对象设计原则
"""

import sys
import os
import numpy as np
import jax.numpy as jnp

# Add HiCosmo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hicosmo.models.lcdm import LCDMModel

def demo_unified_design():
    print("🌟 HiCosmo 统一宇宙学模型演示")
    print("=" * 60)
    
    # 1. 创建不同的宇宙学模型
    print("\n📋 1. 创建多个宇宙学模型")
    
    # Planck 2018 参数
    planck = LCDMModel(
        H0=67.36,
        Omega_m=0.3153,
        Omega_k=0.0,
        sigma8=0.8111,
        n_s=0.9649
    )
    
    # 高H0模型 (SH0ES)
    shoes = LCDMModel(
        H0=73.2,
        Omega_m=0.3153,
        Omega_k=0.0,
        sigma8=0.8111,
        n_s=0.9649
    )
    
    # 开放宇宙模型
    open_universe = LCDMModel(
        H0=67.36,
        Omega_m=0.3153,
        Omega_k=0.05,  # 开放！
        sigma8=0.8111,
        n_s=0.9649
    )
    
    models = {
        'Planck 2018': planck,
        'SH0ES (H0 tension)': shoes,  
        'Open Universe': open_universe
    }
    
    for name, model in models.items():
        print(f"   ✅ {name}: H₀={model.H0_value:.1f}, Ωₘ={model.params['Omega_m']:.4f}, Ωₖ={model.params.get('Omega_k', 0):.3f}")
    
    # 2. 简洁的距离计算
    print("\n📏 2. 简洁的距离计算 (z=1)")
    z = 1.0
    
    for name, model in models.items():
        d_A = model.angular_diameter_distance(z)
        d_L = model.luminosity_distance(z)
        print(f"   {name:20s}: D_A = {d_A:6.1f} Mpc, D_L = {d_L:6.1f} Mpc")
    
    # 3. 批量计算演示
    print("\n📊 3. 批量计算演示")
    z_array = jnp.linspace(0, 2, 11)
    
    print(f"   红移: {z_array}")
    
    planck_dL = planck.luminosity_distance(z_array)
    print(f"   Planck D_L: {planck_dL}")
    
    # 4. H0张力的影响
    print("\n⚡ 4. H0张力对距离的影响")
    z_comparison = jnp.array([0.1, 0.5, 1.0, 1.5, 2.0])
    
    print("   红移  Planck_D_L  SH0ES_D_L   差异%")
    print("   " + "-" * 40)
    for z_val in z_comparison:
        dL_planck = planck.luminosity_distance(z_val)
        dL_shoes = shoes.luminosity_distance(z_val)
        diff_percent = 100 * (dL_shoes - dL_planck) / dL_planck
        print(f"   {z_val:4.1f}   {dL_planck:8.1f}   {dL_shoes:8.1f}   {diff_percent:+5.1f}%")
    
    # 5. 曲率的影响
    print("\n🌍 5. 宇宙曲率的影响 (z=1)")
    z = 1.0
    
    flat_dL = planck.luminosity_distance(z)
    open_dL = open_universe.luminosity_distance(z)
    curvature_effect = 100 * (open_dL - flat_dL) / flat_dL
    
    print(f"   平坦宇宙:   D_L = {flat_dL:.1f} Mpc")
    print(f"   开放宇宙:   D_L = {open_dL:.1f} Mpc")
    print(f"   曲率效应:   {curvature_effect:+.1f}%")
    
    # 6. 完整摘要
    print("\n📋 6. 完整模型摘要")
    print(planck.distance_summary(1.0))
    
    # 7. 设计对比
    print("\n🎯 7. 设计对比")
    print("旧设计 (繁琐且易错):")
    print("```python")
    print("# 需要多层嵌套")
    print("model = LCDMModel()")
    print("bg = BackgroundEvolution(model)")
    print("distances = CosmologicalDistances(bg)")
    print("# 每次都要传参数！")
    print("d_A = distances.angular_diameter_distance(1.0, fiducial_params)")
    print("```")
    print()
    print("新设计 (简洁且直观):")
    print("```python")
    print("# 一步到位，参数封装")
    print("lcdm = LCDMModel(H0=67.4, Omega_m=0.315)")
    print("# 直接调用，无需传参")
    print("d_A = lcdm.angular_diameter_distance(1.0)")
    print("```")
    
    print("\n✅ 统一设计的优势:")
    print("   🔸 参数封装在对象中，避免重复传递")
    print("   🔸 所有相关计算都在一个类中")
    print("   🔸 符合面向对象设计原则")
    print("   🔸 API简洁直观，减少出错机会")
    print("   🔸 支持批量计算")
    print("   🔸 易于比较不同模型")

if __name__ == "__main__":
    demo_unified_design()