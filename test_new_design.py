#!/usr/bin/env python3
"""测试新的集成设计"""

import sys
import os

# Add HiCosmo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hicosmo.models.lcdm import LCDMModel

# 理想的用法 - 所有参数在初始化时指定，所有方法都内置！
print("🚀 测试新的集成设计")
print("=" * 50)

# 创建ΛCDM模型，参数在初始化时指定
lcdm = LCDMModel(
    H0=67.4,
    Omega_m=0.315,
    Omega_k=0.0,  # 平坦宇宙
    sigma8=0.81,
    n_s=0.965
)

print(f"✅ 模型创建成功！")
print(f"   H₀ = {lcdm.H0_value:.1f} km/s/Mpc")
print(f"   Ωₘ = {lcdm.params['Omega_m']:.4f}")
print(f"   ΩΛ = {lcdm.params['Omega_Lambda']:.4f}")

print("\n📏 测试距离计算 - 无需传参数！")
z = 1.0

# 一行调用，无需传参数！
H_z = lcdm.H_z(z)
d_A = lcdm.angular_diameter_distance(z)
d_L = lcdm.luminosity_distance(z)
mu = lcdm.distance_modulus(z)

print(f"   z = {z}")
print(f"   H({z}) = {H_z:.1f} km/s/Mpc")
print(f"   D_A({z}) = {d_A:.1f} Mpc")
print(f"   D_L({z}) = {d_L:.1f} Mpc")
print(f"   μ({z}) = {mu:.2f} mag")

print("\n📊 测试批量计算")
import jax.numpy as jnp
z_array = jnp.array([0.1, 0.5, 1.0, 2.0])
d_L_array = lcdm.luminosity_distance(z_array)
print(f"   z = {z_array}")
print(f"   D_L = {d_L_array}")

print("\n📋 完整摘要")
print(lcdm.distance_summary(1.0))

print("\n🎯 对比：旧设计 vs 新设计")
print("旧设计（繁琐）:")
print("  model = LCDMModel()")
print("  bg = BackgroundEvolution(model)")  
print("  distances = CosmologicalDistances(bg)")
print("  d_A = distances.angular_diameter_distance(1.0, params)")
print()
print("新设计（简洁）:")
print("  lcdm = LCDMModel(H0=67.4, Omega_m=0.315)")
print("  d_A = lcdm.angular_diameter_distance(1.0)")

print("\n✅ 新设计完美工作！")