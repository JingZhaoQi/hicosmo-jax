#!/usr/bin/env python3
"""
测试自动多核配置是否工作
"""

print("🧪 测试HIcosmo自动多核配置")
print("=" * 50)

# 直接导入HIcosmo（应该自动配置多核）
print("导入HIcosmo...")
import hicosmo

print("检查环境变量...")
import os
print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', '未设置')}")

print("检查JAX设备...")
import jax
print(f"JAX设备数量: {jax.local_device_count()}")
print(f"JAX设备列表: {jax.devices()}")

# 检查系统CPU核心数
print(f"系统CPU核心数: {os.cpu_count()}")

if jax.local_device_count() > 1:
    print(f"✅ 成功！自动配置了{jax.local_device_count()}个CPU设备")
else:
    print("❌ 失败！只有1个设备")

# 测试禁用功能
print("\n测试禁用功能...")
os.environ['HICOSMO_DISABLE_AUTO_XLA'] = '1'
print("设置HICOSMO_DISABLE_AUTO_XLA=1，重新导入应该不会设置XLA_FLAGS")