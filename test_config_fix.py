#!/usr/bin/env python3
"""
测试修复后的Config.init()是否能够正确处理JAX重新初始化
"""

print("🧪 测试修复后的Config.init()多核配置")
print("=" * 60)

# 直接导入HIcosmo（这会导致JAX被提前导入）
print("\n🔍 导入HIcosmo模块（会提前导入JAX）...")
from hicosmo.samplers import Config

# 检查JAX是否已被导入
import sys
jax_modules = [name for name in sys.modules.keys() if name.startswith('jax')]
print(f"检测到{len(jax_modules)}个JAX模块已导入: {jax_modules[:5]}...")

# 现在调用Config.init()看它是否能正确处理
print("\n🚀 调用Config.init(cpu_cores=6)...")
Config.init(cpu_cores=6, verbose=True)

# 验证JAX设备数量
print("\n🔍 验证最终JAX配置...")
import jax
print(f"JAX设备数量: {jax.local_device_count()}")
print(f"JAX设备列表: {jax.devices()}")

if jax.local_device_count() == 6:
    print("✅ 成功！Config.init()正确配置了6个CPU设备")
else:
    print(f"❌ 失败！只配置了{jax.local_device_count()}个设备")