#!/usr/bin/env python3
"""
优雅的多核初始化演示

展示如何用一行代码替代之前丑陋的3行初始化
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🎨 HiCosmo 优雅初始化演示")
print("="*60)

print("\n❌ 之前的丑陋方式:")
print("-"*50)
print("""
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
import numpyro
numpyro.set_host_device_count(4)
""")

print("\n✨ 现在的优雅方式:")
print("-"*50)
print("""
from hicosmo.samplers import Config
Config.init()  # 就这么简单！
""")

print("\n📚 多种使用方式:")
print("-"*50)

# 方式1: 最简单 - 自动配置
print("\n1️⃣ 自动配置（推荐）:")
from hicosmo.samplers import Config
Config.reset()  # 仅用于演示，实际使用不需要
Config.init()

# 方式2: 指定核心数
print("\n2️⃣ 指定核心数:")
print("Config.init(cpu_cores=4)")

# 方式3: 静默模式
print("\n3️⃣ 静默模式:")
print("Config.init(verbose=False)")

# 方式4: 使用便捷函数
print("\n4️⃣ 便捷函数:")
print("""
from hicosmo.samplers import init_hicosmo
init_hicosmo()
""")

print("\n🔍 查看配置状态:")
print("-"*50)
status = Config.status()
print(f"初始化状态: {'已初始化' if status['initialized'] else '未初始化'}")
print(f"系统核心数: {status['system_cores']}")
print(f"激活核心数: {status['config'].get('actual_cores', 'N/A')}")
print(f"JAX设备数: {status['jax_devices']}")

print("\n💡 优势总结:")
print("-"*50)
print("✅ 一行代码完成初始化")
print("✅ 自动检测最优配置")
print("✅ 智能处理环境变量")
print("✅ 优雅的错误处理")
print("✅ 支持多种配置模式")

print("\n🚀 立即在你的代码中使用:")
print("-"*50)
print("""
# 在你的脚本开头添加:
from hicosmo.samplers import Config
Config.init()

# 然后正常使用 MCMC:
from hicosmo.samplers import MCMC
mcmc = MCMC(config, likelihood_func)
samples = mcmc.run()
""")

print("\n="*60)
print("🎉 从此告别丑陋的初始化代码！")
print("="*60)