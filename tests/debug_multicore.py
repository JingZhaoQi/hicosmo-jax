#!/usr/bin/env python3
"""
调试多核设置问题
"""

print("=== 第1步：检查系统信息 ===")
import os
print(f"系统CPU核心数: {os.cpu_count()}")

print("\n=== 第2步：设置NumPyro多核 ===")
import numpyro
print("设置4个CPU设备...")
numpyro.set_host_device_count(4)
print("NumPyro设置完成")

print("\n=== 第3步：检查JAX设备 ===")
import jax
print(f"JAX版本: {jax.__version__}")
print(f"JAX设备: {jax.devices()}")
print(f"设备数量: {jax.local_device_count()}")
print(f"设备类型: {[str(d) for d in jax.devices()]}")

print("\n=== 第4步：尝试手动设置JAX ===")
# 尝试手动设置JAX设备
try:
    # 检查JAX配置选项
    print("JAX配置选项:")
    print(f"  XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'None')}")
    print(f"  JAX_PLATFORM_NAME: {os.environ.get('JAX_PLATFORM_NAME', 'None')}")
    
    # 尝试设置环境变量
    print("\n尝试设置环境变量...")
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
    
    # 重新导入JAX（这通常不会生效，因为JAX已经初始化）
    print("重新检查JAX设备...")
    print(f"设备数量: {jax.local_device_count()}")
    
except Exception as e:
    print(f"设置失败: {e}")

print("\n=== 第5步：检查JAX平台信息 ===")
try:
    print("JAX平台信息:")
    for i, device in enumerate(jax.devices()):
        print(f"  设备 {i}: {device}")
        print(f"    平台: {device.platform}")
        print(f"    设备类型: {device.device_kind}")
        
    print(f"默认后端: {jax.default_backend()}")
    
except Exception as e:
    print(f"获取平台信息失败: {e}")

print("\n=== 第6步：测试实际的多设备操作 ===")
try:
    import jax.numpy as jnp
    
    # 创建一个简单的并行操作
    def simple_computation(x):
        return x * 2
    
    # 尝试在多设备上执行
    data = jnp.arange(8)
    print(f"输入数据: {data}")
    
    # 使用pmap进行并行计算
    if len(jax.devices()) > 1:
        print("尝试并行计算...")
        parallel_fn = jax.pmap(simple_computation)
        # 将数据分片到多个设备
        sharded_data = data.reshape(len(jax.devices()), -1)
        result = parallel_fn(sharded_data)
        print(f"并行结果: {result}")
    else:
        print("只有一个设备，无法测试并行")
        
except Exception as e:
    print(f"并行测试失败: {e}")

print("\n=== 诊断总结 ===")
if jax.local_device_count() > 1:
    print("✅ 多核设置成功")
else:
    print("❌ 多核设置失败")
    print("可能的原因:")
    print("1. JAX在NumPyro设置之前已经初始化")
    print("2. 系统不支持多CPU设备")
    print("3. JAX版本或配置问题")
    print("4. 环境变量设置问题")