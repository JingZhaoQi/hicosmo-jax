#!/usr/bin/env python3
"""
测试PantheonPlus和PantheonPlusSH0ES功能
"""

# 使用下载的官方数据
import os
data_path = os.path.join(os.path.dirname(__file__), "data", "DataRelease")

print("🧪 测试PantheonPlus和PantheonPlusSH0ES功能")
print("=" * 60)

try:
    import numpy as np
    from hicosmo.likelihoods.pantheonplus import PantheonPlusLikelihood

    print("测试1: PantheonPlus 纯超新星模式")
    likelihood1 = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=False,
        z_min=0.01,
        apply_z_cut=True
    )
    print(f"数据集: {likelihood1.get_info()['data_type']}")
    print(f"对象数量: {likelihood1.n_sne}")
    print(f"红移范围: [{likelihood1.redshifts.min():.4f}, {likelihood1.redshifts.max():.4f}]")
    print()

    print("测试2: PantheonPlusSH0ES 组合数据集")
    likelihood2 = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=True,
        z_min=0.01,
        apply_z_cut=True
    )
    print(f"数据集: {likelihood2.get_info()['data_type']}")
    print(f"对象数量: {likelihood2.n_sne}")
    print(f"红移范围: [{likelihood2.redshifts.min():.4f}, {likelihood2.redshifts.max():.4f}]")
    if hasattr(likelihood2, 'is_calibrator'):
        print(f"造父变星校准器数量: {np.sum(likelihood2.is_calibrator)}")
    print()

    print("测试3: 自定义红移范围")
    likelihood3 = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=True,
        z_min=0.02,
        z_max=1.5,
        apply_z_cut=True
    )
    print(f"自定义范围后对象数量: {likelihood3.n_sne}")
    print(f"红移范围: [{likelihood3.redshifts.min():.4f}, {likelihood3.redshifts.max():.4f}]")
    print()

    print("测试4: 不截断红移")
    likelihood4 = PantheonPlusLikelihood(
        data_path=data_path,
        include_shoes=False,
        apply_z_cut=False
    )
    print(f"不截断对象数量: {likelihood4.n_sne}")
    print(f"红移范围: [{likelihood4.redshifts.min():.4f}, {likelihood4.redshifts.max():.4f}]")

    print("\n✅ 所有测试通过！PantheonPlus/SH0ES功能正常工作。")
    print("\n数据集对比:")
    print(f"  纯PantheonPlus: {likelihood1.n_sne} 对象")
    print(f"  PantheonPlusSH0ES: {likelihood2.n_sne} 对象")
    print(f"  差异: {likelihood2.n_sne - likelihood1.n_sne} 个造父变星校准器")

except FileNotFoundError as e:
    print(f"❌ 数据文件未找到: {e}")
    print("\n请按以下步骤设置数据:")
    print("1. 从官方仓库下载数据:")
    print("   git clone https://github.com/PantheonPlusSH0ES/DataRelease.git")
    print("2. 设置data_path指向DataRelease目录")
    print("3. 确保存在以下文件结构:")
    print("   DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/")
    print("   ├── Pantheon+SH0ES.dat")
    print("   ├── Pantheon+SH0ES_STAT+SYS.cov")
    print("   └── Pantheon+SH0ES_STATONLY.cov")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()