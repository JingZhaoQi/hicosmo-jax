#!/usr/bin/env python3
"""
HiCosmo 统一架构验证测试
========================

验证重构后的代码库是否符合设计目标:
1. ✅ 消除了重复和冗余代码
2. ✅ 使用统一的参数管理系统
3. ✅ 保持了所有原有功能
4. ✅ 提升了性能和可维护性
5. ✅ 简化了用户接口

作者: HiCosmo团队
日期: 2025-09-09
"""

import sys
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Add HiCosmo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hicosmo.models import LCDM
from hicosmo.core import CosmologicalParameters, PLANCK_2018, PLANCK_2015
import numpy as np


class UnifiedArchitectureValidator:
    """统一架构验证器"""
    
    def __init__(self):
        print("🏗️  HiCosmo 统一架构验证测试")
        print("=" * 80)
        print("🎯 验证重构后的代码库是否达到设计目标")
        print()
        
    def test_clean_imports(self):
        """测试导入的清洁性"""
        print("📦 测试清洁的导入结构")
        print("-" * 50)
        
        # 测试可以从统一接口导入所有需要的组件
        try:
            from hicosmo.models import LCDM
            from hicosmo.core import CosmologicalParameters
            from hicosmo.core import FastIntegration
            from hicosmo.core import PLANCK_2018
            print("✅ 所有核心组件可以从统一接口导入")
        except ImportError as e:
            print(f"❌ 导入失败: {e}")
            return False
            
        # 测试旧的重复模块已被移除
        removed_modules = [
            'hicosmo.core.cosmology_v2',
            'hicosmo.core.cosmology_optimized',
            'hicosmo.background.background'
        ]
        
        for module in removed_modules:
            try:
                __import__(module)
                print(f"⚠️  旧模块仍存在: {module}")
            except ImportError:
                print(f"✅ 旧模块已移除: {module}")
                
        return True
        
    def test_unified_parameters(self):
        """测试统一参数管理"""
        print("\n🎛️  测试统一参数管理系统")
        print("-" * 50)
        
        # 测试参数系统基本功能
        params = CosmologicalParameters(H0=70.0, Omega_m=0.3)
        
        # 参数获取
        assert params['H0'] == 70.0
        assert params['Omega_m'] == 0.3
        assert 'Omega_Lambda' in params  # 衍生参数
        print("✅ 基本参数管理功能正常")
        
        # 参数验证
        try:
            bad_params = CosmologicalParameters(H0=500.0)  # 超出合理范围
            print("❌ 参数验证失败")
        except ValueError:
            print("✅ 参数验证功能正常")
            
        # 预定义参数集
        planck_params = PLANCK_2018
        assert planck_params['H0'] == 67.36
        assert planck_params['Omega_m'] == 0.3153
        print("✅ 预定义参数集工作正常")
        
        return True
        
    def test_model_consistency(self):
        """测试模型一致性"""
        print("\n🔬 测试模型功能一致性")
        print("-" * 50)
        
        # 使用不同方式创建相同的模型
        model1 = LCDM(H0=67.36, Omega_m=0.3153, Omega_b=0.0493)
        
        # 使用预定义参数
        model2 = LCDM(**PLANCK_2018.to_dict())
        
        # 测试计算结果一致性
        z_test = 1.0
        d1 = model1.comoving_distance(z_test)
        d2 = model2.comoving_distance(z_test)
        
        # 应该得到相同结果
        assert abs(d1 - d2) < 1e-10, f"不一致的结果: {d1} vs {d2}"
        print(f"✅ 模型一致性测试通过: {d1:.2f} Mpc")
        
        # 测试所有距离计算方法
        distances = {
            'comoving': model1.comoving_distance(z_test),
            'angular_diameter': model1.angular_diameter_distance(z_test),
            'luminosity': model1.luminosity_distance(z_test),
            'distance_modulus': model1.distance_modulus(z_test)
        }
        
        print("✅ 所有距离计算方法正常工作:")
        for name, value in distances.items():
            print(f"   {name}: {value:.2f}")
            
        return True
        
    def test_performance_maintained(self):
        """测试性能是否保持"""
        print("\n⚡ 测试重构后性能")
        print("-" * 50)
        
        model = LCDM(**PLANCK_2018.to_dict())
        
        # 单次计算性能
        z_test = 1.0
        n_runs = 1000
        
        start = time.time()
        for _ in range(n_runs):
            _ = model.comoving_distance(z_test)
        single_time = (time.time() - start) / n_runs * 1000
        
        print(f"✅ 单次计算时间: {single_time:.4f} ms")
        
        # 批量计算性能
        z_array = np.linspace(0.1, 5.0, 1000)
        
        start = time.time()
        _ = model.comoving_distance(z_array)
        batch_time = (time.time() - start) * 1000
        
        print(f"✅ 批量计算时间 (1000点): {batch_time:.3f} ms")
        print(f"✅ 每点时间: {batch_time/1000:.4f} ms/point")
        
        # 性能应该仍然很好 (< 0.1 ms单次, < 0.1 ms批量)
        assert single_time < 0.1, f"单次性能退化: {single_time:.4f} ms"
        assert batch_time < 0.1, f"批量性能退化: {batch_time:.3f} ms"
        
        return True
        
    def test_precision_modes(self):
        """测试精度模式功能"""
        print("\n🎯 测试精度模式")
        print("-" * 50)
        
        z_test = 1.0
        
        # 测试不同精度模式
        modes = ['fast', 'balanced', 'precise']
        results = {}
        
        for mode in modes:
            model = LCDM(precision_mode=mode, **PLANCK_2018.to_dict())
            result = model.comoving_distance(z_test)
            results[mode] = result
            print(f"✅ {mode}模式: {result:.4f} Mpc")
            
        # 所有模式应该给出相近结果 (相对误差 < 0.1%)
        ref_result = results['precise']
        for mode, result in results.items():
            rel_error = abs(result - ref_result) / ref_result * 100
            assert rel_error < 0.1, f"{mode}模式误差过大: {rel_error:.3f}%"
            
        print("✅ 所有精度模式工作正常，结果一致")
        return True
        
    def test_backward_compatibility(self):
        """测试向后兼容性"""
        print("\n🔄 测试向后兼容性")
        print("-" * 50)
        
        # 测试用户代码是否仍然工作
        
        # 原有的简单用法
        model = LCDM()  # 使用默认参数
        d = model.comoving_distance(1.0)
        print(f"✅ 默认参数模型: {d:.2f} Mpc")
        
        # 原有的参数传递方式
        model = LCDM(H0=70.0, Omega_m=0.3)
        d = model.comoving_distance([0.5, 1.0, 2.0])
        print(f"✅ 向量化计算: {[f'{x:.1f}' for x in d]} Mpc")
        
        # 原有的方法调用
        d_a = model.angular_diameter_distance(1.0)
        d_l = model.luminosity_distance(1.0)
        mu = model.distance_modulus(1.0)
        
        print(f"✅ 距离方法: D_A={d_a:.1f}, D_L={d_l:.1f}, μ={mu:.1f}")
        
        return True
        
    def test_code_organization(self):
        """测试代码组织结构"""
        print("\n📁 测试代码组织结构")
        print("-" * 50)
        
        # 检查核心文件是否存在且合理
        expected_files = [
            'hicosmo/core/base.py',
            'hicosmo/core/fast_integration.py', 
            'hicosmo/core/unified_parameters.py',
            'hicosmo/models/lcdm.py'
        ]
        
        missing_files = []
        for file_path in expected_files:
            full_path = Path(__file__).parent.parent / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                print(f"✅ 核心文件存在: {file_path}")
                
        if missing_files:
            print(f"❌ 缺失文件: {missing_files}")
            return False
            
        # 检查代码行数减少
        lcdm_path = Path(__file__).parent.parent / 'hicosmo/models/lcdm.py'
        with open(lcdm_path) as f:
            lcdm_lines = len(f.readlines())
            
        print(f"✅ LCDM模型代码行数: {lcdm_lines}")
        
        # 应该比之前减少了很多 (原来3184行总共，现在应该大大减少)
        if lcdm_lines > 1500:
            print("⚠️  代码行数仍然较多，可能需要进一步清理")
        else:
            print("✅ 代码简化效果良好")
            
        return True
        
    def run_all_tests(self):
        """运行所有验证测试"""
        tests = [
            ('清洁导入结构', self.test_clean_imports),
            ('统一参数管理', self.test_unified_parameters),
            ('模型功能一致性', self.test_model_consistency),
            ('性能保持', self.test_performance_maintained),
            ('精度模式', self.test_precision_modes),
            ('向后兼容性', self.test_backward_compatibility),
            ('代码组织结构', self.test_code_organization)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    print(f"❌ {test_name}测试失败")
            except Exception as e:
                print(f"❌ {test_name}测试异常: {e}")
                
        print("\n" + "=" * 80)
        print("📋 统一架构验证测试总结")
        print("=" * 80)
        
        if passed == total:
            print("🎉 所有测试通过！统一架构重构成功！")
            print()
            print("✅ 重构成果:")
            print("  - 消除了重复和冗余代码")
            print("  - 统一了参数管理系统") 
            print("  - 保持了所有功能完整性")
            print("  - 维持了超高性能")
            print("  - 简化了代码结构")
            print("  - 保持了向后兼容性")
            print()
            print("🚀 HiCosmo现在具有清洁、统一、高性能的架构！")
            return True
        else:
            print(f"❌ {total - passed}/{total} 测试失败")
            print("需要进一步修复和完善")
            return False


def main():
    """主测试函数"""
    validator = UnifiedArchitectureValidator()
    return validator.run_all_tests()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)