#!/usr/bin/env python3
"""
单核vs多核性能对比协调器
运行独立的单核和多核测试脚本并对比结果
"""

import subprocess
import time
from pathlib import Path
import os

def run_test_script(script_name):
    """运行测试脚本并获取结果"""
    script_path = Path(__file__).parent / script_name
    
    try:
        print(f"\n{'='*60}")
        print(f"运行 {script_name}")
        print('='*60)
        
        start_time = time.time()
        result = subprocess.run(
            ['python', str(script_path)], 
            capture_output=True, 
            text=True, 
            cwd=str(Path(__file__).parent.parent)
        )
        total_time = time.time() - start_time
        
        if result.returncode == 0:
            print(result.stdout)
            if result.stderr:
                print("警告信息:")
                print(result.stderr)
            
            # 尝试从输出中提取执行时间
            output_lines = result.stdout.split('\n')
            execution_time = None
            for line in output_lines:
                if '完成:' in line and 's' in line:
                    try:
                        time_str = line.split('完成:')[1].strip().replace('s', '').replace('(跳过诊断)', '').strip()
                        execution_time = float(time_str)
                        break
                    except:
                        continue
            
            if execution_time is None:
                execution_time = total_time
                
            return {
                'success': True,
                'time': execution_time,
                'output': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ {script_name} 执行失败:")
            print(result.stdout)
            print(result.stderr)
            return {
                'success': False,
                'time': 0,
                'error': result.stderr
            }
            
    except Exception as e:
        print(f"❌ 运行 {script_name} 时出错: {e}")
        return {
            'success': False,
            'time': 0,
            'error': str(e)
        }

def main():
    """主函数"""
    print("HiCosmo 单核 vs 多核性能对比")
    print(f"系统CPU核心数: {os.cpu_count()}")
    print("="*60)
    
    # 运行单核测试
    single_result = run_test_script('single_core_test.py')
    
    # 运行多核测试
    multi_result = run_test_script('multi_core_test.py')
    
    # 分析结果
    print("\n" + "="*60)
    print("📊 性能对比结果")
    print("="*60)
    
    if single_result['success'] and multi_result['success']:
        single_time = single_result['time']
        multi_time = multi_result['time']
        
        print(f"单核单链时间: {single_time:.2f}s")
        print(f"多核四链时间: {multi_time:.2f}s")
        
        # 理论上4条链串行需要的时间
        theoretical_serial = single_time * 4
        
        if multi_time < theoretical_serial:
            speedup = theoretical_serial / multi_time
            efficiency = speedup / 4 * 100
            print(f"\n🎉 多核并行有效!")
            print(f"理论串行时间: {theoretical_serial:.2f}s (单核×4)")
            print(f"实际并行时间: {multi_time:.2f}s")
            print(f"并行加速比: {speedup:.2f}x")
            print(f"并行效率: {efficiency:.1f}%")
            print(f"节省时间: {theoretical_serial - multi_time:.2f}s ({(1 - multi_time/theoretical_serial)*100:.1f}%)")
        else:
            print(f"\n⚠️  多核并行效果有限")
            print(f"理论串行时间: {theoretical_serial:.2f}s")
            print(f"实际并行时间: {multi_time:.2f}s")
            slowdown = multi_time / theoretical_serial
            print(f"相对性能: {slowdown:.2f}x (>1表示比理论串行慢)")
            
        # 直接对比单核vs多核
        if multi_time < single_time:
            direct_speedup = single_time / multi_time
            print(f"\n📈 直接对比:")
            print(f"多核比单核快 {direct_speedup:.2f}x")
        else:
            direct_slowdown = multi_time / single_time  
            print(f"\n📉 直接对比:")
            print(f"多核比单核慢 {direct_slowdown:.2f}x (可能因为链数不同)")
            
    else:
        print("❌ 测试失败，无法进行对比")
        if not single_result['success']:
            print(f"单核测试失败: {single_result.get('error', 'Unknown')}")
        if not multi_result['success']:
            print(f"多核测试失败: {multi_result.get('error', 'Unknown')}")

if __name__ == "__main__":
    main()