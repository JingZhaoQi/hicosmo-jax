# JAX优化使用指南

## 何时使用JAX优化初始化

基于严格的基准测试，**JAX优化默认关闭**，推荐在以下特定场景中启用：

### ✅ 推荐使用的场景

1. **计算昂贵的似然函数** (最重要)
   - 每次似然计算 > 10ms
   - 包含数值积分、微分方程求解
   - 涉及复杂的物理模拟
   - 例子：N体模拟、复杂的宇宙学模型

2. **高维度问题**
   - 参数数量 > 20个
   - 传统warmup需要 > 5000步才能收敛
   - 参数间存在强相关性

3. **多模态分布**
   - 似然函数有多个峰
   - 传统MCMC容易卡在局部最优
   - 需要好的初始点避免收敛失败

4. **收敛困难的问题**
   - 标准warmup后 R̂ > 1.1
   - 需要更好的初始化点

### ❌ 不推荐使用的场景

1. **简单的统计模型**
   - 线性回归、多项式拟合
   - 参数 < 10个
   - 似然计算 < 1ms

2. **常规的宇宙学推断**
   - 标准的距离模数拟合
   - SNe Ia、BAO等数据分析
   - 已经收敛良好的问题

## 使用方法

### 默认使用（推荐）
```python
# 使用标准warmup（默认）
mcmc = AutoMCMC(config, likelihood_func)
samples = mcmc.run()
```

### 启用优化（特殊场景）
```python
# 仅在必要时启用优化
mcmc = AutoMCMC(config, likelihood_func, 
                optimize_init=True,           # 启用JAX优化
                max_opt_iterations=500)       # 适当调整迭代数
samples = mcmc.run()
```

## 性能对比结果

| 似然函数类型 | 传统MCMC | JAX优化 | 加速比 | 推荐 |
|------------|----------|---------|-------|------|
| 简单函数 (< 1ms) | 1.6s | 1.7s | **0.94x** ❌ | 传统 |
| 中等复杂 (1-10ms) | 1.2s | 1.8s | **0.70x** ❌ | 传统 |
| 大规模 (1-10ms) | 4.1s | 4.5s | **0.92x** ❌ | 传统 |
| **昂贵似然** (>10ms) | 169s | ~50s | **~3.4x** ✅ | **优化** |

## 优化建议检查

系统会智能检测何时启用优化有意义：

```python
# 当你启用optimize_init=True时，系统会显示：
🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧
JAX OPTIMIZATION ENABLED
🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧
📝 Optimization is most beneficial when:
  • Likelihood computation time > 10ms per call
  • High-dimensional problems (>20 parameters)
  • Complex multi-modal distributions
  • Convergence problems with traditional warmup

⚠️  Note: For simple problems (<10 parameters), optimization
   overhead may exceed benefits. Consider traditional warmup.

💡 To disable optimization: set optimize_init=False
   This will use standard warmup (recommended for most cases)
```

## 实际测试方法

测试你的似然函数是否适合优化：

```python
import time

# 测试似然函数计算时间
start = time.time()
likelihood_value = your_likelihood(**test_params)
elapsed = time.time() - start

if elapsed > 0.01:  # 10ms
    print("✅ JAX优化可能有用，建议启用")
else:
    print("❌ 直接使用NumPyro标准warmup即可")
```

## 总结

- **默认情况**：使用标准warmup（optimize_init=False）
- **特殊情况**：昂贵似然函数或高维问题时启用优化
- **判断标准**：似然计算时间 > 10ms 或 参数数量 > 20
- **性能期望**：对于适合的问题，可获得 2-4x 加速