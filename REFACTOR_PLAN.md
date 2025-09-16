# HIcosmo Samplers模块重构计划

## 🎯 重构目标

1. **解决硬命名问题**：重命名关键类提高语义清晰度
2. **消除硬编码**：将所有常量参数化或配置化
3. **降低耦合度**：重新设计模块间依赖关系
4. **提高可维护性**：拆分大型类，明确职责边界

## 📋 具体重构步骤

### Step 1: 创建配置常量模块

创建 `hicosmo/samplers/constants.py`：

```python
"""
配置常量定义
"""

# MCMC默认参数
DEFAULT_NUM_SAMPLES = 2000
DEFAULT_NUM_CHAINS = 4
DEFAULT_WARMUP_STANDARD = 2000  # 标准warmup
DEFAULT_WARMUP_OPTIMIZED = 300  # 优化模式warmup

# 优化参数
DEFAULT_MAX_OPTIMIZATION_ITERATIONS = 1000
OPTIMIZATION_PROGRESS_INTERVAL = 100
OPTIMIZATION_PENALTY_FACTOR = 1000

# 检查点参数
DEFAULT_CHECKPOINT_INTERVAL = 1000
DEFAULT_CHECKPOINT_DIR = "mcmc_chains"
CHECKPOINT_FILE_EXTENSION = ".h5"

# 文件和路径
CHECKPOINT_PATTERN_TEMPLATE = "{run_name}_step_*.h5"
FINAL_RESULT_PATTERN = "{run_name}.h5"

# 数值精度
RNG_SEED_MODULO = 2**32
PROGRESS_REPORT_DECIMALS = 1
```

### Step 2: 重命名核心类

**主要重命名映射：**

| 原名称 | 新名称 | 理由 |
|--------|--------|------|
| `AutoMCMC` | `BayesianInference` | 更业务导向，清晰表达用途 |
| `AutoParameter` | `Parameter` | 简化命名，去除冗余前缀 |
| `ParameterConfig` | `InferenceConfig` | 更准确反映配置范围 |

### Step 3: 重构auto.py

将庞大的`AutoMCMC`类拆分为：

1. **`BayesianInference`** (主接口类)
2. **`OptimizationEngine`** (JAX优化逻辑)  
3. **`ConfigurationManager`** (配置管理)
4. **`ModelBuilder`** (NumPyro模型构建)

### Step 4: 消除硬编码

将所有硬编码常量替换为：
1. 配置参数
2. 常量模块引用
3. 可配置的类属性

### Step 5: 简化依赖关系

重新设计模块导入结构：
```
constants.py    (无依赖)
    ↑
config.py      (依赖constants)
    ↑  
core.py        (依赖config, constants)
    ↑
utils.py       (依赖core, config)
    ↑
inference.py   (依赖utils, core, config)  # 原auto.py
```

## 🔧 实施计划

### 阶段1: 基础重构 (Breaking Changes)
- [ ] 创建constants.py
- [ ] 重命名AutoMCMC → BayesianInference
- [ ] 消除主要硬编码问题
- [ ] 更新测试和文档

### 阶段2: 架构优化 (内部重构)
- [ ] 拆分大型类
- [ ] 重新组织模块依赖
- [ ] 简化接口

### 阶段3: 向后兼容 (可选)
- [ ] 提供AutoMCMC别名以保持向后兼容
- [ ] 添加废弃警告
- [ ] 文档迁移指南

## ⚠️ 风险评估

**高风险：**
- 重命名AutoMCMC会破坏现有代码
- 需要更新所有测试和文档

**缓解措施：**
- 先创建新接口，保留旧接口
- 逐步迁移，提供过渡期
- 完整的向后兼容性层

## 🎯 预期收益

1. **可维护性提升**：清晰的职责分离
2. **可配置性增强**：所有参数可调整
3. **语义清晰化**：更好的类和方法命名
4. **扩展性改善**：更容易添加新功能
5. **测试性提高**：更小的类更容易测试

## 📝 待讨论问题

1. 是否立即实施breaking changes还是渐进式重构？
2. BayesianInference这个名称是否合适？
3. 是否需要保持完全向后兼容？
4. 重构的优先级如何安排？