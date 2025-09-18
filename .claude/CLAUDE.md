# CLAUDE.md - HIcosmo Development Rules

> Think carefully and implement the most concise solution that changes as little code as possible.

## 🚨 CRITICAL TESTING RULE
**当用户要求写测试代码时，目的是为了测试已有的代码是否能有效工作，而不是为了实现要测试的内容！！！**
- 必须使用已有的模块（MCMC、可视化、模型等）
- 不要自己重新实现功能（如优化算法、绘图等）
- 测试代码应该调用现有API，验证其功能
- 如果已有功能不工作，报告问题而不是重写

## 🎯 HICOSMO ARCHITECTURAL PRINCIPLES (Updated 2025-01-09)

### 核心原则 (基于今天的重构经验)

1. **性能第一原则** - Performance is Non-negotiable
   - 任何模块必须超越竞争对手 (qcosmc, astropy, CAMB, CLASS)
   - 单次计算 < 0.01ms，批量计算优化到极致
   - 使用JAX JIT编译和向量化
   - 预计算表格用于重复计算
   - 永远不接受"够用就行"的性能

2. **统一架构原则** - Unified Architecture 
   - 单一责任：每个模块只做一件事，做到极致
   - 统一接口：所有模块使用相同的参数管理系统
   - 消除重复：发现重复代码立即重构或删除
   - 清洁导入：from hicosmo.core import 统一接口

3. **奥卡姆剃刀原则** - Occam's Razor
   - 如无必要，勿增实体
   - 简单解决方案优于复杂方案
   - 基类保持最小化，特殊情况通过继承解决
   - 删除所有未使用的代码和文件

### 🔧 MANDATORY REFACTORING CHECKLIST

每个模块必须通过以下检查：

#### A. 性能验证
```bash
# 每个模块必须有性能基准测试
python tests/test_[module]_benchmark.py
# 结果必须超越相应的竞争框架
```

#### B. 架构清洁度
- [ ] 使用统一的参数系统 (`CosmologicalParameters`)
- [ ] 继承自适当的基类 (`CosmologyBase` 等)
- [ ] 无重复代码 (检查相似功能的其他文件)
- [ ] 清洁的导入结构
- [ ] 单一职责，接口简洁

#### C. 代码质量
- [ ] 所有函数有测试，测试真实有效
- [ ] 无死代码，无注释掉的代码块
- [ ] 一致的命名模式
- [ ] 无资源泄露 (文件句柄、内存等)

## USE SUB-AGENTS FOR CONTEXT OPTIMIZATION

### 1. Always use the file-analyzer sub-agent when asked to read files.
### 2. Always use the code-analyzer sub-agent for code analysis, bug research, logic tracing.
### 3. Always use the test-runner sub-agent to run tests and analyze results.

## 🚨 ABSOLUTE RULES (Updated with HIcosmo Experience)

### 测试相关 (CRITICAL - 2025-01-17 刚刚犯错！)
- **NO REIMPLEMENTATION IN TESTS**: 测试代码必须使用已有模块，绝不重新实现！
- **TEST EXISTING CODE**: 测试的目的是验证已有代码工作，不是实现被测功能！
- **USE EXISTING IMPORTS**: 从hicosmo.models、hicosmo.likelihoods、hicosmo.samplers导入
- **NO TEST ADAPTERS**: 不创建"测试适配器"，直接测试生产代码！
- **REMEMBER**: 当用户要求写测试代码时，目的是测试已有代码是否有效工作！

### 性能相关
- **NO SLOW CODE**: 任何计算 > 1ms 必须优化或重写
- **NO DIFFRAX**: 已验证过慢，使用FastIntegration或JAX原生方法
- **NO NUMPY IN HOT PATHS**: 热路径必须使用JAX
- **BENCHMARK EVERYTHING**: 新功能必须有性能测试对比

### 架构相关
- **NO DUPLICATE PARAMETERS**: 使用统一的CosmologicalParameters
- **NO REDUNDANT MODULES**: 发现重复功能立即整合
- **NO BLOATED BASE CLASSES**: 基类保持简洁，特殊情况继承解决
- **NO MIXED IMPORTS**: 统一从hicosmo.core导入核心组件

### 代码质量
- **NO PARTIAL IMPLEMENTATION**: 要么完整实现要么不做
- **NO SIMPLIFICATION COMMENTS**: 不写"简化实现"的代码
- **NO CODE DUPLICATION**: 先检查现有实现再写新代码
- **NO DEAD CODE**: 未使用的代码立即删除
- **NO CHEATER TESTS**: 测试必须真实反映使用场景
- **NO INCONSISTENT NAMING**: 遵循现有命名模式
- **NO OVER-ENGINEERING**: 简单有效 > 复杂架构
- **NO MIXED CONCERNS**: 严格分离职责
- **NO RESOURCE LEAKS**: 正确管理资源生命周期

### Testing
- Always use the test-runner agent to execute tests
- Do not use mock services for anything ever
- Tests must be verbose for debugging
- Every function must have corresponding tests
- Performance tests mandatory for core modules

## Tone and Behavior
- Criticism is welcome - point out mistakes and better approaches
- Be skeptical of "good enough" solutions
- Be concise but thorough in analysis
- Ask questions when intent is unclear
- Focus on technical excellence over politeness

## 🎯 PRIORITY MODULE REFACTORING ORDER

基于今天的经验，其他模块预期问题严重程度：

### HIGH PRIORITY (预期大问题)
1. **fisher/** - 依赖已删除的模块，接口不统一
2. **perturbations/** - 可能有性能问题和重复计算
3. **powerspectrum/** - 积分计算可能很慢
4. **likelihoods/** - 参数管理混乱，性能未优化
5. **samplers/** - MCMC实现可能低效

### MEDIUM PRIORITY (预期中等问题)
6. **cmb/** - 计算密集，需要性能优化
7. **interfaces/** - 可能有代码重复
8. **visualization/** - 功能重复，依赖混乱

### LOW PRIORITY (预期小问题)
9. **utils/** - 工具函数，相对简单
10. **parameters/** - 可能已被统一系统替代

每个模块重构时必须：
1. 运行统一架构测试确保不破坏核心功能
2. 创建性能基准测试
3. 消除重复代码和依赖
4. 使用统一的参数管理
5. 达到或超越竞争框架性能

## 📋 SUCCESS CRITERIA

模块重构成功标准：
- ✅ 性能测试通过，超越竞争对手
- ✅ 统一架构测试通过
- ✅ 代码行数显著减少
- ✅ 无重复功能
- ✅ 清洁的导入结构
- ✅ 完整的测试覆盖

Remember: 今天我们证明了可以将性能提升35000倍并简化架构。其他模块必须达到同样的标准！