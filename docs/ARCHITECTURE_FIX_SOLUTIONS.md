# HIcosmo 架构问题解决方案

## 问题诊断

### 当前错误
```python
TypeError: Error interpreting argument to <function CosmologyBase.H_z> as an abstract array. 
The problematic value is of type <class 'abc.ABCMeta'>
```

### 问题代码
```python
# 基类中的问题代码
class CosmologyBase(ABC):
    @classmethod
    @jit  # JAX无法处理cls参数
    def H_z(cls, z, params):
        return params['H0'] * cls.E_z(z, params)  # cls传递给JIT函数
```

## 解决方案一：纯函数式设计（推荐）

### 核心思想
将计算逻辑与类分离，使用纯函数进行所有数值计算。

### 实现方案

```python
# hicosmo/core/cosmology_functions.py
"""纯函数式宇宙学计算"""

import jax.numpy as jnp
from jax import jit
from typing import Dict, Callable

# ============= 纯函数定义 =============

@jit
def E_z_lcdm(z: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
    """ΛCDM模型的E(z)计算"""
    one_plus_z = 1.0 + z
    matter = params['Omega_m'] * one_plus_z**3
    radiation = params.get('Omega_r', 0.0) * one_plus_z**4
    curvature = params.get('Omega_k', 0.0) * one_plus_z**2
    de = params['Omega_Lambda']
    return jnp.sqrt(matter + radiation + curvature + de)

@jit
def E_z_wcdm(z: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
    """wCDM模型的E(z)计算"""
    one_plus_z = 1.0 + z
    w = params['w']
    matter = params['Omega_m'] * one_plus_z**3
    de = params['Omega_Lambda'] * one_plus_z**(3*(1+w))
    return jnp.sqrt(matter + de)

# 通用函数（使用函数参数而非类方法）
@jit
def H_z(z: jnp.ndarray, params: Dict[str, float], E_z_func: Callable) -> jnp.ndarray:
    """通用哈勃参数计算"""
    return params['H0'] * E_z_func(z, params)

@jit
def comoving_distance_integrand(z: float, params: Dict[str, float], E_z_func: Callable) -> float:
    """共动距离被积函数"""
    return c_km_s / (params['H0'] * E_z_func(z, params))

# ============= 类作为接口层 =============

class LCDM:
    """ΛCDM模型类 - 仅作为接口"""
    
    def __init__(self, **params):
        self.params = params
        self._E_z_func = E_z_lcdm  # 存储对应的纯函数
        
    def E_z(self, z):
        """调用纯函数计算E(z)"""
        return self._E_z_func(z, self.params)
    
    def H_z(self, z):
        """调用纯函数计算H(z)"""
        return H_z(z, self.params, self._E_z_func)
    
    def comoving_distance(self, z, n_steps=1000):
        """使用纯函数进行积分"""
        from jax import vmap
        z_arr = jnp.linspace(0, z, n_steps)
        integrand = vmap(lambda zi: comoving_distance_integrand(zi, self.params, self._E_z_func))
        return jnp.trapz(integrand(z_arr), z_arr)

class wCDM:
    """wCDM模型类 - 仅作为接口"""
    
    def __init__(self, **params):
        self.params = params
        self._E_z_func = E_z_wcdm
        
    def E_z(self, z):
        return self._E_z_func(z, self.params)
    
    def H_z(self, z):
        return H_z(z, self.params, self._E_z_func)
```

### 优点
- ✅ 完全兼容JAX JIT编译
- ✅ 纯函数易于测试和优化
- ✅ 类仅作为接口，保持API友好
- ✅ 支持函数式编程范式

### 缺点
- ❌ 需要重构现有代码
- ❌ 失去部分面向对象的优雅性

---

## 解决方案二：策略模式

### 核心思想
使用策略模式，将不同模型的计算策略分离。

### 实现方案

```python
# hicosmo/core/strategies.py
"""计算策略定义"""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import jit

class CosmologyStrategy(ABC):
    """宇宙学计算策略基类"""
    
    @staticmethod
    @abstractmethod
    def compute_E_z(z: jnp.ndarray, params: dict) -> jnp.ndarray:
        """计算E(z) - 纯静态方法"""
        pass

class LCDMStrategy(CosmologyStrategy):
    """ΛCDM计算策略"""
    
    @staticmethod
    @jit
    def compute_E_z(z: jnp.ndarray, params: dict) -> jnp.ndarray:
        one_plus_z = 1.0 + z
        matter = params['Omega_m'] * one_plus_z**3
        de = params['Omega_Lambda']
        return jnp.sqrt(matter + de)

class wCDMStrategy(CosmologyStrategy):
    """wCDM计算策略"""
    
    @staticmethod
    @jit
    def compute_E_z(z: jnp.ndarray, params: dict) -> jnp.ndarray:
        one_plus_z = 1.0 + z
        w = params['w']
        matter = params['Omega_m'] * one_plus_z**3
        de = params['Omega_Lambda'] * one_plus_z**(3*(1+w))
        return jnp.sqrt(matter + de)

# hicosmo/models/cosmology.py
"""模型类使用策略"""

class CosmologyModel:
    """通用宇宙学模型"""
    
    def __init__(self, strategy: CosmologyStrategy, **params):
        self.strategy = strategy
        self.params = params
    
    def E_z(self, z):
        """使用策略计算E(z)"""
        return self.strategy.compute_E_z(z, self.params)
    
    @jit
    def H_z(self, z):
        """计算H(z) - 可以JIT编译"""
        E_z_val = self.strategy.compute_E_z(z, self.params)
        return self.params['H0'] * E_z_val

# 使用示例
lcdm = CosmologyModel(LCDMStrategy(), H0=70, Omega_m=0.3, Omega_Lambda=0.7)
wcdm = CosmologyModel(wCDMStrategy(), H0=70, Omega_m=0.3, Omega_Lambda=0.7, w=-0.9)
```

### 优点
- ✅ 清晰的关注点分离
- ✅ 易于添加新模型
- ✅ 策略可以独立测试

### 缺点
- ❌ 增加了额外的抽象层
- ❌ 可能过度设计

---

## 解决方案三：工厂模式 + 注册机制

### 核心思想
使用工厂模式创建模型，通过注册机制管理不同的实现。

### 实现方案

```python
# hicosmo/core/registry.py
"""模型注册机制"""

from typing import Dict, Callable
import jax.numpy as jnp
from jax import jit

# 全局注册表
MODEL_REGISTRY: Dict[str, Callable] = {}

def register_E_z(model_name: str):
    """装饰器：注册E(z)函数"""
    def decorator(func):
        MODEL_REGISTRY[model_name] = jit(func)
        return func
    return decorator

# 注册不同模型的E(z)函数
@register_E_z('LCDM')
def E_z_lcdm(z: jnp.ndarray, params: dict) -> jnp.ndarray:
    one_plus_z = 1.0 + z
    matter = params['Omega_m'] * one_plus_z**3
    de = params['Omega_Lambda']
    return jnp.sqrt(matter + de)

@register_E_z('wCDM')
def E_z_wcdm(z: jnp.ndarray, params: dict) -> jnp.ndarray:
    one_plus_z = 1.0 + z
    w = params['w']
    matter = params['Omega_m'] * one_plus_z**3
    de = params['Omega_Lambda'] * one_plus_z**(3*(1+w))
    return jnp.sqrt(matter + de)

# hicosmo/models/factory.py
"""模型工厂"""

class CosmologyFactory:
    """宇宙学模型工厂"""
    
    @staticmethod
    def create(model_type: str, **params):
        """创建模型实例"""
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_type}")
        
        class Model:
            def __init__(self):
                self.params = params
                self.E_z_func = MODEL_REGISTRY[model_type]
                self.model_type = model_type
            
            def E_z(self, z):
                return self.E_z_func(z, self.params)
            
            def H_z(self, z):
                return self.params['H0'] * self.E_z_func(z, self.params)
        
        return Model()

# 使用示例
lcdm = CosmologyFactory.create('LCDM', H0=70, Omega_m=0.3, Omega_Lambda=0.7)
wcdm = CosmologyFactory.create('wCDM', H0=70, Omega_m=0.3, Omega_Lambda=0.7, w=-0.9)
```

### 优点
- ✅ 灵活的注册机制
- ✅ 易于扩展新模型
- ✅ 代码组织清晰

### 缺点
- ❌ 失去了强类型检查
- ❌ 运行时错误而非编译时错误

---

## 解决方案四：混合方法（实用推荐）

### 核心思想
结合纯函数和类接口，保持API友好性的同时确保JAX兼容性。

### 实现方案

```python
# hicosmo/core/cosmology_v2.py
"""改进的宇宙学架构"""

import jax.numpy as jnp
from jax import jit, vmap
from abc import ABC, abstractmethod
from functools import partial

class CosmologyBase(ABC):
    """宇宙学模型基类 - 不使用JIT装饰器"""
    
    def __init__(self, **params):
        self.params = params
        self._validate_params()
        # 预编译常用函数
        self._compile_functions()
    
    @abstractmethod
    def _validate_params(self):
        """验证参数"""
        pass
    
    def _compile_functions(self):
        """预编译JIT函数"""
        # 为实例方法创建JIT编译版本
        self._E_z_jit = jit(self._E_z_impl)
        self._H_z_jit = jit(partial(self._H_z_impl, E_z_func=self._E_z_impl))
    
    @abstractmethod
    def _E_z_impl(self, z: jnp.ndarray, params: dict) -> jnp.ndarray:
        """E(z)的实现 - 子类必须实现"""
        pass
    
    def _H_z_impl(self, z: jnp.ndarray, params: dict, E_z_func) -> jnp.ndarray:
        """H(z)的实现 - 通用"""
        return params['H0'] * E_z_func(z, params)
    
    # 公开API
    def E_z(self, z):
        """计算E(z) - 调用JIT编译版本"""
        return self._E_z_jit(jnp.asarray(z), self.params)
    
    def H_z(self, z):
        """计算H(z) - 调用JIT编译版本"""
        return self._H_z_jit(jnp.asarray(z), self.params)
    
    def comoving_distance(self, z, n_steps=1000):
        """共动距离 - 使用JIT编译的积分"""
        @jit
        def integrate(z_max):
            z_arr = jnp.linspace(0, z_max, n_steps)
            integrand = vmap(lambda zi: c_km_s / self._H_z_jit(zi, self.params))
            return jnp.trapz(integrand(z_arr), z_arr)
        
        return integrate(z)

class LCDM(CosmologyBase):
    """ΛCDM模型实现"""
    
    def _validate_params(self):
        """验证ΛCDM参数"""
        # 参数验证逻辑
        pass
    
    def _E_z_impl(self, z: jnp.ndarray, params: dict) -> jnp.ndarray:
        """ΛCDM的E(z)实现"""
        one_plus_z = 1.0 + z
        matter = params['Omega_m'] * one_plus_z**3
        radiation = params.get('Omega_r', 0.0) * one_plus_z**4
        curvature = params.get('Omega_k', 0.0) * one_plus_z**2
        de = params['Omega_Lambda']
        return jnp.sqrt(matter + radiation + curvature + de)

class wCDM(CosmologyBase):
    """wCDM模型实现"""
    
    def _validate_params(self):
        """验证wCDM参数"""
        if 'w' not in self.params:
            raise ValueError("wCDM requires 'w' parameter")
    
    def _E_z_impl(self, z: jnp.ndarray, params: dict) -> jnp.ndarray:
        """wCDM的E(z)实现"""
        one_plus_z = 1.0 + z
        w = params['w']
        matter = params['Omega_m'] * one_plus_z**3
        radiation = params.get('Omega_r', 0.0) * one_plus_z**4
        de = params['Omega_Lambda'] * one_plus_z**(3*(1+w))
        return jnp.sqrt(matter + radiation + de)
```

### 优点
- ✅ 保持面向对象的接口
- ✅ 完全兼容JAX
- ✅ 预编译提高性能
- ✅ 易于理解和维护

### 缺点
- ❌ 需要小心管理JIT编译
- ❌ 实例方法不能直接用@jit装饰

---

## 推荐方案

对于HIcosmo项目，我推荐**混合方法（方案四）**，原因如下：

1. **保持API友好性**：用户仍然可以使用熟悉的面向对象接口
2. **JAX完全兼容**：通过预编译和函数分离确保JIT正常工作
3. **性能优化**：预编译关键函数，运行时性能最优
4. **易于维护**：代码结构清晰，容易理解和扩展
5. **渐进式重构**：可以逐步迁移现有代码

## 实施步骤

1. **第一步**：创建新的架构文件（不破坏现有代码）
2. **第二步**：实现核心功能的新版本
3. **第三步**：编写测试验证新旧版本一致性
4. **第四步**：逐步迁移其他功能
5. **第五步**：替换旧实现

这样可以确保平滑过渡，不会破坏现有功能。