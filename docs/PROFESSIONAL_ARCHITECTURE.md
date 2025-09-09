# HiCosmo 专业宇宙学架构设计

## 设计理念

HiCosmo 旨在成为一个全面的专业宇宙学计算库，支持从背景演化到结构形成的完整物理过程。

### 核心原则
1. **模块化设计**：不同物理过程分离为独立模块
2. **专业接口**：与 CAMB、CLASS、CCL 兼容的 API 设计
3. **JAX 优化**：全面利用 JAX 的性能优势
4. **扩展性**：易于添加新物理和新模型

## 架构概览

```
HiCosmo Architecture
├── Background (背景演化)
│   ├── Friedmann equations
│   ├── Distance calculations  
│   └── Thermodynamics
├── Perturbations (线性扰动)
│   ├── Scalar perturbations
│   ├── Vector perturbations
│   └── Tensor perturbations
├── Transfer (传递函数)
│   ├── Matter transfer
│   ├── CMB transfer
│   └── Lensing transfer
├── PowerSpectrum (功率谱)
│   ├── Linear P(k)
│   ├── Non-linear corrections
│   └── Cross-correlations
├── CMB (宇宙微波背景)
│   ├── Temperature Cl
│   ├── Polarization Cl
│   └── Lensing Cl
└── HighLevel (高级接口)
    ├── Parameters management
    ├── Model factory
    └── Analysis tools
```

## 详细设计

### 1. Background Module (背景演化)

```python
class BackgroundEvolution:
    \"\"\"
    背景宇宙学计算
    - 哈勃参数 H(z)
    - 距离计算
    - 时间演化
    - 热力学量
    \"\"\"
    
    def __init__(self, cosmology):
        self.cosmo = cosmology
        
    def H_z(self, z): 
        \"\"\"哈勃参数\"\"\"
        
    def comoving_distance(self, z):
        \"\"\"共动距离\"\"\"
        
    def sound_horizon(self, z):
        \"\"\"声学视界\"\"\"
        
    def baryon_drag_epoch(self):
        \"\"\"重子拖拽时期\"\"\"
```

### 2. Perturbations Module (扰动理论)

```python
class LinearPerturbations:
    \"\"\"
    线性扰动理论
    - 爱因斯坦方程
    - 流体方程
    - 玻尔兹曼方程
    \"\"\"
    
    def __init__(self, background):
        self.bg = background
        
    def solve_perturbations(self, k_modes, initial_conditions):
        \"\"\"求解扰动方程\"\"\"
        
    def matter_transfer_function(self, k, z):
        \"\"\"物质传递函数\"\"\"
        
    def photon_transfer_function(self, k, z):
        \"\"\"光子传递函数\"\"\"
```

### 3. PowerSpectrum Module (功率谱)

```python
class PowerSpectrum:
    \"\"\"
    功率谱计算
    - 线性功率谱
    - 非线性修正
    - 交叉功率谱
    \"\"\"
    
    def __init__(self, perturbations):
        self.pert = perturbations
        
    def linear_power(self, k, z):
        \"\"\"线性功率谱 P_lin(k,z)\"\"\"
        
    def nonlinear_power(self, k, z, method='halofit'):
        \"\"\"非线性功率谱 P_nl(k,z)\"\"\"
        
    def sigma8(self, z=0):
        \"\"\"σ8 参数\"\"\"
        
    def sigma_R(self, R, z=0):
        \"\"\"任意尺度上的 σ(R)\"\"\"
```

### 4. CMB Module (宇宙微波背景)

```python
class CMBCalculator:
    \"\"\"
    CMB 功率谱计算
    - 温度功率谱
    - 偏振功率谱  
    - 透镜功率谱
    \"\"\"
    
    def __init__(self, perturbations):
        self.pert = perturbations
        
    def temperature_cl(self, l_max=2500):
        \"\"\"温度功率谱 C_l^TT\"\"\"
        
    def polarization_cl(self, l_max=2500):
        \"\"\"偏振功率谱 C_l^EE, C_l^BB, C_l^TE\"\"\"
        
    def lensing_cl(self, l_max=2500):
        \"\"\"透镜功率谱 C_l^φφ\"\"\"
        
    def acoustic_peaks(self):
        \"\"\"声学峰位置\"\"\"
```

### 5. HighLevel Interface (高级接口)

```python
class Cosmology:
    \"\"\"
    统一的宇宙学计算接口
    整合所有模块，提供简单易用的 API
    \"\"\"
    
    def __init__(self, **params):
        self.params = CosmologyParameters(**params)
        
        # 初始化各个模块
        self.background = BackgroundEvolution(self)
        self.perturbations = LinearPerturbations(self.background)
        self.power_spectrum = PowerSpectrum(self.perturbations)
        self.cmb = CMBCalculator(self.perturbations)
        
    # 简化的接口方法
    def H(self, z):
        return self.background.H_z(z)
        
    def luminosity_distance(self, z):
        return self.background.luminosity_distance(z)
        
    def matter_power_spectrum(self, k, z=0):
        return self.power_spectrum.linear_power(k, z)
        
    def cmb_temperature_spectrum(self):
        return self.cmb.temperature_cl()
```

## 使用示例

### 基础使用
```python
# 创建宇宙学模型
cosmo = Cosmology(
    H0=67.4,
    Omega_m=0.315,
    Omega_b=0.049,
    n_s=0.965,
    sigma8=0.811
)

# 背景演化
z = jnp.linspace(0, 3, 100)
H_z = cosmo.H(z)
d_L = cosmo.luminosity_distance(z)

# 功率谱
k = jnp.logspace(-3, 1, 100)
P_k = cosmo.matter_power_spectrum(k, z=0)

# CMB 功率谱
l, Cl_TT = cosmo.cmb_temperature_spectrum()
```

### 高级使用
```python
# 直接访问模块
bg = cosmo.background
pert = cosmo.perturbations
ps = cosmo.power_spectrum

# 详细计算
transfer = pert.matter_transfer_function(k, z=0)
sigma8_z = ps.sigma8(z=1.0)
acoustic_scale = cosmo.cmb.acoustic_peaks()
```

## 与现有工具的兼容性

### CAMB 兼容接口
```python
def get_background_evolution(cosmo):
    \"\"\"返回类似 CAMB 的背景演化结果\"\"\"
    
def get_matter_power_spectrum(cosmo, kmax=10, npoints=200):
    \"\"\"返回类似 CAMB 的功率谱格式\"\"\"
    
def get_cmb_power_spectra(cosmo, lmax=2500):
    \"\"\"返回类似 CAMB 的 CMB 功率谱\"\"\"
```

### CCL 兼容接口
```python
def ccl_cosmology_from_params(**params):
    \"\"\"从参数创建类似 CCL 的宇宙学对象\"\"\"
    
def ccl_power_spectrum(cosmo, k, a):
    \"\"\"CCL 格式的功率谱计算\"\"\"
```

## 性能优化策略

### JAX 优化
1. **JIT 编译**：所有数值计算函数
2. **向量化**：使用 vmap 进行批量计算
3. **自动微分**：用于参数估计和 Fisher 矩阵
4. **GPU 加速**：大规模计算时的 GPU 支持

### 内存优化
1. **惰性计算**：只在需要时计算
2. **缓存机制**：缓存计算结果
3. **稀疏存储**：对于大矩阵使用稀疏格式

### 数值稳定性
1. **双精度计算**：关键计算使用双精度
2. **稳定的积分算法**：使用适应性积分
3. **边界条件处理**：正确处理极端参数

## 扩展性设计

### 新物理模型
```python
class ModifiedGravity(Cosmology):
    \"\"\"修正引力模型\"\"\"
    
    def effective_newton_constant(self, k, z):
        \"\"\"有效牛顿常数\"\"\"
        
    def modified_growth_rate(self, k, z):
        \"\"\"修正的增长率\"\"\"
```

### 新数据类型
```python
class WeakLensing(Cosmology):
    \"\"\"弱引力透镜计算\"\"\"
    
    def convergence_power_spectrum(self, l, z_source):
        \"\"\"收敛功率谱\"\"\"
        
    def shear_correlation_function(self, theta, z_bins):
        \"\"\"剪切关联函数\"\"\"
```

这个架构设计既保持了简单易用性，又提供了专业级的功能。它将成为 HiCosmo 的核心框架。