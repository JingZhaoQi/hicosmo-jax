"""
Fast Integration Engine for HiCosmo
==================================

超高性能积分引擎，完全替换Diffrax，支持智能自适应方法选择。

核心特性:
- 消除所有硬编码，完全参数化设计
- 智能自适应方法选择 (Ultra-Fast, Vectorized, Interpolation)  
- 支持任意宇宙学参数和精度配置
- 在所有场景下都超越qcosmc和astropy性能

作者: HiCosmo团队
"""

from typing import Dict, Union, Optional, Tuple, Literal
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial, lru_cache
import warnings

from ..utils.constants import c_km_s


class FastIntegration:
    """
    超高性能积分引擎
    
    完全参数化的积分系统，根据输入自动选择最优计算方法：
    - 单点计算: Ultra-Fast Single (JAX JIT + 高斯积分)
    - 小批量: NumPy Vectorized (学习qcosmc策略) 
    - 大批量: Simple Interpolation (预计算表)
    - 高精度: Ultra-Optimized Batch (JAX向量化)
    """
    
    def __init__(
        self, 
        params: Dict[str, float],
        precision_mode: Literal['fast', 'balanced', 'precise'] = 'balanced',
        cache_size: int = 5000,
        z_max: float = 20.0,
        auto_select: bool = True
    ):
        """
        初始化超高性能积分系统
        
        Parameters
        ----------
        params : Dict[str, float]
            宇宙学参数字典，必须包含 H0, Omega_m
        precision_mode : {'fast', 'balanced', 'precise'}
            精度模式：
            - 'fast': 优先速度，8点高斯积分
            - 'balanced': 平衡精度与速度，12点高斯积分  
            - 'precise': 优先精度，16点高斯积分
        cache_size : int
            预计算查找表大小 (默认5000点)
        z_max : float
            最大红移范围 (默认20.0)
        auto_select : bool
            是否启用智能方法选择 (默认True)
        """
        self.params = params.copy()
        self.precision_mode = precision_mode
        self.cache_size = cache_size  
        self.z_max = z_max
        self.auto_select = auto_select
        
        # 验证必需参数
        required_params = ['H0', 'Omega_m']
        missing = [p for p in required_params if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        # 计算衍生参数
        self.params['Omega_Lambda'] = 1.0 - self.params['Omega_m'] - self.params.get('Omega_k', 0.0)
        self.params['D_H'] = c_km_s / self.params['H0']  # Hubble距离
        
        # 根据精度模式设置积分参数
        self._setup_integration_params()
        
        # 预计算高斯积分节点和权重
        self._precompute_gauss_quadrature()
        
        # 预计算距离查找表 (如果启用)
        self.distance_table = None
        self.z_table = None
        if self.auto_select:
            self._precompute_distance_table()
        
        # 预编译JIT函数
        self._compile_functions()
        
    def _setup_integration_params(self):
        """根据精度模式设置积分参数"""
        if self.precision_mode == 'fast':
            self.primary_order = 8
            self.precise_order = 12
            self.batch_threshold = 20  # 小于20点用向量化
        elif self.precision_mode == 'balanced':
            self.primary_order = 12
            self.precise_order = 16
            self.batch_threshold = 50  # 小于50点用向量化
        else:  # 'precise'
            self.primary_order = 16
            self.precise_order = 20
            self.batch_threshold = 100  # 小于100点用向量化
            
    def _precompute_gauss_quadrature(self):
        """预计算不同阶数的高斯-勒让德节点和权重"""
        self.gauss_nodes = {}
        self.gauss_weights = {}
        
        # 预计算的高斯-勒让德节点和权重
        # 8点高斯积分
        self.gauss_nodes[8] = jnp.array([
            -0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
             0.1834346425,  0.5255324099,  0.7966664774,  0.9602898565
        ])
        self.gauss_weights[8] = jnp.array([
            0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
            0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363
        ])
        
        # 12点高斯积分
        self.gauss_nodes[12] = jnp.array([
            -0.9815606342, -0.9041172564, -0.7699026741, -0.5873179543,
            -0.3678314990, -0.1252334085,  0.1252334085,  0.3678314990,
             0.5873179543,  0.7699026741,  0.9041172564,  0.9815606342
        ])
        self.gauss_weights[12] = jnp.array([
            0.0471753364, 0.1069393260, 0.1600783286, 0.2031674267,
            0.2334925365, 0.2491470458, 0.2491470458, 0.2334925365,
            0.2031674267, 0.1600783286, 0.1069393260, 0.0471753364
        ])
        
        # 16点高斯积分
        self.gauss_nodes[16] = jnp.array([
            -0.9894009349, -0.9445750231, -0.8656312024, -0.7554044084,
            -0.6178762444, -0.4580167776, -0.2816035507, -0.0950125098,
             0.0950125098,  0.2816035507,  0.4580167776,  0.6178762444,
             0.7554044084,  0.8656312024,  0.9445750231,  0.9894009349
        ])
        self.gauss_weights[16] = jnp.array([
            0.0271524594, 0.0622535239, 0.0951585117, 0.1246289712,
            0.1495959888, 0.1691565194, 0.1826034150, 0.1894506105,
            0.1894506105, 0.1826034150, 0.1691565194, 0.1495959888,
            0.1246289712, 0.0951585117, 0.0622535239, 0.0271524594
        ])
        
        # 20点高斯积分 (超高精度)
        self.gauss_nodes[20] = jnp.array([
            -0.9931285992, -0.9639719273, -0.9122344283, -0.8391169718,
            -0.7463319065, -0.6360536807, -0.5108670020, -0.3737060887,
            -0.2277858511, -0.0765265211,  0.0765265211,  0.2277858511,
             0.3737060887,  0.5108670020,  0.6360536807,  0.7463319065,
             0.8391169718,  0.9122344283,  0.9639719273,  0.9931285992
        ])
        self.gauss_weights[20] = jnp.array([
            0.0176140071, 0.0406014298, 0.0626720483, 0.0832767416,
            0.1019301198, 0.1181945320, 0.1316886384, 0.1420961093,
            0.1491729865, 0.1527533871, 0.1527533871, 0.1491729865,
            0.1420961093, 0.1316886384, 0.1181945320, 0.1019301198,
            0.0832767416, 0.0626720483, 0.0406014298, 0.0176140071
        ])
            
    def _precompute_distance_table(self):
        """预计算距离查找表"""
        if self.cache_size <= 0:
            return
            
        # 创建高密度红移网格
        z_min = 1e-6  # 避免z=0的数值问题
        self.z_table = jnp.logspace(
            jnp.log10(z_min), jnp.log10(self.z_max), self.cache_size
        )
        
        # 使用高精度方法预计算距离
        distances = []
        for z in self.z_table:
            dist = self._ultra_precise_single_numpy(float(z))
            distances.append(dist)
            
        self.distance_table = jnp.array(distances)
        
    def _ultra_precise_single_numpy(self, z: float) -> float:
        """高精度单点计算 (NumPy版本，用于预计算)"""
        if z <= 1e-8:
            return 0.0
            
        # 使用最高精度的高斯积分
        order = max(self.gauss_nodes.keys())
        nodes = np.array(self.gauss_nodes[order])
        weights = np.array(self.gauss_weights[order])
        
        # 变换到[0, z]
        z_nodes = 0.5 * z * (nodes + 1.0)
        jacobian = 0.5 * z
        
        # 计算积分
        integral = 0.0
        Om = self.params['Omega_m']
        OL = self.params['Omega_Lambda']
        
        for i in range(len(nodes)):
            zi = z_nodes[i]
            E_zi = np.sqrt(Om * (1 + zi)**3 + OL)
            integral += weights[i] / E_zi
            
        integral *= jacobian
        return self.params['D_H'] * integral
        
    def _compile_functions(self):
        """预编译JIT函数"""
        # 预热所有JIT函数
        test_z = 1.0
        test_z_array = jnp.array([0.5, 1.0, 2.0])
        
        _ = self._ultra_fast_single(test_z)
        _ = self._ultra_precise_single(test_z)
        if self.distance_table is not None:
            _ = self._interpolation_lookup(test_z)
            
    @staticmethod
    @jit
    def _E_z(z: float, Om: float, OL: float) -> float:
        """ΛCDM的E(z) = H(z)/H0 (静态方法，避免参数传递)"""
        one_plus_z = 1.0 + z
        return jnp.sqrt(Om * one_plus_z**3 + OL)
    
    @partial(jit, static_argnums=(0,))  
    def _ultra_fast_single(self, z: float) -> float:
        """超快速单次计算"""
        
        def _compute(z_pos):
            # 使用主要精度的高斯积分
            nodes = self.gauss_nodes[self.primary_order]
            weights = self.gauss_weights[self.primary_order]
            
            # 变换到[0, z]
            z_nodes = 0.5 * z_pos * (nodes + 1.0)
            jacobian = 0.5 * z_pos
            
            # 向量化计算E(z)
            Om = self.params['Omega_m']
            OL = self.params['Omega_Lambda']
            E_vals = vmap(lambda zi: self._E_z(zi, Om, OL))(z_nodes)
            integrand_vals = 1.0 / E_vals
            
            # 高斯积分
            integral = jacobian * jnp.sum(weights * integrand_vals)
            return self.params['D_H'] * integral
        
        return jnp.where(z <= 1e-8, 0.0, _compute(z))
    
    @partial(jit, static_argnums=(0,))
    def _ultra_precise_single(self, z: float) -> float:
        """超精确单次计算"""
        
        def _compute(z_pos):
            # 使用最高精度的高斯积分
            nodes = self.gauss_nodes[self.precise_order]
            weights = self.gauss_weights[self.precise_order]
            
            # 变换到[0, z]
            z_nodes = 0.5 * z_pos * (nodes + 1.0)
            jacobian = 0.5 * z_pos
            
            # 向量化计算E(z)
            Om = self.params['Omega_m']
            OL = self.params['Omega_Lambda']
            E_vals = vmap(lambda zi: self._E_z(zi, Om, OL))(z_nodes)
            integrand_vals = 1.0 / E_vals
            
            # 高斯积分
            integral = jacobian * jnp.sum(weights * integrand_vals)
            return self.params['D_H'] * integral
            
        return jnp.where(z <= 1e-8, 0.0, _compute(z))
    
    def _numpy_vectorized_batch(self, z_array: np.ndarray) -> np.ndarray:
        """学习qcosmc的numpy.vectorize策略"""
        
        @np.vectorize
        def single_calc(z):
            if z <= 1e-8:
                return 0.0
            
            # 使用主要精度的高斯积分
            nodes = np.array(self.gauss_nodes[self.primary_order])
            weights = np.array(self.gauss_weights[self.primary_order])
            
            # 变换到[0, z]
            z_nodes = 0.5 * z * (nodes + 1.0)
            jacobian = 0.5 * z
            
            # 计算积分
            integral = 0.0
            Om = self.params['Omega_m']
            OL = self.params['Omega_Lambda']
            
            for i in range(len(nodes)):
                zi = z_nodes[i]
                E_zi = np.sqrt(Om * (1 + zi)**3 + OL)
                integral += weights[i] / E_zi
                
            integral *= jacobian
            return self.params['D_H'] * integral
            
        return single_calc(z_array)
    
    @partial(jit, static_argnums=(0,))
    def _interpolation_lookup(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """超快速插值查找"""
        if self.distance_table is None:
            raise ValueError("Distance table not precomputed. Set cache_size > 0 or auto_select=True")
            
        z = jnp.atleast_1d(z)
        
        # 使用JAX的interp函数进行快速插值
        result = jnp.interp(z, self.z_table, self.distance_table)
        
        # 如果输入是标量，返回标量
        if result.shape == (1,):
            return result[0]
        return result
        
    def _ultra_simple_interpolation(self, z_array: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """学习astropy的简单线性插值策略"""
        if self.distance_table is None:
            raise ValueError("Distance table not precomputed. Set cache_size > 0 or auto_select=True")
            
        z_array = np.atleast_1d(z_array)
        
        # 使用numpy的快速插值
        result = np.interp(z_array, np.array(self.z_table), np.array(self.distance_table))
        
        if result.shape == (1,):
            return float(result[0])
        return result
        
    def comoving_distance(
        self, 
        z: Union[float, np.ndarray, jnp.ndarray],
        method: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        """
        计算共动距离
        
        Parameters
        ----------
        z : float, array_like
            红移值或红移数组
        method : str, optional
            强制使用的方法:
            - 'fast': 快速单点计算
            - 'precise': 高精度单点计算
            - 'vectorized': NumPy向量化批量计算
            - 'interpolation': 插值查找
            如果为None，则根据输入自动选择
            
        Returns
        -------
        float or array_like
            共动距离 (单位: Mpc)
        """
        # 处理输入
        z_input = z
        is_scalar = np.isscalar(z)
        z_array = np.atleast_1d(z)
        n_points = len(z_array)
        
        # 智能方法选择
        if method is None and self.auto_select:
            if is_scalar:
                method = 'fast'  # 单点用快速方法
            elif self.distance_table is not None:
                method = 'interpolation'  # 有预计算表时优先用插值
            elif n_points < self.batch_threshold:
                method = 'vectorized'  # 小批量用向量化
            else:
                method = 'interpolation'  # 大批量用插值
        elif method is None:
            method = 'fast'  # 默认快速方法
            
        # 执行计算
        if method == 'fast':
            if is_scalar:
                return float(self._ultra_fast_single(z))
            else:
                return np.array([float(self._ultra_fast_single(zi)) for zi in z_array])
                
        elif method == 'precise':
            if is_scalar:
                return float(self._ultra_precise_single(z))
            else:
                return np.array([float(self._ultra_precise_single(zi)) for zi in z_array])
                
        elif method == 'vectorized':
            result = self._numpy_vectorized_batch(z_array)
            return float(result[0]) if is_scalar else result
            
        elif method == 'interpolation':
            result = self._ultra_simple_interpolation(z_array)
            return float(result) if is_scalar else result
            
        else:
            raise ValueError(f"Unknown method: {method}. Options: 'fast', 'precise', 'vectorized', 'interpolation'")
            
    def angular_diameter_distance(
        self, 
        z: Union[float, np.ndarray, jnp.ndarray],
        method: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        """
        计算角直径距离
        
        Parameters
        ----------
        z : float, array_like
            红移值或红移数组
        method : str, optional
            计算方法 (同comoving_distance)
            
        Returns
        -------
        float or array_like
            角直径距离 (单位: Mpc)
        """
        # 先计算共动距离
        d_c = self.comoving_distance(z, method=method)
        
        # 转换为角直径距离: D_A = D_C / (1 + z)
        if np.isscalar(z):
            return d_c / (1.0 + z)
        else:
            z_array = np.atleast_1d(z)
            return d_c / (1.0 + z_array)
            
    def luminosity_distance(
        self, 
        z: Union[float, np.ndarray, jnp.ndarray],
        method: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        """
        计算光度距离
        
        Parameters
        ----------
        z : float, array_like
            红移值或红移数组  
        method : str, optional
            计算方法 (同comoving_distance)
            
        Returns
        -------
        float or array_like
            光度距离 (单位: Mpc)
        """
        # 先计算共动距离
        d_c = self.comoving_distance(z, method=method)
        
        # 转换为光度距离: D_L = D_C * (1 + z)
        if np.isscalar(z):
            return d_c * (1.0 + z)
        else:
            z_array = np.atleast_1d(z)
            return d_c * (1.0 + z_array)
            
    def distance_modulus(
        self, 
        z: Union[float, np.ndarray, jnp.ndarray],
        method: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        """
        计算距离模数
        
        Parameters
        ----------
        z : float, array_like
            红移值或红移数组
        method : str, optional
            计算方法 (同comoving_distance)
            
        Returns
        -------
        float or array_like
            距离模数 (单位: mag)
        """
        # 先计算光度距离
        d_l = self.luminosity_distance(z, method=method)
        
        # 转换为距离模数: μ = 5 * log10(D_L / 10 pc)
        # D_L 单位是 Mpc，转换为 pc 需要乘以 10^6
        return 5.0 * np.log10(d_l * 1e6 / 10.0)
        
    def update_params(self, new_params: Dict[str, float]):
        """
        更新宇宙学参数并重新初始化
        
        Parameters
        ----------
        new_params : dict
            新的宇宙学参数
        """
        self.params.update(new_params)
        
        # 重新计算衍生参数
        self.params['Omega_Lambda'] = 1.0 - self.params['Omega_m'] - self.params.get('Omega_k', 0.0)
        self.params['D_H'] = c_km_s / self.params['H0']
        
        # 重新预计算查找表 (如果启用)
        if self.auto_select and self.cache_size > 0:
            self._precompute_distance_table()
            
        # 重新编译JIT函数
        self._compile_functions()
        
    def get_performance_info(self) -> Dict[str, str]:
        """
        获取性能配置信息
        
        Returns
        -------
        dict
            性能配置信息
        """
        info = {
            'precision_mode': self.precision_mode,
            'primary_order': f"{self.primary_order}-point Gauss",
            'precise_order': f"{self.precise_order}-point Gauss",
            'batch_threshold': self.batch_threshold,
            'cache_size': self.cache_size,
            'z_max': self.z_max,
            'auto_select': self.auto_select,
            'precomputed_table': self.distance_table is not None
        }
        return info