# Pantheon+SH0ES 超新星数据说明

## 数据来源

官方数据发布仓库：https://github.com/PantheonPlusSH0ES/DataRelease

**数据版本**: Pantheon+ (Brout et al. 2022) + SH0ES (Riess et al. 2022)

## 文件列表

### 1. Pantheon+SH0ES.dat (1,701 SNe)
**总样本数**: 1,701 个超新星（表头1行 + 数据1,701行 = 1,702行）

**数据组成**:
- **Pantheon+ 样本**: 宇宙学样本，z ∈ [0.001, 2.26]
- **SH0ES Cepheid 宿主星系**: 42 个近邻 SNe Ia，用于距离阶梯标定

**关键列说明**:
- `CID`: 超新星标识符
- `IDSURVEY`: 巡天项目 ID（50=CfA, 51=CSP, 56=PS1, 等）
- `zHD/zCMB/zHEL`: Hubble diagram/CMB frame/heliocentric 红移
- `m_b_corr`: 校正后的 B 波段视星等
- `MU_SH0ES`: SH0ES 距离模数（对 Cepheid 宿主星系）
- `CEPH_DIST`: Cepheid 距离模数
- `IS_CALIBRATOR`: 是否为 Cepheid 宿主星系 (1=是, 0=否)
- `USED_IN_SH0ES_HF`: 是否用于 SH0ES Hubble flow 分析

### 2. Pantheon+SH0ES_STAT+SYS.cov (32 MB)
**完整协方差矩阵**: 1,701 × 1,701
- 包含统计误差 + 全部系统误差
- **推荐用于宇宙学参数约束**

### 3. Pantheon+SH0ES_STATONLY.cov (30 MB)
**仅统计误差协方差矩阵**: 1,701 × 1,701
- 仅包含统计误差
- 用于对比研究或敏感性测试

### 4. README
官方数据说明文件

## 使用方法

### Python 读取示例
```python
import numpy as np

# 读取数据（跳过表头）
data = np.genfromtxt('Pantheon+SH0ES.dat',
                     names=True,  # 使用表头作为列名
                     dtype=None,
                     encoding='utf-8')

# 读取协方差矩阵
cov_full = np.loadtxt('Pantheon+SH0ES_STAT+SYS.cov')
cov_stat = np.loadtxt('Pantheon+SH0ES_STATONLY.cov')

# 提取关键列
z_cmb = data['zCMB']
m_b_corr = data['m_b_corr']
is_calib = data['IS_CALIBRATOR']
```

### HIcosmo PantheonPlusLikelihood 使用
```python
from hicosmo.likelihoods import PantheonPlusLikelihood
from hicosmo.models import LCDM

# 初始化 likelihood（自动加载数据）
pantheon = PantheonPlusLikelihood(
    data_path='data/sne',
    include_shoes=True,        # 包含 SH0ES Cepheid 数据
    marginalize_M_B=True       # 边际化绝对星等
)

# 计算 log-likelihood
model = LCDM(H0=70.0, Omega_m=0.3)
loglike = pantheon.log_likelihood(model)
```

## 数据特性

### 红移分布
- **最小红移**: z_min ≈ 0.001 (SH0ES Cepheid 宿主)
- **最大红移**: z_max ≈ 2.26 (高红移 SNe Ia)
- **Cepheid 宿主数量**: 42 个 (IS_CALIBRATOR=1)
- **Hubble flow SNe 数量**: 1,659 个

### 协方差矩阵维度
- **矩阵大小**: 1,701 × 1,701 = 2,893,401 个元素
- **文件大小**:
  - STAT+SYS: ~32 MB (包含全部误差)
  - STATONLY: ~30 MB (仅统计误差)

## 引用文献

如果使用此数据，请引用：

1. **Pantheon+**:
   Brout, D., et al. 2022, ApJ, 938, 110
   "The Pantheon+ Analysis: Cosmological Constraints"

2. **SH0ES**:
   Riess, A. G., et al. 2022, ApJL, 934, L7
   "A Comprehensive Measurement of the Local Value of the Hubble Constant with 1 km/s/Mpc Uncertainty from the Hubble Space Telescope and the SH0ES Team"

## 更新日期

- 数据下载: 2024年9月17日
- 数据复制到 data/sne/: 2024年11月2日
