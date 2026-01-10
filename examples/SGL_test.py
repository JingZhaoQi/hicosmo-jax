import hicosmo as hc
hc.init(8)


from hicosmo.samplers import MCMC
from hicosmo.models import LCDM
from hicosmo.likelihoods import TDCOSMO,H0LiCOW
from hicosmo.visualization import Plotter

holicow = H0LiCOW(LCDM)
tdcosmo = TDCOSMO(LCDM)

# 创建似然函数

# 参数配置
params = {
    'H0': {'init': 70, 'min': 60, 'max': 80},
    'Omega_m': {'init': 0.3, 'min': 0.1, 'max': 0.5},
}

# 使用 + 运算符组合似然
mcmc = MCMC(params, holicow, chain_name='h0licow_test')
samples = mcmc.run(num_samples=20000)


mcmc1 = MCMC(params, tdcosmo, chain_name='tdcosmo_test')
samples1 = mcmc1.run(num_samples=20000)