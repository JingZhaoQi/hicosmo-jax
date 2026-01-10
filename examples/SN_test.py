import hicosmo as hc
hc.init(8)


from hicosmo.samplers import MCMC
from hicosmo.models import LCDM
from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
from hicosmo.visualization import Plotter



# 创建似然函数
sne = SN_likelihood(LCDM, "pantheon+")
sne_shoes = SN_likelihood(LCDM, "pantheon+shoes")


# 参数配置
params = {
    'H0': {'init': 70, 'min': 60, 'max': 80},
    'Omega_m': {'init': 0.3, 'min': 0.1, 'max': 0.5},
}


mcmc = MCMC(params, sne, chain_name='test_sn')
samples = mcmc.run(num_samples=20000)

mcmc1 = MCMC(params, sne_shoes, chain_name='test_sne_shoes')
samples1 = mcmc1.run(num_samples=20000)

