import hicosmo as hc
hc.init(8)


from hicosmo.samplers import MCMC
from hicosmo.models import LCDM
from hicosmo.likelihoods import SN_likelihood, BAO_likelihood,Planck
from hicosmo.visualization import Plotter



# 创建似然函数
sne = SN_likelihood(LCDM, "pantheon+")
sne_shoes = SN_likelihood(LCDM, "pantheon+shoes")
cmb=Planck(LCDM)

# 参数配置
params = {
    'H0': {'init': 70, 'min': 60, 'max': 80},
    'Omega_m': {'init': 0.3, 'min': 0.1, 'max': 0.5},
}


mcmc = MCMC(params, cmb, chain_name='test_cmb')
samples = mcmc.run(num_samples=20000)


