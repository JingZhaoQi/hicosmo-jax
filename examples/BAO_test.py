import hicosmo as hc
hc.init(8)


from hicosmo.samplers import MCMC
from hicosmo.models import LCDM
from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
from hicosmo.visualization import Plotter



# 创建似然函数
bao = BAO_likelihood(LCDM, "desi2024")

# 使用 BBN 先验
bao_bbn = BAO_likelihood(LCDM, "desi2024", omega_b_mode='bbn_prior')
# nuisance: Omega_b, 并对 Omega_b*h^2 施加 BBN 约束

# 固定 Omega_b（使用 Planck 值）
bao_fixed = BAO_likelihood(LCDM, "desi2024", omega_b_mode='fixed')

# 参数配置
params = {
    'H0': {'init': 70, 'min': 60, 'max': 80},
    'Omega_m': {'init': 0.3, 'min': 0.1, 'max': 0.5},
}



mcmc1 = MCMC(params, bao, chain_name='test_bao')
samples1 = mcmc1.run(num_samples=20000)


mcmc2 = MCMC(params, bao_bbn, chain_name='test_bao_bbn')
samples2 = mcmc2.run(num_samples=20000)


mcmc3 = MCMC(params, bao_fixed, chain_name='test_bao_fixed')
samples3 = mcmc3.run(num_samples=20000)