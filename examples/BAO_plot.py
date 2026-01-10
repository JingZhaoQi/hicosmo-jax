from hicosmo.visualization import Plotter

# Different omega_b_mode chains have different nuisance parameters:
# - h0rd mode: H0, Omega_m, H0_rd
# - bbn_prior mode: H0, Omega_m, Omega_b
# Only common cosmological parameters can be compared across modes
plotter = Plotter(['test_bao', 'test_bao_bbn', 'test_bao_fixed'], labels=['DESI', 'DESI+BBN','DESI+CMB'])
#plotter = Plotter(['test_bao', 'test_bao_bbn'], labels=['DESI', 'DESI+BBN'])
plotter.corner(['H0', 'Omega_m'])  # Only compare common cosmological params
plotter.report()
