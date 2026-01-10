
from hicosmo.visualization import Plotter

plotter = Plotter(['test_sn','test_bao','test_cmb','test_com'],labels=['Pantheon+','DESI','Planck 2018 Prior','SN+BAO+CMB'])
plotter.corner()
plotter.report()
