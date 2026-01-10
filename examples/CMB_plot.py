
from hicosmo.visualization import Plotter

plotter = Plotter(['test_cmb'],labels=['Planck 2018 Prior'])
plotter.corner()
plotter.report()
