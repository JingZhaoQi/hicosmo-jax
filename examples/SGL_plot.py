
from hicosmo.visualization import Plotter

plotter = Plotter(['h0licow_test','tdcosmo_test'],labels=['H0LiCOW','TDCOSMO'])
plotter.corner()
plotter.report()
