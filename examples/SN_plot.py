
from hicosmo.visualization import Plotter

plotter = Plotter(['test_sn','test_sne_shoes'],labels=['Pantheon+','Pantheon_SH0ES'])
plotter.corner()
plotter.report()
