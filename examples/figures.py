
from hicosmo.visualization import Plotter

plotter = Plotter(['test_sne','test_bao'],labels=['Pantheon+','DESIR1'])
plotter.corner()
plotter.report()
