import numpy as np
from pathlib import Path

from hicosmo.forecasts import load_survey, IntensityMappingFisher
from hicosmo.visualization.core import FisherPlot


def test_ska1_fisher_contour_generation(tmp_path):
    survey = load_survey('ska1_mid_band2')
    calculator = IntensityMappingFisher(survey)
    param_result = calculator.parameter_forecast(['H0', 'Omega_m'])

    covariance = param_result['covariance']
    means = [survey.reference['H0'], survey.reference['Omega_m']]

    plotter = FisherPlot(means, covariance, ['H0', 'Omega_m'], legend='SKA1 forecast', nsample=10_000)
    outfile = tmp_path / 'ska1_contour.png'
    fig = plotter.figure(filename=str(outfile))
    assert fig is not None
    fig.canvas.draw()
    assert outfile.exists()
    assert outfile.stat().st_size > 0
