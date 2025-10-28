import numpy as np

from hicosmo.forecasts import (
    list_available_surveys,
    load_survey,
    IntensityMappingFisher,
)


def test_list_available_surveys_reports_ska1():
    surveys = list_available_surveys()
    assert 'ska1_mid_band2' in surveys


def test_forecast_runs_and_returns_positive_errors():
    survey = load_survey('ska1_mid_band2')
    calculator = IntensityMappingFisher(survey)
    fisher = calculator.forecast()

    assert fisher.z.shape[0] == len(survey.redshift_bins)
    assert fisher.covariance_blocks.shape == (len(survey.redshift_bins), 3, 3)
    assert np.all(fisher.sigma_ln_H > 0)
    assert np.all(np.isfinite(fisher.sigma_ln_H))

    param_result = calculator.parameter_forecast(['H0', 'Omega_m', 'w0', 'wa'])
    cov_params = param_result['covariance']
    assert cov_params.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(cov_params) > 0)
