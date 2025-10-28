"""Generate SKA1 forecast figures from Red Book configurations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hicosmo.forecasts import load_survey, IntensityMappingFisher
from hicosmo.visualization.core import FisherPlot

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


def plot_fractional_errors(name: str, fisher_result) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    z = fisher_result.z
    ax.plot(z, 100 * fisher_result.sigma_ln_H, marker='o', label='H(z)')
    ax.plot(z, 100 * fisher_result.sigma_ln_DA, marker='s', label=r'$D_A(z)$')
    ax.plot(z, 100 * fisher_result.sigma_ln_fsigma8, marker='^', label=r'$f\sigma_8(z)$')
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Fractional error [%]')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f'{name}_fractional_errors.png', dpi=200)
    plt.close(fig)


def plot_parameter_contours(name: str, survey, param_result, params=('w0', 'wa')) -> None:
    idx = [param_result['params'].index(p) for p in params]
    covariance = param_result['covariance'][np.ix_(idx, idx)]
    labels = list(params)
    fiducial = [survey.reference.get(p, 0.0) for p in params]
    plot = FisherPlot(fiducial, covariance, labels, legend=name, nsample=50000)
    fig = plot.figure(filename=RESULTS_DIR / f'{name}_parameter_contours.png')
    plt.close(fig)


def main():
    for survey_name in ['ska1_mid_band2', 'ska1_wide_band1']:
        survey = load_survey(survey_name)
        calculator = IntensityMappingFisher(survey)
        fisher_result = calculator.forecast()
        plot_fractional_errors(survey_name, fisher_result)
        param_result = calculator.parameter_forecast(['H0', 'Omega_m', 'w0', 'wa'])
        plot_parameter_contours(survey_name, survey, param_result)


if __name__ == '__main__':
    main()
