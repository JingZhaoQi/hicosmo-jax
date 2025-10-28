"""Generate SKA1 forecast figures from Red Book configurations.

This example demonstrates the CORRECT way to use IntensityMappingFisher:
1. Load survey configuration (hardware ONLY)
2. Define cosmological model separately
3. Pass cosmology explicitly to Fisher calculator

This allows testing multiple models with the same survey.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hicosmo.models import CPL, wCDM, LCDM
from hicosmo.forecasts import load_survey, IntensityMappingFisher
from hicosmo.visualization.core import FisherPlot

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# Fiducial cosmology for CPL model (Bull 2016, Planck 2018)
FIDUCIAL_CPL = {
    'H0': 67.36,
    'Omega_m': 0.316,
    'Omega_b': 0.049,
    'sigma8': 0.834,
    'n_s': 0.962,
    'w0': -1.0,
    'wa': 0.0
}


def plot_fractional_errors(name: str, fisher_result) -> None:
    """Plot fractional errors in H(z), D_A(z), fsigma8(z)."""
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


def plot_parameter_contours(name: str, param_result, fiducial_params: dict, params=('w0', 'wa')) -> None:
    """Plot parameter constraint contours."""
    idx = [param_result['params'].index(p) for p in params]
    covariance = param_result['covariance'][np.ix_(idx, idx)]
    labels = list(params)
    fiducial = [fiducial_params[p] for p in params]  # Use explicit fiducial
    plot = FisherPlot(fiducial, covariance, labels, legend=name, nsample=50000)
    fig = plot.figure(filename=RESULTS_DIR / f'{name}_parameter_contours.png')
    plt.close(fig)


def run_single_survey(survey_name: str, cosmology, fiducial_params: dict):
    """Run Fisher forecast for a single survey with given cosmology.

    Parameters
    ----------
    survey_name : str
        Survey configuration name (e.g., 'ska1_mid_band2')
    cosmology : CosmologyBase
        Cosmological model instance
    fiducial_params : dict
        Fiducial parameter values for plotting

    Returns
    -------
    dict
        Fisher forecast results
    """
    print(f"\n{'='*60}")
    print(f"Running forecast for {survey_name}")
    print(f"Cosmology: {cosmology.__class__.__name__}")
    print(f"Parameters: {fiducial_params}")
    print(f"{'='*60}\n")

    # Load survey (only hardware config)
    survey = load_survey(survey_name)

    # Create Fisher calculator with EXPLICIT cosmology
    calculator = IntensityMappingFisher(survey, cosmology)

    # Run forecasts
    fisher_result = calculator.forecast()
    plot_fractional_errors(survey_name, fisher_result)

    param_result = calculator.parameter_forecast(['H0', 'Omega_m', 'w0', 'wa'])
    plot_parameter_contours(survey_name, param_result, fiducial_params)

    print(f"✓ Forecast completed for {survey_name}")
    print(f"  Figures saved to {RESULTS_DIR}/")

    return param_result


def run_multi_model_comparison(survey_name: str):
    """Compare different cosmological models for the same survey.

    This demonstrates the power of the new architecture:
    - Same hardware configuration
    - Multiple cosmological models
    - Easy comparison
    """
    print(f"\n{'='*60}")
    print(f"Multi-Model Comparison: {survey_name}")
    print(f"{'='*60}\n")

    survey = load_survey(survey_name)

    # Define multiple models with same fiducial
    models = {
        'LCDM': LCDM(H0=67.36, Omega_m=0.316),
        'wCDM': wCDM(H0=67.36, Omega_m=0.316, w=-1.0),
        'CPL': CPL(H0=67.36, Omega_m=0.316, w0=-1.0, wa=0.0),
    }

    results = {}
    for model_name, cosmology in models.items():
        print(f"\nTesting {model_name} model...")
        calculator = IntensityMappingFisher(survey, cosmology)
        results[model_name] = calculator.parameter_forecast(['H0', 'Omega_m'])
        print(f"  ✓ {model_name} constraints computed")

    print(f"\n{'='*60}")
    print("Multi-model comparison completed!")
    print(f"{'='*60}\n")

    return results


def main():
    """Main execution: Run forecasts for SKA1 surveys."""

    # 1. Single model forecasts for both surveys
    print("\n" + "="*60)
    print("SKA1 Fisher Forecasts - CPL Model")
    print("="*60)

    # Create CPL cosmology with fiducial parameters
    cosmo_cpl = CPL(**{k: v for k, v in FIDUCIAL_CPL.items() if k in ['H0', 'Omega_m', 'Omega_b', 'w0', 'wa']})

    for survey_name in ['ska1_mid_band2', 'ska1_wide_band1']:
        run_single_survey(survey_name, cosmo_cpl, FIDUCIAL_CPL)

    # 2. Multi-model comparison (demonstrates new architecture benefit)
    print("\n\n" + "="*60)
    print("BONUS: Multi-Model Comparison")
    print("="*60)
    print("\nDemonstrating the new architecture:")
    print("- Same survey hardware")
    print("- Test LCDM, wCDM, CPL models")
    print("- Easy comparison\n")

    run_multi_model_comparison('ska1_mid_band2')

    print("\n" + "="*60)
    print("All forecasts completed successfully!")
    print(f"Results saved to: {RESULTS_DIR.absolute()}/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
