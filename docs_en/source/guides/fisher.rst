Fisher Forecasts
================

HIcosmo provides professional Fisher matrix analysis tools, supporting 21cm intensity mapping survey forecasts, multi-probe joint analysis, and parameter constraint predictions.

Fisher Matrix Fundamentals
--------------------------

Fisher Information Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~

The Fisher information matrix quantifies the curvature of the likelihood function in parameter space, defined as:

.. math::

   F_{ij} = -\left\langle \frac{\partial^2 \ln \mathcal{L}}{\partial \theta_i \partial \theta_j} \right\rangle

For a Gaussian likelihood, the Fisher matrix can be computed from the covariance matrix :math:`C` and the theoretical prediction :math:`\mu`:

.. math::

   F_{ij} = \frac{1}{2} \text{Tr}\left[ C^{-1} \frac{\partial C}{\partial \theta_i} C^{-1} \frac{\partial C}{\partial \theta_j} \right] + \frac{\partial \mu^T}{\partial \theta_i} C^{-1} \frac{\partial \mu}{\partial \theta_j}

Parameter Constraints
~~~~~~~~~~~~~~~~~~~~~

The :math:`1\sigma` constraints on parameters are given by the covariance matrix:

.. math::

   \text{Cov}(\theta) = F^{-1}

.. math::

   \sigma(\theta_i) = \sqrt{(F^{-1})_{ii}}

**Marginalized constraints**: When nuisance parameters exist, the constraints on target parameters require integration over nuisance parameters:

.. math::

   \text{Cov}_{\text{target}} = (F^{-1})_{\text{target block}}

**Conditional constraints**: Constraints are tighter when other parameters are fixed:

.. math::

   \sigma_{\text{cond}}(\theta_i) = \frac{1}{\sqrt{F_{ii}}}

Quick Start
-----------

**3 lines to complete a Fisher forecast**:

.. code-block:: python

   from hicosmo.models import CPL
   from hicosmo.fisher import IntensityMappingFisher

   result = IntensityMappingFisher.forecast('ska1_mid_band2', CPL(), ['w0', 'wa'])
   print(result)

IntensityMappingFisher Class
----------------------------

``IntensityMappingFisher`` is the core class for 21cm intensity mapping Fisher forecasts, implementing the Bull et al. (2016) method.

Class Method Interface (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import ILCDM
   from hicosmo.fisher import IntensityMappingFisher

   # 3 lines to complete a forecast
   ilcdm = ILCDM(H0=67.36, Omega_m=0.3153, beta=0.001)
   result = IntensityMappingFisher.forecast('tianlai', ilcdm, ['beta', 'H0'])
   print(result)

**Example Output**:

.. code-block:: text

   ======================================================================
   Fisher Forecast Results: tianlai
   ======================================================================
   Cosmology: ILCDM
   Redshift bins: 3
   Parameters: beta, H0

   1σ Constraints:
   ----------------------------------------------------------------------
     σ(beta      ) =    1.340000e-02
     σ(H0        ) =    8.250000e-01
   ======================================================================

Instance Interface
~~~~~~~~~~~~~~~~~~

Use the instance interface when multiple forecasts are needed:

.. code-block:: python

   from hicosmo.fisher import IntensityMappingFisher, load_survey
   from hicosmo.models import CPL

   # Load survey configuration
   survey = load_survey('ska1_mid_band2')
   cosmology = CPL(H0=67.36, Omega_m=0.3153, w0=-1.0, wa=0.0)

   # Create Fisher calculator
   fisher = IntensityMappingFisher(survey, cosmology)

   # Multiple forecasts
   result1 = fisher.parameter_forecast(['w0', 'wa'])
   result2 = fisher.parameter_forecast(['H0', 'Omega_m'])

With Marginalization
~~~~~~~~~~~~~~~~~~~~

Marginalize over nuisance parameters:

.. code-block:: python

   result = IntensityMappingFisher.forecast(
       'ska1_mid_band2',
       cpl_model,
       params=['w0', 'wa'],
       marginalize_over=['H0', 'Omega_m', 'gamma']
   )

   print(f"σ(w0) = {result.errors['w0']:.4f}")
   print(f"σ(wa) = {result.errors['wa']:.4f}")

FisherForecastResult Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The result object returned by ``forecast()`` contains:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Attribute
     - Type
     - Description
   * - ``params``
     - list
     - List of parameter names
   * - ``errors``
     - dict
     - :math:`1\sigma` constraints ``{param: σ}``
   * - ``fisher_matrix``
     - ndarray
     - Fisher matrix
   * - ``covariance``
     - ndarray
     - Covariance matrix
   * - ``correlation``
     - ndarray
     - Correlation matrix
   * - ``survey_name``
     - str
     - Survey name
   * - ``n_bins``
     - int
     - Number of redshift bins

.. code-block:: python

   # Access results
   print(f"σ(w0) = {result.errors['w0']:.4f}")
   print(f"w0-wa correlation coefficient: {result.correlation[0,1]:.3f}")

   # Convert to dictionary (compatible with legacy API)
   data = result.to_dict()

Survey Configurations
---------------------

HIcosmo includes built-in configurations for multiple 21cm intensity mapping surveys.

Available Surveys
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.fisher import list_available_surveys

   surveys = list_available_surveys()
   print(surveys)
   # ['ska1_mid_band2', 'ska1_wide_band1', 'tianlai', 'bingo', 'meerkat', 'chime', ...]

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Survey
     - Redshift Range
     - Features
   * - **ska1_mid_band2**
     - 0.0 - 0.5
     - SKA1-MID medium-deep, Band 2
   * - **ska1_wide_band1**
     - 0.4 - 1.0
     - SKA1-MID wide field, Band 1
   * - **tianlai**
     - 0.8 - 1.2
     - Tianlai experiment, cylinder array
   * - **bingo**
     - 0.13 - 0.45
     - Brazilian 21cm project
   * - **meerkat**
     - 0.0 - 0.6
     - South African MeerKAT array
   * - **chime**
     - 0.8 - 2.5
     - Canadian CHIME cylinder array

Loading a Survey
~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.fisher import load_survey

   survey = load_survey('ska1_mid_band2')

   # View survey information
   print(f"Name: {survey.name}")
   print(f"Description: {survey.description}")
   print(f"Redshift bins: {len(survey.redshift_bins)}")
   print(f"Number of dishes: {survey.instrument.ndish}")
   print(f"Survey area: {survey.instrument.survey_area_deg2} deg2")

Survey Configuration Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Surveys are configured in YAML format (located in ``hicosmo/fisher/surveys/``):

.. code-block:: yaml

   name: ska1_mid_band2
   description: "SKA1-MID Medium-Deep Band 2 survey"

   # Hardware configuration
   instrument:
     telescope_type: single_dish
     ndish: 133
     dish_diameter_m: 15.0
     nbeam: 1

   # Observing strategy
   observing:
     survey_area_deg2: 5000.0
     total_time_hours: 10000.0
     channel_width_hz: 50000.0

   # Noise parameters
   noise:
     system_temperature_K: 13.5
     sky_temperature:
       model: power_law
       T_ref_K: 25.0
       nu_ref_MHz: 408.0
       beta: 2.75

   # HI tracers
   hi_tracers:
     bias:
       model: polynomial
       coefficients: [0.67, 0.18, 0.05]
     density:
       model: polynomial
       coefficients: [0.00048, 0.00039, -0.000065]

   # Redshift binning
   redshift_bins:
     default_delta_z: 0.1
     centers: [0.05, 0.15, 0.25, 0.35, 0.45]

21cm Physics
------------

Brightness Temperature
~~~~~~~~~~~~~~~~~~~~~~

21cm brightness temperature formula (Santos et al. 2015):

.. math::

   T_b(z) = 27 \, \text{mK} \times h \times \frac{\Omega_{\text{HI}}}{10^{-3}} \times \frac{(1+z)^2}{E(z)}

Where :math:`\Omega_{\text{HI}}` is the HI density parameter and :math:`E(z) = H(z)/H_0`.

Power Spectrum
~~~~~~~~~~~~~~

The 21cm power spectrum is:

.. math::

   P_{21}(k, \mu, z) = T_b^2 \, b^2 \, (1 + \beta \mu^2)^2 \, P_m(k, z)

Where:
- :math:`b(z)` is the HI bias factor
- :math:`\beta = f/b` is the redshift-space distortion parameter
- :math:`f = d\ln D/d\ln a` is the growth rate
- :math:`\mu = \cos\theta` is the cosine of the angle to the line of sight
- :math:`P_m(k,z)` is the matter power spectrum

Effective Volume
~~~~~~~~~~~~~~~~

The Fisher information-weighted effective volume:

.. math::

   V_{\text{eff}}(k, \mu, z) = V_{\text{survey}} \left[ \frac{P_{\text{signal}}}{P_{\text{signal}} + P_{\text{noise}}} \right]^2

Noise Power Spectrum
~~~~~~~~~~~~~~~~~~~~

Instrumental noise power spectrum:

.. math::

   P_{\text{noise}} = \frac{\sigma_{\text{pix}}^2 V_{\text{pix}}}{W^2(k_\perp)}

Where :math:`W(k_\perp)` is the beam window function and :math:`\sigma_{\text{pix}}` is the pixel noise.

FisherMatrix Class
------------------

For general Fisher matrix computations (not 21cm-specific), use the ``FisherMatrix`` class.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from hicosmo.fisher import FisherMatrix, FisherMatrixConfig

   # Configuration
   config = FisherMatrixConfig(
       differentiation_method='automatic',  # Use JAX automatic differentiation
       regularization=True
   )

   # Create calculator
   calculator = FisherMatrix(config)

   # Compute Fisher matrix
   fisher_matrix, param_names = calculator.compute_fisher_matrix(
       likelihood_func=my_likelihood,
       parameters=params,
       fiducial_values={'H0': 67.36, 'Omega_m': 0.315}
   )

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.fisher import FisherMatrixConfig

   config = FisherMatrixConfig(
       # Differentiation method
       differentiation_method='automatic',  # 'automatic', 'numerical', 'hybrid'
       step_size=1e-6,
       fallback_to_numerical=True,

       # Regularization
       regularization=True,
       reg_lambda=1e-10,
       svd_threshold=1e-12,

       # Validation
       check_symmetry=True,
       check_positive_definite=True,

       # Performance
       vectorize_derivatives=True,
       cache_derivatives=True
   )

Matrix Combination
~~~~~~~~~~~~~~~~~~

Combine multi-probe Fisher matrices:

.. code-block:: python

   # Compute Fisher matrices for each probe
   fisher_bao, params_bao = calculator.compute_fisher_matrix(bao_likelihood, ...)
   fisher_sn, params_sn = calculator.compute_fisher_matrix(sn_likelihood, ...)

   # Combine
   combined_fisher, combined_params = calculator.combine_fisher_matrices(
       [(fisher_bao, params_bao), (fisher_sn, params_sn)],
       probe_names=['BAO', 'SN']
   )

Parameter Marginalization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Marginalize over nuisance parameters
   marginalized_fisher, remaining_params = calculator.marginalize_parameters(
       fisher_matrix,
       param_names,
       marginalize_over=['M_B', 'delta_M']
   )

Dark Energy Figure of Merit
----------------------------

The constraining power of dark energy experiments is quantified by the Figure of Merit (FoM):

.. math::

   \text{FoM} = \frac{1}{\sqrt{\det(\text{Cov}_{w_0, w_a})}}

This is equivalent to the inverse of the area enclosed by the :math:`1\sigma` contour in the :math:`w_0`-:math:`w_a` plane.

.. code-block:: python

   from hicosmo.models import CPL
   from hicosmo.fisher import IntensityMappingFisher
   import numpy as np

   # Forecast w0-wa constraints
   cpl = CPL(H0=67.36, Omega_m=0.3153, w0=-1.0, wa=0.0)
   result = IntensityMappingFisher.forecast('ska1_mid_band2', cpl, ['w0', 'wa'])

   # Compute FoM
   fom = 1.0 / np.sqrt(np.linalg.det(result.covariance))
   print(f"Dark Energy FoM: {fom:.1f}")

Growth Parameter Forecasts
--------------------------

HIcosmo supports growth parameter forecasts for modified gravity theories.

Growth Parameterization
~~~~~~~~~~~~~~~~~~~~~~~

The growth rate is parameterized as:

.. math::

   f(z) = \Omega_m(z)^{\gamma_0 + \gamma_1 z}

The deviation parameters are:

.. math::

   \eta(z) = \eta_0 + \eta_1 z

In General Relativity: :math:`\gamma_0 = 0.55`, :math:`\gamma_1 = \eta_0 = \eta_1 = 0`.

Growth Scenarios
~~~~~~~~~~~~~~~~

HIcosmo implements the four growth scenarios from Bull et al. (2016):

.. code-block:: python

   from hicosmo.fisher.intensity_mapping import GrowthScenario

   # Available scenarios
   scenarios = [
       'gr',           # General Relativity (no free growth parameters)
       'gamma0_free',  # gamma_0 free, others fixed
       'gamma1_free',  # gamma_1 free, others fixed
       'eta0_free',    # eta_0 free, others fixed
       'eta1_free',    # eta_1 free, others fixed
       'all_free',     # All growth parameters free
   ]

   # Use a growth scenario
   scenario = GrowthScenario('gamma0_free')
   print(f"Free parameters: {scenario.free_params}")
   print(f"Fixed parameters: {scenario.fixed_params}")

Visualization
-------------

Visualizing Fisher forecast results:

.. code-block:: python

   from hicosmo.visualization import Plotter, plot_fisher_contours
   import numpy as np

   # Method 1: Using Plotter.from_fisher
   plotter = Plotter.from_fisher(
       mean=result.cosmology_summary['fiducial'],
       covariance=result.covariance,
       param_names=result.params
   )
   plotter.corner(filename='forecast.pdf')

   # Method 2: Using standalone function
   plot_fisher_contours(
       mean=[-1.0, 0.0],
       covariance=result.covariance,
       param_names=['w0', 'wa'],
       filename='de_forecast.pdf'
   )

Multi-Survey Comparison
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import CPL
   from hicosmo.fisher import IntensityMappingFisher
   from hicosmo.visualization import Plotter

   cpl = CPL()
   surveys = ['ska1_mid_band2', 'tianlai', 'bingo']
   results = []

   for survey in surveys:
       result = IntensityMappingFisher.forecast(survey, cpl, ['w0', 'wa'])
       results.append(result)

   # Comparison plot (using Plotter.from_fisher multiple times)
   # Or manually generate Gaussian samples and use Plotter

Complete Examples
-----------------

SKA1 Dark Energy Forecast
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import CPL
   from hicosmo.fisher import IntensityMappingFisher, list_available_surveys
   import numpy as np

   # Define the cosmological model
   fiducial = CPL(
       H0=67.36,
       Omega_m=0.3153,
       Omega_b=0.0493,
       w0=-1.0,
       wa=0.0,
       sigma8=0.811,
       n_s=0.9649
   )

   # Forecast w0-wa constraints
   result = IntensityMappingFisher.forecast(
       'ska1_mid_band2',
       fiducial,
       params=['w0', 'wa'],
       marginalize_over=['H0', 'Omega_m']
   )

   # Print results
   print(result)

   # Compute FoM
   fom = 1.0 / np.sqrt(np.linalg.det(result.covariance))
   print(f"\nDark Energy FoM: {fom:.1f}")

   # Correlation coefficient
   print(f"w0-wa correlation: {result.correlation[0,1]:.3f}")

Interacting Dark Energy Forecast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import ILCDM
   from hicosmo.fisher import IntensityMappingFisher

   # Interacting dark energy model
   ilcdm = ILCDM(
       H0=67.36,
       Omega_m=0.3153,
       beta=0.001  # Interaction strength
   )

   # Tianlai constraints on beta
   result = IntensityMappingFisher.forecast(
       'tianlai',
       ilcdm,
       params=['beta', 'H0', 'Omega_m']
   )

   print(f"Tianlai constraints on the interaction parameter:")
   print(f"sigma(beta) = {result.errors['beta']:.4e}")

Multi-Survey Joint Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import CPL
   from hicosmo.fisher import IntensityMappingFisher, load_survey, FisherMatrix
   import numpy as np

   cpl = CPL()
   surveys = ['ska1_mid_band2', 'tianlai', 'bingo']

   # Collect Fisher matrices from each survey
   fisher_matrices = []
   for survey_name in surveys:
       result = IntensityMappingFisher.forecast(
           survey_name, cpl, ['w0', 'wa', 'H0', 'Omega_m']
       )
       fisher_matrices.append((result.fisher_matrix, result.params))

   # Combine Fisher matrices
   calculator = FisherMatrix()
   combined_fisher, combined_params = calculator.combine_fisher_matrices(
       fisher_matrices,
       probe_names=surveys
   )

   # Compute joint constraints
   combined_cov = np.linalg.inv(combined_fisher)
   combined_errors = np.sqrt(np.diag(combined_cov))

   print("Joint constraints:")
   for i, param in enumerate(combined_params):
       print(f"  sigma({param}) = {combined_errors[i]:.4e}")

Performance Benchmarks
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Operation
     - Time
     - Notes
   * - Single-survey Fisher forecast
     - ~0.5s
     - 5 redshift bins
   * - Full parameter forecast (with marginalization)
     - ~1.2s
     - 6 parameters
   * - Automatic differentiation Hessian
     - ~0.1s
     - JAX JIT

FAQ
---

Q: How to customize a survey configuration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a YAML file and use ``IntensityMappingSurvey.from_file()``:

.. code-block:: python

   from hicosmo.fisher import IntensityMappingSurvey

   survey = IntensityMappingSurvey.from_file('my_survey.yaml')

Q: What if the Fisher matrix is not positive definite?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable regularization:

.. code-block:: python

   from hicosmo.fisher import FisherMatrixConfig

   config = FisherMatrixConfig(
       regularization=True,
       reg_lambda=1e-8
   )

Q: How to compare different cosmological models?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the same survey configuration with different models:

.. code-block:: python

   from hicosmo.models import LCDM, wCDM, CPL

   survey = 'ska1_mid_band2'

   result_lcdm = IntensityMappingFisher.forecast(survey, LCDM(), ['H0', 'Omega_m'])
   result_wcdm = IntensityMappingFisher.forecast(survey, wCDM(), ['H0', 'Omega_m', 'w0'])
   result_cpl = IntensityMappingFisher.forecast(survey, CPL(), ['H0', 'Omega_m', 'w0', 'wa'])

Next Steps
----------

- `Visualization <visualization.html>`_: Display Fisher forecast results
- `Cosmological Models <models.html>`_: Learn about available models
