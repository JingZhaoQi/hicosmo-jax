Model Selection and Information Criteria
========================================

HIcosmo provides a complete set of model selection tools, including AIC, BIC, and Bayesian evidence computation. These tools help compare the goodness of fit of different cosmological models.

Information Criteria Overview
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30

   * - Criterion
     - Formula
     - Use Case
     - Interpretation
   * - **AIC**
     - :math:`\chi^2_{\min} + 2k`
     - Small samples, exploratory analysis
     - Smaller values indicate a better model
   * - **BIC**
     - :math:`\chi^2_{\min} + k \ln N`
     - Large samples, model selection
     - Stronger penalty on the number of parameters
   * - **Bayesian Evidence**
     - :math:`\int L(\theta) \pi(\theta) d\theta`
     - Bayesian model comparison
     - Accounts for prior volume

Where:

- :math:`k` = number of free parameters
- :math:`N` = number of data points
- :math:`\chi^2_{\min}` = minimum chi-squared value

Quick Start
-----------

**3 lines to compute information criteria**:

.. code-block:: python

   from hicosmo.visualization import information_criteria

   result = information_criteria('my_chain')
   print(f"AIC = {result['aic']:.2f}, BIC = {result['bic']:.2f}")

Formula Details
---------------

Chi-Squared Statistic
~~~~~~~~~~~~~~~~~~~~~

For a Gaussian likelihood function:

.. math::

   \log L = -\frac{\chi^2}{2} + \text{normalization constant}

Where the normalization constant includes:

- :math:`-\frac{1}{2} \log |C|`: determinant of the covariance matrix
- Other constants independent of the parameters

Extracting :math:`\chi^2` from the log-likelihood:

.. math::

   \chi^2 = -2 \times (\log L - \text{normalization constant})

.. note::

   HIcosmo automatically handles the normalization constant, ensuring correct AIC/BIC values. No manual intervention is needed.

AIC (Akaike Information Criterion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Akaike Information Criterion:

.. math::

   \text{AIC} = \chi^2_{\min} + 2k

**Characteristics**:

- Penalty term is independent of sample size
- Suitable for small sample cases
- Tends to favor more complex models

**Interpreting ΔAIC**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ΔAIC
     - Interpretation
   * - < 2
     - No significant difference
   * - 2 - 4
     - Weak evidence in favor of the lower-AIC model
   * - 4 - 7
     - Moderate evidence
   * - > 10
     - Strong evidence

BIC (Bayesian Information Criterion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bayesian Information Criterion:

.. math::

   \text{BIC} = \chi^2_{\min} + k \ln N

**Characteristics**:

- Penalty term increases with sample size
- Stronger penalty against overfitting
- Consistent estimator (selects the true model when the sample is large enough)

**Jeffreys Scale Interpretation of ΔBIC**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ΔBIC
     - Evidence Strength
   * - < 2
     - Not worth more than a bare mention
   * - 2 - 6
     - Positive evidence
   * - 6 - 10
     - Strong evidence
   * - > 10
     - Decisive evidence

Bayesian Evidence
~~~~~~~~~~~~~~~~~

Bayesian evidence (marginal likelihood):

.. math::

   Z = \int L(\theta) \pi(\theta) d\theta

Where :math:`\pi(\theta)` is the prior distribution.

**Bayes Factor**:

.. math::

   B_{12} = \frac{Z_1}{Z_2} = \exp(\log Z_1 - \log Z_2)

**Jeffreys Scale**:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Δlog Z
     - Bayes Factor B
     - Evidence Strength
   * - < 1.0
     - < 2.7
     - Not worth more than a bare mention
   * - 1.0 - 2.5
     - 2.7 - 12
     - Weak evidence
   * - 2.5 - 5.0
     - 12 - 150
     - Moderate evidence
   * - > 5.0
     - > 150
     - Strong evidence

.. warning::

   HIcosmo uses the **Harmonic Mean Estimator** to compute Bayesian evidence. This method can be unstable;
   for accurate Bayesian evidence computation, consider using nested sampling (e.g., dynesty).

Handling the Normalization Constant
-----------------------------------

Why Is the Normalization Constant Needed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the full form of a Gaussian likelihood function:

.. math::

   \log L = -\frac{1}{2} (d - t)^T C^{-1} (d - t) - \frac{1}{2} \log |C| - \frac{N}{2} \log(2\pi)

Where:

- The first term :math:`-\frac{\chi^2}{2}` depends on the model parameters
- The latter two terms are the normalization constant, independent of the model parameters

**Problem**: If :math:`\log L` is used directly to compute AIC/BIC:

.. code-block:: python

   # Wrong!
   aic = 2*k - 2*log_likelihood  # Will yield negative values

**Solution**: Extract :math:`\chi^2` first, then compute:

.. code-block:: python

   # Correct
   chi2 = -2 * (log_likelihood - normalization_constant)
   aic = chi2 + 2*k  # Positive value

HIcosmo's Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

HIcosmo automatically handles the normalization constant; no manual intervention is needed:

1. **Likelihood functions provide the normalization constant**:

.. code-block:: python

   class PantheonPlusLikelihood(Likelihood):
       def get_normalization_constant(self) -> float:
           """Return the normalization constant"""
           return -0.5 * self.log_det_cov

2. **Automatically stored when saving MCMC results**:

.. code-block:: python

   # Normalization constant is automatically included when saving the chain
   mcmc.save_results('my_chain')
   # metadata contains normalization_constant

3. **Automatically read by information criteria**:

.. code-block:: python

   result = information_criteria('my_chain')
   # Automatically reads normalization_constant from metadata

Impact on Model Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key conclusion**: The normalization constant has **no impact on model comparison results**!

.. math::

   \Delta\text{AIC} = \text{AIC}_1 - \text{AIC}_2 = (\chi^2_1 + 2k_1) - (\chi^2_2 + 2k_2)

The normalization constant is the same (using the same data) and cancels out in the difference.

.. code-block:: python

   # Verification
   delta_aic_without_norm = aic1 - aic2
   delta_aic_with_norm = aic1_corrected - aic2_corrected
   # Both are equal!

However, the **absolute values** of AIC/BIC require correct normalization to be meaningful:

- Reduced :math:`\chi^2 = \chi^2_{\min}/N \approx 1` indicates a good fit
- AIC, BIC must be positive numbers

API Reference
-------------

information_criteria Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import information_criteria

   result = information_criteria(
       samples='my_chain',           # Chain name, path, or sample dictionary
       num_params=None,              # Number of free parameters (auto-inferred)
       num_data=None,                # Number of data points (auto-inferred)
       log_likelihood_fn=None,       # Likelihood function (optional)
       evidence_method='harmonic_mean',  # Evidence estimation method
   )

**Return Value**:

.. code-block:: python

   {
       'num_params': 4,                    # Number of free parameters k
       'num_data': 1580,                   # Number of data points N
       'n_samples': 10000,                 # MCMC sample count
       'log_likelihood_max': 2300.68,      # Maximum log-likelihood
       'normalization_constant': 2994.34,  # Normalization constant
       'chi2_min': 1387.32,                # Minimum chi-squared
       'aic': 1395.32,                     # AIC
       'bic': 1417.88,                     # BIC
       'log_evidence': 2298.5,             # log(Bayesian evidence)
       'log_evidence_method': 'harmonic_mean',
   }

aic and bic Functions
~~~~~~~~~~~~~~~~~~~~~

Compute AIC or BIC independently:

.. code-block:: python

   from hicosmo.visualization.information_criteria import aic, bic

   # Compute AIC
   aic_value = aic(
       log_likelihood_max=2300.68,
       num_params=4,
       normalization_constant=2994.34
   )

   # Compute BIC
   bic_value = bic(
       log_likelihood_max=2300.68,
       num_params=4,
       num_data=1580,
       normalization_constant=2994.34
   )

Complete Examples
-----------------

Single-Model Information Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo import MCMC
   from hicosmo.models import LCDM
   from hicosmo.likelihoods import SN_likelihood
   from hicosmo.visualization import information_criteria

   # Run MCMC
   likelihood = SN_likelihood(LCDM, 'pantheon+')
   params = {
       'H0': (67.0, 60.0, 80.0),
       'Omega_m': (0.3, 0.1, 0.5),
   }

   mcmc = MCMC(params, likelihood, chain_name='lcdm_pantheon')
   mcmc.run(num_samples=5000, num_chains=4)

   # Compute information criteria
   result = information_criteria('lcdm_pantheon')

   print(f"Number of free parameters k = {result['num_params']}")
   print(f"Number of data points N = {result['num_data']}")
   print(f"chi2_min = {result['chi2_min']:.2f}")
   print(f"Reduced chi2 = {result['chi2_min']/result['num_data']:.4f}")
   print(f"AIC = {result['aic']:.2f}")
   print(f"BIC = {result['bic']:.2f}")

Multi-Model Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo import MCMC
   from hicosmo.models import LCDM, wCDM, CPL
   from hicosmo.likelihoods import SN_likelihood
   from hicosmo.visualization import information_criteria
   import numpy as np

   # Define three models
   models = {
       'LCDM': (LCDM, ['H0', 'Omega_m']),
       'wCDM': (wCDM, ['H0', 'Omega_m', 'w']),
       'CPL': (CPL, ['H0', 'Omega_m', 'w0', 'wa']),
   }

   results = {}

   for name, (model_class, free_params) in models.items():
       # Build parameter configuration
       params = {p: (67.0 if p == 'H0' else 0.3, 0.1, 0.9) for p in free_params}
       if 'w' in free_params:
           params['w'] = (-1.0, -2.0, 0.0)
       if 'w0' in free_params:
           params['w0'] = (-1.0, -2.0, 0.0)
           params['wa'] = (0.0, -1.0, 1.0)

       # Run MCMC
       likelihood = SN_likelihood(model_class, 'pantheon+')
       mcmc = MCMC(params, likelihood, chain_name=f'{name.lower()}_chain')
       mcmc.run(num_samples=5000, num_chains=4)

       # Compute information criteria
       results[name] = information_criteria(f'{name.lower()}_chain')

   # Print comparison table
   print("\nModel Selection Results")
   print("=" * 70)
   print(f"{'Model':<8} {'k':<4} {'chi2_min':<12} {'AIC':<12} {'BIC':<12} {'DBIC':<10}")
   print("-" * 70)

   bic_ref = results['LCDM']['bic']
   for name, ic in results.items():
       delta_bic = ic['bic'] - bic_ref
       print(f"{name:<8} {ic['num_params']:<4} {ic['chi2_min']:<12.2f} "
             f"{ic['aic']:<12.2f} {ic['bic']:<12.2f} {delta_bic:<+10.2f}")

   print("-" * 70)

   # Interpret results
   print("\nModel Selection Interpretation (Jeffreys Scale):")
   for name, ic in results.items():
       if name == 'LCDM':
           continue
       delta = abs(ic['bic'] - bic_ref)
       if delta < 2:
           strength = "Not worth more than a bare mention"
       elif delta < 6:
           strength = "Positive evidence"
       elif delta < 10:
           strength = "Strong evidence"
       else:
           strength = "Decisive evidence"
       better = "LCDM" if ic['bic'] > bic_ref else name
       print(f"  {name} vs LCDM: |DBIC| = {delta:.1f} -> {strength} in favor of {better}")

Bayesian Evidence Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import information_criteria

   # Assume MCMC has been run for multiple models
   chains = ['lcdm_chain', 'wcdm_chain', 'cpl_chain']
   names = ['LCDM', 'wCDM', 'CPL']

   print("\nBayesian Evidence Comparison")
   print("=" * 60)
   print(f"{'Model':<8} {'log(Z)':<15} {'Dlog(Z)':<15} {'B':<10}")
   print("-" * 60)

   results = [information_criteria(c) for c in chains]
   log_z_ref = results[0]['log_evidence']

   for name, ic in zip(names, results):
       delta = ic['log_evidence'] - log_z_ref
       bayes_factor = np.exp(delta) if abs(delta) < 100 else float('inf')
       print(f"{name:<8} {ic['log_evidence']:<15.4f} {delta:<+15.4f} {bayes_factor:<10.2f}")

   print("-" * 60)
   print("\nJeffreys Scale: |Dlog(Z)| > 5 indicates strong evidence")

FAQ
---

Q: Are negative AIC/BIC values correct?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**No.** AIC = chi2 + 2k and BIC = chi2 + k*ln(N) must be positive, because chi2 >= 0.

If you see negative values, it may be due to using an uncorrected log-likelihood. HIcosmo handles this automatically.

Q: Should I use AIC or BIC?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **AIC**: Suitable for small samples and exploratory analysis; lighter penalty on complex models
- **BIC**: Suitable for large samples and model selection; stronger penalty against overfitting

In cosmology, **BIC** is generally recommended, as datasets are typically large (N > 1000).

Q: What if the Bayesian evidence estimate is unstable?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Harmonic Mean Estimator can indeed be unstable. Recommendations:

1. Increase the number of MCMC samples
2. Average over multiple chains
3. For accurate computation, use nested sampling (e.g., dynesty, polychord)

Q: Does the normalization constant affect model comparison?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**No.** When comparing models using the same data, the normalization constant is identical and cancels in DAIC and DBIC.

However, the normalization constant does affect **absolute values**:

- Ensures chi2 > 0
- Ensures AIC, BIC are positive
- Reduced chi2 ~ 1 indicates a good fit

Next Steps
----------

- `Visualization <visualization.html>`_: Plot constraint figures
- `Samplers <samplers.html>`_: Learn about MCMC configuration
- `Fisher Forecasts <fisher.html>`_: Perform survey predictions
