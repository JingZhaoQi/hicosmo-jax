Visualization
=============

HIcosmo provides a professional-grade visualization system based on GetDist, supporting MCMC sample visualization, multi-chain comparison, and Fisher forecast plotting.

Visualization Overview
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Plot Type
     - Purpose
     - Use Case
   * - **Corner Plot**
     - Joint parameter constraints
     - Display correlations and marginal distributions between parameters
   * - **2D Contour**
     - Two-parameter constraints
     - Highlight confidence intervals for specific parameter pairs
   * - **1D Marginal**
     - Single-parameter constraints
     - Display the posterior distribution of individual parameters
   * - **Trace Plot**
     - Convergence diagnostics
     - Check convergence of MCMC chains
   * - **Fisher Contour**
     - Forecast constraints
     - Visualize Fisher matrix forecast results

Quick Start
-----------

**3 lines to draw a corner plot**:

.. code-block:: python

   from hicosmo.visualization import Plotter

   plotter = Plotter('my_chain')
   plotter.corner(['H0', 'Omega_m'], filename='corner.pdf')

Plotter Class Interface
-----------------------

The ``Plotter`` class is the unified entry point for HIcosmo visualization, supporting multiple data input formats and rich plotting functionality.

Initialization
~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import Plotter

   # Method 1: Load from chain name (simplest)
   plotter = Plotter('my_chain')  # Automatically loads from mcmc_chains/my_chain.h5

   # Method 2: Multi-chain comparison
   plotter = Plotter(
       ['chain1', 'chain2', 'chain3'],
       labels=['Planck', 'SH0ES', 'DESI']  # Legend labels
   )

   # Method 3: Create from dictionary
   plotter = Plotter({
       'H0': samples_h0,      # 1D array
       'Omega_m': samples_om  # 1D array
   })

   # Method 4: Create from MCMC object
   plotter = Plotter.from_mcmc(mcmc)

   # Method 5: Create from Fisher results
   plotter = Plotter.from_fisher(
       mean=[67.36, 0.315],
       covariance=[[0.5, 0.01], [0.01, 0.007]],
       param_names=['H0', 'Omega_m']
   )

**Initialization Parameters**:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Type
     - Description
   * - ``data``
     - Various formats
     - Chain name, path, dictionary, array, or MCSamples
   * - ``labels``
     - dict or list
     - Parameter LaTeX labels
   * - ``ranges``
     - dict
     - Plot ranges ``{name: (min, max)}``
   * - ``chain_labels``
     - list
     - Legend labels for multi-chain mode
   * - ``style``
     - str
     - Color scheme (``'modern'`` or ``'classic'``)
   * - ``results_dir``
     - str or Path
     - Directory for saving results

Corner Plot
~~~~~~~~~~~

A corner plot (triangle plot) is the standard way to visualize cosmological parameter constraints, displaying both marginal distributions and two-dimensional joint distributions simultaneously.

.. code-block:: python

   # Plot all parameters
   fig = plotter.corner()

   # Select specific parameters (by name)
   fig = plotter.corner(['H0', 'Omega_m', 'rd_h'])

   # Select specific parameters (by index, 0-based)
   fig = plotter.corner([0, 1, 2])  # First three parameters

   # Save to file
   fig = plotter.corner(['H0', 'Omega_m'], filename='corner.pdf')

**Key Features**:

- Diagonal: 1D marginal distributions (filled density)
- Off-diagonal: 2D contours (68% and 95% confidence intervals)
- Automatic LaTeX labels (read from chain file metadata)
- Smart tick optimization (prevents label overlap)

2D Contour Plot
~~~~~~~~~~~~~~~

Plot the 2D constraints for two parameters individually:

.. code-block:: python

   # Plot two-parameter contours
   fig = plotter.plot_2d('H0', 'Omega_m', filename='h0_om.pdf')

   # Use parameter indices
   fig = plotter.plot_2d(0, 1)

   # Unfilled
   fig = plotter.plot_2d('w0', 'wa', filled=False)

1D Marginal Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

Plot the posterior distribution of a single parameter:

.. code-block:: python

   # Single parameter distribution
   fig = plotter.plot_1d('H0', filename='h0_posterior.pdf')

   # Multi-parameter grid layout
   fig = plotter.marginals(['H0', 'Omega_m', 'Omega_b'])

Chain Trace Plot
~~~~~~~~~~~~~~~~

Trace plots for MCMC convergence diagnostics:

.. code-block:: python

   # Default: show first 6 parameters
   fig = plotter.traces()

   # Select specific parameters
   fig = plotter.traces(['H0', 'Omega_m'], filename='traces.pdf')

**Convergence Diagnostics Key Points**:

- Good convergence: chains should fluctuate randomly around the mean
- Non-convergence signal: obvious trends or periodicity
- Warm-up issues: initial portion clearly deviates from equilibrium

Statistical Summary
~~~~~~~~~~~~~~~~~~~

Retrieve statistical information for parameters:

.. code-block:: python

   summary = plotter.get_summary()

   # Return structure:
   # {
   #     'H0': {'mean': 67.36, 'std': 0.42, 'median': 67.35, 'q16': 66.94, 'q84': 67.78},
   #     'Omega_m': {'mean': 0.315, 'std': 0.007, ...}
   # }

   # Print results
   for param, stats in summary.items():
       print(f"{param}: {stats['mean']:.3f} ± {stats['std']:.3f}")

Analysis Report
~~~~~~~~~~~~~~~

Generate a Markdown report containing parameter constraints and diagnostic information:

.. code-block:: python

   # Single-chain mode: generate one report
   plotter = Plotter('my_chain')
   plotter.report()  # Automatically saved as my_chain_report.md

   # Custom title
   plotter.report(title='LCDM Analysis Report')

   # Without LaTeX-formatted constraints
   plotter.report(include_latex=False)

**Report Contents**:

- **Run Information**: Chain name, sampler, number of chains, warm-up steps, sample count
- **Likelihood Functions**: Class name, signature, description
- **Cosmological Parameters**: Priors, bounds, whether free
- **Nuisance Parameters**: Priors, bounds, description
- **Parameter Constraints**: Mean +/- sigma, median, 68%/95% confidence intervals
- **LaTeX Format**: 1-sigma and 2-sigma constraint LaTeX expressions
- **MCMC Diagnostics**: Total samples, effective sample size (ESS)

Multi-Chain Comparison
----------------------

HIcosmo supports comparative visualization of multiple MCMC chains, which is very useful when comparing different datasets or models.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import Plotter

   # Create from multiple chain names
   plotter = Plotter(
       ['planck_chain', 'shoes_chain', 'desi_chain'],
       labels=['Planck 2018', 'SH0ES', 'DESI DR1']
   )

   # Plot comparison corner plot
   fig = plotter.corner(['H0', 'Omega_m'], filename='comparison.pdf')

Automatic File Naming
~~~~~~~~~~~~~~~~~~~~~

In multi-chain mode, corner plots automatically use ``+`` to concatenate all chain names for the filename:

.. code-block:: python

   plotter = Plotter(['test_sne', 'test_bao'], labels=['Pantheon+', 'DESI R1'])
   plotter.corner()  # Automatically saved as test_sne+test_bao_corner.pdf

Automatic Parameter Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calling ``corner()`` without specifying parameters will automatically plot **all parameters from all chains**, including nuisance parameters:

.. code-block:: python

   # Suppose test_sne has [H0, Omega_m]
   # Suppose test_bao has [H0, Omega_m, rd_fid_Mpc] (with nuisance parameter)

   plotter = Plotter(['test_sne', 'test_bao'], labels=['Pantheon+', 'DESI R1'])
   plotter.corner()  # Automatically plots [H0, Omega_m, rd_fid_Mpc] three parameters

   # To specify parameters, pass them explicitly
   plotter.corner(['H0', 'Omega_m'])  # Only plot these two parameters

Multi-Chain Report Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In multi-chain mode, ``report()`` generates **separate independent reports** for each chain:

.. code-block:: python

   plotter = Plotter(['test_sne', 'test_bao'], labels=['Pantheon+', 'DESI R1'])
   plotter.report()
   # Generates two files:
   #   - results/test_sne_report.md
   #   - results/test_bao_report.md

This design is because each chain may use different likelihood functions and different nuisance parameters; merging reports would cause confusion.

Create from Dictionaries
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prepare multi-chain data
   chain1 = {'H0': samples1_h0, 'Omega_m': samples1_om}
   chain2 = {'H0': samples2_h0, 'Omega_m': samples2_om}
   chain3 = {'H0': samples3_h0, 'Omega_m': samples3_om}

   plotter = Plotter(
       [chain1, chain2, chain3],
       chain_labels=['CMB only', 'CMB + BAO', 'CMB + BAO + SN']
   )

   fig = plotter.corner(filename='joint_comparison.pdf')

Color Schemes
~~~~~~~~~~~~~

HIcosmo provides two professional color schemes:

**Modern (default)**:

.. code-block:: python

   MODERN_COLORS = [
       '#2E86AB',  # Dark blue
       '#A23B72',  # Purple-red
       '#F18F01',  # Orange
       '#C73E1D',  # Red
       '#592E83',  # Purple
       '#0F7173',  # Teal
       '#7B2D26',  # Brown
   ]

**Classic**:

.. code-block:: python

   CLASSIC_COLORS = [
       '#348ABD',  # Blue
       '#7A68A6',  # Purple
       '#E24A33',  # Red
       '#467821',  # Green
       '#ffb3a6',  # Pink
       '#188487',  # Teal
       '#A60628',  # Dark red
   ]

Usage:

.. code-block:: python

   plotter = Plotter('my_chain', style='classic')

Fisher Forecast Visualization
-----------------------------

HIcosmo supports generating forecast contour plots from Fisher matrix results.

Create from Fisher Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import Plotter
   import numpy as np

   # Suppose you have Fisher analysis results
   mean = np.array([67.36, 0.315])  # Fiducial values
   covariance = np.array([
       [0.5, 0.01],
       [0.01, 0.007]
   ])  # Covariance matrix

   # Create Plotter
   plotter = Plotter.from_fisher(
       mean=mean,
       covariance=covariance,
       param_names=['H0', 'Omega_m']
   )

   # Plot forecast constraints
   fig = plotter.corner(filename='fisher_forecast.pdf')

**How It Works**:

1. Generate samples from a Gaussian distribution :math:`\mathcal{N}(\mu, \Sigma)`
2. By default, 200,000 samples are generated to ensure smooth contours
3. Use GetDist to draw contours

Dark Energy Figure of Merit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The constraining power on the dark energy equation of state is measured by the Figure of Merit (FoM):

.. math::

   \text{FoM} = \frac{1}{\sqrt{\det(\text{Cov}_{w_0, w_a})}}

.. code-block:: python

   from hicosmo.fisher import IntensityMappingFisher

   # Get the w0-wa submatrix
   cov_de = fisher.marginalize(['w0', 'wa']).covariance
   fom = 1.0 / np.sqrt(np.linalg.det(cov_de))

   print(f"Dark Energy FoM: {fom:.1f}")

Function Interface
------------------

In addition to the ``Plotter`` class, HIcosmo also provides standalone function interfaces for quick plotting.

plot_corner
~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import plot_corner

   # Single chain
   fig = plot_corner(
       'results/mcmc_chain.pkl',
       params=['H0', 'Omega_m'],
       style='modern',
       filename='corner.pdf'
   )

   # Multi-chain comparison
   fig = plot_corner(
       [chain1, chain2],
       labels=['Dataset A', 'Dataset B'],
       filename='comparison.pdf'
   )

plot_chains
~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import plot_chains

   fig = plot_chains(
       'results/mcmc_chain.pkl',
       params=[0, 1, 2],  # First three parameters
       filename='traces.pdf'
   )

plot_1d
~~~~~~~

.. code-block:: python

   from hicosmo.visualization import plot_1d

   fig = plot_1d(
       'results/mcmc_chain.pkl',
       filename='marginals.pdf'
   )

plot_fisher_contours
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import plot_fisher_contours

   fig = plot_fisher_contours(
       mean=[67.36, 0.315],
       covariance=[[0.5, 0.01], [0.01, 0.007]],
       param_names=['H0', 'Omega_m'],
       filename='forecast.pdf'
   )

Supported Data Formats
----------------------

HIcosmo automatically recognizes multiple data formats:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Format
     - Extension
     - Description
   * - **HDF5**
     - ``.h5``, ``.hdf5``
     - Recommended format, supports metadata
   * - **Pickle**
     - ``.pkl``, ``.pickle``
     - Python native format
   * - **NumPy**
     - ``.npy``, ``.npz``
     - Array format
   * - **Text**
     - ``.txt``, ``.dat``
     - ASCII format
   * - **Dictionary**
     - In-memory
     - ``{'param': array}``
   * - **MCSamples**
     - In-memory
     - GetDist native object

Automatic Metadata Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain files saved with ``MCMC.save_results()`` contain complete metadata:

.. code-block:: python

   # Save chain (metadata automatically included)
   mcmc.save_results('my_chain')  # Saved to mcmc_chains/my_chain.h5

   # Automatically extracted upon loading
   plotter = Plotter('my_chain')
   # - LaTeX labels read from file
   # - Parameter ranges read from file
   # - Derived parameters (e.g., rd_h) automatically available

Style Customization
-------------------

Global Style
~~~~~~~~~~~~

HIcosmo automatically applies professional styles, including:

- **Font sizes**: Axis labels 14pt, ticks 11pt, legend 12pt
- **Line widths**: Contours 2pt, grid 1.2pt
- **Background**: White (publication-friendly)
- **Legend**: No frame

Manual Style Adjustments
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Modify global style
   plt.rcParams.update({
       'axes.labelsize': 16,
       'legend.fontsize': 14,
       'lines.linewidth': 2.5,
   })

   # Create plotter
   plotter = Plotter('my_chain')
   fig = plotter.corner()

   # Further adjustments (returns a matplotlib Figure)
   for ax in fig.axes:
       ax.tick_params(labelsize=10)

Direct GetDist Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced users who need more GetDist control:

.. code-block:: python

   from getdist import plots, MCSamples
   import numpy as np

   # Create GetDist MCSamples
   samples = MCSamples(
       samples=np.column_stack([h0_samples, om_samples]),
       names=['H0', 'Omega_m'],
       labels=[r'H_0', r'\Omega_m']
   )

   # Use GetDist native API
   g = plots.get_subplot_plotter()
   g.settings.figure_legend_frame = False
   g.triangle_plot(samples, filled=True)

Save Options
------------

File Formats
~~~~~~~~~~~~

.. code-block:: python

   # PDF (default, vector graphics, recommended for publication)
   plotter.corner(filename='corner.pdf')

   # PNG (raster graphics, for web display)
   plotter.corner(filename='corner.png')

   # SVG (vector graphics, editable)
   plotter.corner(filename='corner.svg')

Resolution Settings
~~~~~~~~~~~~~~~~~~~

The default DPI is 300, suitable for publication. Adjustable via matplotlib:

.. code-block:: python

   fig = plotter.corner()
   fig.savefig('corner.png', dpi=600)  # High resolution

Complete Examples
-----------------

MCMC Results Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo import hicosmo
   from hicosmo.visualization import Plotter

   # Run MCMC
   inf = hicosmo(
       cosmology='LCDM',
       likelihood=['sn', 'bao'],
       free_params=['H0', 'Omega_m'],
       num_samples=4000,
       num_chains=4
   )
   samples = inf.run()

   # Save chain
   inf.mcmc.save_results('lcdm_sn_bao')

   # Visualization
   plotter = Plotter('lcdm_sn_bao')

   # Corner plot
   plotter.corner(['H0', 'Omega_m'], filename='lcdm_corner.pdf')

   # Convergence diagnostics
   plotter.traces(filename='lcdm_traces.pdf')

   # 1D distributions
   plotter.marginals(filename='lcdm_marginals.pdf')

   # Print statistics
   for param, stats in plotter.get_summary().items():
       print(f"{param}: {stats['mean']:.4f} ± {stats['std']:.4f}")

Multi-Probe Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.visualization import Plotter

   # Run multiple analyses (assume already completed)
   # ...

   # Create comparison plot
   plotter = Plotter(
       ['sn_only', 'bao_only', 'cmb_only', 'joint_all'],
       labels=[
           'Pantheon+',
           'DESI BAO',
           'Planck 2018',
           'Joint'
       ]
   )

   # Plot comparison (automatically includes all parameters, including nuisance)
   plotter.corner()  # Automatically saved as sn_only+bao_only+cmb_only+joint_all_corner.pdf

   # Or plot only cosmological parameters
   plotter.corner(['H0', 'Omega_m'], filename='multi_probe_comparison.pdf')

   # Compare H0 individually
   plotter.plot_1d('H0', filename='h0_tension.pdf')

   # Generate individual analysis reports
   plotter.report()
   # Generates four independent reports:
   #   - sn_only_report.md
   #   - bao_only_report.md
   #   - cmb_only_report.md
   #   - joint_all_report.md

Fisher Forecast Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.fisher import IntensityMappingFisher, load_survey
   from hicosmo.models import CPL
   from hicosmo.visualization import Plotter
   import numpy as np

   # Fisher analysis
   survey = load_survey('ska1_mid_band2')
   fiducial = CPL(H0=67.36, Omega_m=0.3153, w0=-1.0, wa=0.0)
   fisher = IntensityMappingFisher(survey, fiducial)
   result = fisher.parameter_forecast(['w0', 'wa'])

   # Visualization
   plotter = Plotter.from_fisher(
       mean=result.mean,
       covariance=result.covariance,
       param_names=['w0', 'wa']
   )

   plotter.corner(filename='ska1_forecast.pdf')

   # Compute FoM
   fom = 1.0 / np.sqrt(np.linalg.det(result.covariance))
   print(f"SKA1 Dark Energy FoM: {fom:.1f}")

FAQ
---

Q: What if GetDist is not installed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install getdist

Q: How to change contour colors?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use classic color scheme
   plotter = Plotter('my_chain', style='classic')

   # Or customize (requires using the GetDist API)
   from getdist import plots
   g = plots.get_subplot_plotter()
   g.triangle_plot(samples, contour_colors=['red', 'blue'])

Q: How to add truth value markers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig = plotter.corner(['H0', 'Omega_m'])

   # Add truth value lines
   import matplotlib.pyplot as plt
   for ax in fig.axes:
       # Add reference lines to each subplot
       pass  # Needs to be added based on specific subplot positions

Next Steps
----------

- `Fisher Forecasts <fisher.html>`_: Perform survey predictions
- `Samplers <samplers.html>`_: Learn about MCMC configuration
