Dataset Management
==================

HIcosmo provides automatic dataset discovery, allowing users to easily view all available observational data.
No hardcoding is needed -- when you add new datasets to the data directory, the system will automatically detect them.

Viewing Available Datasets
--------------------------

Quick View
~~~~~~~~~~

The simplest way is to call ``show_available_datasets()``:

.. code-block:: python

   from hicosmo.likelihoods import show_available_datasets

   # Show all available datasets
   show_available_datasets()

Output example::

   ======================================================================
                       HIcosmo Available Datasets
   ======================================================================

     BAO (Baryon Acoustic Oscillations)
     --------------------------------------------------
       • desi_2024        : DESI 2024 DR1 BAO measurements (z=0.3-2.3)
       • boss_dr12        : BOSS DR12 Luminous Red Galaxy BAO
       • sdss_dr12        : SDSS DR12 consensus BAO
       • sdss_dr16        : SDSS DR16 LRG and QSO BAO
       • sixdf            : 6dF Galaxy Survey BAO (z=0.1)

     SN (Type Ia Supernovae)
     --------------------------------------------------
       • pantheon+shoes   : Pantheon+ with SH0ES Cepheid calibration

     CMB (Cosmic Microwave Background)
     --------------------------------------------------
       • planck2018_distance : Planck 2018 distance priors (built-in)

     H0 (Direct Hubble Measurements)
     --------------------------------------------------
       • h0licow          : H0LiCOW strong lensing time delays (6 lenses)

     GW (Gravitational Waves)
     --------------------------------------------------
       • gwtc-3           : GWTC-3 gravitational wave catalog

     Lensing (Strong Gravitational Lensing)
     --------------------------------------------------
       • tdcosmo          : TDCOSMO hierarchical strong lensing (7 lenses)
       • tdcosmo2025      : TDCOSMO 2025 updated analysis

   ======================================================================

Programmatic Access
~~~~~~~~~~~~~~~~~~~

If you need to use the dataset list in code:

.. code-block:: python

   from hicosmo.likelihoods import list_all_datasets

   # Get all datasets (as a dictionary)
   datasets = list_all_datasets()

   # View BAO datasets
   print(datasets['bao'])
   # ['desi_2024', 'boss_dr12', 'sdss_dr12', 'sdss_dr16', 'sixdf']

   # View SN datasets
   print(datasets['sn'])
   # ['pantheon+shoes']

   # Dynamically select datasets
   for bao_name in datasets['bao']:
       print(f"Available BAO dataset: {bao_name}")

View by Category
~~~~~~~~~~~~~~~~~

View datasets of a specific category only:

.. code-block:: python

   # Show BAO datasets only
   show_available_datasets('bao')

   # Show gravitational wave datasets only
   show_available_datasets('gw')

Class Method Queries
~~~~~~~~~~~~~~~~~~~~

Each Likelihood class also provides an ``available_datasets()`` method:

.. code-block:: python

   from hicosmo.likelihoods.bao import DESI2024BAO
   from hicosmo.likelihoods.sn import PantheonPlusLikelihood

   # Available datasets for the BAO class
   print(DESI2024BAO.available_datasets())
   # ['desi_2024', 'boss_dr12', 'sdss_dr12', 'sdss_dr16', 'sixdf']

   # Available datasets for the SN class
   print(PantheonPlusLikelihood.available_datasets())
   # ['pantheon+shoes']


Dataset Categories
------------------

HIcosmo supports the following categories of observational data:

.. list-table:: Dataset Categories
   :header-rows: 1
   :widths: 15 35 50

   * - Category
     - Full Name
     - Included Datasets
   * - ``bao``
     - Baryon Acoustic Oscillations
     - DESI 2024, BOSS DR12, SDSS DR12/16, 6dF
   * - ``sn``
     - Type Ia Supernovae
     - Pantheon+, Pantheon+SH0ES
   * - ``cmb``
     - Cosmic Microwave Background
     - Planck 2018 distance priors
   * - ``h0``
     - Direct H0 Measurements
     - H0LiCOW, SH0ES
   * - ``gw``
     - Gravitational Waves
     - GWTC-3
   * - ``lensing``
     - Strong Gravitational Lensing
     - TDCOSMO, TDCOSMO 2025


Using Discovered Datasets
-------------------------

Use discovered datasets for MCMC analysis:

.. code-block:: python

   from hicosmo.likelihoods import (
       BAO_likelihood, SN_likelihood,
       list_all_datasets
   )
   from hicosmo.models import LCDM

   # 1. View available datasets
   datasets = list_all_datasets()
   print("BAO datasets:", datasets['bao'])

   # 2. Select a dataset to create a likelihood
   bao = BAO_likelihood(LCDM, 'desi_2024')
   sn = SN_likelihood(LCDM, 'pantheon+shoes')

   # 3. Joint analysis
   joint = bao + sn

   # 4. Run MCMC
   from hicosmo.samplers import MCMC
   params = {
       'H0': (70.0, 60.0, 80.0),
       'Omega_m': (0.3, 0.1, 0.5),
   }
   mcmc = MCMC(params, joint, chain_name='joint_analysis')
   mcmc.run(num_samples=2000)


Default Data Directory
----------------------

Data file storage location:

- **Default path**: ``hicosmo/data/``
- **Environment variable**: You can specify a custom path via ``HICOSMO_DATA``

Directory Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   hicosmo/data/
   ├── bao_data/
   │   ├── desi_2024/
   │   │   ├── desi_2024_gaussian_bao_ALL_GCcomb_mean.txt
   │   │   └── desi_2024_gaussian_bao_ALL_GCcomb_cov.txt
   │   ├── boss_dr12/
   │   ├── sdss_dr12/
   │   ├── sdss_dr16/
   │   └── sixdf/
   ├── sne/
   │   ├── Pantheon+SH0ES.dat
   │   └── Pantheon+SH0ES_STAT+SYS.cov
   ├── h0licow/
   ├── tdcosmo/
   └── gwtc-3/


Adding New Datasets
-------------------

Adding new datasets is very simple -- just place the data files in the correct directory:

1. **BAO data**: Create a new directory under ``bao_data/``
2. **SN data**: Place in the ``sne/`` directory
3. **Others**: Place in the corresponding category directory

Example: Adding a new BAO dataset

.. code-block:: bash

   # Create a new dataset directory
   mkdir hicosmo/data/bao_data/my_new_survey

   # Copy data files
   cp my_data.txt hicosmo/data/bao_data/my_new_survey/

Then refresh the data registry:

.. code-block:: python

   from hicosmo.likelihoods import list_all_datasets

   # Force refresh to discover new datasets
   datasets = list_all_datasets(refresh=True)
   print(datasets['bao'])
   # [..., 'my_new_survey']  # New dataset automatically discovered!


DataRegistry Advanced Usage
---------------------------

For more complex needs, you can use the ``DataRegistry`` class directly:

.. code-block:: python

   from hicosmo.data_registry import DataRegistry

   # Create a registry instance
   registry = DataRegistry()

   # Get detailed information
   info = registry.get_info('bao', 'desi_2024')
   print(f"Name: {info.name}")
   print(f"Description: {info.description}")
   print(f"Number of files: {len(info.files)}")
   print(f"Path: {info.path}")

   # Export as dictionary (serializable to JSON)
   data = registry.to_dict()


Complete Example
----------------

.. code-block:: python

   """Complete example of dataset discovery and usage"""
   from hicosmo.likelihoods import (
       show_available_datasets,
       list_all_datasets,
       BAO_likelihood,
       SN_likelihood,
   )
   from hicosmo.models import LCDM

   # 1. View all available datasets
   print("=" * 50)
   print("Step 1: View available datasets")
   print("=" * 50)
   show_available_datasets()

   # 2. Programmatic access
   print("\nStep 2: Get dataset list")
   datasets = list_all_datasets()
   for category, names in datasets.items():
       if names:
           print(f"  {category}: {names}")

   # 3. Create likelihood
   print("\nStep 3: Create Likelihood")
   bao = BAO_likelihood(LCDM, datasets['bao'][0])  # Use the first BAO dataset
   print(f"  Created: {bao}")

   # 4. Test likelihood
   print("\nStep 4: Test Likelihood")
   log_L = bao(H0=70.0, Omega_m=0.3)
   print(f"  log(L) = {log_L:.2f}")
