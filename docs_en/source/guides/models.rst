Cosmological Models
===================

HIcosmo provides multiple cosmological models, covering the standard cosmological model and its dynamic dark energy extensions. All models are built on JAX with support for JIT compilation and automatic differentiation.

Model Overview
--------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Model
     - Dark Energy Equation of State
     - Description
   * - **LCDM**
     - :math:`w = -1`
     - Standard :math:`\Lambda\text{CDM}` model, cosmological constant
   * - **wCDM**
     - :math:`w = w_0`
     - Constant dark energy equation of state
   * - **CPL**
     - :math:`w(a) = w_0 + w_a(1-a)`
     - Chevallier-Polarski-Linder parameterization
   * - **ILCDM**
     - :math:`w = -1` + interaction
     - Interacting dark energy model

Basic Theory
------------

Dimensionless Hubble Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core of all models is the dimensionless Hubble parameter :math:`E(z) = H(z)/H_0`. It determines the expansion history of the universe and all distance measurements.

**General form**:

.. math::

   E^2(z) = \Omega_m (1+z)^3 + \Omega_r (1+z)^4 + \Omega_k (1+z)^2 + \Omega_{\rm DE}(z)

where:

- :math:`\Omega_m`: Matter density parameter
- :math:`\Omega_r`: Radiation density parameter
- :math:`\Omega_k`: Curvature density parameter
- :math:`\Omega_{\rm DE}(z)`: Dark energy density evolution

**Normalization condition**: At :math:`z=0`, :math:`E(0) = 1`, which requires:

.. math::

   \Omega_m + \Omega_r + \Omega_k + \Omega_\Lambda = 1

Distance Calculations
~~~~~~~~~~~~~~~~~~~~~

All cosmological distances can be derived from :math:`E(z)`:

Comoving distance:

.. math::

   d_C(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}

Transverse comoving distance (accounting for curvature):

.. math::

   D_M(z) = \begin{cases}
   D_H / \sqrt{\Omega_k} \sinh(\sqrt{\Omega_k} \, d_C / D_H) & \Omega_k > 0 \\
   d_C & \Omega_k = 0 \\
   D_H / \sqrt{|\Omega_k|} \sin(\sqrt{|\Omega_k|} \, d_C / D_H) & \Omega_k < 0
   \end{cases}

where :math:`D_H = c/H_0` is the Hubble distance.

Luminosity distance:

.. math::

   d_L(z) = (1+z) \, D_M(z)

Angular diameter distance:

.. math::

   D_A(z) = \frac{D_M(z)}{1+z}

Distance modulus:

.. math::

   \mu = 5 \log_{10}\left(\frac{d_L}{\text{Mpc}}\right) + 25

LCDM Model
-----------

The standard :math:`\Lambda\text{CDM}` model assumes dark energy is a cosmological constant, :math:`w = -1`.

LCDM Mathematical Form
~~~~~~~~~~~~~~~~~~~~~~~

Hubble parameter:

.. math::

   E^2(z) = \Omega_m (1+z)^3 + \Omega_r (1+z)^4 + \Omega_k (1+z)^2 + \Omega_\Lambda

Dark energy equation of state:

.. math::

   w(z) = -1 \quad \text{(constant)}

LCDM Parameters
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 12 15 55

   * - Parameter
     - Default
     - Range
     - Description
   * - ``H0``
     - 67.36
     - [50, 100]
     - Hubble constant in km/s/Mpc
   * - ``Omega_m``
     - 0.3153
     - [0.01, 0.99]
     - Total matter density parameter
   * - ``Omega_b``
     - 0.0493
     - [0.01, 0.1]
     - Baryon density parameter
   * - ``Omega_k``
     - 0.0
     - [-0.3, 0.3]
     - Curvature density parameter (0 = flat universe)
   * - ``Omega_r``
     - Auto-computed
     - --
     - Radiation density parameter, computed from :math:`T_{\rm CMB}`
   * - ``sigma8``
     - 0.8111
     - [0.5, 1.2]
     - Matter power spectrum normalization
   * - ``n_s``
     - 0.9649
     - [0.8, 1.1]
     - Primordial power spectrum scalar spectral index
   * - ``T_cmb``
     - 2.7255
     - --
     - CMB temperature (K)
   * - ``N_eff``
     - 3.046
     - --
     - Effective number of neutrino species

LCDM Basic Usage
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import LCDM

   # Use Planck 2018 default parameters
   model = LCDM()

   # Or specify parameters
   model = LCDM(H0=70.0, Omega_m=0.3, Omega_k=0.01)

   # Compute distances
   z = 1.0
   d_L = model.luminosity_distance(z)      # Luminosity distance [Mpc]
   D_A = model.angular_diameter_distance(z)  # Angular diameter distance [Mpc]
   D_M = model.transverse_comoving_distance(z)  # Transverse comoving distance [Mpc]
   mu = model.distance_modulus(z)          # Distance modulus

   print(f"d_L(z=1) = {d_L:.2f} Mpc")
   print(f"D_A(z=1) = {D_A:.2f} Mpc")
   print(f"mu(z=1) = {mu:.2f}")

Parameter Consistency and Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All model parameters use ``model.params`` as the single source of truth
(internally ``CosmologicalParameters``). The recommended way to update
parameters is to **rebuild the instance**: this matches JAX's immutable
style and prevents already-JIT-compiled distance methods from silently
using stale parameter values.

.. code-block:: python

   from hicosmo.models import LCDM, wCDM, CPL, ILCDM

   model = LCDM(H0=70.0, Omega_m=0.3)

   # NOTE: the wCDM equation-of-state parameter is named w0 (NOT w).
   # Misspelled names are ignored by the physics kernel; the likelihood
   # layer emits a UserWarning when it sees an unrecognized name.
   w_model = wCDM(w0=-1.0)
   w_model = wCDM(w0=-0.9)          # updating a parameter = rebuilding

   cpl = CPL(w0=-0.95, wa=0.2)

   ilcdm = ILCDM(beta=0.05)

.. warning::

   Do not update parameters through attribute assignment (e.g.
   ``w_model.w0 = -0.9``): it merely creates an unrelated instance
   attribute and leaves ``params`` unchanged.

Sound Horizon Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~

The LCDM model includes the calculation of the sound horizon at the drag epoch :math:`r_d`, a key quantity for BAO analysis.

Sound horizon (Eisenstein & Hu 1998 fitting formula):

.. math::

   r_d = \frac{44.5 \ln(9.83 / \Omega_m h^2)}{\sqrt{1 + 10 (\Omega_b h^2)^{3/4}}} \text{ Mpc}

.. code-block:: python

   # Get the sound horizon
   rd = model.sound_horizon()  # Mpc
   rd_h = model.rd_h           # r_d * h (dimensionless)

   print(f"r_d = {rd:.2f} Mpc")

Growth Functions
~~~~~~~~~~~~~~~~

The LCDM model provides calculations of the linear growth factor and growth rate.

Growth factor D(z) (Carroll, Press & Turner 1992 approximation):

.. math::

   D(z) \propto \frac{5 \Omega_m E(z)}{2} \int_z^\infty \frac{(1+z') dz'}{E^3(z')}

Growth rate:

.. math::

   f(z) = \frac{d \ln D}{d \ln a} \approx \Omega_m(z)^\gamma, \quad \gamma \approx 0.55

.. code-block:: python

   # Growth factor (normalized to z=0)
   D = model.growth_factor(z=0.5)

   # Growth rate
   f = model.growth_rate(z=0.5)

   # f*sigma8 parameter (commonly used in RSD analysis)
   fsig8 = model.fsigma8(z=0.5)

   print(f"D(z=0.5) = {D:.4f}")
   print(f"f(z=0.5) = {f:.4f}")
   print(f"f*sigma8(z=0.5) = {fsig8:.4f}")

wCDM Model
-----------

The wCDM model generalizes the dark energy equation of state to a constant :math:`w \neq -1`.

wCDM Mathematical Form
~~~~~~~~~~~~~~~~~~~~~~~

Hubble parameter:

.. math::

   E^2(z) = \Omega_m (1+z)^3 + \Omega_r (1+z)^4 + \Omega_k (1+z)^2 + \Omega_\Lambda (1+z)^{3(1+w)}

Dark energy equation of state:

.. math::

   w(z) = w_0 \quad \text{(constant)}

wCDM Parameters
~~~~~~~~~~~~~~~~

wCDM inherits all LCDM parameters and adds:

.. list-table::
   :header-rows: 1
   :widths: 18 12 20 50

   * - Parameter
     - Default
     - Prior Range
     - Description
   * - ``w`` (or ``w0``)
     - -1.0
     - [-2.5, 0.0]
     - Constant dark energy equation of state

Physical constraints:

- :math:`w = -1`: Cosmological constant (LCDM)
- :math:`w > -1`: Quintessence-type dark energy
- :math:`w < -1`: Phantom dark energy (violates energy conditions)

wCDM Basic Usage
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import wCDM

   # Create a wCDM model
   model = wCDM(H0=70.0, Omega_m=0.3, w=-0.9)

   # All distance methods inherited from LCDM
   d_L = model.luminosity_distance(1.0)

   # View the equation of state
   w = model.w_z(0.5)  # Returns constant w
   print(f"w(z=0.5) = {w:.2f}")

   # E(z) now includes the w parameter
   E = model.E_z(1.0)
   print(f"E(z=1) = {E:.4f}")

CPL Model
---------

The CPL (Chevallier-Polarski-Linder) parameterization allows the dark energy equation of state to evolve with time.

CPL Mathematical Form
~~~~~~~~~~~~~~~~~~~~~

Equation of state parameterization:

.. math::

   w(a) = w_0 + w_a (1 - a) = w_0 + w_a \frac{z}{1+z}

where :math:`a = 1/(1+z)` is the scale factor.

Dark energy density evolution:

.. math::

   f_{\rm DE}(z) = \exp\left( \frac{3 w_a z}{1+z} \right) \cdot (1+z)^{3(1 + w_0 + w_a)}

Hubble parameter:

.. math::

   E^2(z) = \Omega_m (1+z)^3 + \Omega_r (1+z)^4 + \Omega_k (1+z)^2 + \Omega_\Lambda \, f_{\rm DE}(z)

CPL Parameters
~~~~~~~~~~~~~~

CPL inherits all LCDM parameters and adds:

.. list-table::
   :header-rows: 1
   :widths: 15 12 18 55

   * - Parameter
     - Default
     - Prior Range
     - Description
   * - ``w0``
     - -1.0
     - [-2.5, 0.5]
     - Dark energy equation of state today (:math:`z=0`)
   * - ``wa``
     - 0.0
     - [-3.0, 2.0]
     - Equation of state evolution parameter

Special cases:

- :math:`w_0 = -1, w_a = 0`: LCDM
- :math:`w_a = 0`: wCDM (constant :math:`w`)
- :math:`w_a \neq 0`: Dynamic dark energy

CPL Basic Usage
~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import CPL

   # Create a CPL model
   model = CPL(H0=67.36, Omega_m=0.3, w0=-1.0, wa=0.1)

   # View the evolving equation of state
   z_values = [0.0, 0.5, 1.0, 2.0]
   for z in z_values:
       w = model.w_z(z)
       print(f"w(z={z}) = {w:.3f}")

   # Output:
   # w(z=0.0) = -1.000
   # w(z=0.5) = -0.967
   # w(z=1.0) = -0.950
   # w(z=2.0) = -0.933

   # Distance calculation
   d_L = model.luminosity_distance(1.0)
   print(f"d_L(z=1) = {d_L:.2f} Mpc")

Dark Energy Figure of Merit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CPL parameterization is commonly used to compute the dark energy Figure of Merit (FoM):

.. math::

   \text{FoM} = \frac{1}{\sqrt{\det(\text{Cov}(w_0, w_a))}}

A larger FoM indicates tighter constraints on the dark energy equation of state.

ILCDM Model
------------

The ILCDM (Interacting Lambda-CDM) model assumes energy exchange between dark matter and dark energy.

Physical Picture
~~~~~~~~~~~~~~~~

In standard LCDM, dark matter and dark energy evolve independently. ILCDM introduces an interaction term:

.. math::

   Q = \beta H \rho_c

where :math:`\beta` is a dimensionless coupling constant and :math:`\rho_c` is the cold dark matter density.

Energy conservation equations:

.. math::

   \dot{\rho}_c + 3H\rho_c = -Q = -\beta H \rho_c

.. math::

   \dot{\rho}_\Lambda = +Q = \beta H \rho_c

Density evolution:

.. math::

   \rho_c(z) = \rho_{c,0} (1+z)^{3 - \beta}

ILCDM Mathematical Form
~~~~~~~~~~~~~~~~~~~~~~~~

Hubble parameter:

.. math::

   E^2(z) = \Omega_\Lambda + \Omega_b (1+z)^3 + \Omega_r (1+z)^4 + \Omega_c \left[ \frac{\beta}{\beta - 1} + \frac{1}{1-\beta} (1+z)^{3-3\beta} \right]

where :math:`\Omega_c = \Omega_m - \Omega_b` is the cold dark matter density parameter.

ILCDM Parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 12 18 55

   * - Parameter
     - Default
     - Prior Range
     - Description
   * - ``beta``
     - 0.0
     - [-1.0, 1.0]
     - Dark energy--dark matter coupling constant

Physical meaning:

- :math:`\beta = 0`: Standard LCDM
- :math:`\beta > 0`: Dark matter decays into dark energy
- :math:`\beta < 0`: Dark energy decays into dark matter

**Note**: ILCDM assumes a flat universe (:math:`\Omega_k = 0`).

ILCDM Basic Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import ILCDM

   # Create an ILCDM model
   model = ILCDM(H0=67.36, Omega_m=0.3, beta=0.05)

   # Distance calculation
   d_L = model.luminosity_distance(1.0)

   # Compare different beta values
   for beta in [-0.1, 0.0, 0.1]:
       m = ILCDM(H0=67.36, Omega_m=0.3, beta=beta)
       d_L = m.luminosity_distance(1.0)
       print(f"beta={beta:+.1f}: d_L(z=1) = {d_L:.2f} Mpc")

Traced-Aware Interface
-----------------------

For compatibility with the NumPyro MCMC sampler, all models provide a ``compute_grid_traced`` static method that supports JAX tracer propagation.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import LCDM, CPL
   import jax.numpy as jnp

   # Define a redshift grid
   z_grid = jnp.linspace(0.01, 2.0, 1000)

   # Parameter dictionary (for MCMC sampling)
   # Only H0 / Omega_m and model-specific parameters are needed
   params = {'H0': 70.0, 'Omega_m': 0.3}

   # Batch-compute all distances
   grid = LCDM.compute_grid_traced(z_grid, params)

   # The returned dictionary contains multiple quantities
   d_L = grid['d_L']     # Luminosity distance
   D_M = grid['D_M']     # Transverse comoving distance
   D_H = grid['D_H']     # Hubble distance (c/H(z))
   E_z = grid['E_z']     # Dimensionless Hubble parameter
   d_C = grid['d_C']     # Comoving distance
   dVc_dz = grid['dVc_dz']  # Comoving volume element

CPL Model Example
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # CPL requires additional parameters
   cpl_params = {
       'H0': 67.36,
       'Omega_m': 0.3,
       'w0': -1.0,
       'wa': 0.1
   }

   grid = CPL.compute_grid_traced(z_grid, cpl_params)
   d_L = grid['d_L']

wCDM / ILCDM Parameter Dictionary Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hicosmo.models import wCDM, ILCDM

   w_params = {'H0': 70.0, 'Omega_m': 0.3, 'w': -0.9}
   grid_w = wCDM.compute_grid_traced(z_grid, w_params)

   ilcdm_params = {'H0': 70.0, 'Omega_m': 0.3, 'beta': 0.05}
   grid_ilcdm = ILCDM.compute_grid_traced(z_grid, ilcdm_params)

Using with NumPyro
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpyro
   import numpyro.distributions as dist

   def model(z_obs, mu_obs, mu_err):
       # Sample cosmological parameters
       H0 = numpyro.sample('H0', dist.Uniform(60, 80))
       Omega_m = numpyro.sample('Omega_m', dist.Uniform(0.1, 0.5))

       params = {'H0': H0, 'Omega_m': Omega_m}

       # Compute theoretical distance modulus
       grid = LCDM.compute_grid_traced(z_obs, params)
       d_L = grid['d_L']
       mu_theory = 5 * jnp.log10(d_L) + 25

       # Likelihood
       numpyro.sample('obs', dist.Normal(mu_theory, mu_err), obs=mu_obs)

Preset Parameters
-----------------

HIcosmo provides commonly used cosmological parameter presets:

Planck 2018
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Value
     - Description
   * - :math:`H_0`
     - 67.36
     - Hubble constant [km/s/Mpc]
   * - :math:`\Omega_m`
     - 0.3153
     - Matter density parameter
   * - :math:`\Omega_b`
     - 0.0493
     - Baryon density parameter
   * - :math:`\sigma_8`
     - 0.8111
     - Matter power spectrum normalization
   * - :math:`n_s`
     - 0.9649
     - Primordial power spectrum scalar spectral index

.. code-block:: python

   # Use Planck 2018 default parameters
   model = LCDM()  # Default is Planck 2018

   # Or explicitly use a preset
   from hicosmo.parameters import ParameterRegistry
   registry = ParameterRegistry.from_defaults('planck2018')

WMAP9
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Value
     - Description
   * - :math:`H_0`
     - 70.0
     - Hubble constant [km/s/Mpc]
   * - :math:`\Omega_m`
     - 0.279
     - Matter density parameter
   * - :math:`\Omega_b`
     - 0.046
     - Baryon density parameter

Adding New Cosmological Models
------------------------------

HIcosmo adopts a minimalist model architecture design, allowing users to create new cosmological models by defining only the core physics formulas. This section describes in detail how to add custom models.

Design Principles
~~~~~~~~~~~~~~~~~

HIcosmo's model architecture follows these core principles:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Principle
     - Description
   * - **Minimal subclass**
     - Subclasses only define differential physics + necessary parameters
   * - **Unified signature**
     - All core computation functions use the ``func(z, params)`` signature
   * - **User-first naming**
     - Public API uses the intuitive name ``E_z``
   * - **Decorator generation**
     - Boilerplate code is auto-generated by decorators
   * - **Physics-framework separation**
     - Subclasses handle physics only; framework logic lives in base class/decorator

Model Inheritance Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   CosmologyBase (abstract base class)
       |
       └── LCDM (reference implementation)
               |
               ├── wCDM (constant w)
               ├── CPL  (evolving w(z))
               └── ILCDM (interaction)

All custom models should inherit from ``LCDM`` or its subclasses.

Minimal Subclass Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~

A minimal cosmological model requires only the following:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Must Define
     - May Define
     - Never Define
   * - ``__init__``
     - ``w_z()`` (if different)
     - ``_E_z_static``
   * - ``@staticmethod E_z(z, params)``
     - ``get_parameters()`` (if new parameters exist)
     - Distance calculation methods
   * -
     -
     - Factory methods

**Core idea**: Only define the physics formulas that differ from the parent class; everything else is handled automatically by the decorator and base class.

Complete Example: Creating a Quintessence Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below we create a simple Quintessence dark energy model with the equation of state:

.. math::

   w(z) = w_0 + w_1 \ln(1+z)

**Step 1: Create the model class**

.. code-block:: python

   # hicosmo/models/quintessence.py

   from typing import Dict, Union
   import jax.numpy as jnp

   from .lcdm import LCDM
   from .base import register_cosmology_model


   @register_cosmology_model
   class Quintessence(LCDM):
       """
       Quintessence dark energy model.

       Equation of state parameterization: w(z) = w0 + w1 * ln(1+z)

       Physical background:
       - Effective equation of state under scalar field slow-roll approximation
       - w1 > 0: dark energy is "stiffer" in the early universe
       - w1 < 0: dark energy is "softer" in the early universe
       """

       def __init__(self, w0: float = -1.0, w1: float = 0.0, **kwargs):
           """
           Initialize the Quintessence model.

           Parameters
           ----------
           w0 : float
               Dark energy equation of state today (z=0), default -1.0
           w1 : float
               Logarithmic evolution coefficient of the equation of state, default 0.0
           **kwargs
               Other parameters passed to LCDM (H0, Omega_m, etc.)
           """
           super().__init__(w0=w0, w1=w1, **kwargs)

       @staticmethod
       def E_z(z: Union[float, jnp.ndarray], params: Dict) -> jnp.ndarray:
           """
           Compute the dimensionless Hubble parameter E(z) = H(z)/H0.

           This is the core of the model: just define the physics formula!

           Mathematical form:
           For w(z) = w0 + w1*ln(1+z), the dark energy density evolves as:
           f_DE(z) = (1+z)^{3(1+w0)} * exp(3*w1*[(1+z)*ln(1+z) - z])
           """
           z_arr = jnp.asarray(z)
           one_plus_z = 1.0 + z_arr

           # Get parameters from the params dictionary
           Omega_m = params['Omega_m']
           Omega_r = params.get('Omega_r', 0.0)
           Omega_k = params.get('Omega_k', 0.0)
           Omega_Lambda = params.get('Omega_Lambda', 1.0 - Omega_m - Omega_k - Omega_r)
           w0 = params.get('w0', -1.0)
           w1 = params.get('w1', 0.0)

           # Dark energy density evolution factor
           # Obtained by integrating w(z) = w0 + w1*ln(1+z)
           log_term = one_plus_z * jnp.log(one_plus_z) - z_arr
           f_DE = one_plus_z**(3.0 * (1.0 + w0)) * jnp.exp(3.0 * w1 * log_term)

           # E^2(z) = Omega_m*(1+z)^3 + Omega_r*(1+z)^4 + Omega_k*(1+z)^2 + Omega_Lambda*f_DE
           E_squared = (
               Omega_m * one_plus_z**3 +
               Omega_r * one_plus_z**4 +
               Omega_k * one_plus_z**2 +
               Omega_Lambda * f_DE
           )
           return jnp.sqrt(E_squared)

       def w_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
           """
           Dark energy equation of state w(z) = w0 + w1 * ln(1+z).

           Only override this method when the equation of state differs from LCDM.
           """
           z_arr = jnp.asarray(z)
           w0 = self.params.get('w0', -1.0)
           w1 = self.params.get('w1', 0.0)
           return w0 + w1 * jnp.log(1.0 + z_arr)

       @classmethod
       def get_parameters(cls):
           """
           Return the model parameter list.

           Only override this method when the model introduces new parameters.
           """
           from ..parameters import Parameter

           # Get parent class parameters
           params = LCDM.get_parameters()

           # Add new parameters
           params.extend([
               Parameter(
                   name='w0',
                   value=-1.0,
                   free=False,
                   prior={'dist': 'uniform', 'min': -2.0, 'max': 0.0},
                   latex_label=r'$w_0$',
                   description='Dark energy equation of state at z=0'
               ),
               Parameter(
                   name='w1',
                   value=0.0,
                   free=False,
                   prior={'dist': 'uniform', 'min': -1.0, 'max': 1.0},
                   latex_label=r'$w_1$',
                   description='Logarithmic evolution coefficient'
               ),
           ])
           return params

**Step 2: Register the model (optional)**

If you want the model to be available in ``hicosmo.models``, add it to ``hicosmo/models/__init__.py``:

.. code-block:: python

   from .quintessence import Quintessence

   __all__ = ['LCDM', 'wCDM', 'CPL', 'ILCDM', 'Quintessence']

**Step 3: Use the new model**

.. code-block:: python

   from hicosmo.models import Quintessence

   # Create model instance
   model = Quintessence(H0=67.36, Omega_m=0.3, w0=-0.9, w1=0.1)

   # All distance methods are automatically available (inherited from LCDM)
   d_L = model.luminosity_distance(1.0)
   D_A = model.angular_diameter_distance(1.0)

   # View the evolving equation of state
   import jax.numpy as jnp
   z = jnp.array([0.0, 0.5, 1.0, 2.0])
   w = model.w_z(z)
   print(f"w(z) = {w}")

   # Use in MCMC
   params = {'H0': 70.0, 'Omega_m': 0.3, 'w0': -0.9, 'w1': 0.1}
   z_grid = jnp.linspace(0.01, 2.0, 100)
   grid = Quintessence.compute_grid_traced(z_grid, params)
   d_L_array = grid['d_L']

What the Decorator Does
~~~~~~~~~~~~~~~~~~~~~~~

The ``@register_cosmology_model`` decorator automatically generates the following for your model:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Generated Item
     - Description
   * - ``_E_z_kernel``
     - Converted from the user-defined ``E_z``
   * - ``_E_z_static``
     - JIT-compiled version for high-performance computation
   * - ``E_z(self, z)``
     - Instance method wrapper, automatically passes ``self.params``
   * - ``compute_grid_traced``
     - MCMC factory method, supports JAX tracer
   * - ``sound_horizon_traced``
     - Sound horizon computation factory method
   * - ``sound_horizon_drag_traced``
     - Drag epoch sound horizon factory method
   * - ``recombination_redshift_traced``
     - Recombination redshift computation factory method

**Core advantage**: You only write the physics formula; the decorator handles all framework code!

Parameter Dictionary Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``params`` in ``E_z(z, params)`` is a dictionary containing all cosmological parameters:

**Required parameters**:

.. code-block:: python

   params = {
       'Omega_m': 0.3,    # Required: matter density parameter
   }

**Optional parameters** (with defaults):

.. code-block:: python

   params = {
       'Omega_m': 0.3,
       'Omega_r': 0.0,              # Radiation density, default 0
       'Omega_k': 0.0,              # Curvature, default 0 (flat)
       'Omega_Lambda': 0.7,         # Dark energy density, default 1-Omega_m-Omega_k-Omega_r
       # Model-specific parameters
       'w': -1.0,                   # wCDM
       'w0': -1.0, 'wa': 0.0,       # CPL
       'beta': 0.0,                 # ILCDM
   }

**Recommended pattern**: Use ``params.get('key', default)`` to provide defaults:

.. code-block:: python

   Omega_r = params.get('Omega_r', 0.0)  # Use 0.0 if not present

Testing New Models
~~~~~~~~~~~~~~~~~~

After creating a new model, write tests to verify its correctness:

.. code-block:: python

   # tests/test_quintessence.py

   import pytest
   import jax.numpy as jnp
   from hicosmo.models import Quintessence, LCDM


   class TestQuintessence:
       """Quintessence model test suite"""

       def test_reduces_to_lcdm(self):
           """Should reduce to LCDM when w0=-1, w1=0"""
           quint = Quintessence(H0=67.36, Omega_m=0.3, w0=-1.0, w1=0.0)
           lcdm = LCDM(H0=67.36, Omega_m=0.3)

           z = jnp.array([0.5, 1.0, 2.0])

           # E(z) should be identical
           E_quint = quint.E_z(z)
           E_lcdm = lcdm.E_z(z)
           assert jnp.allclose(E_quint, E_lcdm, rtol=1e-10)

           # Distances should also be identical
           d_L_quint = quint.luminosity_distance(1.0)
           d_L_lcdm = lcdm.luminosity_distance(1.0)
           assert jnp.isclose(d_L_quint, d_L_lcdm, rtol=1e-10)

       def test_equation_of_state(self):
           """Test equation of state w(z) = w0 + w1*ln(1+z)"""
           quint = Quintessence(w0=-0.9, w1=0.1)

           # At z=0, w = w0
           assert jnp.isclose(quint.w_z(0.0), -0.9, rtol=1e-10)

           # At z=e-1, w = w0 + w1*ln(e) = w0 + w1
           z_e = jnp.e - 1
           assert jnp.isclose(quint.w_z(z_e), -0.9 + 0.1, rtol=1e-6)

       def test_compute_grid_traced(self):
           """Test the traced interface"""
           params = {'H0': 70.0, 'Omega_m': 0.3, 'w0': -0.9, 'w1': 0.1}
           z_grid = jnp.linspace(0.01, 2.0, 100)

           grid = Quintessence.compute_grid_traced(z_grid, params)

           # Check return values
           assert 'd_L' in grid
           assert 'D_M' in grid
           assert 'E_z' in grid

           # Check shapes
           assert grid['d_L'].shape == z_grid.shape

       def test_jax_gradients(self):
           """Test JAX automatic differentiation"""
           from jax import grad

           def loss(H0):
               params = {'H0': H0, 'Omega_m': 0.3, 'w0': -1.0, 'w1': 0.0}
               grid = Quintessence.compute_grid_traced(jnp.array([1.0]), params)
               return grid['d_L'][0]

           # Gradient should exist and be finite
           grad_H0 = grad(loss)(70.0)
           assert jnp.isfinite(grad_H0)

Run tests:

.. code-block:: bash

   pytest tests/test_quintessence.py -v

Common Errors and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error 1: Forgetting the decorator**

.. code-block:: python

   # Wrong: no decorator
   class MyModel(LCDM):
       @staticmethod
       def E_z(z, params):
           ...

   # Correct: use the decorator
   @register_cosmology_model
   class MyModel(LCDM):
       @staticmethod
       def E_z(z, params):
           ...

**Error 2: ``E_z`` is not a static method**

.. code-block:: python

   # Wrong: instance method
   @register_cosmology_model
   class MyModel(LCDM):
       def E_z(self, z, params):  # Should not have self
           ...

   # Correct: static method
   @register_cosmology_model
   class MyModel(LCDM):
       @staticmethod
       def E_z(z, params):
           ...

**Error 3: Wrong parameter signature**

.. code-block:: python

   # Wrong: expanded parameters
   @staticmethod
   def E_z(z, Omega_m, Omega_Lambda, w):
       ...

   # Correct: use params dictionary
   @staticmethod
   def E_z(z, params):
       Omega_m = params['Omega_m']
       Omega_Lambda = params.get('Omega_Lambda', 1.0 - Omega_m)
       w = params.get('w', -1.0)
       ...

**Error 4: Redefining framework methods in subclass**

.. code-block:: python

   # Wrong: these methods do not need to be redefined
   @register_cosmology_model
   class MyModel(LCDM):
       @staticmethod
       def E_z(z, params):
           ...

       @staticmethod
       def _E_z_static(z, params):  # The decorator generates this
           ...

       def compute_grid_traced(self, z, params):  # The decorator generates this
           ...

Performance Optimization
------------------------

JIT Compilation
~~~~~~~~~~~~~~~

All models' core computations are optimized with JAX JIT compilation:

.. code-block:: python

   from jax import jit
   import jax.numpy as jnp
   from hicosmo.models import LCDM

   model = LCDM()

   # First call: compilation
   %timeit model.luminosity_distance(1.0)  # ~100ms (includes compilation)

   # Subsequent calls: compiled
   %timeit model.luminosity_distance(1.0)  # ~0.01ms

Vectorized Computation
~~~~~~~~~~~~~~~~~~~~~~

Use ``vmap`` for efficient batch computation:

.. code-block:: python

   from jax import vmap

   z_array = jnp.linspace(0.01, 2.0, 1000)

   # Vectorized distance computation
   d_L_array = vmap(model.luminosity_distance)(z_array)

Or use ``compute_grid_traced`` directly:

.. code-block:: python

   grid = LCDM.compute_grid_traced(z_array, {'H0': 70.0, 'Omega_m': 0.3})
   d_L_array = grid['d_L']

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Operation
     - scipy (qcosmc)
     - HIcosmo (JAX)
     - Speedup
   * - Distance calculation (1000 points)
     - 150 ms
     - 20 ms
     - **7.5x**
   * - E(z) computation (10000 points)
     - 50 ms
     - 2 ms
     - **25x**
   * - Growth factor
     - 120 ms
     - 15 ms
     - **8x**

Best Practices
--------------

1. **Choose the appropriate model**

   - Use LCDM for standard analyses
   - Use CPL to explore dark energy properties
   - Use ILCDM to study dark matter--dark energy interaction

2. **Use preset parameters**

   .. code-block:: python

      # Use Planck priors
      model = LCDM()  # Default Planck 2018

3. **Batch computation optimization**

   .. code-block:: python

      # Recommended: compute multiple redshifts at once
      grid = LCDM.compute_grid_traced(z_array, params)

      # Avoid: loop calls
      for z in z_array:
          d_L = model.luminosity_distance(z)

4. **Use traced interface for MCMC**

   .. code-block:: python

      # Use static methods in NumPyro models
      grid = LCDM.compute_grid_traced(z_obs, params)

Next Steps
----------

- `Likelihood Functions <likelihoods.html>`_: Learn how to use observational data
- `Samplers <samplers.html>`_: Learn about MCMC parameter estimation
- `Fisher Forecasts <fisher.html>`_: Perform survey predictions
