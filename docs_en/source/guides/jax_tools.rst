JAX Numerical Tools Reference
=============================

The ``hicosmo.utils.jax_tools`` module provides a complete set of JAX-optimized numerical tools, including integration, differentiation, root-finding, and ODE solvers.
All functions support JIT compilation and automatic differentiation.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start
-----------

.. code-block:: python

   from hicosmo.utils import (
       integrate_simpson,      # Simpson integration
       gauss_legendre_integrate,  # Gauss-Legendre integration
       cumulative_trapezoid,   # Cumulative trapezoidal integration
       gradient_1d,            # 1D gradient
       newton_root,            # Newton's method root-finding
       odeint_rk4,             # RK4 ODE solver
   )


Integration Tools
-----------------

Trapezoidal Integration
^^^^^^^^^^^^^^^^^^^^^^^

.. function:: trapezoid(y, x, axis=-1)

   JAX trapezoidal integration wrapper.

   :param y: Array of function values
   :param x: Array of independent variable values
   :param axis: Integration axis
   :return: Integration result

   **Example**:

   .. code-block:: python

      import jax.numpy as jnp
      from hicosmo.utils import trapezoid

      x = jnp.linspace(0, 1, 100)
      y = x**2
      result = trapezoid(y, x)  # ≈ 1/3


Simpson Integration
^^^^^^^^^^^^^^^^^^^

.. function:: simpson(y, x)

   Composite Simpson integration for discretely sampled data.

   :param y: Array of function values (at least 3 points)
   :param x: Array of independent variable values
   :return: Integration result
   :raises ValueError: If fewer than 3 points are provided

.. function:: integrate_simpson(func, a, b, *, num=512)

   Integrate a function over the interval [a, b] using Simpson's rule.

   :param func: Integrand f(x)
   :param a: Lower integration limit
   :param b: Upper integration limit
   :param num: Number of sampling points (default 512, automatically adjusted to even)
   :return: Integration result

   **Example**:

   .. code-block:: python

      from hicosmo.utils import integrate_simpson
      import jax.numpy as jnp

      # Compute integral of x^2 from 0 to 1 = 1/3
      result = integrate_simpson(lambda x: x**2, 0, 1)
      print(f"Result: {result:.6f}")  # ≈ 0.333333


Log-Space Integration
^^^^^^^^^^^^^^^^^^^^^

.. function:: integrate_logspace(func, k_min, k_max, *, num=512)

   Integrate on a logarithmic-space grid, suitable for power-law functions with positive domains.

   :param func: Integrand f(k)
   :param k_min: Lower integration limit (must be > 0)
   :param k_max: Upper integration limit
   :param num: Number of sampling points
   :return: Integration result

   **Example**:

   .. code-block:: python

      from hicosmo.utils import integrate_logspace

      # Integrate a power-law function
      result = integrate_logspace(lambda k: k**(-2), 0.01, 100)


Gauss-Legendre Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

High-precision quadrature method, highly effective for smooth functions.

.. function:: gauss_legendre_nodes_weights(order)

   Return Gauss-Legendre nodes and weights for the specified order.

   :param order: Order (supports 8, 12, 16)
   :return: (nodes, weights) tuple

.. function:: gauss_legendre_integrate(integrand, a, b, *, order=12)

   Perform Gauss-Legendre integration over the interval [a, b].

   :param integrand: Integrand function
   :param a: Lower integration limit
   :param b: Upper integration limit
   :param order: Integration order (8, 12, or 16)
   :return: Integration result

.. function:: gauss_legendre_integrate_batch(integrand, z_array, *, z_min=0.0, order=12)

   Batch Gauss-Legendre integration, computing simultaneously for multiple upper limits.

   :param integrand: Integrand function
   :param z_array: Array of upper limits
   :param z_min: Lower integration limit
   :param order: Integration order
   :return: Array of integration results

   **Example**:

   .. code-block:: python

      from hicosmo.utils import gauss_legendre_integrate_batch
      import jax.numpy as jnp

      # Compute integrals for multiple upper limits
      z_values = jnp.array([0.5, 1.0, 2.0])
      results = gauss_legendre_integrate_batch(
          lambda z: 1.0 / jnp.sqrt(0.3 * (1 + z)**3 + 0.7),
          z_values,
          z_min=0.0,
          order=12
      )


Cumulative Integration
^^^^^^^^^^^^^^^^^^^^^^

.. function:: cumulative_trapezoid(y, x)

   Cumulative trapezoidal integration, returning cumulative integral values starting from the first point.

   :param y: Array of function values
   :param x: Array of independent variable values
   :return: Cumulative integral array (first element is 0)

   **Example**:

   .. code-block:: python

      from hicosmo.utils import cumulative_trapezoid
      import jax.numpy as jnp

      x = jnp.linspace(0, 1, 100)
      y = jnp.ones_like(x)  # Constant function
      cumulative = cumulative_trapezoid(y, x)
      # cumulative[-1] ≈ 1.0


.. function:: integrate_batch_cumulative(func, z_array, *, z_min=0.0, z_max=None, n_grid=4096)

   Batch integration using a cumulative grid and interpolation. Suitable for a large number of query points.

   :param func: Integrand function
   :param z_array: Array of query points
   :param z_min: Lower integration limit
   :param z_max: Upper grid limit (defaults to the maximum of z_array)
   :param n_grid: Number of grid points
   :return: Array of integration results

   **Performance tip**: When there are many query points, this method is faster than ``gauss_legendre_integrate_batch``.


Segmented Integration
^^^^^^^^^^^^^^^^^^^^^

.. function:: integrate_segmented(func, a, b, *, n_segments=8, order=12)

   Segmented Gauss-Legendre integration, suitable for large integration intervals.

   :param func: Integrand function
   :param a: Lower integration limit
   :param b: Upper integration limit
   :param n_segments: Number of segments
   :param order: Gauss-Legendre order per segment
   :return: Integration result

   **Use case**: When the integration interval is very large (e.g., [0, 1000]), a single
   Gauss-Legendre integration may lack precision; segmented integration improves accuracy.


Adaptive Simpson Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: integrate_adaptive_simpson(func, a, b, *, num=64, max_num=4096, tol=1e-6)

   Adaptive Simpson integration, automatically adjusting grid density.

   :param func: Integrand function
   :param a: Lower integration limit
   :param b: Upper integration limit
   :param num: Initial number of grid points (must be even)
   :param max_num: Maximum number of grid points (must be an integer multiple of num)
   :param tol: Convergence tolerance
   :return: Integration result


Differentiation Tools
---------------------

1D Gradient
^^^^^^^^^^^

.. function:: gradient_1d(values, coords)

   Compute the derivative on a 1D grid, supporting non-uniform grids.

   :param values: Array of function values
   :param coords: Array of coordinates
   :return: Array of derivatives

   **Features**:
   - Endpoints use 3-point Lagrange interpolation (second-order accuracy)
   - Interior points use second-order central differences
   - Supports non-uniform grids

   **Example**:

   .. code-block:: python

      from hicosmo.utils import gradient_1d
      import jax.numpy as jnp

      x = jnp.linspace(0, 2 * jnp.pi, 100)
      y = jnp.sin(x)
      dy_dx = gradient_1d(y, x)  # ≈ cos(x)


Finite Difference Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: finite_difference_grad(func, x, *, step=1e-5)

   Compute the gradient of a scalar function using central finite differences.

   :param func: Scalar function f(x) -> scalar
   :param x: Evaluation point (array)
   :param step: Difference step size
   :return: Gradient array

   **Note**: For differentiable functions, using ``jax.grad`` for automatic differentiation is recommended.


Automatic Differentiation Wrappers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: grad(func)

   Return the gradient function of a function (``jax.grad`` wrapper).

.. function:: jacobian(func)

   Return the Jacobian matrix function of a function (``jax.jacobian`` wrapper).

.. function:: hessian(func)

   Return the Hessian matrix function of a function (``jax.hessian`` wrapper).

   **Example**:

   .. code-block:: python

      from hicosmo.utils import grad, hessian
      import jax.numpy as jnp

      def f(x):
          return jnp.sum(x**2)

      grad_f = grad(f)
      hess_f = hessian(f)

      x = jnp.array([1.0, 2.0])
      print(grad_f(x))  # [2., 4.]


Root-Finding Tools
------------------

Newton-Raphson Method
^^^^^^^^^^^^^^^^^^^^^

.. function:: newton_root(func, x0, *, tol=1e-8, max_iter=64, deriv=None)

   Solve f(x) = 0 using the Newton-Raphson method.

   :param func: Target function f(x)
   :param x0: Initial guess
   :param tol: Convergence tolerance
   :param max_iter: Maximum number of iterations
   :param deriv: Derivative function (optional, defaults to automatic differentiation)
   :return: Approximate root

   **Example**:

   .. code-block:: python

      from hicosmo.utils import newton_root
      import jax.numpy as jnp

      # Solve x^2 - 2 = 0
      root = newton_root(lambda x: x**2 - 2, jnp.array(1.5))
      print(f"sqrt(2) ≈ {root:.10f}")  # 1.4142135624


Bisection Method
^^^^^^^^^^^^^^^^

.. function:: bisection_root(func, a, b, *, tol=1e-8, max_iter=128)

   Solve f(x) = 0 using the bisection method (requires opposite signs at interval endpoints).

   :param func: Target function f(x)
   :param a: Left endpoint of the interval
   :param b: Right endpoint of the interval
   :param tol: Convergence tolerance
   :param max_iter: Maximum number of iterations
   :return: Approximate root (returns NaN if endpoints have the same sign)

   **Example**:

   .. code-block:: python

      from hicosmo.utils import bisection_root
      import jax.numpy as jnp

      # Solve x^3 - x - 1 = 0
      root = bisection_root(
          lambda x: x**3 - x - 1,
          jnp.array(1.0),
          jnp.array(2.0)
      )


ODE Solvers
-----------

RK4 Single Step
^^^^^^^^^^^^^^^

.. function:: rk4_step(func, t, y, dt)

   Perform one step of fourth-order Runge-Kutta integration.

   :param func: ODE right-hand side f(t, y)
   :param t: Current time
   :param y: Current state
   :param dt: Time step
   :return: State at the next time step


RK4 Integrator
^^^^^^^^^^^^^^

.. function:: odeint_rk4(func, y0, t_grid)

   Solve ODE y' = f(t, y) on a fixed grid using the RK4 method.

   :param func: ODE right-hand side f(t, y)
   :param y0: Initial condition
   :param t_grid: Time grid
   :return: Solution array (one row per time point)

   **Example**:

   .. code-block:: python

      from hicosmo.utils import odeint_rk4
      import jax.numpy as jnp

      # Solve y' = -y, y(0) = 1 (analytical solution: y = e^(-t))
      def rhs(t, y):
          return -y

      t = jnp.linspace(0, 5, 100)
      y = odeint_rk4(rhs, jnp.array(1.0), t)


.. function:: solve_ivp_rk4(func, y0, t0, t1, *, n_steps=256)

   Solve an initial value problem on the interval [t0, t1] using fixed-step RK4.

   :param func: ODE right-hand side f(t, y)
   :param y0: Initial condition
   :param t0: Start time
   :param t1: End time
   :param n_steps: Number of steps
   :return: (t_grid, y_grid) tuple


Cosmology-Specific Tools
-------------------------

Sound Horizon Computation
^^^^^^^^^^^^^^^^^^^^^^^^^

Based on the Eisenstein & Hu (1998) fitting formula.

.. function:: sound_horizon_drag_eh98(H0, Omega_m, Omega_b, T_cmb)

   Compute the sound horizon at the drag epoch r_d.

   :param H0: Hubble constant [km/s/Mpc]
   :param Omega_m: Matter density parameter
   :param Omega_b: Baryon density parameter
   :param T_cmb: CMB temperature [K]
   :return: Sound horizon r_d [Mpc]

   **Example**:

   .. code-block:: python

      from hicosmo.utils.jax_tools import sound_horizon_drag_eh98

      r_d = sound_horizon_drag_eh98(
          H0=67.36,
          Omega_m=0.3153,
          Omega_b=0.0493,
          T_cmb=2.7255
      )
      print(f"r_d = {r_d:.2f} Mpc")  # ≈ 147 Mpc


.. function:: sound_horizon_eh98(H0, Omega_m, Omega_b, T_cmb, z)

   Compute the sound horizon at an arbitrary redshift z.

   :param H0: Hubble constant [km/s/Mpc]
   :param Omega_m: Matter density parameter
   :param Omega_b: Baryon density parameter
   :param T_cmb: CMB temperature [K]
   :param z: Redshift
   :return: Sound horizon r_s(z) [Mpc]


Performance Comparison
----------------------

Applicable scenarios for each integration method:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Method
     - Accuracy
     - Use Case
   * - ``trapezoid``
     - O(h²)
     - Quick rough estimates
   * - ``simpson``
     - O(h⁴)
     - Standard choice for smooth functions
   * - ``gauss_legendre``
     - High
     - High-precision integration over small intervals
   * - ``integrate_batch_cumulative``
     - O(h²)
     - Batch integration with many query points
   * - ``integrate_segmented``
     - High
     - High-precision integration over large intervals


JIT Compilation Notes
---------------------

All functions are compatible with JAX JIT compilation:

.. code-block:: python

   from jax import jit
   from hicosmo.utils import integrate_simpson

   @jit
   def my_function(params):
       result = integrate_simpson(
           lambda x: params[0] * x**2 + params[1],
           0, 1
       )
       return result

**Notes**:

1. Integration limits [a, b] can be dynamic values
2. Grid point counts (num, n_grid) must be static values
3. Use ``static_argnums`` for parameters that need to be static


Complete Example
----------------

Computing Comoving Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   from hicosmo.utils import integrate_batch_cumulative

   # Cosmological parameters
   H0 = 70.0  # km/s/Mpc
   Omega_m = 0.3
   c = 299792.458  # km/s

   def E_z(z):
       """Hubble parameter E(z) = H(z)/H0"""
       return jnp.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

   # Compute comoving distance for multiple redshifts
   z_array = jnp.array([0.1, 0.5, 1.0, 2.0])
   integrals = integrate_batch_cumulative(
       lambda z: 1.0 / E_z(z),
       z_array,
       n_grid=4096
   )
   d_c = (c / H0) * integrals

   for z, d in zip(z_array, d_c):
       print(f"z = {z:.1f}: d_c = {d:.1f} Mpc")


References
----------

- Eisenstein, D. J., & Hu, W. (1998). Baryonic Features in the Matter Transfer Function. ApJ, 496, 605.
