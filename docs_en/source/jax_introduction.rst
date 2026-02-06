JAX Technical Deep Dive
=======================

This chapter provides a detailed introduction to JAX's core concepts and technical advantages, as well as why HIcosmo chose JAX as its computation engine.
If you are unfamiliar with JAX, this chapter will help you understand why JAX is a revolutionary tool for modern scientific computing.

What is JAX?
------------

JAX is a high-performance numerical computing library developed by Google. It can be thought of as **"differentiable, compilable, parallelizable NumPy"**.
Its name derives from **J**\ ust-in-time compilation + **A**\ utomatic differentiation + **X**\ LA (Accelerated Linear Algebra).

**Official definition**:

   JAX is NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research.

**In plain terms**:

- If you know NumPy, you know JAX (the APIs are nearly identical)
- JAX can automatically compute derivatives of any function (including higher-order derivatives)
- JAX can automatically compile code into efficient machine code
- JAX can transparently run on CPUs, GPUs, and TPUs

JAX is regarded as a **differentiable programming language** that uses Python as a "metaprogramming language" for building XLA computation graphs.

Functional Programming: The Core Philosophy of JAX
---------------------------------------------------

The design core of JAX is **Functional Programming (FP)**, which fundamentally differs from traditional object-oriented programming (OOP) frameworks such as PyTorch.

Why Functional Programming?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main reasons JAX adopts functional programming include:

1. **Mathematical consistency**: Automatic differentiation is inherently a transformation of functions. Functional programming allows JAX to freely nest and compose ``grad`` (differentiation), ``jit`` (compilation), and ``vmap`` (vectorization) as higher-order functions, e.g.: ``jit(vmap(grad(f)))``

2. **Easier compiler optimization**: The XLA compiler requires a static, side-effect-free computation graph to perform operator fusion and memory optimization. State mutation in OOP (such as modifying class member variables) breaks the construction of such static graphs

3. **Deterministic random state**: Traditional OOP frameworks often use global random state, while JAX mandates explicit passing of pseudo-random number generator (PRNG) keys, ensuring code reproducibility

4. **Separation of state and logic**: Under the functional paradigm, model parameters are passed as function inputs rather than stored inside objects, making parameter distribution and updates more transparent

Pure Functions
~~~~~~~~~~~~~~

Pure functions are the cornerstone of JAX's ability to achieve high-performance computation.

**Definition**: A pure function is a function that **does not alter external state** and has **no side effects**.

**Property**: For the same input, a pure function **must always produce the same output**.

**Why it matters**: The predictability of pure functions allows JAX's XLA compiler to deeply optimize code, enable just-in-time compilation (JIT), and easily achieve operator parallelism and sharding on GPUs/TPUs.

.. code-block:: python

   import jax.numpy as jnp

   # Pure function: same input -> same output, no side effects
   def pure_function(x, y):
       return jnp.sin(x) + jnp.cos(y)

   # Impure function: depends on external state
   global_state = 0
   def impure_function(x):
       global global_state
       global_state += 1  # Side effect: modifies external state
       return x + global_state

Immutable Arrays
~~~~~~~~~~~~~~~~

In JAX, arrays **cannot be modified** once created.

**Difference from NumPy**: Traditional NumPy allows in-place array updates; JAX arrays are immutable.

**How it works**: If you need to modify an element in an array, JAX does not directly overwrite the original memory. Instead, it creates a new array containing the modified content.

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp

   # NumPy: in-place modification
   np_arr = np.array([1, 2, 3])
   np_arr[0] = 99  # Direct modification

   # JAX: immutable, requires creating a new array
   jax_arr = jnp.array([1, 2, 3])
   # jax_arr[0] = 99  # TypeError!
   jax_arr = jax_arr.at[0].set(99)  # Creates a new array

**Design rationale**: Immutability unlocks compiler optimizations, allowing XLA to safely perform operator fusion and memory reuse.

Gradient Computation Comparison: PyTorch vs JAX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below are the core differences between PyTorch (object-oriented) and JAX (functional) in handling gradients:

**PyTorch (Object-Oriented Paradigm)**:

Gradients are treated as an attribute of tensor objects, implemented through state mutation.

.. code-block:: python

   import torch

   def f(x):
       return x**2 + 3*x + 5

   x = torch.tensor([2.0], requires_grad=True)
   loss = f(x)
   loss.backward()  # Triggers AutoGrad and traces back through the computation graph

   # Gradients are "stored" in the .grad attribute of variable x (state change)
   print(x.grad.item())  # Output: 7.0

**JAX (Functional Paradigm)**:

Differentiation is a **function transformation** that does not alter the state of any input or object.

.. code-block:: python

   import jax

   def f(x):
       return x**2 + 3*x + 5

   # jax.grad takes a function and returns another function (function transformation)
   df_dx = jax.grad(f)

   x_value = 2.0
   # The gradient is returned directly as a return value, without modifying the original variable
   derivative_value = df_dx(x_value)

   print(derivative_value)  # Output: 7.0

**Comparison summary**:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - Object-Oriented (PyTorch)
     - Functional (JAX)
   * - **State management**
     - State stored inside objects, allows in-place modification
     - Pure functions, state passed via parameters, immutable
   * - **Gradient handling**
     - Stored in ``.grad`` attribute via ``loss.backward()``
     - Uses ``jax.grad`` to transform functions, returns gradient values directly
   * - **Random numbers**
     - Relies on implicit global random state
     - Requires explicit PRNG key passing
   * - **Underlying core**
     - Dynamic computation graph, easy to debug
     - XLA-compiled static computation graph, pursuing maximum performance

The Four Core Features of JAX
------------------------------

1. NumPy-Compatible API
~~~~~~~~~~~~~~~~~~~~~~~

JAX provides an API that is almost fully compatible with NumPy:

.. code-block:: python

   # NumPy code
   import numpy as np

   x = np.array([1.0, 2.0, 3.0])
   y = np.sin(x) + np.cos(x)
   z = np.dot(x, y)

   # JAX code (just change the import)
   import jax.numpy as jnp

   x = jnp.array([1.0, 2.0, 3.0])
   y = jnp.sin(x) + jnp.cos(x)
   z = jnp.dot(x, y)

**Key differences**:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - NumPy
     - JAX
   * - **Mutability**
     - Arrays are mutable (``x[0] = 5``)
     - Arrays are immutable (``x = x.at[0].set(5)``)
   * - **Random numbers**
     - Global state (``np.random.rand()``)
     - Explicit key (``jax.random.uniform(key)``)
   * - **Execution mode**
     - Eager execution
     - Can be deferred (traced)
   * - **Hardware support**
     - CPU only
     - CPU / GPU / TPU

2. Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatic differentiation is one of JAX's most powerful features. It can automatically compute exact derivatives of arbitrarily complex functions.

**What is automatic differentiation?**

In scientific computing, we frequently need to compute derivatives (gradients) of functions. There are three traditional methods:

1. **Symbolic differentiation**: Manual derivation using mathematical formulas, error-prone and unable to handle complex functions
2. **Numerical differentiation**: Finite difference approximation, :math:`f'(x) \approx \frac{f(x+h) - f(x)}{h}`, low precision and low efficiency
3. **Automatic differentiation**: Automatically traces the computation graph via the chain rule, exact and efficient

**JAX automatic differentiation example**:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   # Define a complex function
   def f(x):
       return jnp.sin(x) * jnp.exp(-x**2) + jnp.log(1 + x**2)

   # Automatically obtain derivative functions
   df = jax.grad(f)      # First derivative
   d2f = jax.grad(df)    # Second derivative
   d3f = jax.grad(d2f)   # Third derivative

   # Compute derivative values
   x = 1.0
   print(f"f(x)   = {f(x):.6f}")
   print(f"f'(x)  = {df(x):.6f}")
   print(f"f''(x) = {d2f(x):.6f}")
   print(f"f'''(x)= {d3f(x):.6f}")

**Jacobian and Hessian of vector functions**:

.. code-block:: python

   # Hessian matrix (used in optimization and Fisher matrix)
   def scalar_func(x):
       return jnp.sum(x**2)

   hessian = jax.hessian(scalar_func)

**Advantages of automatic differentiation**:

- **Exact**: Results are exact mathematical derivatives, not numerical approximations
- **Efficient**: Complexity is the same as the original function, no multiple function evaluations needed
- **General**: Supports arbitrarily complex function compositions, including conditionals and loops

3. JIT Compilation and XLA Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JIT compilation is the core source of JAX's performance. It compiles Python code into optimized machine code.

**How the XLA Compiler Works**

XLA (Accelerated Linear Algebra) is a compiler specifically designed for linear algebra operations. It transforms programs written in JAX into optimized executable kernels for specific hardware (CPU, GPU, or TPU).

The workflow consists of the following stages:

1. **Tracing**: When you apply ``jax.jit`` to a Python function and call it for the first time, JAX uses a "tracing" mechanism to record the sequence of operations performed by the function on arrays with specific shapes and types

2. **Building the computation graph**: The traced information is transformed into an XLA computation graph (HLO, High-Level Optimizer representation)

3. **Hardware compilation and optimization**: The XLA compiler performs comprehensive analysis on the graph, executing optimization strategies including dead code elimination, operator fusion, and more, finally compiling it into binary machine code that runs directly on hardware accelerators

**Operator Fusion**

Operator fusion is one of XLA's most powerful optimization techniques.

**Principle**: In traditional vectorized programming, each simple operation (such as ``A * B + C``) typically generates many intermediate temporary arrays. These intermediate results need to be written back to memory and then read again, causing significant memory bandwidth waste.

**Implementation**: XLA identifies consecutive operations in the computation graph and "fuses" them into a single GPU kernel. This means intermediate computation results are kept directly in the processor's fast registers or cache, rather than frequently exchanging data with main memory.

**Benefit**: Significantly reduces memory bandwidth usage and memory allocation pressure, particularly avoiding OOM (out of memory) errors when processing high-dimensional cosmological data.

**Analogy**:

   Think of traditional Python operations like ordering at a fast-food counter: every time you order a burger, the server runs back to the kitchen (calls a C library), makes it, brings it to you, then you order fries, and the server runs back again.

   The XLA compiler and JIT are like a smart head chef: you tell them you want a combo meal (the entire function), they analyze the menu, decide to fry the fries and grill the patty simultaneously (operator fusion), and finally deliver the entire hot combo meal to you at once.

   This "one-stop" processing greatly reduces the waiting time in transit (Python overhead and memory read/write).

**JIT compilation example**:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import time

   # Define a compute-intensive function
   def slow_function(x):
       for _ in range(100):
           x = jnp.sin(x) + jnp.cos(x)
       return x

   # JIT-compiled version
   fast_function = jax.jit(slow_function)

   x = jnp.ones(10000)

   # First call: includes compilation time
   start = time.time()
   _ = fast_function(x)
   print(f"First call (with compilation): {time.time() - start:.4f}s")

   # Subsequent calls: uses cached compiled result
   start = time.time()
   for _ in range(100):
       _ = fast_function(x)
   print(f"Subsequent 100 calls: {time.time() - start:.4f}s")

4. Vectorization with vmap
~~~~~~~~~~~~~~~~~~~~~~~~~~

``vmap`` is JAX's vectorization transformation that automatically converts scalar functions into batched functions.

**Example**:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   # Define a function that processes a single sample
   def single_sample_loglike(theta, data_point):
       mu, sigma = theta
       return -0.5 * ((data_point - mu) / sigma)**2 - jnp.log(sigma)

   # Automatically batch using vmap
   batch_loglike = jax.vmap(single_sample_loglike, in_axes=(None, 0))

   theta = (0.0, 1.0)
   data = jnp.array([0.1, 0.5, -0.3, 0.8, -0.2])

   # Compute log-likelihood for all data points at once
   log_likes = batch_loglike(theta, data)

**Performance comparison** (1000 data points):

- Python loop: ~50ms
- NumPy vectorized: ~2ms
- JAX vmap + JIT: ~0.1ms

PRNG Random Number Management
------------------------------

JAX's pseudo-random number generator (PRNG) management mechanism is a core embodiment of its functional programming philosophy. Unlike traditional frameworks such as NumPy or PyTorch that use "global random state", JAX employs an **explicit, stateless, and splittable** random number management system.

Core Mechanism: Explicit Keys and Splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In JAX, randomness is not controlled by a global seed that changes over time, but is managed through a **random key (PRNGKey)**.

- **PRNGKey**: This is an array object containing the random state. All random functions (such as ``jax.random.normal``) require a Key as input and generate deterministic output based on that Key

- **Key splitting**: To generate multiple uncorrelated random number streams, JAX uses the ``jax.random.split`` function to decompose a master Key into multiple sub-Keys. This is similar to a tree structure: starting from a root seed, continuously branching into independent random substreams

Why Explicit Key Passing?
~~~~~~~~~~~~~~~~~~~~~~~~~

JAX requires explicit key passing fundamentally because of its pursuit of **pure functions**:

1. **Eliminating side effects**: Traditional global random state is essentially a "global variable". Every call to a random function changes this global state, which constitutes a "side effect". For JIT and operator fusion, functions must be "pure"

2. **Reproducibility**: Explicit keys ensure that regardless of where or in what order computation is executed, as long as the Key is the same, results are completely identical

3. **Parallelization and vectorization**: In large-scale parallel computation (such as ``pmap`` or ``vmap``), global state leads to races and unpredictable results. Explicit keys make PRNG generation easily parallelizable across multiple hardware devices

**Comparison**:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - Traditional Frameworks (PyTorch/NumPy)
     - JAX
   * - **State management**
     - Implicit, global. A background global state exists
     - Explicit, local. Keys are passed as parameters between functions
   * - **Side effects**
     - Yes. Each random function call changes the global state
     - None. Calling functions does not change the input key, only returns results
   * - **Parallel safety**
     - Difficult. Synchronizing global state in parallel environments is complex
     - Natively supported. Split provides independent Keys for each branch
   * - **Determinism**
     - Depends on execution order
     - Depends only on the passed Key, independent of execution order

Code Example
~~~~~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp

   # 1. Create an initial key
   key = jax.random.PRNGKey(42)

   # 2. Using the same key multiple times yields identical results
   val1 = jax.random.normal(key)
   val2 = jax.random.normal(key)
   assert jnp.all(val1 == val2)  # Identical!

   # 3. Correct approach: split the key
   key, subkey = jax.random.split(key)
   random_val = jax.random.normal(subkey)

   # 4. Generate multiple independent random numbers
   key, *subkeys = jax.random.split(key, 5)
   random_vals = [jax.random.normal(k) for k in subkeys]

**Analogy**:

   A traditional random generator is like an **"ATM machine"** -- every time you withdraw money, the bank's backend balance (global state) decreases, and you cannot return to the state before the withdrawal.

   JAX's PRNG mechanism is like a **"duplicable treasure map"** -- the key is the coordinates on the map. If you copy the map (Split) and give it to another person, and you both follow the same coordinates and steps, the treasure (random numbers) you find will be exactly the same.

   This makes it clear in large-scale distributed search (parallel computing) that each person knows exactly which area they are responsible for, without confusion.

JAX in Cosmology
-----------------

JAX's application in cosmological computation is triggering a paradigm revolution, driving the field from traditional "numerical integration-driven" approaches toward **"fully differentiable inference (Differentiable Universe)"**.

Specific Application Areas
~~~~~~~~~~~~~~~~~~~~~~~~~~

Through its automatic differentiation and GPU acceleration capabilities, JAX has permeated every core aspect of cosmological research:

1. **High-dimensional Bayesian inference**: When processing next-generation astronomical observation data (such as Stage IV surveys), model parameters often exceed 150. JAX makes the entire likelihood function fully differentiable, supporting efficient exploration of high-dimensional spaces

2. **Cosmological prediction emulators**: Using JAX-based neural networks (such as CosmoPower-JAX) to replace traditional Boltzmann solvers (such as CAMB/CLASS). These emulators can predict matter power spectra and other observational effects at sub-millisecond speeds

3. **Differentiable cosmological simulations**: JAX is used to write fully differentiable N-body simulations and gravitational lensing simulations (such as JAXtronomy). This enables researchers to perform field-level inference

4. **Fisher matrix computation**: Traditional methods rely on unstable numerical differentiation. JAX can automatically compute exact second-order derivatives via ``jax.hessian``

How Does It Accelerate Parameter Inference and MCMC Sampling?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX's acceleration is not a single-dimensional improvement, but achieved through a combination of hardware optimization and algorithmic improvements:

- **From "random blind walk" to "gradient navigation"**: Traditional MCMC (such as Metropolis-Hastings) randomly explores parameter space, with efficiency dropping dramatically as dimensionality increases. JAX enables the use of **Hamiltonian Monte Carlo (HMC)** or NUTS samplers, using gradients to guide the sampler along high-probability regions

- **XLA compiler and operator fusion**: Through ``jax.jit``, the XLA compiler transforms Python code into machine code optimized for GPU/TPU, eliminating Python interpretation latency

- **Neural network acceleration**: Neural network emulators like CosmoPower-JAX are essentially linear algebra operations, which is exactly what GPUs excel at

**Analogy**:

   Traditional cosmological sampling is like blindly groping through a fog-shrouded valley (random scattering of points), spending enormous effort at each step to probe the depth.

   JAX is like providing researchers with a **GPS navigation system and a powerful spotlight** -- gradient information indicates the fastest path to the summit, while GPU hardware acceleration is like equipping researchers with high-speed skateboards, allowing journeys that would take years to be completed in just days.

Cosmological JAX Ecosystem
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cosmology community has already formed a rich JAX software ecosystem:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Project
     - Core Use
     - Features
   * - **jax-cosmo**
     - Core cosmology library
     - Provides differentiable background evolution, power spectra, and likelihood computation
   * - **CosmoPower-JAX**
     - Neural network emulator
     - High-speed prediction of CMB and matter power spectra, supports HMC sampling with hundreds of parameters
   * - **candl**
     - CMB likelihood analysis
     - Specialized for analyzing CMB power spectrum measurements (such as SPT, ACT)
   * - **JAXtronomy**
     - Gravitational lensing simulation
     - JAX port of lenstronomy, supports GPU acceleration
   * - **microJAX**
     - Microlensing simulation
     - First fully differentiable microlensing modeling framework
   * - **DISCO-DJ**
     - Differentiable N-body simulation
     - Implements fully automatic differentiable cosmological simulations
   * - **PyBird-JAX**
     - Effective Field Theory (EFT)
     - For fast processing of LSS data with EFT predictions

Performance Data and Benchmarks
--------------------------------

Based on research data, JAX/XLA demonstrates acceleration capabilities far exceeding traditional frameworks:

Cosmological Inference Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 25 25 10

   * - Task
     - Traditional Method
     - JAX Method
     - Speedup
   * - High-dimensional inference (157 parameters)
     - 12 years (48-core CPU, nested sampling)
     - 8 days (24 GPUs, gradient sampling)
     - **10^5x**
   * - Cosmic shear analysis (37 parameters)
     - Weeks
     - Hours
     - **~1000x**
   * - Central Bank of Chile economic model
     - 12 hours (industrial server)
     - Seconds (consumer GPU)
     - **~1000x**

Scientific Computing Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 20 20 10

   * - Operation
     - NumPy/SciPy
     - JAX JIT
     - Speedup
   * - Supernova likelihood computation
     - ~50 ms
     - ~0.05 ms
     - **1000x**
   * - Distance calculation (1000 points)
     - 150 ms
     - 20 ms
     - **7.5x**
   * - JAXtronomy ray tracing (vs CPU)
     - Baseline
     - GPU accelerated
     - **120-140x**
   * - Solving 10,000-dimensional PDE
     - Baseline
     - JAX optimized
     - **1000x + 30x memory savings**

Why HIcosmo Chose JAX
----------------------

1. MCMC Sampling Requires Efficient Gradient Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modern MCMC samplers (such as NUTS) rely on Hamiltonian Monte Carlo methods, requiring gradient computation of parameters at each step.

**Problems with traditional numerical differentiation**:

- For 10 parameters: 20 function evaluations needed
- For 100 parameters: 200 function evaluations needed
- Precision affected by epsilon choice

**Advantages of JAX automatic differentiation**:

- Regardless of the number of parameters, only ~2 equivalent function evaluations needed
- Results are exact mathematical derivatives
- NumPyro NUTS sampler directly uses JAX's automatic differentiation

2. Cosmological Computation Requires High-Precision Numerical Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cosmological distance calculations involve integrals:

.. math::

   d_C(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}

**HIcosmo approach**: Pre-compute high-precision integration tables, extremely fast after JIT compilation

.. code-block:: python

   from hicosmo.models import LCDM
   import jax.numpy as jnp

   z_grid = jnp.linspace(0.01, 2.0, 1000)
   params = {'H0': 70.0, 'Omega_m': 0.3}

   # First call: compilation ~100ms
   grid = LCDM.compute_grid_traced(z_grid, params)

   # Subsequent calls: cached ~0.1ms
   grid = LCDM.compute_grid_traced(z_grid, params)

3. Fisher Matrix Requires Second-Order Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Fisher matrix is the expectation value of the Hessian of the likelihood function:

.. math::

   F_{ij} = -\left\langle \frac{\partial^2 \ln L}{\partial \theta_i \partial \theta_j} \right\rangle

**Traditional method**: Numerical second derivatives require :math:`O(n^2)` function evaluations and are unstable

**JAX method**:

.. code-block:: python

   import jax

   # Automatically obtain the Hessian matrix
   hessian = jax.hessian(log_likelihood)

   # Fisher matrix
   fisher_matrix = -hessian(best_fit_params)

JAX vs Other Frameworks
------------------------

Comparison with NumPy
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - NumPy
     - JAX
   * - **API**
     - Native Python
     - Nearly identical to NumPy
   * - **Autodiff**
     - Not supported
     - Full support
   * - **JIT compilation**
     - Not supported
     - Supported
   * - **GPU support**
     - Requires CuPy
     - Transparent support
   * - **Learning curve**
     - Low
     - Low (if you know NumPy)

Comparison with TensorFlow/PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Feature
     - TensorFlow
     - PyTorch
     - JAX
   * - **Design goal**
     - Deep learning
     - Deep learning
     - General scientific computing
   * - **API style**
     - Computation graph
     - Dynamic graph
     - Function transformations
   * - **Scientific computing friendly**
     - Average
     - Average
     - **Excellent**
   * - **NumPy compatible**
     - Partial
     - Partial
     - **Full**
   * - **Functional programming**
     - Limited
     - Limited
     - **Core design**

**Why doesn't HIcosmo use TensorFlow/PyTorch?**

1. **API complexity**: TensorFlow/PyTorch APIs are designed for neural networks, overly complex for pure numerical computation
2. **NumPy compatibility**: JAX can directly use NumPy-style code with low migration cost
3. **Functional programming**: Cosmological computations are pure functions, perfectly matching JAX's function transformation philosophy
4. **Compilation efficiency**: The XLA compiler has specialized optimizations for numerical computation

Common JAX Pitfalls and Solutions
----------------------------------

1. Immutable Arrays
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Wrong: JAX arrays are immutable
   x = jnp.array([1, 2, 3])
   x[0] = 5  # TypeError!

   # Correct: use .at[].set()
   x = x.at[0].set(5)

2. Random Numbers Require Explicit Keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Wrong: no global random state
   x = jax.random.normal()  # Error!

   # Correct: explicitly pass a key
   key = jax.random.PRNGKey(42)
   x = jax.random.normal(key, shape=(10,))

   # Generate a new key
   key, subkey = jax.random.split(key)
   y = jax.random.normal(subkey, shape=(10,))

3. Dynamic Shapes in JIT
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Wrong: dynamic shape in JIT function
   @jax.jit
   def bad_func(x):
       return jnp.zeros(len(x))  # len(x) unknown at compile time

   # Correct: use x.shape
   @jax.jit
   def good_func(x):
       return jnp.zeros(x.shape)

4. Python Control Flow
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Wrong: Python if in JIT
   @jax.jit
   def bad_func(x, flag):
       if flag:  # Python conditional
           return x + 1
       return x

   # Correct: use jnp.where
   @jax.jit
   def good_func(x, flag):
       return jnp.where(flag, x + 1, x)

HIcosmo's JAX Usage Patterns
------------------------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   # HIcosmo wraps JAX; users typically do not need to use JAX directly
   from hicosmo import hicosmo

   # Behind this line of code, JAX is used for:
   # - JIT compilation to accelerate distance calculations
   # - Automatic differentiation to provide gradients for the NUTS sampler
   # - vmap to parallelize multiple MCMC chains
   inf = hicosmo("LCDM", "sn", ["H0", "Omega_m"])
   samples = inf.run()

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from hicosmo.models import LCDM

   # Directly use JAX's automatic differentiation
   def my_likelihood(params):
       grid = LCDM.compute_grid_traced(z_obs, params)
       d_L = grid['d_L']
       mu_theory = 5 * jnp.log10(d_L) + 25
       chi2 = jnp.sum(((mu_obs - mu_theory) / mu_err)**2)
       return -0.5 * chi2

   # Automatically obtain gradients
   grad_likelihood = jax.grad(my_likelihood)

   # Automatically obtain the Hessian (for Fisher matrix)
   hessian_likelihood = jax.hessian(my_likelihood)

Learning Resources
------------------

**Official documentation**:

- `JAX Official Documentation <https://jax.readthedocs.io/>`_
- `JAX Quickstart <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_
- `JAX Common Gotchas <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_

**Recommended tutorials**:

- `Thinking in JAX <https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html>`_
- `JAX 101 <https://jax.readthedocs.io/en/latest/jax-101/index.html>`_

**Community resources**:

- `Awesome JAX <https://github.com/n2cholas/awesome-jax>`_: Curated list of JAX resources
- `NumPyro <https://num.pyro.ai/>`_: JAX-based probabilistic programming library (used by HIcosmo)

**Cosmological JAX projects**:

- `jax-cosmo <https://github.com/DifferentiableUniverseInitiative/jax_cosmo>`_: Differentiable cosmology
- `CosmoPower-JAX <https://github.com/dpiras/cosmopower-jax>`_: Neural network emulator
- `JAXtronomy <https://github.com/Autolens/JAXtronomy>`_: Gravitational lensing simulation

Summary
-------

JAX provides HIcosmo with the following core capabilities:

1. **Automatic differentiation**: NUTS sampler requires no manual gradients, supports arbitrarily complex likelihood functions
2. **JIT compilation**: 10-1000x performance improvement, XLA operator fusion eliminates memory bottlenecks
3. **Vectorization**: vmap enables efficient batch computation and multi-chain parallelism
4. **GPU support**: Code runs transparently on GPUs without modification
5. **Composability**: grad/jit/vmap can be arbitrarily combined
6. **Functional programming**: Pure function design ensures reproducibility and parallel safety

Through this functional architecture, JAX transforms complex physical or mathematical models into optimizable objects, achieving **10^3 to 10^5x** performance improvements over traditional methods in high-dimensional inference tasks such as cosmology.

For users, HIcosmo has encapsulated JAX's complexity. You can use HIcosmo like a regular Python library while enjoying the performance benefits that JAX provides. Only when you need to customize likelihood functions or perform advanced analysis will you need to use the JAX API directly.

Next Steps
----------

- `Core Concepts <concepts.html>`_: Understand HIcosmo's architecture design
- `Samplers <guides/samplers.html>`_: Learn about MCMC configuration
- `Fisher Forecasts <guides/fisher.html>`_: Learn about Fisher matrix analysis
