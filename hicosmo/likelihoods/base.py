"""
Base likelihood class compatible with Cobaya and optimized for JAX.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
import yaml
import inspect


class NuisanceList(list):
    """Callable list wrapper for nuisance parameters (property/method compatibility)."""

    def __call__(self):
        return self


class Likelihood(ABC):
    """
    Base class for all likelihoods in HIcosmo.

    Compatible with Cobaya's interface while leveraging JAX for autodiff and GPU.

    Minimal Custom Likelihood Example
    ----------------------------------
    >>> from hicosmo.likelihoods import Likelihood
    >>> from hicosmo.models import LCDM
    >>> import jax.numpy as jnp
    >>>
    >>> class MyLikelihood(Likelihood):
    ...     def __init__(self, cosmology_class=LCDM, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self._cosmology_class = cosmology_class
    ...         self.z_data = jnp.array([0.5, 1.0, 1.5])
    ...         self.mu_obs = jnp.array([42.5, 44.1, 45.2])
    ...         self.sigma = jnp.array([0.1, 0.15, 0.12])
    ...
    ...     def log_likelihood(self, model, **kwargs):
    ...         mu_theory = model.distance_modulus(self.z_data)
    ...         chi2 = jnp.sum(((self.mu_obs - mu_theory) / self.sigma)**2)
    ...         return -0.5 * chi2
    >>>
    >>> # Usage
    >>> lik = MyLikelihood()
    >>> log_L = lik(H0=70, Omega_m=0.3)  # Cosmology auto-constructed
    >>>
    >>> # Combine with other likelihoods
    >>> from hicosmo.likelihoods import BAO_likelihood
    >>> joint = lik + BAO_likelihood(LCDM, "desi2024")

    Key Points
    ----------
    - Only override what you need: `log_likelihood(model, **kwargs)` is sufficient
    - Set `self._cosmology_class` to enable auto model construction in `__call__`
    - Use `+` operator to combine likelihoods for joint analysis
    - Override `nuisance_parameters` property to add sampled nuisance params
    - Use `available_datasets()` classmethod to discover available data
    """

    # Category identifier for data registry lookup
    _data_category: Optional[str] = None
    
    def __init__(self, name: Optional[str] = None,
                 data_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize the likelihood.

        Args:
            name: Name of the likelihood
            data_path: Path to data files
            **kwargs: Additional configuration parameters
        """
        self.name = name or self.__class__.__name__
        self.data_path = data_path
        self.config = kwargs
        self._initialized = False
        self._nuisance_params = {}
        self._requirements = {}
        
    def initialize(self) -> None:
        """
        Initialize the likelihood by loading data and covariance matrix.
        This method should be called before first use.
        """
        if self._initialized:
            return
            
        self._load_data()
        self._setup_covariance()
        self._initialize_nuisance()
        self._initialized = True
    
    def _load_data(self) -> None:
        """
        Load observational data.
        Override in subclasses that load external data files.
        """
        pass

    def _setup_covariance(self) -> None:
        """
        Setup covariance matrix and its inverse.
        Override in subclasses that use covariance-based chi2 computation.
        """
        pass

    def _initialize_nuisance(self) -> None:
        """
        Initialize nuisance parameters.
        Override in subclasses that have nuisance parameters.
        """
        pass

    def get_requirements(self) -> Dict[str, Any]:
        """
        Return the cosmological quantities required by this likelihood.

        Override this method to tell the sampler what cosmological quantities
        need to be computed. Default returns empty dict.

        Example return value:
        {
            'luminosity_distance': {'z': self.z_data},
            'H': {'z': self.z_data}
        }

        Returns:
            Dictionary of required quantities
        """
        return {}
    
    def get_nuisance_params(self) -> Dict[str, Dict[str, Any]]:
        """
        Return the nuisance parameters for this likelihood (legacy interface).

        Returns:
            Dictionary of nuisance parameters with their properties
            Example: {'M': {'prior': {'min': -20, 'max': -18}, 'ref': -19.3}}

        Notes
        -----
        This is the old dict-based interface. New code should use the
        `nuisance_parameters` property which returns Parameter objects.
        """
        return self._nuisance_params

    @property
    def nuisance_parameters(self) -> List:
        """
        Return nuisance parameters as Parameter objects.

        This property enables automatic parameter discovery by ParameterRegistry.
        Subclasses should override this to declare their nuisance parameters.

        Returns
        -------
        List[Parameter]
            List of nuisance Parameter objects. Default is empty list.

        Examples
        --------
        >>> # In a subclass:
        >>> @property
        >>> def nuisance_parameters(self):
        >>>     from ..parameters import Parameter
        >>>     return [
        >>>         Parameter('M_B', value=-19.23, free=True,
        >>>                   prior={'dist': 'uniform', 'min': -20, 'max': -18},
        >>>                   latex_label=r'$\\mathcal{M}_B$')
        >>>     ]

        Notes
        -----
        This enables the workflow:
        >>> registry = ParameterRegistry.from_defaults('planck2018')
        >>> registry.set_free(['H0', 'Omega_m'])
        >>> pantheon = SN_likelihood(LCDM, "pantheon+")
        >>> registry.add_from_likelihood(pantheon)  # Auto-adds M_B
        """
        # Default: no nuisance parameters
        # Subclasses override this property to declare their parameters
        return NuisanceList()

    def _wrap_nuisance(self, params: List) -> NuisanceList:
        """Wrap nuisance list to be callable for compatibility."""
        if isinstance(params, NuisanceList):
            return params
        return NuisanceList(params)

    def _get_cosmology_class(self):
        """Return associated cosmology class if present."""
        return getattr(self, "_cosmology_class", None) or getattr(self, "cosmology_class", None)

    def _prepare_params_for_jax(
        self,
        params: Dict[str, Any],
        dtype=None,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Convert parameter dict to JAX arrays with tracer support.

        This helper handles the common pattern of converting parameters for
        JIT-compiled likelihood computation, supporting JAX Tracers during MCMC.

        Parameters
        ----------
        params : dict
            Input parameters (may contain int, float, arrays, or JAX Tracers)
        dtype : optional
            JAX dtype for arrays (default: jnp.float64)
        default_params : dict, optional
            Default values for missing parameters

        Returns
        -------
        dict
            Parameters converted to JAX arrays
        """
        if dtype is None:
            dtype = jnp.float64

        cosmo_class = self._get_cosmology_class()
        if cosmo_class is not None and hasattr(cosmo_class, "normalize_params"):
            params = cosmo_class.normalize_params(params, dtype=dtype)

        # Add defaults for missing parameters
        if default_params:
            for key in ["Omega_b", "T_cmb", "N_eff"]:
                if key not in params and key in default_params:
                    params[key] = default_params[key]

        result = {}
        for k, v in params.items():
            if isinstance(v, (int, float)):
                result[k] = jnp.asarray(v, dtype=dtype)
            elif isinstance(v, jax.core.Tracer):
                result[k] = v  # Keep tracer as-is
            elif hasattr(v, "shape"):
                result[k] = jnp.asarray(v, dtype=dtype)
        return result

    def _call_log_likelihood(self, model, params: Dict[str, Any]) -> float:
        """Call log_likelihood with smart argument matching."""
        if not hasattr(self, "log_likelihood"):
            raise AttributeError("log_likelihood not implemented")

        try:
            sig = inspect.signature(self.log_likelihood)
            param_items = list(sig.parameters.values())
            kwargs = {}
            used = set()
            if param_items:
                first_param = param_items[0]
                if first_param.name in {"model", "cosmology"}:
                    kwargs[first_param.name] = model
                    used.add(first_param.name)
            for param in param_items[1:]:
                if param.kind == param.VAR_KEYWORD:
                    for key, value in params.items():
                        if key not in kwargs:
                            kwargs[key] = value
                    break
                name = param.name
                if name in params:
                    kwargs[name] = params[name]
            if kwargs:
                return self.log_likelihood(**kwargs)
            return self.log_likelihood(model)
        except (TypeError, AttributeError):
            # Fallback: signature inspection failed, call with model only
            return self.log_likelihood(model)

    def log_likelihood(self, *args, **kwargs) -> float:
        """Default log_likelihood implementation (alias to logp)."""
        return self.logp(*args, **kwargs)

    def __call__(self, **params) -> float:
        """Unified callable interface: build model and call log_likelihood."""
        cosmo_class = self._get_cosmology_class()
        if cosmo_class is None:
            return self.logp(**params)

        # Filter to cosmology constructor signature when nuisance parameters present.
        try:
            sig = inspect.signature(cosmo_class.__init__)
            valid = {k: v for k, v in params.items() if k in sig.parameters and k != "self"}
            model = cosmo_class(**valid)
        except (TypeError, ValueError):
            # Signature filtering failed, pass all params
            model = cosmo_class(**params)

        return self._call_log_likelihood(model, params)

    def derived_parameters(self, **params) -> Dict[str, float]:
        """
        Compute derived parameters relevant to this likelihood's data.

        Each likelihood can define derived parameters that are physically
        meaningful for its measurements. For example:
        - BAO likelihood: rd, rd_h (sound horizon related)
        - CMB likelihood: omega_b_h2, omega_c_h2 (physical densities)
        - SNe likelihood: typically none needed

        Parameters
        ----------
        **params : dict
            Sampled cosmological parameters (H0, Omega_m, etc.)

        Returns
        -------
        Dict[str, float]
            Dictionary of derived parameter names and values.
            Default implementation returns empty dict.

        Notes
        -----
        Subclasses should override this method to compute derived parameters
        that are relevant to their specific measurements. The derived parameters
        will be automatically recorded during MCMC sampling.

        Examples
        --------
        >>> class MyBAOLikelihood(Likelihood):
        ...     def derived_parameters(self, **params):
        ...         model = self._cosmology_class(**params)
        ...         return model.derived_parameters()  # rd, rd_h, H0_rd
        """
        return {}

    def get_normalization_constant(self) -> float:
        """
        Return the log-likelihood normalization constant.

        The normalization constant is the portion of log-likelihood that does
        NOT depend on model parameters. For Gaussian likelihoods:

            log(L) = -χ²/2 + normalization_constant

        where the normalization constant typically includes:
        - -0.5 * log|C| (covariance determinant)
        - -N/2 * log(2π) (Gaussian prefactor, often omitted)
        - Other data-dependent constants

        This constant is needed for information criteria (AIC, BIC) which use χ²:

            χ² = -2 * (log_likelihood - normalization_constant)
            AIC = χ²_min + 2k
            BIC = χ²_min + k * ln(N)

        Returns
        -------
        float
            The normalization constant. Default is 0.0 (pure χ²/2 likelihood).

        Notes
        -----
        Subclasses that include normalization terms in their log_likelihood
        MUST override this method to return the correct constant. Without this,
        information criteria will be incorrect.

        Examples
        --------
        >>> class MySNeLikelihood(Likelihood):
        ...     def get_normalization_constant(self):
        ...         return -0.5 * self.log_det_cov  # Gaussian normalization
        """
        return 0.0

    def get_num_data(self) -> Optional[int]:
        """
        Return the number of data points used by this likelihood.

        This is needed for BIC calculation: BIC = χ²_min + k * ln(N)

        Returns
        -------
        int or None
            Number of data points. None if not available.

        Notes
        -----
        Subclasses should override this to return the actual data count.
        """
        return None

    def theory(self, **kwargs) -> jnp.ndarray:
        """
        Compute theoretical prediction given cosmological parameters.

        Override this method for likelihoods that use the chi2 pattern.
        Alternatively, override `logp()` directly for custom likelihood computation.

        Args:
            **kwargs: Cosmological quantities computed by the theory code

        Returns:
            Theory vector matching the data vector shape
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either theory() or logp()"
        )
    
    @jit
    def chi2(self, theory_vec: jnp.ndarray, data_vec: jnp.ndarray, 
             inv_cov: jnp.ndarray) -> float:
        """
        Compute chi-squared statistic.
        
        Args:
            theory_vec: Theoretical predictions
            data_vec: Observational data
            inv_cov: Inverse covariance matrix
            
        Returns:
            Chi-squared value
        """
        residual = data_vec - theory_vec
        return jnp.dot(residual, jnp.dot(inv_cov, residual))
    
    def logp(self, **params_values) -> float:
        """
        Compute log-likelihood.
        
        This is the main method called by the sampler.
        
        Args:
            **params_values: Dictionary of all parameters (cosmological + nuisance)
            
        Returns:
            Log-likelihood value
        """
        if not self._initialized:
            self.initialize()
        theory_vec = self.theory(**params_values)
        chi2_val = self.chi2(theory_vec, self.data_vec, self.inv_cov)
        return -0.5 * chi2_val
    
    def logp_grad(self, **params_values) -> Tuple[float, Dict[str, float]]:
        """
        Compute log-likelihood and its gradient.
        
        This method uses JAX's automatic differentiation.
        
        Args:
            **params_values: Dictionary of all parameters
            
        Returns:
            Tuple of (log-likelihood, gradient dictionary)
        """
        param_names = list(params_values.keys())
        param_array = jnp.array([params_values[name] for name in param_names])

        def logp_func(arr):
            return self.logp(**{n: arr[i] for i, n in enumerate(param_names)})

        logp_val, grad_array = jax.value_and_grad(logp_func)(param_array)
        return logp_val, {n: grad_array[i] for i, n in enumerate(param_names)}
    
    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'Likelihood':
        """
        Create likelihood instance from YAML configuration file.
        
        Args:
            yaml_file: Path to YAML configuration file
            
        Returns:
            Likelihood instance
        """
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    def to_yaml(self, yaml_file: str) -> None:
        """
        Save likelihood configuration to YAML file.
        
        Args:
            yaml_file: Path to output YAML file
        """
        config = {
            'name': self.name,
            'data_path': self.data_path,
            **self.config
        }
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    @classmethod
    def available_datasets(cls) -> List[str]:
        """
        Return list of available datasets for this likelihood type.

        This method auto-discovers datasets by scanning the data directory.
        No hardcoding required - new datasets are automatically detected
        when added to the appropriate data folder.

        Returns
        -------
        List[str]
            List of available dataset names.

        Examples
        --------
        >>> from hicosmo.likelihoods.bao import DESI2024BAO
        >>> DESI2024BAO.available_datasets()
        ['desi_2024', 'boss_dr12', 'sdss_dr12', 'sdss_dr16', 'sixdf']

        >>> from hicosmo.likelihoods.sn import PantheonPlusLikelihood
        >>> PantheonPlusLikelihood.available_datasets()
        ['pantheon+shoes']

        Notes
        -----
        Subclasses should set `_data_category` class attribute to enable
        automatic dataset discovery:
        - 'bao' for BAO likelihoods
        - 'sn' for supernova likelihoods
        - 'cmb' for CMB likelihoods
        - 'h0' for H0 likelihoods
        - 'gw' for gravitational wave likelihoods
        - 'lensing' for strong lensing likelihoods
        """
        from ..data_registry import DataRegistry

        registry = DataRegistry()
        category = getattr(cls, "_data_category", None)

        if category is None:
            # Try to infer category from class name
            name_lower = cls.__name__.lower()
            if "bao" in name_lower:
                category = "bao"
            elif "sn" in name_lower or "pantheon" in name_lower:
                category = "sn"
            elif "planck" in name_lower or "cmb" in name_lower:
                category = "cmb"
            elif "h0licow" in name_lower or "shoes" in name_lower:
                category = "h0"
            elif "gw" in name_lower or "siren" in name_lower:
                category = "gw"
            elif "tdcosmo" in name_lower or "lens" in name_lower:
                category = "lensing"
            else:
                # Return all datasets if category unknown
                all_datasets = registry.list_all()
                return [name for names in all_datasets.values() for name in names]

        # Get datasets for this category
        method = getattr(registry, category, None)
        if method is not None:
            return method()
        return []

    @classmethod
    def show_available_datasets(cls) -> None:
        """
        Pretty print available datasets for this likelihood type.

        Examples
        --------
        >>> from hicosmo.likelihoods.bao import DESI2024BAO
        >>> DESI2024BAO.show_available_datasets()
        Available BAO datasets:
          • desi_2024      : DESI 2024 DR1 BAO measurements (z=0.3-2.3)
          • boss_dr12      : BOSS DR12 Luminous Red Galaxy BAO
          ...
        """
        from ..data_registry import DataRegistry

        registry = DataRegistry()
        category = getattr(cls, "_data_category", None)

        # Infer category if not set
        if category is None:
            name_lower = cls.__name__.lower()
            if "bao" in name_lower:
                category = "bao"
            elif "sn" in name_lower or "pantheon" in name_lower:
                category = "sn"
            elif "planck" in name_lower or "cmb" in name_lower:
                category = "cmb"
            elif "h0licow" in name_lower or "shoes" in name_lower:
                category = "h0"
            elif "gw" in name_lower or "siren" in name_lower:
                category = "gw"
            elif "tdcosmo" in name_lower or "lens" in name_lower:
                category = "lensing"

        registry.show(category)

    def __add__(self, other):
        """
        Combine two likelihoods for joint analysis.

        Parameters
        ----------
        other : callable
            Another likelihood object with __call__(**params) interface.

        Returns
        -------
        CombinedLikelihood
            A new likelihood that returns the sum of both log-likelihoods.

        Examples
        --------
        >>> sne = SN_likelihood(LCDM, "pantheon+")
        >>> bao = BAO_likelihood(LCDM, "desi2024")
        >>> joint = sne + bao
        >>> MCMC(params, joint, chain_name='joint').run(...)
        """
        from .combined import CombinedLikelihood
        return CombinedLikelihood([self, other])

    def __radd__(self, other):
        """Support sum() and other + self."""
        if other == 0:
            return self
        return self.__add__(other)


class GaussianLikelihood(Likelihood):
    """
    Base class for Gaussian likelihoods with simple chi2 computation.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_vec = None
        self.cov = None
        self.inv_cov = None
        
    def _setup_covariance(self) -> None:
        """
        Setup covariance matrix and compute its inverse.
        Assumes self.cov has been set by _load_data().
        """
        if self.cov is None:
            raise ValueError("Covariance matrix not loaded")
        
        # Convert to JAX array
        self.cov = jnp.array(self.cov)
        
        # Compute inverse (with regularization for numerical stability)
        try:
            self.inv_cov = jnp.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            # Add small regularization
            eps = 1e-10
            self.inv_cov = jnp.linalg.inv(self.cov + eps * jnp.eye(len(self.cov)))
    
    def validate_data(self) -> bool:
        """
        Validate loaded data and covariance matrix.
        
        Returns:
            True if data is valid, raises ValueError otherwise
        """
        if self.data_vec is None:
            raise ValueError("Data vector not loaded")
        
        if self.cov is None:
            raise ValueError("Covariance matrix not loaded")
        
        n_data = len(self.data_vec)
        if self.cov.shape != (n_data, n_data):
            raise ValueError(f"Covariance shape {self.cov.shape} doesn't match data length {n_data}")
        
        # Check if covariance is symmetric
        if not jnp.allclose(self.cov, self.cov.T):
            raise ValueError("Covariance matrix is not symmetric")
        
        # Check if covariance is positive definite
        eigenvals = jnp.linalg.eigvalsh(self.cov)
        if jnp.any(eigenvals <= 0):
            raise ValueError("Covariance matrix is not positive definite")
        
        return True
