"""
Combined Likelihood for Multi-Probe Analysis
=============================================

Provides a simple way to combine multiple likelihoods using the + operator.

Example
-------
>>> sne = SN_likelihood(LCDM, "pantheon+")
>>> bao = BAO_likelihood(LCDM, "desi2024")
>>> joint = sne + bao
>>> MCMC(params, joint, chain_name='joint').run(...)
"""

from typing import List, Callable, Dict, Any

from .base import Likelihood, NuisanceList
import jax.numpy as jnp


class CombinedLikelihood(Likelihood):
    """
    Combined likelihood for multi-probe joint analysis.

    This class allows combining multiple likelihood objects using the + operator,
    enabling intuitive joint analysis syntax:

        joint = sne + bao + cmb
        MCMC(params, joint, chain_name='joint').run(...)

    Parameters
    ----------
    likelihoods : List[Callable]
        List of likelihood objects with __call__(**params) interface.

    Examples
    --------
    >>> from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
    >>> from hicosmo.models import LCDM
    >>> from hicosmo.samplers import MCMC
    >>>
    >>> sne = SN_likelihood(LCDM, "pantheon+")
    >>> bao = BAO_likelihood(LCDM, "desi2024")
    >>> joint = sne + bao
    >>>
    >>> params = {
    ...     'H0': {'init': 70, 'min': 60, 'max': 80},
    ...     'Omega_m': {'init': 0.3, 'min': 0.1, 'max': 0.5},
    ... }
    >>> MCMC(params, joint, chain_name='lcdm_joint').run(num_samples=10000)
    """

    def __init__(self, likelihoods: List[Callable]):
        """
        Initialize combined likelihood.

        Parameters
        ----------
        likelihoods : List[Callable]
            List of likelihood objects.
        """
        super().__init__(name="CombinedLikelihood")
        self._likelihoods = []

        # Flatten nested CombinedLikelihoods
        for lik in likelihoods:
            if isinstance(lik, CombinedLikelihood):
                self._likelihoods.extend(lik._likelihoods)
            else:
                self._likelihoods.append(lik)

    def __call__(self, **params) -> float:
        """
        Compute combined log-likelihood.

        Parameters
        ----------
        **params : dict
            Cosmological parameters passed to all likelihoods.

        Returns
        -------
        float
            Sum of log-likelihoods from all components.
        """
        total_log_L = 0.0
        for lik in self._likelihoods:
            total_log_L = total_log_L + lik(**params)
        return total_log_L

    def _load_data(self) -> None:
        return

    def _setup_covariance(self) -> None:
        return

    def get_requirements(self) -> Dict[str, Any]:
        requirements: Dict[str, Any] = {}
        for lik in self._likelihoods:
            if hasattr(lik, "get_requirements"):
                try:
                    req = lik.get_requirements()
                    if isinstance(req, dict):
                        requirements.update(req)
                except (TypeError, AttributeError, NotImplementedError):
                    # Skip likelihoods that fail to provide requirements
                    continue
        return requirements

    def theory(self, **kwargs) -> jnp.ndarray:
        raise NotImplementedError(
            "CombinedLikelihood does not define a single theory vector."
        )

    @property
    def nuisance_parameters(self):
        nuisance_items = []
        for lik in self._likelihoods:
            nuisance = getattr(lik, "nuisance_parameters", None)
            if nuisance is None:
                continue
            nuisance = nuisance() if callable(nuisance) else nuisance
            if isinstance(nuisance, dict):
                nuisance_items.extend(nuisance.values())
            else:
                nuisance_items.extend(list(nuisance))
        return NuisanceList(nuisance_items)

    # __add__ and __radd__ inherited from Likelihood base class

    def __repr__(self):
        """String representation."""
        lik_names = []
        for lik in self._likelihoods:
            name = getattr(lik, "__class__", type(lik)).__name__
            lik_names.append(name)
        return f"CombinedLikelihood({' + '.join(lik_names)})"

    def __len__(self):
        """Number of component likelihoods."""
        return len(self._likelihoods)

    @property
    def likelihoods(self) -> List[Callable]:
        """Get list of component likelihoods."""
        return self._likelihoods.copy()
