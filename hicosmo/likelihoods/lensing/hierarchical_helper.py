"""High-level helper that reuses the icarogw-compatible likelihood."""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from ..gw.standard_siren import (
    GWStandardSirenLikelihood,
    GWCatalogData,
    GWEventData,
    GWInjectionSet,
)


def _dict_to_event(name: str, data: Dict[str, Any]) -> GWEventData:
    posterior = np.column_stack(
        [
            np.asarray(data["distance"]),
            np.asarray(data["redshift"]),
            np.asarray(data["mass_1_source"]),
            np.asarray(data["mass_2_source"]),
        ]
    )
    prior = data.get("prior", np.power(data["distance"], 2.0))
    return GWEventData(
        name=name,
        posterior_samples=posterior,
        prior_samples=np.asarray(prior),
    )


def _dict_to_injections(data: Dict[str, Any]) -> GWInjectionSet:
    injections = np.column_stack(
        [
            np.asarray(data["distance"]),
            np.asarray(data["redshift"]),
            np.asarray(data["mass_1"]),
            np.asarray(data["mass_2"]),
        ]
    )
    return GWInjectionSet(
        injections=injections,
        weights=np.asarray(data["prior"]),
        n_total=int(data["ntotal"]),
        V_T=float(data.get("V_T", 0.0)),
        T_obs=float(data["Tobs"]),
        snr=np.asarray(data.get("snr")) if "snr" in data else None,
        snr_threshold=float(data.get("snr_threshold", 0.0)),
    )


class HierarchicalGWLikelihood:
    """Simple proxy that shares implementation with GWStandardSirenLikelihood."""

    def __init__(self, catalog: GWCatalogData, **kwargs):
        self._like = GWStandardSirenLikelihood(catalog=catalog, **kwargs)

    @classmethod
    def from_raw(
        cls,
        events_data: Dict[str, Dict[str, Any]],
        injections_data: Dict[str, Any],
        **kwargs,
    ):
        events = [_dict_to_event(name, data) for name, data in events_data.items()]
        injections = _dict_to_injections(injections_data)
        catalog = GWCatalogData(events=events, injections=injections)
        return cls(catalog, **kwargs)

    def log_likelihood(self, cosmology, **kwargs) -> float:
        return self._like.log_likelihood(cosmology, **kwargs)

    def log_likelihood_traced(self, cosmo_params: Dict[str, Any], **kwargs):
        if not hasattr(self._like, "log_likelihood_traced"):
            raise AttributeError(
                "Underlying likelihood does not support traced evaluations."
            )
        return self._like.log_likelihood_traced(cosmo_params, **kwargs)
