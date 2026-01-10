"""Legacy-compatible injections helper backed by HIcosmo's JAX sampler."""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from .standard_siren import GWInjectionSet, InjectionSampler


class Injections:
    """Thin compatibility wrapper over :class:`InjectionSampler`."""

    def __init__(self, injections_data: Dict, cosmology: Optional[Any] = None, *, snr_threshold: float = 11.0, **_: Any):
        distance = np.asarray(injections_data['distance'])
        mass_1 = np.asarray(injections_data['mass_1'])
        mass_2 = np.asarray(injections_data['mass_2'])
        prior = np.asarray(injections_data['prior'])

        if prior.shape != distance.shape:
            raise ValueError('Injection prior must match the number of injections.')

        snr = np.asarray(injections_data.get('snr')) if 'snr' in injections_data else None

        injection_set = GWInjectionSet(
            injections={
                'luminosity_distance': distance,
                'mass_1': mass_1,
                'mass_2': mass_2,
            },
            weights=prior,
            n_total=int(injections_data['ntotal']),
            V_T=float(injections_data.get('V_T', 0.0)),
            T_obs=float(injections_data['Tobs']),
            snr=snr,
            snr_threshold=snr_threshold,
        )

        self._sampler = InjectionSampler(injection_set)
        self.ntotal = injection_set.n_total
        self.Tobs = injection_set.T_obs
        self.log_weights: Optional[np.ndarray] = None
        self.pseudo_rate: Optional[float] = None

    def update_weights(self, rate_model) -> None:
        self._sampler.update_weights(rate_model)
        self.log_weights = self._sampler.log_weights
        self.pseudo_rate = self._sampler.pseudo_rate

    def expected_number_detections(self) -> float:
        return self._sampler.expected_number_detections()

    def effective_injections_number(self) -> float:
        return self._sampler.effective_injections_number()

    # Convenience properties -------------------------------------------------
    @property
    def log_weights_array(self):  # pragma: no cover - legacy alias
        return self.log_weights

    @property
    def pseudo_rate_value(self):  # pragma: no cover - legacy alias
        return self.pseudo_rate
