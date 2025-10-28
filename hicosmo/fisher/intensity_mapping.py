"""Compatibility wrapper for intensity mapping forecasts.

The implementation now lives in :mod:`hicosmo.forecasts.intensity_mapping`.
This module re-exports the public classes/functions so that existing imports
continue to work.
"""

from ..forecasts.intensity_mapping import (  # noqa: F401
    IntensityMappingSurvey,
    IntensityMappingFisher,
    list_available_surveys,
    load_survey,
    run_forecast,
)

__all__ = [
    'IntensityMappingSurvey',
    'IntensityMappingFisher',
    'list_available_surveys',
    'load_survey',
    'run_forecast',
]
