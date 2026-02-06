#!/usr/bin/env python3
"""
MCMC Utilities and Helper Functions.

Simplified module providing only essential mapping structures.
Removed: ParameterMapper, FunctionInspector, fuzzy matching (~550 lines deleted).
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class MappingResult:
    """
    Result of parameter mapping (simplified).

    Attributes
    ----------
    parameter_mapping : Dict[str, str]
        Mapping from function parameters to config parameters.
        For direct mapping, this is always {name: name}.
    data_arguments : Dict[str, Any]
        Data arguments to pass to function.
    missing_parameters : List[str]
        Required parameters that couldn't be mapped.
    unused_parameters : List[str]
        Config parameters that aren't used.
    warnings : List[str]
        Warning messages about the mapping.
    success : bool
        Whether the mapping was successful.
    """

    parameter_mapping: Dict[str, str]
    data_arguments: Dict[str, Any]
    missing_parameters: List[str]
    unused_parameters: List[str]
    warnings: List[str]
    success: bool = True


def create_direct_mapping(
    param_names: List[str], data_kwargs: Dict[str, Any] = None
) -> MappingResult:
    """
    Create a simple direct parameter mapping.

    Parameters
    ----------
    param_names : List[str]
        Parameter names from config.
    data_kwargs : Dict[str, Any], optional
        Data arguments to pass to function.

    Returns
    -------
    MappingResult
        Direct identity mapping {name: name}.
    """
    return MappingResult(
        parameter_mapping={name: name for name in param_names},
        data_arguments=data_kwargs or {},
        missing_parameters=[],
        unused_parameters=[],
        warnings=[],
        success=True,
    )
