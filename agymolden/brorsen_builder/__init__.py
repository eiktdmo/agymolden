"""
External modular package for newBRORSEN calculations input building.
Decoupled from newBRORSEN wrapper engine script.
"""

from .schema import (
    UIOptionDef,
    RUN_TYPES,
    METHODS,
    SECTORS,
    INTEGRAL_MODES,
    COORD_UNITS,
    OPTION_DEPENDENCIES,
)
from .generator import BrorsenStage, BrorsenInputGenerator
from .dialog import BrorsenInputBuilderDialog

__all__ = [
    "UIOptionDef",
    "RUN_TYPES",
    "METHODS",
    "SECTORS",
    "INTEGRAL_MODES",
    "COORD_UNITS",
    "OPTION_DEPENDENCIES",
    "BrorsenStage",
    "BrorsenInputGenerator",
    "BrorsenInputBuilderDialog",
]
