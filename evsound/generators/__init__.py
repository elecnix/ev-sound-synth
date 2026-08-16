"""The generators, in the order the render CLI writes them."""

from .avas import AvasForward, AvasReverse
from .base import Generator, GeneratorResult
from .inverter import InverterPwm
from .mixes import CabinMix, ExteriorPassBy
from .motor import MotorOrders
from .road import RoadWindNoise
from .boost import BoostPerformance

ALL_GENERATORS = [
    AvasForward,
    AvasReverse,
    MotorOrders,
    InverterPwm,
    BoostPerformance,
    RoadWindNoise,
    CabinMix,
    ExteriorPassBy,
]

__all__ = [
    "ALL_GENERATORS",
    "AvasForward",
    "AvasReverse",
    "CabinMix",
    "ExteriorPassBy",
    "Generator",
    "GeneratorResult",
    "InverterPwm",
    "MotorOrders",
    "RoadWindNoise",
    "BoostPerformance",
]
