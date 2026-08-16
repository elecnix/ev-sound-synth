"""The two mixes: what a driver hears, and what a bystander hears.

Keeping these as mixes of the same generators - rather than as separate
synthesis - is the point. The cabin hears the drive unit through glass and
trim with the performance sound layered over it; the kerb hears the pedestrian
alert over tyre noise and almost none of the motor. Same sources, different
balance and different microphone.

Both mixes normalise for comfortable listening, so both record the gain they
applied and shift their stated calibration by it. A file that lies about its
own level is worse than a quiet one.
"""

from __future__ import annotations

import numpy as np

from .. import dsp, vehicle
from .avas import AvasForward
from .base import CABIN_SPL_AT_FULL_SCALE, EXTERIOR_SPL_AT_FULL_SCALE, Generator
from .inverter import InverterPwm
from .motor import MotorOrders
from .road import RoadWindNoise
from .boost import BoostPerformance


class CabinMix(Generator):
    """Driver's ear: the drive unit, the inverter, boost mode and road noise."""

    name = "cabin_mix"
    description = (
        "What the driver hears through a full launch cycle: drive-unit orders "
        "and inverter comb behind the boost performance sound, over tyre and wind "
        "noise. Balanced for the driver's ear, not the kerb."
    )
    spl_at_full_scale = CABIN_SPL_AT_FULL_SCALE
    seed = 101

    LAYERS = [(MotorOrders, 1.0), (InverterPwm, 0.30), (BoostPerformance, 0.95),
              (RoadWindNoise, 0.75)]

    def synthesize(self, cycle, rng):
        mix = np.zeros((len(cycle), 2))
        for cls, gain in self.LAYERS:
            mix += gain * cls().synthesize(cycle, np.random.default_rng(cls.seed))

        # Glass and trim: the cabin never hears the top octave raw.
        for channel in range(2):
            mix[:, channel] = dsp.fft_filter(
                mix[:, channel], cycle.sr, dsp.lowpass_response(11_000.0, skirt_oct=0.8)
            )

        mix, self._gain_db = dsp.normalize_peak(mix, peak_db=-1.5)
        return mix


class ExteriorPassBy(Generator):
    """Kerbside: the pedestrian alert over tyre noise, motor barely audible."""

    name = "exterior_passby"
    description = (
        "What a bystander hears as the car creeps past: the FMVSS pedestrian "
        "alert dominant at walking pace, tyre noise taking over as it speeds up, "
        "and only a trace of the drive unit through the body."
    )
    spl_at_full_scale = EXTERIOR_SPL_AT_FULL_SCALE
    seed = 103

    def default_cycle(self, duration, sr):
        return vehicle.avas_staircase(duration, sr)

    def synthesize(self, cycle, rng):
        alert = AvasForward().synthesize(cycle, np.random.default_rng(AvasForward.seed))
        road = RoadWindNoise().synthesize(cycle, np.random.default_rng(RoadWindNoise.seed))
        motor = MotorOrders().synthesize(cycle, np.random.default_rng(MotorOrders.seed))

        mix = alert + 0.55 * road + 0.12 * motor
        mix, self._gain_db = dsp.normalize_peak(mix, peak_db=-3.0)
        return mix
