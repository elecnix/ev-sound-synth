"""Tyre and wind noise - the sound an EV cannot switch off.

This layer is why FMVSS No. 141 stops asking for a pedestrian alert at
30 km/h: by then the tyres are louder than any alert would be.

Two exponents do the work. Tyre noise grows about 30*log10(v), wind noise
about 60*log10(v), so the balance tips from a mid-band roar to a high hiss as
speed rises. Rather than sweeping a filter over time - which needs a per-
sample loop - the bed is built as five fixed noise bands whose gains are
crossfaded by the speed curve.
"""

from __future__ import annotations

import numpy as np

from .. import dsp
from .base import Generator

REFERENCE_KMH = 50.0

# (low Hz, high Hz, level at 50 km/h in dB, growth exponent in dB per decade)
BANDS = [
    (30.0, 120.0, -12.0, 20.0),      # body and suspension rumble
    (120.0, 400.0, -8.0, 25.0),      # tyre carcass
    (400.0, 1200.0, 0.0, 30.0),      # tread pattern - the loudest band
    (1200.0, 3500.0, -6.0, 40.0),    # tread edge and airborne
    (3500.0, 12000.0, -14.0, 55.0),  # wind over mirrors and seals
]

GATE_KMH = 3.0
"""Below this the car is stopped and the bed is silent."""


class RoadWindNoise(Generator):
    name = "road_wind"
    description = (
        "Tyre and wind noise from five crossfaded noise bands. Tyre content "
        "grows 30*log10(v) and wind content 60*log10(v), so the balance shifts "
        "from mid-band roar to high hiss as the car speeds up."
    )
    seed = 29

    def synthesize(self, cycle, rng):
        kmh = np.maximum(np.abs(cycle.speed_kmh), 0.5)
        decades = np.log10(kmh / REFERENCE_KMH)
        gate = np.clip(np.abs(cycle.speed_kmh) / GATE_KMH, 0.0, 1.0)

        channels = []
        for channel in range(2):
            # Independent noise per side: a stereo image made of two different
            # noises, not one noise pushed around by a delay.
            source = dsp.pink_noise(len(cycle), rng)
            out = np.zeros(len(cycle))
            for lo, hi, level_db, exponent in BANDS:
                bed = dsp.fft_filter(source, cycle.sr,
                                     dsp.bandpass_response(lo, hi, skirt_oct=0.35))
                out += bed * dsp.db_to_amp(level_db + exponent * decades)
            channels.append(out * gate)

        stereo = np.stack(channels, axis=1)
        return 0.055 * stereo
