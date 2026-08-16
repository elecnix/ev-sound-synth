"""Boost mode - the interior performance sound.

Manufacturers describe a maximum-power mode's sound as a high frequency over a
low rumble, composed to mark the moment the mode engages. This generator builds
exactly that: two layers that can be measured separately, plus a stinger at
engagement.

The important design choice is what the pitch hangs on. There is no engine
speed, so the sound follows a virtual rev that rises mostly with road speed
but jumps a little with pedal. That jump is what makes the sound feel like a
response to the driver rather than a speedometer read aloud.

Nothing here is a recording. Every layer is additive synthesis over the drive
cycle, so it retunes for any speed profile.
"""

from __future__ import annotations

import numpy as np

from .. import dsp
from .base import Generator

TOP_SPEED_KMH = 170.0
LEAD_BASE_HZ = 42.0
LEAD_SPAN = 6.5
"""Virtual rev of 1.0 puts the lead 7.5 times its base frequency."""

RUMBLE_DIVISOR = 3.4
RUMBLE_RANGE_HZ = (26.0, 110.0)

LEAD_PARTIALS = 14
BRIGHT_FROM = 7
"""Partials at or above this index only come up under throttle."""

OUTPUT_ROLLOFF_HZ = 8500.0

STINGER_START_S = 1.9
STINGER_LENGTH_S = 1.0


def _virtual_rev(cycle) -> np.ndarray:
    speed_norm = np.clip(np.abs(cycle.speed_kmh) / TOP_SPEED_KMH, 0.0, 1.0)
    throttle = dsp.smooth(np.clip(cycle.pedal, 0.0, 1.0), 0.08, cycle.sr)
    return np.clip(0.12 + 0.88 * speed_norm**0.7 + 0.15 * throttle, 0.0, 1.15)


def _stinger(cycle, rng) -> np.ndarray:
    """The engagement sound: a dive, then a climb. Signals that boost mode is live."""
    t = cycle.t
    window = (t >= STINGER_START_S) & (t < STINGER_START_S + STINGER_LENGTH_S)
    if not window.any():
        return np.zeros(len(cycle))

    u = np.clip((t - STINGER_START_S) / STINGER_LENGTH_S, 0.0, 1.0)
    dive = 1200.0 * np.exp(-6.0 * np.minimum(u, 0.35) / 0.35 * 0.34)
    climb = 160.0 * (1.0 + 7.0 * np.clip((u - 0.35) / 0.65, 0.0, 1.0) ** 1.6)
    freq = np.where(u < 0.35, dive, climb)

    body = dsp.osc_sine(freq, 1.0, cycle.sr) + 0.4 * dsp.osc_sine(2.0 * freq, 1.0, cycle.sr)
    air = dsp.fft_filter(dsp.white_noise(len(cycle), rng), cycle.sr,
                         dsp.bandpass_response(900.0, 5000.0, skirt_oct=0.6))
    body += 0.5 * air / max(float(np.std(air)), 1e-9)

    env = np.where(window, np.sin(np.pi * np.clip(u, 0.0, 1.0)) ** 1.4, 0.0)
    return 0.30 * body * env


class BoostPerformance(Generator):
    name = "boost_performance"
    description = (
        "Interior boost-mode sound: a low rumble under a bright harmonic "
        "lead, both tracking a virtual rev that rises with speed and jumps with "
        "pedal, plus the engagement stinger when the mode arms."
    )
    seed = 41

    def synthesize(self, cycle, rng):
        rev = _virtual_rev(cycle)
        speed_norm = np.clip(np.abs(cycle.speed_kmh) / TOP_SPEED_KMH, 0.0, 1.0)
        throttle = dsp.smooth(np.clip(cycle.pedal, 0.0, 1.0), 0.08, cycle.sr)
        drive = np.clip(0.25 * speed_norm + 0.85 * throttle, 0.0, 1.0)
        intensity = 0.08 + 0.92 * drive

        lead_f0 = LEAD_BASE_HZ * (1.0 + LEAD_SPAN * rev)
        lead_phase = dsp.phase_ramp(lead_f0, cycle.sr)

        dark = np.zeros(len(cycle))
        bright = np.zeros(len(cycle))
        for n in range(1, LEAD_PARTIALS + 1):
            amp = (1.25 if n % 2 else 1.0) / n**1.15
            partial = amp * np.sin(n * lead_phase + rng.uniform(0.0, 6.28))
            if n < BRIGHT_FROM:
                dark += partial
            else:
                bright += partial
        lead = 0.16 * intensity * (dark + bright * (0.25 + 0.75 * drive))

        rumble_f0 = np.clip(lead_f0 / RUMBLE_DIVISOR, *RUMBLE_RANGE_HZ)
        rumble_phase = dsp.phase_ramp(rumble_f0, cycle.sr)
        rumble = sum(
            amp * np.sin(n * rumble_phase + rng.uniform(0.0, 6.28))
            for n, amp in ((1, 1.0), (2, 0.45), (3, 0.22))
        )
        # A small floor only - an armed car at a standstill hums, it does not
        # rumble. Most of the rumble has to arrive with the throttle, or the
        # mode engaging is not an event.
        rumble *= 0.24 * (0.10 + 0.90 * intensity)

        # Turbine-like air over the top, so full throttle reads as effort.
        air = dsp.fft_filter(dsp.white_noise(len(cycle), rng), cycle.sr,
                             dsp.bandpass_response(1800.0, 6000.0, skirt_oct=0.7))
        air = 0.05 * drive * air / max(float(np.std(air)), 1e-9)

        mono = lead + rumble + air + _stinger(cycle, rng)
        mono = dsp.fft_filter(mono, cycle.sr,
                              dsp.lowpass_response(OUTPUT_ROLLOFF_HZ, skirt_oct=0.6))
        return dsp.stereo_widen(mono, cycle.sr, delay_ms=0.5, amount=0.16)
