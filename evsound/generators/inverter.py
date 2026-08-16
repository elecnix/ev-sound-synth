"""Traction inverter noise: the fixed PWM carrier and its sidebands.

A traction inverter chops DC at a fixed switching frequency. The chopping
beats against the motor's electrical fundamental, so the radiated spectrum is
a strong tone at the carrier with a comb of sidebands at the carrier plus and
minus whole multiples of the electrical frequency. Those sidebands are what
make an EV sound electronic rather than mechanical: the carrier stays put
while the comb around it opens and closes with road speed.

A `spread` option jitters the carrier, which is what random-PWM inverters do
to trade a piercing tone for a wider, softer hiss.
"""

from __future__ import annotations

import numpy as np

from .. import dsp
from .base import Generator

# Sideband index -> amplitude relative to the carrier. Loosely the Bessel
# pattern of a phase-modulated carrier: the first pair is not the loudest.
SIDEBAND_AMPLITUDES = {1: 0.55, 2: 0.75, 3: 0.40, 4: 0.28, 5: 0.18, 6: 0.12}

NYQUIST_MARGIN_HZ = 2000.0


class InverterPwm(Generator):
    name = "inverter_pwm"
    description = (
        "Traction inverter whine: a 10 kHz PWM carrier with sidebands at the "
        "carrier plus and minus multiples of the electrical fundamental, so the "
        "comb widens with road speed. Set spread=True for random-PWM hiss."
    )
    seed = 13

    def __init__(self, spread: bool = False):
        self.spread = spread

    def synthesize(self, cycle, rng):
        fsw = cycle.spec.inverter_switching_hz
        f_elec = np.maximum(cycle.electrical_hz, 0.5)

        carrier_freq = np.full(len(cycle), fsw)
        if self.spread:
            jitter = dsp.smooth(rng.standard_normal(len(cycle)), 0.002, cycle.sr) * 40.0
            carrier_freq = carrier_freq + 0.06 * fsw * jitter

        carrier_phase = dsp.phase_ramp(carrier_freq, cycle.sr)
        elec_phase = dsp.phase_ramp(f_elec, cycle.sr)
        limit = cycle.sr / 2.0 - NYQUIST_MARGIN_HZ

        out = np.sin(carrier_phase)
        for k, amp in SIDEBAND_AMPLITUDES.items():
            for sign in (1, -1):
                if fsw + sign * k * float(np.max(f_elec)) > limit:
                    continue
                out += amp * np.sin(carrier_phase + sign * k * elec_phase
                                    + rng.uniform(0.0, 6.28))

        # Second carrier group, quieter and only its close sidebands.
        if 2 * fsw < limit:
            second = 2.0 * carrier_phase
            out += 0.18 * np.sin(second)
            for k in (1, 2):
                for sign in (1, -1):
                    if 2 * fsw + sign * k * float(np.max(f_elec)) > limit:
                        continue
                    out += 0.10 * np.sin(second + sign * k * elec_phase)

        # Magnetostriction hiss under the comb.
        hiss = dsp.fft_filter(dsp.white_noise(len(cycle), rng), cycle.sr,
                              dsp.bandpass_response(3000.0, 14000.0, skirt_oct=0.8))
        out += 0.25 * hiss / max(np.std(hiss), 1e-9)

        # Current draw, not speed, sets the level: an inverter under load is
        # loud even at a standstill.
        load = 0.20 + 0.80 * dsp.smooth(np.abs(cycle.pedal), 0.10, cycle.sr)
        out *= 0.10 * load

        return dsp.stereo_widen(out, cycle.sr, delay_ms=0.25, amount=0.25)
