"""Order-based synthesis of the drive unit: what the car actually makes.

An electric drive unit has no combustion, so its tonal content is entirely
"orders" - components locked to a fixed multiple of shaft speed. Three
families matter:

- Electromagnetic orders from the stator. The 48th order (one per slot) is the
  dominant radial force and is the sound people call "EV whine". Torque ripple
  adds the 6th electrical harmonic, which is the 24th mechanical order.
- Gear mesh orders, one per reduction stage, at the tooth count of each pinion.
- The electrical fundamental itself, felt as a low hum.

Every partial is locked to the shaft frequency the drive cycle dictates, so
the pitch tracks the car instead of drifting.
"""

from __future__ import annotations

import numpy as np

from .. import dsp
from .base import Generator

ELECTROMAGNETIC_ORDERS = [
    (4.0, 0.22),    # electrical fundamental - 4 pole pairs
    (24.0, 0.30),   # 6th electrical harmonic, torque ripple
    (48.0, 1.00),   # slot passing - the dominant whine
    (96.0, 0.25),   # 2nd slot harmonic
]

CABIN_ROLLOFF_HZ = 9000.0
"""Glass and trim roll the top off before it reaches the cabin."""

WANDER_FRACTION = 0.0006
"""Shaft-speed wobble. Keeps the whine alive without moving its pitch."""


def slow_noise(n: int, seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance noise band-limited to roughly 1/`seconds` Hz."""
    x = dsp.smooth(rng.standard_normal(n), seconds, sr)
    return x / max(float(np.std(x)), 1e-12)


class MotorOrders(Generator):
    name = "motor_orders"
    description = (
        "Order-based synthesis of the drive unit: the 48th slot order (the EV "
        "whine), the 6th electrical harmonic from torque ripple, and both gear "
        "mesh stages, every partial locked to shaft speed."
    )
    seed = 5

    def _orders(self, spec) -> list[tuple[float, float]]:
        stage1 = float(spec.motor_pinion_teeth)
        stage2 = spec.motor_pinion_teeth / spec.stage1_wheel_teeth * spec.stage2_pinion_teeth
        return ELECTROMAGNETIC_ORDERS + [(stage1, 0.32), (stage2, 0.26)]

    def synthesize(self, cycle, rng):
        shaft = np.maximum(cycle.shaft_hz, 0.5)
        wander = 1.0 + WANDER_FRACTION * slow_noise(len(cycle), 0.25, cycle.sr, rng)
        base_phase = dsp.phase_ramp(shaft * wander, cycle.sr)

        out = np.zeros(len(cycle))
        for order, amp in self._orders(cycle.spec):
            out += amp * np.sin(order * base_phase + rng.uniform(0.0, 6.28))

        # Load, not speed, sets how hard the machine sings; the floor keeps a
        # coasting car from going silent.
        load = 0.35 + 0.65 * dsp.smooth(np.abs(cycle.pedal), 0.15, cycle.sr)
        moving = np.clip(np.abs(cycle.speed_kmh) / 4.0, 0.0, 1.0)
        out *= 0.16 * load * moving

        out = dsp.fft_filter(out, cycle.sr, dsp.lowpass_response(CABIN_ROLLOFF_HZ, skirt_oct=1.0))
        return dsp.stereo_widen(out, cycle.sr, delay_ms=0.35, amount=0.2)
