"""The vehicle model that drives every generator.

An EV's sound has no engine speed to hang on, so everything is derived from
road speed through a fixed single-speed reduction. That chain - road speed to
wheel speed to motor shaft speed to electrical frequency to motor order - is
what makes a synthesised sound track the car instead of drifting free of it.

The reference vehicle is a dual-motor performance SUV of the kind sold today:
615 hp, 650 lb-ft, 0-60 mph in 3.4 s on 275/40R21 tyres. Manufacturers do not
publish driveline internals, so the gearing, pole count and slot count are
stated assumptions. Each one sets pitch; none affects whether the code is
correct, and all of them live in this one class, so another vehicle can be
described by editing it alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dsp


@dataclass(frozen=True)
class VehicleSpec:
    """Driveline geometry, in the terms the synthesis needs."""

    name: str = "reference dual-motor performance EV"

    # 275/40R21: 533.4 mm rim + 2 x 110 mm sidewall.
    tire_radius_m: float = 0.3767
    # Single-speed reduction, both drive units. ASSUMED - not published.
    final_drive: float = 11.1

    # Permanent-magnet synchronous machine: 8 poles, 48 stator slots. ASSUMED.
    pole_pairs: int = 4
    stator_slots: int = 48

    # Two-stage helical reduction; tooth counts ASSUMED, chosen to give
    # realistic mesh orders rather than to match a specific gearset.
    motor_pinion_teeth: int = 23
    stage1_wheel_teeth: int = 71
    stage2_pinion_teeth: int = 19

    # Fixed-carrier PWM. 10 kHz is typical for a traction inverter.
    inverter_switching_hz: float = 10_000.0

    peak_power_hp: float = 615.0
    peak_torque_lbft: float = 650.0
    zero_to_60mph_s: float = 3.4  # boost mode

    def wheel_hz(self, speed_mps):
        return np.asarray(speed_mps, dtype=np.float64) / (2 * np.pi * self.tire_radius_m)

    def shaft_hz(self, speed_mps):
        """Motor shaft revolutions per second."""
        return self.wheel_hz(speed_mps) * self.final_drive

    def motor_rpm(self, speed_mps):
        return self.shaft_hz(speed_mps) * 60.0

    def electrical_hz(self, speed_mps):
        """Fundamental of the stator current."""
        return self.shaft_hz(speed_mps) * self.pole_pairs

    def slot_order_hz(self, speed_mps):
        """The dominant radial force order - the classic EV whine."""
        return self.shaft_hz(speed_mps) * self.stator_slots

    def gear_mesh_hz(self, speed_mps):
        """Mesh frequency of each reduction stage."""
        shaft = self.shaft_hz(speed_mps)
        stage1 = shaft * self.motor_pinion_teeth
        inter = shaft * self.motor_pinion_teeth / self.stage1_wheel_teeth
        return stage1, inter * self.stage2_pinion_teeth


REFERENCE_EV = VehicleSpec()


@dataclass(frozen=True)
class Plateau:
    """A window where speed is held, so a level can be measured over it."""

    label: str
    t_start: float
    t_end: float
    speed_kmh: float


@dataclass
class DriveCycle:
    """Sampled speed and pedal over time, plus the derived motor frequencies."""

    t: np.ndarray
    speed_mps: np.ndarray
    pedal: np.ndarray
    sr: int
    spec: VehicleSpec = REFERENCE_EV
    plateaus: list[Plateau] = field(default_factory=list)
    label: str = ""

    @property
    def speed_kmh(self) -> np.ndarray:
        return self.speed_mps * 3.6

    @property
    def abs_speed_mps(self) -> np.ndarray:
        return np.abs(self.speed_mps)

    @property
    def shaft_hz(self) -> np.ndarray:
        return self.spec.shaft_hz(self.abs_speed_mps)

    @property
    def electrical_hz(self) -> np.ndarray:
        return self.spec.electrical_hz(self.abs_speed_mps)

    @property
    def motor_rpm(self) -> np.ndarray:
        return self.spec.motor_rpm(self.abs_speed_mps)

    @property
    def throttle(self) -> np.ndarray:
        """Positive pedal only - the drive-side demand."""
        return np.clip(self.pedal, 0.0, 1.0)

    @property
    def regen(self) -> np.ndarray:
        return np.clip(-self.pedal, 0.0, 1.0)

    def __len__(self) -> int:
        return len(self.t)


def _smoothstep_keyframes(t: np.ndarray, keys: list[tuple[float, float]]) -> np.ndarray:
    """Interpolate keyframes with a smoothstep, so speed is continuous and so
    is its first derivative. A linear ramp would kink the pitch audibly at
    every keyframe."""
    out = np.empty_like(t)
    out[:] = keys[0][1]

    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        m = (t >= t0) & (t <= t1)
        u = (t[m] - t0) / max(t1 - t0, 1e-9)
        out[m] = v0 + (v1 - v0) * (u * u * (3.0 - 2.0 * u))

    out[t > keys[-1][0]] = keys[-1][1]
    return out


_PEDAL_REFERENCE_ACCEL = 9.0  # m/s^2 mapped to full pedal


def _pedal_from_speed(speed_mps: np.ndarray, sr: int) -> np.ndarray:
    accel = np.gradient(speed_mps) * sr
    return np.clip(dsp.smooth(accel, 0.08, sr) / _PEDAL_REFERENCE_ACCEL, -1.0, 1.0)


def _cycle(keys, duration, sr, plateaus=(), label="", spec=REFERENCE_EV) -> DriveCycle:
    t = dsp.time_axis(duration, sr)
    speed = _smoothstep_keyframes(t, keys) / 3.6
    return DriveCycle(
        t=t,
        speed_mps=speed,
        pedal=_pedal_from_speed(speed, sr),
        sr=sr,
        spec=spec,
        plateaus=list(plateaus),
        label=label,
    )


def performance_cycle(duration: float = 30.0, sr: int = dsp.SR) -> DriveCycle:
    """Launch, two pulls, a lift with regen, then a hard stop.

    The first pull is the published boost-mode figure: 0-96.6 km/h (60 mph) in
    3.4 s, starting at t = 2 s so the file opens on a stationary car.
    """
    keys = [
        (0.0, 0.0),
        (2.0, 0.0),
        (5.4, 96.6),     # 0-60 mph in 3.4 s
        (9.0, 150.0),
        (12.0, 168.0),
        (14.5, 168.0),   # cruise
        (18.0, 95.0),    # lift and regen
        (19.5, 95.0),
        (23.0, 150.0),   # second pull
        (24.0, 152.0),
        (duration, 0.0), # regen plus friction to a stop
    ]
    return _cycle(keys, duration, sr, label="performance launch cycle")


def avas_staircase(duration: float = 30.0, sr: int = dsp.SR) -> DriveCycle:
    """The four FMVSS No. 141 test conditions, each held long enough to
    measure: stationary, 10, 20 and 30 km/h."""
    keys = [
        (0.0, 0.0), (6.0, 0.0),
        (7.5, 10.0), (13.5, 10.0),
        (15.0, 20.0), (21.0, 20.0),
        (22.5, 30.0), (duration, 30.0),
    ]
    plateaus = [
        Plateau("stationary", 0.0, 6.0, 0.0),
        Plateau("10 km/h", 7.5, 13.5, 10.0),
        Plateau("20 km/h", 15.0, 21.0, 20.0),
        Plateau("30 km/h", 22.5, duration, 30.0),
    ]
    return _cycle(keys, duration, sr, plateaus, label="FMVSS 141 speed staircase")


def reverse_cycle(duration: float = 30.0, sr: int = dsp.SR) -> DriveCycle:
    """Backing out of a space: a pause, then a steady 8 km/h reverse."""
    keys = [(0.0, 0.0), (2.0, 0.0), (4.0, -8.0), (26.0, -8.0), (duration, -3.0)]
    plateaus = [
        Plateau("stationary", 0.0, 2.0, 0.0),
        Plateau("reverse", 4.0, 26.0, -8.0),
    ]
    return _cycle(keys, duration, sr, plateaus, label="reverse manoeuvre")


def constant_speed_cycle(speed_kmh: float, duration: float = 30.0, sr: int = dsp.SR,
                         pedal: float = 0.3) -> DriveCycle:
    """Steady cruise. Used by the tests that need a stationary spectrum."""
    t = dsp.time_axis(duration, sr)
    return DriveCycle(
        t=t,
        speed_mps=np.full_like(t, speed_kmh / 3.6),
        pedal=np.full_like(t, pedal),
        sr=sr,
        plateaus=[Plateau(f"{speed_kmh:g} km/h", 0.0, duration, speed_kmh)],
        label=f"steady {speed_kmh:g} km/h",
    )
