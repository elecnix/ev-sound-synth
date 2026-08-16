"""The drive cycle is what every generator is modulated by, so its speed,
rpm and order frequencies have to be right before any sound is judged."""

import numpy as np
import pytest

from evsound import vehicle


def test_spec_maps_road_speed_to_motor_rpm():
    spec = vehicle.REFERENCE_EV

    # 100 km/h = 27.78 m/s. Wheel turns 27.78 / (2*pi*r) rev/s.
    rpm = spec.motor_rpm(100.0 / 3.6)
    wheel_rps = (100.0 / 3.6) / (2 * np.pi * spec.tire_radius_m)
    assert rpm == pytest.approx(wheel_rps * 60.0 * spec.final_drive, rel=1e-9)
    assert 6000.0 < rpm < 9000.0  # sane for a single-speed EV at 100 km/h


def test_electrical_fundamental_is_pole_pairs_times_shaft_speed():
    spec = vehicle.REFERENCE_EV
    speed = 100.0 / 3.6

    shaft = spec.shaft_hz(speed)
    assert spec.electrical_hz(speed) == pytest.approx(shaft * spec.pole_pairs)


def test_slot_order_lands_in_the_audible_whine_range():
    """The 48th mechanical order is the classic EV whine; at motorway speed
    it must sit in the few-kHz range where the ear is most sensitive."""
    spec = vehicle.REFERENCE_EV
    whine = spec.shaft_hz(100.0 / 3.6) * spec.stator_slots

    assert 4000.0 < whine < 9000.0


def test_performance_cycle_starts_stopped_and_ends_stopped():
    cycle = vehicle.performance_cycle(duration=30.0, sr=8000)

    assert cycle.speed_mps[0] == pytest.approx(0.0, abs=1e-6)
    assert cycle.speed_mps[-1] == pytest.approx(0.0, abs=0.5)
    assert cycle.speed_kmh.max() > 150.0


def test_performance_cycle_launches_to_60mph_in_about_34_seconds():
    """Reference performance EV in boost mode: 0-60 mph in 3.4 s."""
    cycle = vehicle.performance_cycle(duration=30.0, sr=8000)
    launch = cycle.speed_kmh >= 96.6

    t_hit = cycle.t[np.argmax(launch)]
    t_start = cycle.t[np.argmax(cycle.speed_kmh > 0.5)]
    assert t_hit - t_start == pytest.approx(3.4, abs=0.6)


def test_performance_cycle_speed_is_continuous():
    """A step in speed would be an audible click in every speed-driven layer."""
    sr = 8000
    cycle = vehicle.performance_cycle(duration=30.0, sr=sr)

    accel = np.diff(cycle.speed_mps) * sr
    assert np.max(np.abs(accel)) < 14.0  # m/s^2, well past any road car


def test_pedal_is_positive_under_acceleration_and_negative_on_regen():
    cycle = vehicle.performance_cycle(duration=30.0, sr=8000)

    assert cycle.pedal.max() > 0.8
    assert cycle.pedal.min() < -0.3
    assert np.all(cycle.pedal >= -1.0001) and np.all(cycle.pedal <= 1.0001)


def test_avas_staircase_holds_the_four_fmvss_test_conditions():
    cycle = vehicle.avas_staircase(duration=30.0, sr=8000)

    labels = [p.label for p in cycle.plateaus]
    assert labels == ["stationary", "10 km/h", "20 km/h", "30 km/h"]

    for plateau, expected in zip(cycle.plateaus, [0.0, 10.0, 20.0, 30.0]):
        window = (cycle.t >= plateau.t_start) & (cycle.t < plateau.t_end)
        held = cycle.speed_kmh[window]
        assert held.min() == pytest.approx(expected, abs=0.05)
        assert held.max() == pytest.approx(expected, abs=0.05)
        assert plateau.t_end - plateau.t_start >= 3.0


def test_reverse_cycle_is_negative_speed():
    cycle = vehicle.reverse_cycle(duration=30.0, sr=8000)

    assert cycle.speed_kmh.min() < -5.0
    assert cycle.speed_kmh.max() <= 0.0001
