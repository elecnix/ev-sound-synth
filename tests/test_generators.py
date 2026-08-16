"""Every generator has to deliver a clean, correctly-shaped 30 s render, and
each one has to actually synthesise the thing it claims to."""

import numpy as np
import pytest

from evsound import analysis, dsp, vehicle
from evsound.generators import (
    ALL_GENERATORS,
    BoostPerformance,
    InverterPwm,
    MotorOrders,
    RoadWindNoise,
)

DURATION = 30.0


@pytest.fixture(scope="module")
def renders():
    return {g.name: g.render(duration=DURATION) for g in (cls() for cls in ALL_GENERATORS)}


def test_there_are_several_distinct_generators():
    names = [cls().name for cls in ALL_GENERATORS]
    assert len(names) == len(set(names))
    assert len(names) >= 6


@pytest.mark.parametrize("cls", ALL_GENERATORS, ids=lambda c: c().name)
def test_render_is_exactly_30_seconds_of_clean_stereo(cls, renders):
    result = renders[cls().name]

    assert result.sr == dsp.SR
    assert result.audio.shape == (int(DURATION * dsp.SR), 2)
    assert np.isfinite(result.audio).all()
    assert np.max(np.abs(result.audio)) <= 1.0
    assert np.sqrt(np.mean(result.audio**2)) > 1e-4, "render is effectively silent"


@pytest.mark.parametrize("cls", ALL_GENERATORS, ids=lambda c: c().name)
def test_render_starts_and_ends_at_silence(cls, renders):
    """Without fades a 30 s file clicks on loop."""
    result = renders[cls().name]
    edge = int(0.002 * result.sr)

    assert np.max(np.abs(result.audio[:edge])) < 1e-3
    assert np.max(np.abs(result.audio[-edge:])) < 1e-3


@pytest.mark.parametrize("cls", ALL_GENERATORS, ids=lambda c: c().name)
def test_render_has_a_name_and_a_description(cls):
    g = cls()
    assert g.name and " " not in g.name
    assert len(g.description) > 20


def test_motor_orders_whine_tracks_the_shaft_speed():
    """The dominant partial must be the 48th mechanical order of whatever the
    motor is doing at that instant - that is what makes it sound like a motor
    and not a fixed drone."""
    spec = vehicle.REFERENCE_EV
    cycle = vehicle.constant_speed_cycle(80.0, duration=4.0)
    result = MotorOrders().render(duration=4.0, cycle=cycle)

    mid = result.audio[int(1.0 * result.sr) : int(3.0 * result.sr)].mean(axis=1)
    expected = spec.shaft_hz(80.0 / 3.6) * spec.stator_slots

    assert analysis.dominant_frequency(mid, result.sr, fmin=1000.0) == pytest.approx(
        expected, rel=0.02
    )


def test_motor_orders_pitch_rises_with_speed():
    fast = MotorOrders().render(duration=4.0, cycle=vehicle.constant_speed_cycle(120.0, 4.0))
    slow = MotorOrders().render(duration=4.0, cycle=vehicle.constant_speed_cycle(40.0, 4.0))

    def whine(r):
        return analysis.dominant_frequency(
            r.audio[int(1.5 * r.sr) : int(2.5 * r.sr)].mean(axis=1), r.sr, fmin=500.0
        )

    assert whine(fast) == pytest.approx(3.0 * whine(slow), rel=0.05)


def test_inverter_puts_sidebands_around_the_switching_frequency():
    """A fixed-carrier PWM inverter radiates at fsw plus and minus multiples of
    the electrical fundamental. Those sidebands are the whole character."""
    spec = vehicle.REFERENCE_EV
    speed = 80.0 / 3.6
    cycle = vehicle.constant_speed_cycle(80.0, duration=4.0)
    result = InverterPwm().render(duration=4.0, cycle=cycle)

    mid = result.audio[int(1.0 * result.sr) : int(3.0 * result.sr)].mean(axis=1)
    fsw = spec.inverter_switching_hz
    f_elec = spec.electrical_hz(speed)

    carrier = analysis.peak_level_near(mid, result.sr, fsw, tol_hz=25.0)
    sideband = analysis.peak_level_near(mid, result.sr, fsw + 2 * f_elec, tol_hz=25.0)
    gap = analysis.peak_level_near(mid, result.sr, fsw + 1.11 * f_elec, tol_hz=15.0)

    assert carrier - gap > 20.0
    assert sideband - gap > 12.0


def test_road_noise_grows_with_speed_like_tyre_noise():
    """Tyre noise rises roughly 30*log10(v): tripling speed adds ~14 dB."""
    slow = RoadWindNoise().render(duration=4.0, cycle=vehicle.constant_speed_cycle(40.0, 4.0))
    fast = RoadWindNoise().render(duration=4.0, cycle=vehicle.constant_speed_cycle(120.0, 4.0))

    def level(r):
        return analysis.overall_level(
            r.audio[int(1.0 * r.sr) : int(3.0 * r.sr)].mean(axis=1), r.sr, r.spl_at_full_scale
        )

    assert 10.0 < level(fast) - level(slow) < 24.0


def test_road_noise_is_broadband_not_tonal():
    r = RoadWindNoise().render(duration=4.0, cycle=vehicle.constant_speed_cycle(100.0, 4.0))
    mid = r.audio[int(1.0 * r.sr) : int(3.0 * r.sr)].mean(axis=1)

    levels = analysis.third_octave_levels(mid, r.sr, r.spl_at_full_scale)
    band_values = [levels[fc] for fc in dsp.THIRD_OCTAVE_CENTERS]

    # No single third-octave band may tower over its neighbours.
    assert max(band_values) - np.median(band_values) < 12.0


def test_boost_mode_delivers_both_a_low_rumble_and_a_high_lead():
    """Boost mode is a high frequency over a low rumble. Both layers have to be
    measurable, not just the bright one."""
    r = BoostPerformance().render(duration=DURATION)
    sr = r.sr
    mid = r.audio[int(8 * sr) : int(12 * sr)].mean(axis=1)

    f, p = dsp.power_spectrum(mid, sr)

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return 10.0 * np.log10(max(p[m].sum(), 1e-30))

    rumble = band(25.0, 120.0)
    lead = band(600.0, 4000.0)
    floor = band(12_000.0, 20_000.0)

    assert rumble - floor > 20.0
    assert lead - floor > 20.0


def test_boost_mode_gets_louder_and_brighter_under_full_throttle():
    r = BoostPerformance().render(duration=DURATION)
    sr = r.sr

    def rms(a, b):
        return np.sqrt(np.mean(r.audio[int(a * sr) : int(b * sr)] ** 2))

    idle = rms(0.5, 1.8)      # stationary, before the launch
    launch = rms(3.0, 5.0)    # boost launch
    assert launch > 4.0 * idle


def test_stereo_channels_are_decorrelated_but_not_out_of_phase(renders):
    """A wide render must not collapse to silence when a phone sums to mono."""
    for name, result in renders.items():
        left, right = result.audio[:, 0], result.audio[:, 1]
        mono = 0.5 * (left + right)
        wide = np.sqrt(np.mean(left**2) + np.mean(right**2))
        assert np.sqrt(np.mean(mono**2)) > 0.35 * wide, f"{name} cancels in mono"
