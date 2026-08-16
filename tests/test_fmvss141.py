"""The pedestrian alert is the one generator with a legal ground truth.

FMVSS No. 141 (49 CFR 571.141) says an EV must emit, at each test condition,
enough sound in enough of the thirteen one-third octave bands from 315 Hz to
5000 Hz, and must get louder as it speeds up. These tests measure the rendered
audio against the tables in the rule.
"""

import numpy as np
import pytest

from evsound import analysis, fmvss141, vehicle
from evsound.generators import AvasForward, AvasReverse

WINDOW_INSET_S = 0.5


def _mid(result, t_start, t_end):
    """Mono sum of a window of the render, which is what a microphone hears."""
    sr = result.sr
    lo = int((t_start + WINDOW_INSET_S) * sr)
    hi = int((t_end - WINDOW_INSET_S) * sr)
    return result.audio[lo:hi].mean(axis=1)


@pytest.fixture(scope="module")
def forward():
    return AvasForward().render(duration=30.0)


@pytest.fixture(scope="module")
def reverse():
    return AvasReverse().render(duration=30.0)


def test_the_rule_tables_are_transcribed_for_all_thirteen_bands():
    for table in fmvss141.ALL_TABLES.values():
        assert sorted(table) == sorted(fmvss141.BANDS)


def test_four_band_option_is_satisfied_at_every_test_condition(forward):
    """S5.1.3 four-band option: at least four non-adjacent bands each at or
    above the table minimum, spanning no fewer than 9 of the 13 bands."""
    for plateau in forward.cycle.plateaus:
        table = fmvss141.table_for_condition(plateau.label)
        levels = analysis.third_octave_levels(
            _mid(forward, plateau.t_start, plateau.t_end),
            forward.sr,
            spl_at_full_scale=forward.spl_at_full_scale,
        )
        passing = [fc for fc in fmvss141.BANDS if levels[fc] >= table[fc]]

        chosen = fmvss141.largest_non_adjacent_subset(passing)
        assert len(chosen) >= 4, f"{plateau.label}: only {passing} met the minimum"
        span = fmvss141.BANDS.index(chosen[-1]) - fmvss141.BANDS.index(chosen[0]) + 1
        assert span >= 9, f"{plateau.label}: bands {chosen} span only {span}"


def test_sound_gets_at_least_3db_louder_at_every_speed_step(forward):
    """S5.4 relative volume change: >= 3 dB from one condition to the next."""
    levels = [
        analysis.overall_level(
            _mid(forward, p.t_start, p.t_end),
            forward.sr,
            spl_at_full_scale=forward.spl_at_full_scale,
        )
        for p in forward.cycle.plateaus
    ]

    steps = np.diff(levels)
    assert np.all(steps >= 3.0), f"levels {levels} -> steps {steps}"


def test_the_alert_fades_out_above_the_32kmh_cutoff():
    """FMVSS 141 stops requiring sound above 30 km/h, where tyre noise takes
    over. Leaving the alert on at motorway speed would be a design fault."""
    result = AvasForward().render(duration=30.0, cycle=vehicle.performance_cycle(30.0))
    sr = result.sr

    slow = result.audio[int(1.0 * sr) : int(2.0 * sr)].mean(axis=1)
    fast = result.audio[int(11.0 * sr) : int(12.0 * sr)].mean(axis=1)

    assert np.sqrt(np.mean(fast**2)) < 0.05 * np.sqrt(np.mean(slow**2))


def test_reverse_alert_meets_the_reverse_table(reverse):
    plateau = reverse.cycle.plateaus[-1]
    levels = analysis.third_octave_levels(
        _mid(reverse, plateau.t_start, plateau.t_end),
        reverse.sr,
        spl_at_full_scale=reverse.spl_at_full_scale,
    )
    table = fmvss141.TABLE_2_REVERSE
    passing = [fc for fc in fmvss141.BANDS if levels[fc] >= table[fc]]

    chosen = fmvss141.largest_non_adjacent_subset(passing)
    assert len(chosen) >= 4
    span = fmvss141.BANDS.index(chosen[-1]) - fmvss141.BANDS.index(chosen[0]) + 1
    assert span >= 9


def _envelope_of_last_plateau(result):
    """Amplitude contour over a window held at one speed, so what is measured
    is the sound's own modulation and not a speed change."""
    plateau = result.cycle.plateaus[-1]
    sr = result.sr
    x = result.audio[int((plateau.t_start + 1.5) * sr) : int((plateau.t_end - 1.0) * sr)]
    return np.sqrt(np.maximum(0.0, analysis.envelope(x.mean(axis=1), sr, 0.02)))


def test_reverse_alert_is_distinguishable_from_the_forward_alert(forward, reverse):
    """A pedestrian must be able to tell a reversing car apart by ear. The
    reverse alert pulses; the forward alert only shimmers."""
    reverse_env = _envelope_of_last_plateau(reverse)
    forward_env = _envelope_of_last_plateau(forward)

    assert reverse_env.std() / reverse_env.mean() > 0.35
    assert forward_env.std() / forward_env.mean() < 0.25


def test_the_reverse_pulse_runs_at_its_designed_rate(reverse):
    env = _envelope_of_last_plateau(reverse)
    rate = analysis.dominant_frequency(env - env.mean(), reverse.sr, fmin=0.4, fmax=6.0)

    assert rate == pytest.approx(AvasReverse.PULSE_HZ, abs=0.15)


def test_largest_non_adjacent_subset_picks_the_widest_legal_set():
    assert fmvss141.largest_non_adjacent_subset([315, 400, 500]) == [315, 500]
    assert fmvss141.largest_non_adjacent_subset([]) == []
    picked = fmvss141.largest_non_adjacent_subset(fmvss141.BANDS)
    assert len(picked) == 7
    assert picked[0] == 315 and picked[-1] == 5000
