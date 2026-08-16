"""The analyser is the ruler every other test measures with, so it is
calibrated against a signal whose level is known exactly."""

import numpy as np
import pytest

from evsound import analysis, dsp


def test_full_scale_sine_reads_the_declared_calibration_level():
    """A full-scale 1 kHz sine defines the calibration point.

    A-weighting is ~0 dB at 1 kHz, so the reading is the calibration constant.
    """
    n = 2 * dsp.SR
    x = dsp.osc_sine(np.full(n, 1000.0), amp=1.0, sr=dsp.SR)

    levels = analysis.third_octave_levels(x, dsp.SR, spl_at_full_scale=100.0)

    assert levels[1000] == pytest.approx(100.0, abs=0.3)


def test_a_tone_stays_inside_its_own_third_octave_band():
    n = 2 * dsp.SR
    x = dsp.osc_sine(np.full(n, 1000.0), amp=1.0, sr=dsp.SR)

    levels = analysis.third_octave_levels(x, dsp.SR, spl_at_full_scale=100.0)

    assert levels[1000] - levels[800] > 40.0
    assert levels[1000] - levels[1250] > 40.0


def test_halving_the_amplitude_drops_the_level_by_6db():
    n = dsp.SR
    loud = dsp.osc_sine(np.full(n, 1000.0), amp=0.5, sr=dsp.SR)
    quiet = dsp.osc_sine(np.full(n, 1000.0), amp=0.25, sr=dsp.SR)

    a = analysis.overall_level(loud, dsp.SR, spl_at_full_scale=100.0)
    b = analysis.overall_level(quiet, dsp.SR, spl_at_full_scale=100.0)

    assert a - b == pytest.approx(6.02, abs=0.1)


def test_overall_level_applies_a_weighting():
    """100 Hz is attenuated 19.1 dB by A-weighting; unweighted it is not."""
    n = dsp.SR
    x = dsp.osc_sine(np.full(n, 100.0), amp=0.5, sr=dsp.SR)

    weighted = analysis.overall_level(x, dsp.SR, 100.0, weighting="A")
    flat = analysis.overall_level(x, dsp.SR, 100.0, weighting="Z")

    assert flat - weighted == pytest.approx(19.1, abs=0.6)


def test_dominant_frequency_finds_the_loudest_partial():
    n = dsp.SR
    x = dsp.osc_sine(np.full(n, 220.0), 0.2, sr=dsp.SR)
    x += dsp.osc_sine(np.full(n, 3300.0), 0.6, sr=dsp.SR)

    assert analysis.dominant_frequency(x, dsp.SR) == pytest.approx(3300.0, rel=0.01)


def test_peak_level_near_reports_a_present_and_an_absent_partial():
    n = dsp.SR
    x = dsp.osc_sine(np.full(n, 4000.0), 0.5, sr=dsp.SR)

    present = analysis.peak_level_near(x, dsp.SR, 4000.0, tol_hz=20.0)
    absent = analysis.peak_level_near(x, dsp.SR, 4600.0, tol_hz=20.0)

    assert present - absent > 40.0
