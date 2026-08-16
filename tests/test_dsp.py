"""Ground-truth tests for the DSP primitives.

Each test measures a property of the output, not a status flag: a filter is
judged by its measured rejection, a noise generator by its measured slope.
"""

import numpy as np
import pytest

from evsound import dsp


def test_write_wav_produces_a_30s_48k_24bit_stereo_file(tmp_path):
    n = int(30.0 * dsp.SR)
    audio = np.zeros((n, 2), dtype=np.float64)
    path = tmp_path / "silence.wav"

    dsp.write_wav(path, audio, dsp.SR)

    import wave

    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 48_000
        assert w.getnchannels() == 2
        assert w.getsampwidth() == 3
        assert w.getnframes() == n


def test_wav_roundtrip_preserves_the_samples(tmp_path):
    rng = np.random.default_rng(7)
    audio = rng.uniform(-0.5, 0.5, size=(1024, 2))
    path = tmp_path / "noise.wav"

    dsp.write_wav(path, audio, dsp.SR)
    back, sr = dsp.read_wav(path)

    assert sr == dsp.SR
    # 24-bit quantisation step is 2**-23; allow one step of error.
    assert np.max(np.abs(back - audio)) < 2.0 ** -22


def test_swept_sine_is_phase_continuous():
    """A frequency sweep built on a phase ramp must never step-discontinue.

    The per-sample delta of a sine is bounded by 2*amp*sin(pi*f/sr). If the
    oscillator restarted its phase on a frequency change, some delta would
    blow past that bound.
    """
    n = dsp.SR  # 1 s
    freq = np.linspace(80.0, 6000.0, n)
    x = dsp.osc_sine(freq, amp=1.0, sr=dsp.SR)

    bound = 2.0 * np.sin(np.pi * freq.max() / dsp.SR)
    assert np.max(np.abs(np.diff(x))) < bound * 1.02


def test_bandpass_rejects_an_octave_outside_the_band():
    rng = np.random.default_rng(1)
    x = dsp.white_noise(dsp.SR, rng)
    y = dsp.fft_filter(x, dsp.SR, dsp.bandpass_response(800.0, 1600.0, skirt_oct=0.5))

    def band_db(lo, hi):
        f, p = dsp.power_spectrum(y, dsp.SR)
        m = (f >= lo) & (f < hi)
        return 10.0 * np.log10(max(p[m].sum(), 1e-30))

    passband = band_db(900.0, 1500.0)
    below = band_db(200.0, 400.0)
    above = band_db(3200.0, 6400.0)

    assert passband - below > 40.0
    assert passband - above > 40.0


def test_pink_noise_has_equal_power_per_octave():
    rng = np.random.default_rng(2)
    x = dsp.pink_noise(8 * dsp.SR, rng)
    f, p = dsp.power_spectrum(x, dsp.SR)

    levels = []
    for lo in (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0):
        m = (f >= lo) & (f < 2 * lo)
        levels.append(10.0 * np.log10(p[m].sum()))

    assert np.max(levels) - np.min(levels) < 1.5


def test_soft_clip_stays_inside_unity():
    x = np.linspace(-8.0, 8.0, 10_001)
    y = dsp.soft_clip(x)

    assert np.max(np.abs(y)) < 1.0
    assert np.all(np.diff(y) >= 0.0)  # monotonic: no fold-back distortion
    # Near-linear for small signals.
    small = np.abs(x) < 0.1
    assert np.max(np.abs(y[small] - x[small])) < 0.01


@pytest.mark.parametrize(
    "freq,expected",
    [(31.5, -39.4), (100.0, -19.1), (1000.0, 0.0), (4000.0, 1.0), (10_000.0, -2.5)],
)
def test_a_weighting_matches_iec_61672(freq, expected):
    assert dsp.a_weighting_db(np.array([freq]))[0] == pytest.approx(expected, abs=0.4)


def test_third_octave_centers_are_the_thirteen_fmvss_bands():
    assert dsp.THIRD_OCTAVE_CENTERS[0] == 315
    assert dsp.THIRD_OCTAVE_CENTERS[-1] == 5000
    assert len(dsp.THIRD_OCTAVE_CENTERS) == 13


def test_smooth_preserves_the_mean_and_removes_the_ripple():
    """Interior only: a moving average cannot cancel a ripple inside its own
    first half-window, because there is nothing yet to average against. Every
    caller here either fades the edges or gates them to zero.
    """
    t = np.linspace(0.0, 1.0, dsp.SR, endpoint=False)
    x = 5.0 + np.sin(2 * np.pi * 50.0 * t)
    y = dsp.smooth(x, 0.1, dsp.SR)

    edge = int(0.05 * dsp.SR)
    assert np.mean(y) == pytest.approx(5.0, abs=0.01)
    assert np.max(np.abs(y[edge:-edge] - 5.0)) < 0.05
    # The edges may ripple but must never overshoot the input range.
    assert y.min() >= x.min() and y.max() <= x.max()
