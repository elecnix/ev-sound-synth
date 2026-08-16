"""Signal primitives shared by every generator.

Two rules shape this module:

- Every oscillator is driven by an instantaneous-frequency array and a phase
  accumulator, so a pitch that glides with road speed never clicks.
- Every filter is applied in the frequency domain. A recursive filter would
  need a per-sample Python loop over 1.4 million samples; masking an rFFT is
  exact for a fixed response and runs in milliseconds.

Where a filter genuinely has to change over time, the generator crossfades
between fixed bands instead of sweeping a filter.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 48_000
"""Sample rate, in Hz. 48 kHz keeps the 10 kHz inverter carrier and its upper
sidebands well below Nyquist."""

THIRD_OCTAVE_CENTERS = [315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]
"""The thirteen one-third octave bands FMVSS No. 141 measures in."""

_BAND_EDGE = 2.0 ** (1.0 / 6.0)


# --------------------------------------------------------------------------
# time and oscillators
# --------------------------------------------------------------------------


def time_axis(duration: float, sr: int = SR) -> np.ndarray:
    return np.arange(int(round(duration * sr)), dtype=np.float64) / sr


def phase_ramp(freq: np.ndarray, sr: int = SR, phase0: float = 0.0) -> np.ndarray:
    """Integrate an instantaneous frequency into a continuous phase."""
    return phase0 + 2.0 * np.pi * np.cumsum(np.asarray(freq, dtype=np.float64)) / sr


def osc_sine(freq, amp=1.0, sr: int = SR, phase0: float = 0.0) -> np.ndarray:
    return np.asarray(amp) * np.sin(phase_ramp(np.asarray(freq, dtype=np.float64), sr, phase0))


def osc_stack(freq, partials, sr: int = SR, rng: np.random.Generator | None = None) -> np.ndarray:
    """Additive stack. `partials` is a sequence of (ratio, amplitude) pairs.

    Each partial gets its own random start phase, which keeps the summed
    waveform from spiking on every fundamental period.
    """
    freq = np.asarray(freq, dtype=np.float64)
    rng = rng or np.random.default_rng(0)
    out = np.zeros_like(freq)
    for ratio, amp in partials:
        out += osc_sine(freq * ratio, amp, sr, phase0=rng.uniform(0.0, 2 * np.pi))
    return out


# --------------------------------------------------------------------------
# noise
# --------------------------------------------------------------------------


def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Noise with equal power per octave, made by tilting white noise by 1/sqrt(f)."""
    x = white_noise(n, rng)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    tilt = np.ones_like(f)
    tilt[1:] = 1.0 / np.sqrt(f[1:])
    tilt[0] = 0.0
    y = np.fft.irfft(spec * tilt, n=n)
    return y / max(np.std(y), 1e-12)


def narrowband_noise(fc: float, n: int, rng: np.random.Generator, width_oct: float = 1 / 12,
                     sr: int = SR) -> np.ndarray:
    """Noise confined to a slice around `fc`, narrow enough to stay inside one
    third-octave band even after the raised-cosine skirts."""
    x = white_noise(n, rng)
    lo = fc * 2.0 ** (-width_oct / 2)
    hi = fc * 2.0 ** (width_oct / 2)
    y = fft_filter(x, sr, bandpass_response(lo, hi, skirt_oct=width_oct))
    return y / max(np.std(y), 1e-12)


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------


def fft_filter(x: np.ndarray, sr: int, response) -> np.ndarray:
    """Apply a zero-phase magnitude response to a whole signal at once."""
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(spec * response(f), n=n)


def _rc_edge(u: np.ndarray) -> np.ndarray:
    """Raised-cosine ramp from 0 to 1 over u in [0, 1]."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))


def bandpass_response(lo: float, hi: float, skirt_oct: float = 0.5):
    def response(f: np.ndarray) -> np.ndarray:
        lf = np.log2(np.maximum(f, 1e-6))
        rise = _rc_edge((lf - (np.log2(lo) - skirt_oct)) / skirt_oct)
        fall = _rc_edge(((np.log2(hi) + skirt_oct) - lf) / skirt_oct)
        return rise * fall

    return response


def lowpass_response(cutoff: float, skirt_oct: float = 1.0):
    def response(f: np.ndarray) -> np.ndarray:
        lf = np.log2(np.maximum(f, 1e-6))
        return _rc_edge(((np.log2(cutoff) + skirt_oct) - lf) / skirt_oct)

    return response


def highpass_response(cutoff: float, skirt_oct: float = 1.0):
    def response(f: np.ndarray) -> np.ndarray:
        lf = np.log2(np.maximum(f, 1e-6))
        return _rc_edge((lf - (np.log2(cutoff) - skirt_oct)) / skirt_oct)

    return response


def tilt_response(db_per_octave: float, pivot: float = 1000.0):
    def response(f: np.ndarray) -> np.ndarray:
        lf = np.log2(np.maximum(f, 1e-6) / pivot)
        return 10.0 ** (db_per_octave * lf / 20.0)

    return response


# --------------------------------------------------------------------------
# envelopes and shaping
# --------------------------------------------------------------------------


def smooth(x: np.ndarray, seconds: float, sr: int = SR) -> np.ndarray:
    """Boxcar average in O(n). Used to de-ripple control signals."""
    n = len(x)
    w = max(1, int(round(seconds * sr)))
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - w // 2)
    hi = np.minimum(n, idx + w // 2 + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def soft_clip(x: np.ndarray) -> np.ndarray:
    """tanh saturation: monotonic, so it warms rather than folds."""
    return np.tanh(np.asarray(x, dtype=np.float64))


def fade(x: np.ndarray, sr: int = SR, fade_in: float = 0.02, fade_out: float = 0.05) -> np.ndarray:
    """Raised-cosine fades at both ends, so the file neither clicks nor loops badly."""
    y = np.array(x, dtype=np.float64, copy=True)
    n = y.shape[0]
    n_in = min(int(fade_in * sr), n // 2)
    n_out = min(int(fade_out * sr), n // 2)
    if n_in:
        ramp = _rc_edge(np.arange(n_in) / n_in)
        y[:n_in] *= ramp if y.ndim == 1 else ramp[:, None]
    if n_out:
        ramp = _rc_edge(np.arange(n_out)[::-1] / n_out)
        y[-n_out:] *= ramp if y.ndim == 1 else ramp[:, None]
    return y


def db_to_amp(db) -> np.ndarray:
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)


def normalize_peak(x: np.ndarray, peak_db: float = -1.0) -> tuple[np.ndarray, float]:
    """Scale to a target peak. Returns the audio and the gain applied, in dB,
    so a caller can undo the change when reporting calibrated levels."""
    peak = float(np.max(np.abs(x)))
    if peak < 1e-12:
        return x, 0.0
    gain = db_to_amp(peak_db) / peak
    return x * gain, 20.0 * np.log10(gain)


# --------------------------------------------------------------------------
# stereo
# --------------------------------------------------------------------------


def stereo_widen(x: np.ndarray, sr: int = SR, delay_ms: float = 0.45,
                 amount: float = 0.18) -> np.ndarray:
    """Add a small anti-phase delayed copy to each side.

    The mid channel is exactly the input, so summing to mono loses nothing -
    a plain one-sided delay would comb-filter the mono sum instead.
    """
    d = int(delay_ms * 1e-3 * sr)
    delayed = np.concatenate((np.zeros(d), x[:-d])) if d else np.zeros_like(x)
    return np.stack((x + amount * delayed, x - amount * delayed), axis=1)


def stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack((left, right), axis=1)


# --------------------------------------------------------------------------
# weighting and spectra
# --------------------------------------------------------------------------


def a_weighting_db(f: np.ndarray) -> np.ndarray:
    """IEC 61672 A-weighting curve, normalised to 0 dB at 1 kHz."""
    f = np.maximum(np.asarray(f, dtype=np.float64), 1e-6)
    f2 = f * f
    num = (12194.0**2) * f2 * f2
    den = (
        (f2 + 20.6**2)
        * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
        * (f2 + 12194.0**2)
    )
    return 20.0 * np.log10(num / den) + 2.0


def power_spectrum(x: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed power per bin, scaled so the bins sum to the mean square.

    A full-scale sine therefore sums to 0.5, i.e. -3.01 dBFS, which is what
    makes the SPL calibration in `analysis` exact.
    """
    n = len(x)
    win = np.hanning(n)
    spec = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(n, 1.0 / sr)

    fold = np.full(spec.shape, 2.0)
    fold[0] = 1.0
    if n % 2 == 0:
        fold[-1] = 1.0
    power = fold * np.abs(spec) ** 2 / (n * np.sum(win**2))
    return f, power


def third_octave_edges(fc: float) -> tuple[float, float]:
    return fc / _BAND_EDGE, fc * _BAND_EDGE


# --------------------------------------------------------------------------
# file i/o
# --------------------------------------------------------------------------


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SR) -> Path:
    """Write 24-bit PCM. 24 bits keeps the AVAS stationary condition, which
    sits near -35 dBFS, well clear of the quantisation floor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    a = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    if a.shape[0] < a.shape[1] and a.ndim == 2 and a.shape[0] <= 2:
        a = a.T
    channels = a.shape[1]

    clipped = np.clip(a, -1.0, 1.0 - 2.0**-23)
    ints = np.round(clipped * (2**23 - 1)).astype("<i4").reshape(-1)
    packed = np.frombuffer(ints.tobytes(), dtype=np.uint8).reshape(-1, 4)[:, :3]

    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(packed.tobytes())
    return path


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        channels, width, sr, frames = (
            w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        )
        raw = w.readframes(frames)

    if width != 3:
        raise ValueError(f"expected 24-bit PCM, got {width * 8}-bit")

    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    padded = np.concatenate((np.zeros((b.shape[0], 1), dtype=np.uint8), b), axis=1)
    ints = np.frombuffer(padded.tobytes(), dtype="<i4") >> 8
    return ints.reshape(-1, channels).astype(np.float64) / (2**23 - 1), sr
