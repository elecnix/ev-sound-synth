"""Measurement, kept separate from synthesis so the tests judge the audio
rather than the code that made it.

Digital samples carry no absolute level, so every reading needs a calibration
point. The convention here: `spl_at_full_scale` is the dB SPL a full-scale
sine would produce at the relevant microphone position. Each generator
declares its own, because the FMVSS microphone sits 2 m from the bumper while
the cabin mix is measured at the driver's ear.
"""

from __future__ import annotations

import numpy as np

from . import dsp

_FULL_SCALE_SINE_DBFS = 10.0 * np.log10(0.5)  # -3.0103


def _calibration(spl_at_full_scale: float) -> float:
    return spl_at_full_scale - _FULL_SCALE_SINE_DBFS


def _weighted_power(x: np.ndarray, sr: int, weighting: str):
    f, power = dsp.power_spectrum(x, sr)
    if weighting.upper() == "A":
        power = power * 10.0 ** (dsp.a_weighting_db(f) / 10.0)
    elif weighting.upper() != "Z":
        raise ValueError(f"unknown weighting {weighting!r}")
    return f, power


def third_octave_levels(x: np.ndarray, sr: int, spl_at_full_scale: float,
                        weighting: str = "A") -> dict[int, float]:
    """Band levels in dB SPL for the thirteen FMVSS bands."""
    f, power = _weighted_power(np.asarray(x, dtype=np.float64), sr, weighting)
    cal = _calibration(spl_at_full_scale)

    levels: dict[int, float] = {}
    for fc in dsp.THIRD_OCTAVE_CENTERS:
        lo, hi = dsp.third_octave_edges(fc)
        band = power[(f >= lo) & (f < hi)]
        levels[fc] = 10.0 * np.log10(max(band.sum(), 1e-30)) + cal
    return levels


def overall_level(x: np.ndarray, sr: int, spl_at_full_scale: float,
                  weighting: str = "A") -> float:
    """Wideband level in dB SPL."""
    _, power = _weighted_power(np.asarray(x, dtype=np.float64), sr, weighting)
    return 10.0 * np.log10(max(power.sum(), 1e-30)) + _calibration(spl_at_full_scale)


def dominant_frequency(x: np.ndarray, sr: int, fmin: float = 20.0,
                       fmax: float | None = None) -> float:
    """Frequency of the strongest bin, refined by parabolic interpolation so a
    reading is accurate to well under one bin."""
    f, power = dsp.power_spectrum(np.asarray(x, dtype=np.float64), sr)
    fmax = fmax if fmax is not None else sr / 2.0

    mask = (f >= fmin) & (f <= fmax)
    idx = int(np.argmax(np.where(mask, power, 0.0)))
    if idx <= 0 or idx >= len(power) - 1:
        return float(f[idx])

    a, b, c = (np.log(max(power[i], 1e-30)) for i in (idx - 1, idx, idx + 1))
    denom = a - 2 * b + c
    offset = 0.0 if abs(denom) < 1e-12 else 0.5 * (a - c) / denom
    return float(f[idx] + offset * (f[1] - f[0]))


def peak_level_near(x: np.ndarray, sr: int, freq: float, tol_hz: float = 20.0) -> float:
    """dB of the strongest bin within +/- tol_hz of `freq`."""
    f, power = dsp.power_spectrum(np.asarray(x, dtype=np.float64), sr)
    band = power[(f >= freq - tol_hz) & (f <= freq + tol_hz)]
    return 10.0 * np.log10(max(band.max() if band.size else 0.0, 1e-30))


def envelope(x: np.ndarray, sr: int, seconds: float = 0.02) -> np.ndarray:
    """Short-term mean square: the amplitude contour, used to detect pulsing."""
    return dsp.smooth(np.asarray(x, dtype=np.float64) ** 2, seconds, sr)
