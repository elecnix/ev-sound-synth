"""What every generator has in common.

A generator turns a drive cycle into calibrated stereo audio. It does not
normalise: the level it returns is the level it means, because the pedestrian
alert is only legal at a stated sound pressure. Making files comfortable to
listen to is the renderer's job, and the renderer records the gain it applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import dsp, vehicle

FADE_IN_S = 0.12
FADE_OUT_S = 0.30

CABIN_SPL_AT_FULL_SCALE = 100.0
"""dB SPL a full-scale sample would make at the driver's ear."""

EXTERIOR_SPL_AT_FULL_SCALE = 80.0
"""dB SPL a full-scale sample would make at the FMVSS microphone, 2 m out."""


@dataclass
class GeneratorResult:
    name: str
    description: str
    audio: np.ndarray  # shape (n, 2), float in [-1, 1]
    sr: int
    spl_at_full_scale: float
    cycle: vehicle.DriveCycle

    @property
    def duration(self) -> float:
        return self.audio.shape[0] / self.sr

    @property
    def peak_dbfs(self) -> float:
        return 20.0 * np.log10(max(float(np.max(np.abs(self.audio))), 1e-12))


class Generator:
    """Base class. Subclasses set the metadata and implement `synthesize`."""

    name: str = ""
    description: str = ""
    spl_at_full_scale: float = CABIN_SPL_AT_FULL_SCALE
    seed: int = 0

    _gain_db: float = 0.0
    """Set by `synthesize` when it normalises. The stated calibration moves by
    the same amount, so the file never misreports its own sound pressure."""

    def default_cycle(self, duration: float, sr: int) -> vehicle.DriveCycle:
        return vehicle.performance_cycle(duration, sr)

    def synthesize(self, cycle: vehicle.DriveCycle, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def render(self, duration: float = 30.0, sr: int = dsp.SR,
               cycle: vehicle.DriveCycle | None = None) -> GeneratorResult:
        cycle = cycle if cycle is not None else self.default_cycle(duration, sr)
        audio = np.asarray(self.synthesize(cycle, np.random.default_rng(self.seed)),
                           dtype=np.float64)

        if audio.ndim == 1:
            audio = dsp.stereo_widen(audio, cycle.sr)
        audio = dsp.fade(audio, cycle.sr, FADE_IN_S, FADE_OUT_S)

        peak = float(np.max(np.abs(audio)))
        if peak > 0.99:
            # tanh rather than a hard clip: monotonic, so it warms instead of
            # tearing. Only reached if a mix stacks up further than expected.
            audio = dsp.soft_clip(audio * (0.99 / peak) * 1.2) * 0.83

        return GeneratorResult(
            name=self.name,
            description=self.description,
            audio=audio,
            sr=cycle.sr,
            spl_at_full_scale=self.spl_at_full_scale - self._gain_db,
            cycle=cycle,
        )


def speed_gain(cycle: vehicle.DriveCycle, anchors_kmh, anchors_db) -> np.ndarray:
    """Piecewise-linear-in-dB gain against road speed.

    Working in dB is what makes the FMVSS step rule provable: if every layer
    rises by at least 3 dB between two conditions, so does their sum.
    """
    kmh = np.abs(cycle.speed_kmh)
    db = np.interp(kmh, anchors_kmh, anchors_db)
    return dsp.db_to_amp(db)


def tone_amplitude(level_db: float, freq: float, spl_at_full_scale: float) -> float:
    """Amplitude of a sine that reads `level_db` dB(A) SPL in its own band."""
    return float(10.0 ** ((level_db - spl_at_full_scale - dsp.a_weighting_db(np.array([freq]))[0]) / 20.0))
