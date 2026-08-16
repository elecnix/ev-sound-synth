"""The pedestrian alert - the exterior sound, and the only one with a law
behind it.

Design, and why:

- Four one-third octave bands, not a broadband whoosh. FMVSS No. 141 measures
  in bands, so putting the energy where it is measured buys compliance at the
  lowest total loudness. Quieter is better: this sound plays in car parks.
- Bands 400 / 800 / 1600 / 3150 Hz forward. No two are neighbours and they
  reach across ten of the thirteen bands, which is the four-band option.
- Reverse uses 500 / 1000 / 2000 / 4000 Hz and pulses at 1.6 Hz. A pedestrian
  has to hear the difference between a car coming and a car backing out.
- Every band's level follows the rule's own tables plus a 4 dB margin, so the
  required 3 dB step per speed increment falls out of the design rather than
  being tuned in.
- Pitch glides about 5% with speed and the whole thing fades out by 34 km/h,
  above which tyre noise does the job.
"""

from __future__ import annotations

import numpy as np

from .. import dsp, fmvss141, vehicle
from .base import EXTERIOR_SPL_AT_FULL_SCALE, Generator

MARGIN_DB = 4.0
"""Headroom over the table minimum, to survive measurement scatter."""

_MAIN_POWER = 0.88
_PARTNER_POWER = 0.06
_NOISE_POWER = 0.06


def _band_layer(cycle, fc, level_db, beat_hz, glide, rng, spl_ref):
    """One third-octave band: a tone, a slightly detuned partner for movement,
    and a whisper of narrowband noise so it does not sound like a test tone."""
    a_weight = dsp.a_weighting_db(np.array([float(fc)]))[0]
    amp = 10.0 ** ((np.asarray(level_db) - spl_ref - a_weight) / 20.0)

    vibrato = 1.0 + 0.008 * np.sin(2 * np.pi * 4.3 * cycle.t + rng.uniform(0, 6.28))
    freq = fc * glide * vibrato

    out = np.sqrt(_MAIN_POWER) * amp * dsp.osc_sine(freq, 1.0, cycle.sr,
                                                    phase0=rng.uniform(0, 6.28))
    out += np.sqrt(_PARTNER_POWER) * amp * dsp.osc_sine(
        freq + beat_hz, 1.0, cycle.sr, phase0=rng.uniform(0, 6.28)
    )
    # The tone carries power amp^2/2; unit-variance noise carries its variance
    # directly, hence the extra factor of two.
    noise = dsp.narrowband_noise(fc, len(cycle), rng, width_oct=1 / 10, sr=cycle.sr)
    out += np.sqrt(_NOISE_POWER / 2.0) * amp * noise
    return out


class _Avas(Generator):
    spl_at_full_scale = EXTERIOR_SPL_AT_FULL_SCALE
    bands: list[int] = []
    beats: list[float] = []
    seed = 11

    def _level_db(self, cycle: vehicle.DriveCycle, fc: int) -> np.ndarray:
        raise NotImplementedError

    def _glide(self, cycle: vehicle.DriveCycle) -> np.ndarray:
        raise NotImplementedError

    def _shape(self, cycle: vehicle.DriveCycle, x: np.ndarray) -> np.ndarray:
        return x

    def synthesize(self, cycle, rng):
        out = np.zeros(len(cycle))
        glide = self._glide(cycle)
        for fc, beat in zip(self.bands, self.beats):
            out += _band_layer(cycle, fc, self._level_db(cycle, fc), beat, glide, rng,
                               self.spl_at_full_scale)
        return dsp.stereo_widen(self._shape(cycle, out), cycle.sr, delay_ms=0.6, amount=0.12)


class AvasForward(_Avas):
    """Forward pedestrian alert, FMVSS No. 141 four-band option."""

    name = "avas_forward"
    description = (
        "Pedestrian alert for forward motion. Four third-octave bands at "
        "400/800/1600/3150 Hz, levels tracking the FMVSS No. 141 tables with a "
        "4 dB margin, pitch gliding 5% with speed, silent above 34 km/h."
    )
    bands = [400, 800, 1600, 3150]
    beats = [0.7, 1.1, 1.5, 1.9]

    _ANCHORS_KMH = [0.0, 10.0, 20.0, 30.0]
    _TABLES = [
        fmvss141.TABLE_1_STATIONARY,
        fmvss141.TABLE_3_10KMH,
        fmvss141.TABLE_4_20KMH,
        fmvss141.TABLE_5_30KMH,
    ]

    def default_cycle(self, duration, sr):
        return vehicle.avas_staircase(duration, sr)

    def _level_db(self, cycle, fc):
        targets = [table[fc] + MARGIN_DB for table in self._TABLES]
        return np.interp(np.abs(cycle.speed_kmh), self._ANCHORS_KMH, targets)

    def _glide(self, cycle):
        norm = np.clip(np.abs(cycle.speed_kmh) / 30.0, 0.0, 1.0)
        return 0.95 + 0.10 * norm

    def _shape(self, cycle, x):
        """Fade out through the 30-34 km/h band, where the rule stops asking."""
        kmh = np.abs(cycle.speed_kmh)
        u = np.clip((fmvss141.CUTOFF_KMH + 2.0 - kmh) / 4.0, 0.0, 1.0)
        return x * (0.5 - 0.5 * np.cos(np.pi * u))


class AvasReverse(_Avas):
    """Reverse alert - deliberately unlike the forward sound."""

    name = "avas_reverse"
    description = (
        "Reverse pedestrian alert. A separate band set at 500/1000/2000/4000 Hz "
        "pulsing at 1.6 Hz, at the FMVSS No. 141 reverse-table levels, so a "
        "pedestrian can tell a reversing car from an approaching one by ear."
    )
    bands = [500, 1000, 2000, 4000]
    beats = [0.9, 1.3, 1.7, 2.1]
    seed = 23

    PULSE_HZ = 1.6
    PULSE_DEPTH = 0.7

    def default_cycle(self, duration, sr):
        return vehicle.reverse_cycle(duration, sr)

    def _level_db(self, cycle, fc):
        return np.full(len(cycle), fmvss141.TABLE_2_REVERSE[fc] + MARGIN_DB)

    def _glide(self, cycle):
        # A slow downward drift as the car creeps back, opposite in sense to
        # the forward alert's rise.
        norm = np.clip(np.abs(cycle.speed_kmh) / 10.0, 0.0, 1.0)
        return 1.03 - 0.05 * norm

    def _shape(self, cycle, x):
        pulse = 1.0 + self.PULSE_DEPTH * np.sin(2 * np.pi * self.PULSE_HZ * cycle.t)
        return x * pulse
