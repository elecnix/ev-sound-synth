"""Electric vehicle sound synthesis.

Synthesises the two sound systems every electric vehicle has - the pedestrian
alert required by FMVSS No. 141, and the interior performance sound - from a
model of the drive unit, with nothing sampled.

    python3 -m evsound.render --all
"""

from . import analysis, dsp, fmvss141, vehicle

__all__ = ["analysis", "dsp", "fmvss141", "vehicle"]
__version__ = "1.0.0"
