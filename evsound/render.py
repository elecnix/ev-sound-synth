"""Render the generators to 30-second WAV files.

    python3 -m evsound.render --all
    python3 -m evsound.render boost_performance --duration 12

Files are written at a comfortable listening level, which for the pedestrian
alert means a large boost: its calibrated level is only about 45 dB SPL at
rest. Every file's manifest entry records the boost and the resulting
calibration, so a measurement can be undone back to real sound pressure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import analysis, dsp
from .generators import ALL_GENERATORS

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "out"
LISTENING_PEAK_DBFS = -3.0


def _by_name() -> dict[str, type]:
    return {cls.name: cls for cls in ALL_GENERATORS}


def render_one(cls, duration: float, sr: int, out_dir: Path, index: int) -> dict:
    generator = cls()
    result = generator.render(duration=duration, sr=sr)

    listening, gain_db = dsp.normalize_peak(result.audio, LISTENING_PEAK_DBFS)
    path = out_dir / f"{index:02d}_{result.name}.wav"
    dsp.write_wav(path, listening, result.sr)

    mono = result.audio.mean(axis=1)
    return {
        "file": path.name,
        "name": result.name,
        "description": result.description,
        "duration_s": round(result.duration, 3),
        "sample_rate": result.sr,
        "drive_cycle": result.cycle.label,
        "listening_gain_db": round(gain_db, 2),
        "spl_at_full_scale_db": round(result.spl_at_full_scale - gain_db, 2),
        "calibrated_peak_dbfs": round(result.peak_dbfs, 2),
        "level_dba": round(analysis.overall_level(mono, result.sr, result.spl_at_full_scale), 1),
    }


def main(argv: list[str] | None = None) -> int:
    names = _by_name()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("generators", nargs="*", choices=[*names, []], default=[],
                        help="generator names; omit with --all for every one")
    parser.add_argument("--all", action="store_true", help="render every generator")
    parser.add_argument("--duration", type=float, default=30.0, help="seconds per file")
    parser.add_argument("--sample-rate", type=int, default=dsp.SR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    selected = ALL_GENERATORS if args.all or not args.generators else [
        names[n] for n in args.generators
    ]
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for index, cls in enumerate(selected, start=1):
        entry = render_one(cls, args.duration, args.sample_rate, args.out, index)
        manifest.append(entry)
        print(
            f"{entry['file']:<32} {entry['duration_s']:>6.2f} s  "
            f"{entry['level_dba']:>6.1f} dB(A)  boost {entry['listening_gain_db']:+.1f} dB"
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{len(manifest)} files in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
