"""Measure the rendered files, not the code that made them.

The test suite checks in-memory arrays. This checks the WAVs that actually
shipped: it reads them back, undoes the listening boost recorded in the
manifest, and re-measures the pedestrian alert against the FMVSS No. 141
tables. A render that quietly clipped, or a manifest that drifted from its
files, shows up here and nowhere else.

    python3 -m evsound.verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import analysis, dsp, fmvss141
from .generators import AvasForward, AvasReverse
from .render import DEFAULT_OUT

WINDOW_INSET_S = 0.5


def _load(out_dir: Path) -> tuple[list[dict], dict[str, Path]]:
    manifest = json.loads((out_dir / "manifest.json").read_text())
    return manifest, {e["name"]: out_dir / e["file"] for e in manifest}


def check_files(manifest: list[dict], paths: dict[str, Path]) -> list[str]:
    problems = []
    for entry in manifest:
        audio, sr = dsp.read_wav(paths[entry["name"]])
        peak = float(np.max(np.abs(audio)))
        duration = audio.shape[0] / sr

        if abs(duration - entry["duration_s"]) > 0.001:
            problems.append(f"{entry['file']}: {duration:.3f} s, manifest says {entry['duration_s']}")
        if sr != entry["sample_rate"]:
            problems.append(f"{entry['file']}: {sr} Hz, manifest says {entry['sample_rate']}")
        if peak > 0.9999:
            problems.append(f"{entry['file']}: peaks at digital full scale - clipped")
        if audio.shape[1] != 2:
            problems.append(f"{entry['file']}: {audio.shape[1]} channels, expected stereo")

        print(
            f"{entry['file']:<32} {duration:6.2f} s  {sr} Hz  "
            f"peak {20 * np.log10(max(peak, 1e-12)):+6.2f} dBFS  "
            f"{entry['level_dba']:6.1f} dB(A)"
        )
    return problems


def check_pedestrian_alert(manifest: list[dict], paths: dict[str, Path]) -> list[str]:
    problems = []
    by_name = {e["name"]: e for e in manifest}

    # The 3 dB step rule (S5.4, Table 7) covers the four forward conditions
    # only. Reverse has a single requirement, Table 2, at any reverse speed -
    # so a reversing car is not asked to get louder as it backs up faster.
    for generator, condition_of, check_steps in (
        (AvasForward(), lambda p: p.label, True),
        (AvasReverse(), lambda p: "reverse", False),
    ):
        entry = by_name[generator.name]
        audio, sr = dsp.read_wav(paths[generator.name])
        mono = audio.mean(axis=1)
        spl_ref = entry["spl_at_full_scale_db"]
        cycle = generator.default_cycle(entry["duration_s"], sr)

        print(f"\n{generator.name} - measured from the rendered file")
        overall = []
        for plateau in cycle.plateaus:
            lo = int((plateau.t_start + WINDOW_INSET_S) * sr)
            hi = int((plateau.t_end - WINDOW_INSET_S) * sr)
            window = mono[lo:hi]

            table = fmvss141.table_for_condition(condition_of(plateau))
            levels = analysis.third_octave_levels(window, sr, spl_ref)
            chosen = fmvss141.largest_non_adjacent_subset(
                [fc for fc in fmvss141.BANDS if levels[fc] >= table[fc]]
            )
            ok = fmvss141.satisfies_four_band_option(levels, table)
            total = analysis.overall_level(window, sr, spl_ref)
            overall.append(total)

            print(
                f"  {plateau.label:<12} {total:5.1f} dB(A)   "
                f"{len(chosen)} non-adjacent bands, span {fmvss141.band_span(chosen)}   "
                f"{'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                problems.append(f"{generator.name} at {plateau.label}: four-band option not met")

        steps = np.diff(overall)
        if check_steps and len(steps):
            shown = ", ".join(f"{s:+.1f}" for s in steps)
            ok = bool(np.all(steps >= fmvss141.MINIMUM_STEP_DB))
            print(f"  speed steps   {shown} dB   {'PASS' if ok else 'FAIL'}")
            if not ok:
                problems.append(f"{generator.name}: a speed step is under 3 dB")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    manifest, paths = _load(args.out)
    problems = check_files(manifest, paths) + check_pedestrian_alert(manifest, paths)

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"\nAll {len(manifest)} rendered files verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
