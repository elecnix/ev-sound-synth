"""FMVSS No. 141 - Minimum Sound Requirements for Hybrid and Electric Vehicles.

The tables below are transcribed from 49 CFR 571.141. They are the reason the
pedestrian alert generator sounds the way it does: a designer picks bands and
levels to clear these minima at the least annoying cost.

What the rule demands, in short:

- Thirteen one-third octave bands, 315 Hz to 5000 Hz.
- Either two non-adjacent bands from 315 to 3150 Hz spanning at least three
  bands, or four non-adjacent bands spanning at least nine of the thirteen.
  This project uses the four-band option.
- At least 3 dB more sound at each speed step: stationary, 10, 20, 30 km/h.
- A distinct sound in reverse.

Source: https://www.law.cornell.edu/cfr/text/49/571.141
"""

from __future__ import annotations

BANDS = [315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]

TABLE_1_STATIONARY = dict(zip(BANDS, [39, 39, 40, 40, 41, 41, 42, 39, 39, 37, 34, 32, 31]))
TABLE_2_REVERSE = dict(zip(BANDS, [42, 41, 43, 43, 44, 44, 45, 41, 42, 40, 37, 35, 33]))
TABLE_3_10KMH = dict(zip(BANDS, [45, 44, 46, 46, 47, 47, 48, 44, 45, 43, 40, 38, 36]))
TABLE_4_20KMH = dict(zip(BANDS, [52, 51, 52, 53, 53, 54, 54, 51, 51, 50, 47, 45, 43]))
TABLE_5_30KMH = dict(zip(BANDS, [56, 55, 57, 57, 58, 58, 59, 55, 55, 54, 51, 49, 47]))

ALL_TABLES = {
    "stationary": TABLE_1_STATIONARY,
    "reverse": TABLE_2_REVERSE,
    "10 km/h": TABLE_3_10KMH,
    "20 km/h": TABLE_4_20KMH,
    "30 km/h": TABLE_5_30KMH,
}

MINIMUM_STEP_DB = 3.0
"""S5.4: minimum relative volume change between consecutive conditions."""

CUTOFF_KMH = 32.0
"""Above this the rule stops requiring sound - tyre noise is loud enough."""

FORWARD_CONDITIONS = ["stationary", "10 km/h", "20 km/h", "30 km/h"]


def table_for_condition(label: str) -> dict[int, int]:
    try:
        return ALL_TABLES[label]
    except KeyError:
        raise ValueError(f"no FMVSS table for condition {label!r}") from None


def largest_non_adjacent_subset(passing: list[int]) -> list[int]:
    """Pick as many bands as possible with no two of them neighbours.

    Greedy from the lowest band is optimal here: the bands form a path, and on
    a path the leftmost-first greedy gives a maximum independent set.
    """
    chosen: list[int] = []
    last_index = -2
    for fc in sorted(set(passing), key=BANDS.index):
        index = BANDS.index(fc)
        if index > last_index + 1:
            chosen.append(fc)
            last_index = index
    return chosen


def band_span(chosen: list[int]) -> int:
    """How many of the thirteen bands the chosen set reaches across."""
    if not chosen:
        return 0
    return BANDS.index(chosen[-1]) - BANDS.index(chosen[0]) + 1


def satisfies_four_band_option(levels: dict[int, float], table: dict[int, int]) -> bool:
    chosen = largest_non_adjacent_subset([fc for fc in BANDS if levels[fc] >= table[fc]])
    return len(chosen) >= 4 and band_span(chosen) >= 9
