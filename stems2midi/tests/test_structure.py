"""Tests for structure.py — song parts from the audio (--parts auto)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structure as st  # noqa: E402


def _song(sr=22050, bpm=120):
    """intro(4) A(8) B(8) A(8) B(8) C(4) bars: A/B/C differ in chords and timbre."""
    beat = 60 / bpm
    plan = [("I", 4), ("A", 8), ("B", 8), ("A", 8), ("B", 8), ("C", 4)]
    chords = {"I": [[57, 64]], "A": [[48, 52, 55], [53, 57, 60]], "B": [[50, 54, 57], [55, 59, 62]],
              "C": [[46, 50, 53]]}
    bright = {"I": 1, "A": 3, "B": 8, "C": 1}
    rng = np.random.default_rng(0)
    out, t = [], 0.0
    beats = []
    for name, nbars in plan:
        for b in range(nbars):
            ch = chords[name][b % len(chords[name])]
            for k in range(4):
                beats.append(t)
                n = int(beat * sr)
                tt = np.arange(n) / sr
                env = np.exp(-tt * 3)
                x = sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * h * tt) / h ** (4 / bright[name])
                        for m in ch for h in range(1, 6))
                out.append(0.2 * env * x + 0.002 * rng.standard_normal(n))
                t += beat
    beats.append(t)
    return np.concatenate(out).astype(np.float32), sr, beats


def test_finds_repeats_and_names_them():
    pytest.importorskip("librosa")
    y, sr, beats = _song()
    downs = list(range(0, len(beats) - 1, 4))
    segs = st.analyze(y, sr, beats, downs)
    letters = [s["letter"] for s in segs]
    names = [s["name"] for s in segs]
    # the two A's share a letter, the two B's share another, A ≠ B
    a = [s for s in segs if 7.5 <= s["start"] <= 9 or 39.5 <= s["start"] <= 41]
    b = [s for s in segs if 23.5 <= s["start"] <= 25 or 55.5 <= s["start"] <= 57]
    assert len(a) == 2 and a[0]["letter"] == a[1]["letter"], st.report(segs)
    assert len(b) == 2 and b[0]["letter"] == b[1]["letter"], st.report(segs)
    assert a[0]["letter"] != b[0]["letter"]
    assert names[0] == "intro" and {"verse", "chorus"} <= set(names), st.report(segs)
    nums = {}
    for s in segs:
        nums[s["name"]] = nums.get(s["name"], 0) + 1
        assert s["number"] == nums[s["name"]]
    assert letters[0] == "A"                       # letters in order of appearance


def test_parts_spec_round_trips_through_chart_tools():
    import chart_tools as ct
    segs = [{"name": "intro", "start": 0.5, "end": 8.0}, {"name": "verse", "start": 8.0, "end": 24.0},
            {"name": "chorus", "start": 24.0, "end": 40.25}, {"name": "verse", "start": 40.25, "end": 56.0}]
    spec = st.parts_spec(segs)
    assert spec.startswith("intro=0:00.50-0:08.00, verse=0:08.00-0:24.00")
    assert [n for n, _a, _b in ct.parse_parts(spec)] == ["intro", "verse", "chorus", "verse"]
