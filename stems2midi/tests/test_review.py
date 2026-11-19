"""Bar-by-bar review of a transcription against the bass stem."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s2m_review", HERE / "review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

pytest.importorskip("librosa")

SR = 22050
BEAT = 0.5          # 120 BPM


def _pluck_line(n_bars=8, midi=40):
    """One plucked note per beat (with harmonics, decaying)."""
    y = np.zeros(int(SR * (n_bars * 4 * BEAT + 1.0)), np.float32)
    notes = []
    f = 440 * 2 ** ((midi - 69) / 12)
    tt = np.arange(int(SR * BEAT * 0.9)) / SR
    tone = sum(np.sin(2 * np.pi * f * k * tt) / k for k in (1, 2, 3)) * np.exp(-tt * 4)
    for i in range(n_bars * 4):
        t = 0.5 + i * BEAT
        s = int(t * SR)
        y[s:s + len(tone)] += 0.5 * tone.astype(np.float32)
        notes.append({"t": round(t, 3), "d": BEAT * 0.85, "m": midi})
    bars = [(0.5 + b * 4 * BEAT, 0.5 + (b + 1) * 4 * BEAT) for b in range(n_bars)]
    return y, notes, bars


def test_model_file_matches_features():
    m = json.loads((HERE / "review_model.json").read_text(encoding="utf-8"))
    assert len(m["coef"]) == len(m["mean"]) == len(m["scale"]) == len(review.FEATURES)
    assert 0 < m["check"] < m["bad"] < 1


def test_bars_from_beat_map():
    bm = [(0.0, 1), (0.5, 0), (1.0, 0), (1.5, 0), (2.0, 2), (2.5, 0), (3.0, 0), (3.5, 0), (4.0, 3)]
    assert review.bars_from_beat_map(bm) == [(0.0, 2.0), (2.0, 4.0)]


def test_missing_notes_make_a_bar_suspect():
    y, notes, bars = _pluck_line()
    gap = [n for n in notes if not bars[4][0] <= n["t"] < bars[4][1]]
    out = review.review(y, SR, gap, bars)
    assert [b["bar"] for b in out] == list(range(1, 9))
    p = [b["p"] for b in out]
    others = p[1:4] + p[5:7]          # first/last bars sit at the audio edges
    assert p[4] > max(others)
    assert out[4]["level"] != "ok"
    assert any("no note" in w for w in out[4]["why"])


def test_report_lists_flagged_bars_with_eof_times():
    bars = [
        {"t": 1.0, "end": 3.0, "bar": 1, "p": 0.2, "level": "ok", "why": []},
        {"t": 3.0, "end": 5.0, "bar": 2, "p": 0.9, "level": "bad", "why": ["attacks with no note"]},
        {"t": 5.0, "end": 7.0, "bar": 3, "p": 0.6, "level": "check", "why": []},
    ]
    txt = review.report(bars, shift=5)
    assert "1 likely wrong, 1 worth a check, 1 look fine (of 3)" in txt
    lines = [ln for ln in txt.splitlines() if ln.strip().startswith("bar")]
    assert lines[0].split()[1] == "2" and "[0:08.0]" in lines[0] and "attacks with no note" in lines[0]
    assert lines[1].split()[1] == "3"


def test_hard_stem_is_ranked_within_the_song():
    y, notes, bars = _pluck_line(n_bars=10)
    model = json.loads((HERE / "review_model.json").read_text(encoding="utf-8"))
    model = dict(model, check=0.0001, bad=0.0002)       # everything "bad" by the usual standard
    out = review.review(y, SR, notes, bars, model)
    assert all(b.get("relative") for b in out)
    assert 1 <= sum(b["level"] == "bad" for b in out) <= 3
    assert "worst bars of this song" in review.report(out)
