"""Tests for slapdetect.py — the experimental slap-section guesser."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fretting as fr  # noqa: E402
import slapdetect as sd  # noqa: E402


@dataclass
class N:
    start: float
    end: float
    pitch: int


def test_model_shape():
    m = sd.load_model()
    n = 2 * len(m["feature_index"]) + 1
    assert len(m["coef"]) == len(m["mean"]) == len(m["scale"]) == n
    assert 0 < m["threshold"] < 1 and m["window_s"] > 0


def test_sections_from_mask_needs_four_notes_and_merges_gaps():
    notes = [N(i * 0.25, i * 0.25 + 0.2, 40) for i in range(40)]
    on = np.zeros(40, bool)
    on[2:4] = True            # too short
    on[10:16] = True          # 2.5–3.95
    on[20:26] = True          # 5.0–6.45: gap < 2 s → merged
    secs = sd.sections_from_mask(notes, on)
    assert secs == [(2.45, 6.5)]


def test_format_round_trips_through_parse_sections():
    secs = [(3.2, 26.04), (65.5, 95.1), (130.0, 162.37)]
    assert fr.parse_sections(sd.format_sections(secs)) == secs


def test_detect_sections_runs_on_audio():
    pytest.importorskip("librosa")
    sr = 22050
    t = np.arange(int(sr * 6)) / sr
    y = np.zeros_like(t)
    notes = []
    for i in range(24):
        s = i * 0.25
        m = (t >= s) & (t < s + 0.22)
        y[m] += 0.5 * np.sin(2 * np.pi * 55 * (t[m] - s)) * np.exp(-(t[m] - s) * 8)
        notes.append(N(s, s + 0.22, 33))
    secs = sd.detect_sections(y.astype(np.float32), sr, notes)
    assert isinstance(secs, list)
    for a, b in secs:
        assert 0 <= a < b <= 6.5
