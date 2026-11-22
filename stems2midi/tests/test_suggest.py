"""Second-pass hints (suggest.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("librosa")

import suggest as sg  # noqa: E402

SR = 11025


def _tone(sec, midi=40, amp=0.5, pluck=False):
    t = np.arange(int(SR * sec)) / SR
    f = 440 * 2 ** ((midi - 69) / 12)
    y = sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 2, 3)) * amp
    if pluck:
        y *= np.exp(-t * 6)
    return y.astype(np.float32)


def test_long_note_cut_in_pieces_suggests_sustained():
    y = np.concatenate([np.zeros(SR // 2, np.float32), _tone(12.0), np.zeros(SR // 2, np.float32)])
    notes = [{"t": 0.5 + i * 0.25, "d": 0.25, "m": 40} for i in range(48)]
    hints = sg.suggest(y, SR, notes, 70.0)
    assert [h["option"] for h in hints] == ["--sustained"]


def test_plucked_notes_suggest_nothing():
    y = np.concatenate([_tone(0.5, pluck=True) for _ in range(40)])
    notes = [{"t": i * 0.5, "d": 0.45, "m": 40} for i in range(40)]
    assert sg.suggest(y, SR, notes, 120.0) == []


def test_quiet_leak_suggests_only():
    quiet = np.concatenate([_tone(0.5, amp=0.01, pluck=True) for _ in range(80)])     # 40 s, −34 dB
    loud = np.concatenate([_tone(0.5, amp=0.5, pluck=True) for _ in range(120)])      # 60 s
    y = np.concatenate([quiet, loud])
    notes = [{"t": i * 0.5, "d": 0.45, "m": 40} for i in range(200)]
    hints = sg.suggest(y, SR, notes, 120.0)
    only = [h for h in hints if h["option"] == "--only"]
    assert only and only[0]["value"].startswith("0:40-")
    assert not sg.suggest(y, SR, notes, 120.0, used={"only": "0:40-1:40"})


def test_slow_tempo_with_sixteenths_suggests_double():
    y = np.concatenate([_tone(0.19, pluck=True) for _ in range(100)])
    notes = [{"t": i * 0.19, "d": 0.17, "m": 40} for i in range(100)]
    hints = sg.suggest(y, SR, notes, 80.0)
    assert any(h["option"] == "--bpm" and h["value"] == "160.0" for h in hints)


def test_hard_review_suggests_notes_from_and_command():
    y = np.concatenate([_tone(0.5, pluck=True) for _ in range(40)])
    notes = [{"t": i * 0.5, "d": 0.45, "m": 40} for i in range(40)]
    review = [{"level": "bad", "relative": True}] * 3 + [{"level": "ok", "relative": True}] * 17
    hints = sg.suggest(y, SR, notes, 120.0, review)
    assert [h["option"] for h in hints] == ["--notes-from"]
    cmd = sg.command(hints, "C:/x y", "--shift 5")
    assert cmd.endswith('"C:/x y" --notes-from "<file.gp5|.musicxml>" --shift 5')
    assert "Worth trying" in sg.report(hints, cmd) and cmd in sg.report(hints, cmd)


def test_bass_missing_from_its_stem_suggests_another_model():
    rng = np.random.default_rng(1)
    low = _tone(20.0, midi=40, amp=0.5)                       # the bass, ended up in "guitar"
    stems = {"bass": (low * 0.03)[None, :], "guitar": (low * 0.97)[None, :],
             "drums": (rng.normal(0, 0.05, len(low)).astype(np.float32))[None, :]}
    h = sg.weak_stem(stems, SR, "htdemucs_6s")
    assert h["option"] == "--model" and h["value"] == "htdemucs_ft" and h.get("source")
    assert '"guitar"' in h["why"]
    assert sg.weak_stem(stems, SR, "htdemucs_ft")["option"] == "--skip-separation"
    ok = {"bass": low[None, :], "drums": stems["drums"]}
    assert sg.weak_stem(ok, SR, "htdemucs_6s") is None
    cmd = sg.command([h], "C:/out/song_stems", source="C:/m/song.mp3")
    assert cmd.endswith('"C:/m/song.mp3" -o "C:/out/song_stems" --model htdemucs_ft')
    assert "Run the song again" in sg.report([h], cmd)


def test_bass_resting_half_the_song_is_not_a_weak_stem():
    rng = np.random.default_rng(2)
    orch = _tone(20.0, midi=36, amp=0.5)
    bass = np.concatenate([np.zeros(len(orch) // 2, np.float32), _tone(10.0, midi=40, amp=0.5)])
    orch[len(orch) // 2:] *= 0.1
    stems = {"bass": bass[None, :], "other": orch[None, :],
             "drums": rng.normal(0, 0.05, len(orch)).astype(np.float32)[None, :]}
    assert sg.weak_stem(stems, SR, "htdemucs_6s") is None
