"""One example of each part from Guitar Pro (gpsections.py, --sections-from)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("librosa")

import gpsections as gs  # noqa: E402

SR = 11025
VERSE = [(0, .5, 40), (.5, .5, 40), (1, 1, 43), (2, .5, 45), (2.5, .5, 47), (3, 1, 45), (4, 1, 40), (5, 1, 38),
         (6, 2, 40)]                                                   # 2 bars, in beats
CHORUS = [(0, 1, 45), (1, 1, 45), (2, 1, 48), (3, 1, 50), (4, 2, 43), (6, 1, 43), (7, 1, 42)]


def _pluck(m, d):
    t = np.arange(int(SR * d)) / SR
    f = 440 * 2 ** ((m - 69) / 12)
    return sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 2, 3, 4)) * np.exp(-t * 3) * 0.3


def _song(order, bpm=118.0, lead=1.3):
    beat = 60 / bpm
    y = np.zeros(int(SR * (lead + 8 * len(order) * beat + 3)))
    truth = []
    for i, part in enumerate(order):
        for b, d, m in (VERSE if part == "V" else CHORUS):
            t = lead + (i * 8 + b) * beat
            seg = _pluck(m, d * beat * 0.95)
            k = int(t * SR)
            y[k:k + len(seg)] += seg
            truth.append((t, m))
    y += np.random.default_rng(0).normal(0, 0.003, len(y))
    return y.astype(np.float32), truth, lead + np.arange(8 * len(order)) * beat


def _gp(bpm=90.0, empty_bar_at_end=True):
    """What notesimport.read_gp gives for a tab with a 'Verso' and a 'Refrão' marker."""
    gb = 60 / bpm
    off = 4 * gb                                                       # a bar of rest first
    raw = [(off + b * gb, d * gb, m, {"s": 0, "f": m - 28}) for b, d, m in VERSE]
    raw += [(off + (8 + b) * gb, d * gb, m, {"s": 1, "f": m - 33, "lg": 1}) for b, d, m in CHORUS]
    n = 4 + 16 + (4 if empty_bar_at_end else 0)
    beats = [(i * gb, i // 4 + 1 if i % 4 == 0 else -1) for i in range(n)]
    return raw, {"beats": beats, "sections": [("Verso", off), ("Refrão", off + 8 * gb)]}


def test_marker_names():
    assert gs.section_name("Verse 2") == "verse"
    assert gs.section_name("Refrão") == "chorus" and gs.section_name("Pré-refrão") == "prechorus"
    assert gs.section_name("chorus3") == "chorus"
    assert gs.section_name("Wheeled out once a year", allowed={"verse", "riff"}) == "riff"


def test_examples_in_beats_without_trailing_rests():
    raw, info = _gp()
    ex = gs.examples_from(raw, info["beats"], info["sections"])
    assert [(e.name, e.label, e.beats) for e in ex] == [("verse", "Verso", 8), ("chorus", "Refrão", 8)]
    assert [round(n["b"], 3) for n in ex[0].notes][:3] == [0.0, 0.5, 1.0]
    assert ex[1].notes[0]["s"] == 1 and ex[1].notes[0]["lg"] == 1


def test_examples_found_across_the_song():
    y, truth, tracked = _song("VVVVCCVV")
    raw, info = _gp()
    res = gs.place(raw, info, y, SR, tracked, 118.0)
    assert [(o["name"], o["times"]) for o in res["occurrences"]] == [("verse", 4), ("chorus", 2), ("verse", 2)]
    got = res["notes"]
    found = sum(any(abs(n["t"] - t) < 0.03 and n["m"] == m for n in got) for t, m in truth)
    assert found == len(truth) and len(got) == len(truth)
    assert res["parts"].startswith("verse=1.") and ", chorus=" in res["parts"]


def test_wrong_tracked_tempo_falls_back_to_a_constant_grid():
    y, truth, tracked = _song("VVCCVV", bpm=112.0)
    raw, info = _gp(bpm=100.0)
    wrong = 1.3 + np.arange(60) * 60 / 117.5                      # the tracker heard 117.5 BPM
    res = gs.place(raw, info, y, SR, wrong, 117.5)
    got = res["notes"]
    found = sum(any(abs(n["t"] - t) < 0.04 and n["m"] == m for n in got) for t, m in truth)
    assert found >= 0.9 * len(truth), res["report"]
    assert res["grid"].startswith("constant")


def test_an_example_not_in_the_song_is_not_used():
    y, truth, tracked = _song("VVVV")
    raw, info = _gp()
    res = gs.place(raw, info, y, SR, tracked, 118.0)
    assert {o["name"] for o in res["occurrences"]} == {"verse"}
    assert "not found in the song: Refrão" in res["report"]


def test_pipeline_sections_from(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("mido")
    import stems2midi as s2m
    y, truth, tracked = _song("VVVVCCVV")
    src = tmp_path / "Band - Song.wav"
    sf.write(str(src), y, SR)
    gp = tmp_path / "examples.gp5"
    gp.write_bytes(b"GP")
    raw, info = _gp()
    info.update({"track": "Bass", "track_no": 1, "tracks": ["1 Bass (bass)"], "tuning": [28, 33, 38, 43],
                 "octave": 0, "muted": 0, "capo": 0, "repeats_once": False})
    monkeypatch.setattr(s2m._sibling("notesimport"), "read_gp", lambda p, t=None: (raw, info))
    res = s2m.run_pipeline(src, tmp_path / "out", s2m.Options(skip_separation=True, bpm=118.0, review=False,
                                                              suggest=False, dlc_project=False,
                                                              sections_from=str(gp)))
    data = json.loads((tmp_path / "out" / "bass.notes.json").read_text())
    assert data["engine"] == "pyin+sections" and data["sections_from"]["file"] == "examples.gp5"
    chorus = [n for n in data["notes"] if 18.0 < n["t"] < 25.5]
    assert chorus and all(n["s"] == 1 for n in chorus)              # strings as in the tab
    xml = (tmp_path / "out" / "bass_RS2.xml").read_text(encoding="utf-8")
    assert 'name="verse"' in xml and 'name="chorus"' in xml
    assert any("examples.gp5: Examples placed over" in w for w in res.warnings)


def test_cli_checks_the_file(tmp_path):
    import stems2midi as s2m
    song = tmp_path / "a.wav"
    song.write_bytes(b"")
    with pytest.raises(SystemExit):
        s2m.main([str(song), "--sections-from", str(tmp_path / "x.musicxml")])
    with pytest.raises(SystemExit):
        s2m.main([str(song), "--sections-from", str(tmp_path / "missing.gp5")])
