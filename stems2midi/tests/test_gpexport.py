"""The transcription as Guitar Pro 5 (gpexport.py) and the BS-RoFormer separator."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))
pytest.importorskip("guitarpro")

import gpexport as ge  # noqa: E402
import rsexport as rx  # noqa: E402


def _payload(bpm=100.0, bars=12, lead=1.0):
    beat = 60 / bpm
    beats = [lead + i * beat for i in range(bars * 4)]
    notes = []
    for b in range(bars):
        for k, (q, f, s) in enumerate([(0, 3, 0), (1, 5, 0), (2, 0, 1), (3.5, 2, 1)]):
            n = {"t": lead + (b * 4 + q) * beat, "d": beat * 0.4, "m": [28, 33][s] + f, "s": s, "f": f, "v": 90}
            if b == 2 and k == 3:
                n["lg"] = 1
            if b == 3:
                n["tech"] = "slap"
            notes.append(n)
    return {"bpm": bpm, "beats": beats, "duration": lead + bars * 4 * beat + 1, "notes": notes,
            "tuning": {"strings": [28, 33, 38, 43]}, "source": "Band - Song.mp3"}


def test_split_lengths():
    assert ge._split(16) == [16] and ge._split(5) == [4, 1] and ge._split(7) == [6, 1]


def test_whole_song_round_trip(tmp_path):
    import guitarpro as gp
    import notesimport as ni
    pl = _payload()
    shift = 5.0
    bm = rx.beat_map_for(pl, shift=shift)
    info = ge.export(tmp_path / "bass.gp5", pl, bm, shift=shift - rx.ADVANCE_S, markers={4: "verse 1 (A)"})
    song = gp.parse(str(tmp_path / "bass.gp5"))
    assert len(song.measureHeaders) == info["bars"] and song.measureHeaders[3].marker.title == "verse 1 (A)"
    assert [s.value for s in song.tracks[0].strings] == [43, 38, 33, 28]
    raw, rinfo = ni.read_gp(tmp_path / "bass.gp5")
    t1 = [t for t, m in bm if m == 1][0]
    want = sorted(n["t"] + shift - rx.ADVANCE_S for n in pl["notes"])
    got = sorted(r[0] + t1 for r in raw)
    assert len(got) == len(want) and max(abs(a - b) for a, b in zip(got, want)) < 0.04
    frets = [(r[3]["s"], r[3]["f"]) for r in raw[:4]]
    assert frets == [(0, 3), (0, 5), (1, 0), (1, 2)]
    assert sum(1 for r in raw if r[3].get("tech") == "slap") == 4
    assert sum(1 for r in raw if r[3].get("lg")) == 1
    assert rinfo["sections"][0][0] == "verse1(a)"


def test_examples_file_has_only_those_bars(tmp_path):
    import notesimport as ni
    pl = _payload()
    bm = rx.beat_map_for(pl)
    ge.export(tmp_path / "ex.gp5", pl, bm, markers={2: "verse", 6: "chorus"}, bars=[2, 3, 6, 7])
    raw, info = ni.read_gp(tmp_path / "ex.gp5")
    assert len(raw) == 16 and [n for n, _t in info["sections"]] == ["verse", "chorus"]


def test_pipeline_writes_gp_files(tmp_path):
    import stems2midi as s2m
    pl = _payload()
    folder = tmp_path / "Band - Song_stems"
    folder.mkdir()
    pl["review"] = [{"bar": i + 1, "t": 1.0 + i * 2.4, "p": 0.1, "level": "ok"} for i in range(12)]
    pl["parts_auto"] = [{"letter": "A", "name": "verse", "number": 1, "b0": 0, "b1": 6, "bars": 6,
                         "start": 1.0, "end": 15.4},
                        {"letter": "B", "name": "chorus", "number": 1, "b0": 6, "b1": 12, "bars": 6,
                         "start": 15.4, "end": 29.8}]
    files, warns = s2m.write_gp_files(folder, pl, 0.0)
    assert files == {"song": "bass.gp5", "examples": "bass.examples.gp5"}, warns
    assert (folder / "bass.gp5").is_file() and (folder / "bass.examples.gp5").is_file()


def test_roformer_separator_through_audio_separator(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    import stems2midi as s2m
    calls = {}

    class Separator:
        def __init__(self, **kw):
            calls["init"] = kw

        def load_model(self, model_filename):
            calls["model"] = model_filename

        def separate(self, path, names):
            out = Path(calls["init"]["output_dir"])
            for stem in ("Bass", "Drums", "Other", "Vocals", "Guitar", "Piano"):
                sf.write(str(out / f"{names[stem]}.wav"), np.zeros((4410, 2), np.float32), 44100)
            return [f"{names[s]}.wav" for s in ("Bass", "Drums", "Other", "Vocals", "Guitar", "Piano")]

    mod = types.ModuleType("audio_separator.separator")
    mod.Separator = Separator
    monkeypatch.setitem(sys.modules, "audio_separator", types.ModuleType("audio_separator"))
    monkeypatch.setitem(sys.modules, "audio_separator.separator", mod)
    monkeypatch.setenv("STEMS2MIDI_MODELS", str(tmp_path / "models"))
    sf.write(str(tmp_path / "song.wav"), np.zeros((4410, 2), np.float32), 44100)
    stems, sr, dev = s2m.separate(tmp_path / "song.wav", "roformer", device="cpu")
    assert calls["init"]["use_soundfile"] is True           # no ffmpeg needed
    assert calls["model"] == "BS-Roformer-SW.ckpt" and calls["init"]["model_file_dir"] == str(tmp_path / "models")
    assert sorted(stems) == sorted(s2m.ALL_STEMS) and stems["bass"].shape == (2, 4410) and sr == 44100


def test_review_hint_suggests_roformer():
    pytest.importorskip("librosa")
    import suggest as sg
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.05, 11025 * 30).astype(np.float32)
    notes = [{"t": i * 0.5, "d": 0.4, "m": 40} for i in range(50)]
    review = [{"level": "bad", "p": 0.9, "relative": True}] * 10
    opts = [h["option"] for h in sg.suggest(y, 11025, notes, 120.0, review, used={"model": "htdemucs_6s"})]
    assert "--model" in opts and "--notes-from" in opts
    opts = [h["option"] for h in sg.suggest(y, 11025, notes, 120.0, review, used={"model": "roformer"})]
    assert "--model" not in opts


def test_edited_gp_file_is_not_overwritten(tmp_path):
    import stems2midi as s2m
    pl = _payload()
    folder = tmp_path / "Band - Song_stems"
    folder.mkdir()
    files, _w = s2m.write_gp_files(folder, pl, 0.0)
    assert files == {"song": "bass.gp5"}
    files, _w = s2m.write_gp_files(folder, pl, 0.0)          # ours: rewritten in place
    assert files == {"song": "bass.gp5"}
    (folder / "bass.gp5").write_bytes((folder / "bass.gp5").read_bytes() + b"\0")   # "edited in Guitar Pro"
    files, warns = s2m.write_gp_files(folder, pl, 0.0)
    assert files == {"song": "bass (new).gp5"} and any("your edits" in w for w in warns)
