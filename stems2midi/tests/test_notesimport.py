"""Notes from another transcriber, lined up with the stem."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import notesimport as ni  # noqa: E402

MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Baixo</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>2</divisions>
<transpose><chromatic>0</chromatic><octave-change>-1</octave-change></transpose></attributes>
<direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>120</per-minute></metronome></direction-type></direction>
{notes}</measure></part></score-partwise>"""


def _note(step, octave, dur=1):
    return (f"<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>"
            f"<duration>{dur}</duration><type>eighth</type></note>")


def test_musicxml_metronome_and_transposition(tmp_path):
    p = tmp_path / "a.musicxml"
    p.write_text(MUSICXML.format(notes=_note("E", 2) + _note("B", 2) + _note("D", 3, 2)), encoding="utf-8")
    notes = ni.read_notes(p)
    assert [m for _t, _d, m in notes] == [28, 35, 38]          # written an octave up
    assert [t for t, _d, _m in notes] == [0.0, 0.25, 0.5]       # 120 BPM, eighths


def test_midi_file(tmp_path):
    mido = pytest.importorskip("mido")
    mf = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mf.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000))
    for k, m in enumerate((28, 31, 33)):
        tr.append(mido.Message("note_on", note=m, velocity=90, time=0 if k == 0 else 240))
        tr.append(mido.Message("note_off", note=m, velocity=0, time=240))
    p = tmp_path / "a.mid"
    mf.save(str(p))
    notes = ni.read_notes(p)
    assert [m for *_x, m in notes] == [28, 31, 33]
    assert np.allclose([t for t, *_x in notes], [0.0, 0.5, 1.0])


def test_line_up_finds_tempo_drift_and_offset():
    rng = np.random.default_rng(1)
    file_t = np.arange(0, 200, 0.5)                              # eighths at 120 BPM in the file
    # the song: 1.5 % faster, 4.5 s later, plus a slow wander of ±60 ms
    song_t = 0.985 * file_t + 4.5 + 0.06 * np.sin(file_t / 25)
    onsets = np.sort(np.r_[song_t + rng.normal(0, 0.005, song_t.size),
                           rng.uniform(4, 200, 40)])            # plus some extra attacks
    line = rng.integers(28, 40, file_t.size)
    notes = [(float(t), 0.4, int(m)) for t, m in zip(file_t, line)]

    def pitch_score(times, pitches):     # the "stem" plays the line on song_t
        k = np.clip(np.searchsorted(song_t, times - 0.05), 0, len(song_t) - 1)
        return float(np.mean((line[k] == pitches) & (np.abs(song_t[k] - times) < 0.1)))
    out, info = ni.line_up(notes, onsets, pitch_score=pitch_score)
    assert abs(info["ratio"] - 0.985) < 0.002 and abs(info["offset"] - 4.5) < 0.15
    got = np.array([t for t, _d, _m in out])
    assert np.median(np.abs(got - song_t)) < 0.01 and info["share_on_attack"] > 0.95
    assert [m for *_x, m in out] == [int(m) for m in line]
    assert all(b[0] >= a[0] + a[1] - 1e-6 for a, b in zip(out, out[1:]))   # no overlaps


def test_bad_extension(tmp_path):
    p = tmp_path / "a.gp5"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        ni.read_notes(p)


def test_notes_end_where_the_stem_goes_quiet():
    pytest.importorskip("librosa")
    sr = 11025
    y = np.zeros(sr * 3, np.float32)
    t = np.arange(int(0.3 * sr)) / sr
    y[sr // 2: sr // 2 + t.size] = 0.5 * np.sin(2 * np.pi * 82.4 * t)    # sounds 0.5–0.8 s
    notes = [(0.5, 1.2, 40)]                                             # written to 1.7 s
    out, cut = ni.trim_to_sound(notes, y, sr)
    assert cut == 1 and 0.25 < out[0][1] < 0.4
    keep, cut2 = ni.trim_to_sound([(0.5, 0.25, 40)], y, sr)
    assert cut2 == 0 and keep[0][1] == 0.25
