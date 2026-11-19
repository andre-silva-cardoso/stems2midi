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
    p = tmp_path / "a.txt"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        ni.read_notes(p)


# ── Guitar Pro ───────────────────────────────────────────────────────────

def _gp_song(gp, tracks, bars=4, tempo=120):
    """tracks: [(name, program, strings high→low, [(fret, gp string)] per bar)]."""
    song = gp.Song()
    song.tempo = tempo
    for _ in range(len(tracks) - 1):
        t = gp.Track(song)
        t.measures = [gp.Measure(t, h) for h in song.measureHeaders]
        song.tracks.append(t)
    for _ in range(bars - 1):
        song.newMeasure()
    for tr, (name, program, strings, pattern) in zip(song.tracks, tracks):
        tr.name = name
        tr.channel.instrument = program
        tr.strings = [gp.GuitarString(i + 1, v) for i, v in enumerate(strings)]
        for m in tr.measures:
            v = m.voices[0]
            for fret, st in pattern:
                b = gp.Beat(v, duration=gp.Duration(value=4))
                b.status = gp.BeatStatus.normal
                b.notes.append(gp.Note(b, value=fret, string=st, type=gp.NoteType.normal))
                v.beats.append(b)
    return song


BASS = ("Baixo", 33, (43, 38, 33, 28), ((3, 4), (5, 3), (0, 2), (2, 1)))
GUITAR = ("Guitar", 29, (64, 59, 55, 50, 45, 40), ((0, 6), (2, 5), (2, 4), (1, 3)))


def test_guitar_pro_bass_track_strings_and_frets(tmp_path):
    gp = pytest.importorskip("guitarpro")
    p = tmp_path / "song.gp5"
    gp.write(_gp_song(gp, [GUITAR, BASS]), str(p))
    notes, info = ni.read_gp(p)
    assert info["track"] == "Baixo" and info["track_no"] == 2 and info["tuning"] == [28, 33, 38, 43]
    assert len(notes) == 16
    assert [(m, x["s"], x["f"]) for _t, _d, m, x in notes[:4]] == [(31, 0, 3), (38, 1, 5), (38, 2, 0), (45, 3, 2)]
    assert np.allclose([t for t, *_x in notes[:4]], [0.0, 0.5, 1.0, 1.5])         # quarters at 120
    assert ni.read_gp(p, "guitar")[1]["track"] == "Guitar"
    assert ni.read_gp(p, "1")[1]["track_no"] == 1
    with pytest.raises(ValueError):
        ni.read_gp(p, "9")


def test_guitar_pro_bass_on_a_guitar_tuned_track(tmp_path):
    gp = pytest.importorskip("guitarpro")
    p = tmp_path / "song.gp4"
    bass_up = ("Baixo", 33, (64, 59, 55, 50, 45, 40), ((3, 6), (5, 5)))
    gp.write(_gp_song(gp, [bass_up], bars=2), str(p))
    notes, info = ni.read_gp(p)
    assert info["octave"] == -12 and info["tuning"] is None
    assert [m for _t, _d, m, _x in notes[:2]] == [31, 38]
    assert "s" not in notes[0][3]                    # not a bass tuning: fingering left to the tool


def test_not_a_guitar_pro_file(tmp_path):
    pytest.importorskip("guitarpro")
    p = tmp_path / "a.gp5"
    p.write_bytes(b"not a tab")
    with pytest.raises(ValueError):
        ni.read_notes(p)


def test_tab_written_at_double_tempo():
    rng = np.random.default_rng(3)
    song_t = np.cumsum(rng.choice([0.25, 0.5, 0.75], 300)) + 6.0
    file_t = (song_t - 6.0) / 2.0                    # the tab: twice as fast, starts at 0
    assert ni.ratio_centers(file_t, song_t) == [1.0, 2.0]
    notes = [(float(t), 0.1, 30) for t in file_t]
    out, info = ni.line_up(notes, song_t)
    assert abs(info["ratio"] - 2.0) < 0.01 and "double" in info["tempo_note"]
    assert np.median(np.abs(np.array([t for t, *_x in out]) - song_t)) < 0.01


def test_segment_refit_follows_a_jump():
    rng = np.random.default_rng(5)
    t = np.cumsum(rng.choice([0.25, 0.5, 0.75], 400)) + 2.0
    song = t + np.where(t > t[200], 0.13, 0.0)       # the band holds a beat 130 ms longer halfway
    notes = [(float(x), 0.1, 30) for x in t]
    out, info = ni.line_up(notes, song)
    got = np.array([x for x, *_r in out])
    assert np.mean(np.abs(got - song) < 0.02) > 0.95
    assert info["src"] == list(range(len(t)))


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


GPIF = """<GPIF><Score><Title>T</Title><Artist>A</Artist></Score>
<Tracks><Track id="0"><Name>Bass</Name><Property name="Tuning"><Pitches>28 33 38 43</Pitches></Property></Track></Tracks>
<MasterBars><MasterBar><Time>4/4</Time><Bars>0</Bars>{repeat}</MasterBar></MasterBars>
<Bars><Bar id="0"><Voices>0</Voices></Bar></Bars>
<Voices><Voice id="0"><Beats>0 1</Beats></Voice></Voices>
<Beats><Beat id="0"><Rhythm ref="r0"/><Notes>0</Notes></Beat><Beat id="1"><Rhythm ref="r0"/><Notes>1</Notes></Beat></Beats>
<Notes><Note id="0"><Property name="String"><String>0</String></Property><Property name="Fret"><Fret>3</Fret></Property></Note>
<Note id="1"><Property name="String"><String>1</String></Property><Property name="Fret"><Fret>2</Fret></Property></Note></Notes>
<Rhythms><Rhythm id="r0"><NoteValue>Quarter</NoteValue></Rhythm></Rhythms></GPIF>"""


def test_guitar_pro_6_to_8(monkeypatch):
    """.gpx / .gp through FeedBack's gp2rs_gpx (when it's there)."""
    import xml.etree.ElementTree as ET
    pytest.importorskip("guitarpro")
    try:
        gx = ni._gp2rs_gpx()
    except ValueError:
        pytest.skip("FeedBack's lib/gp2rs_gpx.py not found")
    monkeypatch.setattr(gx, "_load_gpif", lambda _p: ET.fromstring(GPIF.format(repeat="")))
    notes, info = ni.read_gp("song.gp")
    assert [(m, x["s"], x["f"]) for _t, _d, m, x in notes] == [(31, 0, 3), (35, 1, 2)]
    assert info["tuning"] == [28, 33, 38, 43] and not info["repeats_once"]
    monkeypatch.setattr(gx, "_load_gpif", lambda _p: ET.fromstring(GPIF.format(repeat='<Repeat start="true" end="true" count="2"/>')))
    assert ni.read_gp("song.gpx")[1]["repeats_once"]


def test_low_strings_track_is_not_taken_for_the_bass():
    tracks = [{"index": 0, "name": "Lead Guitar", "is_bass": False, "notes": 80},
              {"index": 1, "name": "Strings 2 - Tremolo Strings", "is_bass": True, "notes": 60}]
    with pytest.raises(ValueError, match="no bass track"):
        ni.pick_track(tracks)
    assert ni.pick_track(tracks, "2")["index"] == 1
    tracks.append({"index": 2, "name": "Fretless", "is_bass": True, "notes": 300})
    assert ni.pick_track(tracks)["index"] == 2


def test_tempo_changing_against_the_file_follows_the_notes():
    """The file keeps one tempo, the band changes it mid-song (no single ratio
    fits): the notes are lined up with a quick transcription of the stem."""
    rng = np.random.default_rng(7)
    file_t = np.cumsum(rng.choice([0.25, 0.5], 500))
    line = rng.integers(28, 45, file_t.size)
    half = file_t[250]
    song_t = np.where(file_t < half, 3.0 + file_t, 3.0 + half + 0.7 * (file_t - half))   # 30 % faster later
    onsets = np.sort(np.r_[song_t, rng.uniform(3, song_t.max(), 60)])

    def pitch_score(times, pitches):
        k = np.clip(np.searchsorted(song_t, times - 0.04), 0, len(song_t) - 1)
        return float(np.mean((line[k] == pitches) & (np.abs(song_t[k] - times) < 0.05)))
    # the "quick transcription": right times, a few wrong / missing notes
    keep = rng.random(song_t.size) > 0.1
    ref = [(float(t), int(m) if rng.random() > 0.1 else int(m) + 5) for t, m, k in zip(song_t, line, keep) if k]
    notes = [(float(t), 0.2, int(m)) for t, m in zip(file_t, line)]
    out, info = ni.line_up(notes, onsets, pitch_score=pitch_score, ref_notes=lambda: ref)
    assert info["method"] == "notes"
    got = np.array([t for t, *_x in out])
    assert np.mean(np.abs(got - song_t) < 0.03) > 0.9
