"""Tests for rsexport.py — transcription → Rocksmith 2014 bass XML."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rsexport as rx  # noqa: E402


def _gp2rs_or_skip():
    try:
        return rx.load_gp2rs()
    except RuntimeError as e:
        pytest.skip(str(e))


def _payload(n_bars=6, bpm=120.0, jitter=0.0, tech=None):
    """A bass line: root on beat 1 (long, low), passing notes on 2-4."""
    per = 60 / bpm
    t0 = 1.0
    rng = np.random.default_rng(1)
    beats = [round(t0 + i * per + (rng.normal(0, jitter) if jitter else 0), 4) for i in range(n_bars * 4)]
    notes = []
    for i in range(n_bars * 4):
        t = t0 + i * per
        down = i % 4 == 0
        notes.append({"t": round(t, 4), "d": round(per * (0.9 if down else 0.4), 4),
                      "m": 28 if down else 40, "v": 110 if down else 80,
                      "s": 0 if down else 2, "f": 0 if down else 2})
    notes[5]["lg"] = 1                      # D2 fret 2 → hammer-on? (prev note is on another string)
    notes.insert(6, {"t": notes[5]["t"] + 0.2, "d": 0.1, "m": 42, "v": 70, "s": 2, "f": 4, "lg": 1})
    if tech:
        for n in notes[:8]:
            n["tech"] = "slap" if n["s"] < 2 else "pop"
    return {"version": 2, "bpm": bpm, "duration": t0 + n_bars * 4 * per + 2, "beats": beats,
            "tuning": {"strings": [28, 33, 38, 43], "offsets": [0, 0, 0, 0]},
            "anchors": [{"t": 1.0, "f": 1, "w": 4}], "notes": notes}


def test_regularize_fills_gaps_and_drops_doubles():
    b = np.array([0, .5, 1.0, 1.02, 1.5, 2.5, 3.0])
    r = rx._regularize(b)
    assert np.allclose(np.diff(r), 0.5, atol=0.03)
    assert len(r) == 7


def test_beat_map_downbeats_and_coverage():
    p = _payload(jitter=0.01)
    bm = rx.make_beat_map(p["beats"], p["bpm"], p["notes"], p["duration"])
    times = np.array([t for t, _ in bm])
    assert times[0] >= 0 and times[-1] >= p["duration"] - 0.5
    assert np.allclose(np.diff(times), 0.5, atol=0.02)          # smoothed: no wobble
    downs = [t for t, m in bm if m > 0]
    assert any(abs(d - 1.0) < 0.02 for d in downs)              # bar starts on the low root
    assert [m for _, m in bm if m > 0] == list(range(1, len(downs) + 1))


def test_beat_map_constant_without_tracked_beats():
    p = _payload()
    bm = rx.make_beat_map([], 120.0, p["notes"], p["duration"])
    assert np.allclose(np.diff([t for t, _ in bm]), 0.5, atol=1e-6)


def test_export_xml_structure(tmp_path):
    _gp2rs_or_skip()
    p = _payload(tech=True)
    xml_text, warns = rx.export_xml(p, title="Canção", artist="Delfins", album="X", year="1996")
    root = ET.fromstring(xml_text.encode("utf-8"))
    assert root.tag == "song" and root.get("version") == "7"
    assert root.findtext("title") == "Canção" and root.findtext("albumYear") == "1996"
    assert float(root.findtext("offset")) == pytest.approx(-float(root.findtext("startBeat")), abs=1e-3)
    tun = root.find("tuning")
    assert tun.get("stringCount") == "4" and tun.get("string0") == "0"
    notes = root.findall("levels/level/notes/note")
    assert len(notes) == len(p["notes"])
    assert sum(n.get("slap") == "1" for n in notes) and sum(n.get("pluck") == "1" for n in notes)
    assert notes[6].get("hammerOn") == "1"                       # same string, higher fret, legato
    assert notes[5].get("hammerOn") == "0"                       # legato but came from another string
    assert notes[0].get("sustain") != "0.000" and notes[1].get("sustain") == "0.000"
    phrases = [ph.get("name") for ph in root.findall("phrases/phrase")]
    assert phrases[0] == "COUNT" and phrases[-1] == "END" and "riff1" in phrases
    its = root.findall("phraseIterations/phraseIteration")
    assert [float(i.get("time")) for i in its] == sorted(float(i.get("time")) for i in its)
    assert root.find("sections/section[@name='riff']") is not None
    props = root.find("arrangementProperties")
    assert props.get("pathBass") == "1" and props.get("slapPop") == "1" and props.get("hopo") == "1"
    assert root.findall("levels/level/anchors/anchor")
    assert not warns


def test_export_reads_back_in_feedback(tmp_path):
    gp2rs = _gp2rs_or_skip()
    lib = [d for d in rx._lib_dirs() if (d / "song.py").is_file()]
    if lib and str(lib[0]) not in sys.path:
        sys.path.insert(0, str(lib[0]))
    try:
        import song
    except ImportError:
        pytest.skip("FeedBack lib/song.py not importable")
    x = tmp_path / "arr.xml"
    x.write_text(rx.export_xml(_payload(tech=True), title="t")[0], encoding="utf-8")
    arr = song.parse_arrangement(str(x))
    assert len(arr.notes) == 25 and arr.path_bass
    assert any(n.slap for n in arr.notes) and any(n.pluck for n in arr.notes)
    assert gp2rs  # used


def test_export_needs_tab():
    _gp2rs_or_skip()
    with pytest.raises(ValueError):
        rx.export_xml({"bpm": 120, "notes": [{"t": 0, "d": 1, "m": 40, "v": 90}]})


def test_five_string_becomes_four_string_b_standard():
    _gp2rs_or_skip()
    p = _payload()
    p["tuning"] = {"strings": [23, 28, 33, 38, 43], "offsets": [0, 0, 0, 0, 0]}
    p["notes"][1].update(s=4, f=2)                 # A2 on the G string …
    p["notes"][2].update(s=0, f=0)                 # … and a low B
    xml, warns = rx.export_xml(p)
    assert any("B standard" in w and "+7 +7 +7 +7" in w and "-1200" in w for w in warns)
    root = ET.fromstring(xml.encode("utf-8"))
    tun = root.find("tuning")
    assert tun.get("stringCount") in (None, "4") or int(tun.get("stringCount")) == 4
    assert [int(tun.get(f"string{i}")) for i in range(4)] == [7, 7, 7, 7]      # as in 5-string CDLC
    assert root.findtext("centOffset") == "-1200"
    notes = root.findall("./levels/level/notes/note")
    assert max(int(n.get("string")) for n in notes) <= 3
    t1 = round(p["notes"][1]["t"] - rx.ADVANCE_S, 3)
    moved = [n for n in notes if abs(float(n.get("time")) - t1) < 0.002]
    assert moved and moved[0].get("string") == "3" and moved[0].get("fret") == "7"   # D string, fret 7 = A2
    assert all(int(a.get("fret")) >= 1 for a in root.iter("anchor"))


def test_cli(tmp_path):
    _gp2rs_or_skip()
    src = tmp_path / "bass.notes.json"
    src.write_text(json.dumps(_payload()), encoding="utf-8")
    assert rx.main([str(src), "--artist", "A", "--title", "B"]) == 0
    out = tmp_path / rx.XML_NAME
    assert ET.parse(out).getroot().findtext("artistName") == "A"


def test_works_without_numpy_and_guitarpro(tmp_path):
    """The export must run in any Python: FeedBack's (no numpy?) or the
    engine's (no PyGuitarPro)."""
    import subprocess
    _gp2rs_or_skip()
    src = tmp_path / "bass.notes.json"
    src.write_text(json.dumps(_payload(tech=True)), encoding="utf-8")
    code = (
        "import sys; sys.modules['numpy'] = None; sys.modules['guitarpro'] = None; "
        f"sys.path.insert(0, {str(Path(rx.__file__).parent)!r}); "
        "import rsexport; sys.modules.pop('guitarpro'); sys.modules['guitarpro'] = None; "
        f"print(rsexport.write_xml({str(tmp_path)!r})[0])"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / rx.XML_NAME).is_file()


def test_write_xml_folder_and_meta_from_file_name(tmp_path):
    _gp2rs_or_skip()
    p = _payload()
    p["source"] = "Delfins - Canção de Engate.mp3"
    (tmp_path / "bass.notes.json").write_text(json.dumps(p), encoding="utf-8")
    out, _ = rx.write_xml(tmp_path, year="1996")
    root = ET.parse(out).getroot()
    assert out.name == "bass_RS2.xml"
    assert root.findtext("artistName") == "Delfins" and root.findtext("title") == "Cancao de Engate"   # plain letters for EOF / Builder
    assert root.findtext("albumYear") == "1996"
    out2, _ = rx.write_xml(tmp_path / "bass.notes.json", tmp_path / "x.xml", title="Outro")
    assert ET.parse(out2).getroot().findtext("title") == "Outro"


def test_guess_meta():
    assert rx.guess_meta("A - B - C.mp3") == {"artist": "A", "title": "B - C"}
    assert rx.guess_meta("song.wav") == {"artist": "", "title": "song"}


def _eof_ms(text):
    """EOF's parse_xml_rs_timestamp(): digits after the point are read as
    milliseconds, padded to 3 but never cut."""
    whole, _, frac = text.partition(".")
    return int(whole or 0) * 1000 + (int(frac.ljust(3, "0")) if frac else 0)


def test_times_read_by_eof_like_rocksmith():
    """EOF reads 0.314906 as 314.906 s: every time must have ≤ 3 decimals,
    and once read the EOF way no note may sit before the first beat."""
    _gp2rs_or_skip()
    p = _payload(jitter=0.01)
    p["beats"] = [b + 0.000123 for b in p["beats"]]
    root = ET.fromstring(rx.export_xml(p, shift=4.6)[0].encode())
    times = [e.get("time") for e in root.iter("ebeat")] + [root.findtext("startBeat")]
    times += [n.get("time") for n in root.iter("note")] + [n.get("sustain") for n in root.iter("note")]
    assert all(len(t.partition(".")[2]) <= 3 for t in times), [t for t in times if len(t.partition(".")[2]) > 3][:3]
    beats = [_eof_ms(e.get("time")) for e in root.iter("ebeat")]
    assert beats == sorted(beats)
    assert min(_eof_ms(n.get("time")) for n in root.iter("note")) >= beats[0]


def test_shift_moves_notes_and_beats():
    _gp2rs_or_skip()
    p = _payload()
    r0 = ET.fromstring(rx.export_xml(p)[0].encode())
    r1 = ET.fromstring(rx.export_xml(p, shift=4.6)[0].encode())
    n0 = [float(n.get("time")) for n in r0.iter("note")]
    n1 = [float(n.get("time")) for n in r1.iter("note")]
    assert all(abs(b - a - 4.6) < 0.002 for a, b in zip(n0, n1))
    d0 = [float(e.get("time")) for e in r0.iter("ebeat") if e.get("measure") != "-1"]
    d1 = [float(e.get("time")) for e in r1.iter("ebeat") if e.get("measure") != "-1"]
    assert any(abs(t - (d0[0] + 4.6)) < 0.01 for t in d1)          # same bar lines, later
    assert float(r1.findtext("songLength")) >= float(r0.findtext("songLength")) + 4.5


def test_metadata_without_accents_for_eof_and_builder():
    _gp2rs_or_skip()
    pl = _payload()
    xml, _w = rx.render(pl, title="Canção de Engate", artist="Xutos & Pontapés", album="Ao Vivo — Lisboa")
    assert "<title>Cancao de Engate</title>" in xml
    assert "<artistName>Xutos &amp; Pontapes</artistName>" in xml
    assert "<albumName>Ao Vivo - Lisboa</albumName>" in xml
    xml2, _w = rx.render(pl, title="Canção", keep_accents=True)
    assert "<title>Canção</title>" in xml2


def test_five_string_part_also_as_rhythm(tmp_path):
    _gp2rs_or_skip()
    p = _payload(n_bars=8)
    p["tuning"] = {"strings": [23, 28, 33, 38, 43], "offsets": [0, 0, 0, 0, 0]}
    for n in p["notes"][::2]:
        n.update(s=4, f=2, tech="pop")                  # many notes on the G string
    assert rx.wants_rhythm5(p) and not rx.wants_rhythm5(p, "bass")
    (tmp_path / "bass.notes.json").write_text(json.dumps(p), encoding="utf-8")
    out, warns = rx.write_xml(tmp_path, title="T", artist="A")
    assert any(rx.RHYTHM_XML in w for w in warns)
    r = ET.parse(str(tmp_path / rx.RHYTHM_XML)).getroot()
    assert r.findtext("arrangement") == "Rhythm" and r.findtext("centOffset") == "-1200"
    t = r.find("tuning")
    assert [int(t.get(f"string{i}")) for i in range(6)] == [-5, -5, -5, -5, -4, -9]
    props = r.find("arrangementProperties")
    assert props.get("pathRhythm") == "1" and props.get("pathBass") == "0" and props.get("bonusArr") == "1"
    assert r.findtext("tonebase") == "bass"
    notes = r.findall("./levels/level/notes/note")
    assert {n.get("string") for n in notes} >= {"4"} and not any(n.get("slap") or n.get("pluck") for n in notes)
    b = ET.parse(str(out)).getroot()                    # the bass XML stays 4-string
    assert max(int(n.get("string")) for n in b.iter("note")) <= 3
    few = _payload()
    few["tuning"] = {"strings": [23, 28, 33, 38, 43], "offsets": [0, 0, 0, 0, 0]}
    few["notes"][0].update(s=4, f=1)
    assert not rx.wants_rhythm5(few) and rx.wants_rhythm5(few, "rhythm")


def test_names_without_what_stem_separators_add():
    import rsexport as rx
    assert rx.guess_meta("Delfins - Bandeira-bass-E minor-130bpm-442hz v2.musicxml") == \
        {"artist": "Delfins", "title": "Bandeira"}
    assert rx.guess_meta("Delfins - Bandeira-bass-E-minor-130bpm.mp3")["title"] == "Bandeira"
    assert rx.clean_name("Delfins - 1 lugar ao sol-bass-G minor-110bpm-442hz") == "Delfins - 1 lugar ao sol"
    assert rx.clean_name("Queen - Who Wants to Live Forever (Bass)") == "Queen - Who Wants to Live Forever"
    assert rx.guess_meta("Jay-Z - 99 Problems.mp3") == {"artist": "Jay-Z", "title": "99 Problems"}
    assert rx.guess_meta("a-ha - Take On Me.wav") == {"artist": "a-ha", "title": "Take On Me"}
    assert rx.guess_meta("05 - Sétima Legião - Auto de Fé - Noutro Lugar.mp3") == \
        {"artist": "Sétima Legião", "album": "Auto de Fé", "title": "Noutro Lugar"}
    assert rx.guess_meta("Band - Song - Live 1986.mp3") == {"artist": "Band", "title": "Song - Live 1986"}
