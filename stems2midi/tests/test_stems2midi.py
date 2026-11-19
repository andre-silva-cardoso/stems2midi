"""Tests for the stems2midi plugin.

Run from the repo root:  pytest plugins/stems2midi/tests -v
Pure-logic tests need only numpy; transcription tests skip without librosa,
route tests skip without fastapi/httpx. Nothing here downloads Demucs weights.
"""

import json
import xml.etree.ElementTree as ET
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))

import stems2midi as s2m  # noqa: E402


# ── note segmentation ───────────────────────────────────────────────────────

HOP = 0.01


def _track(spec, n):
    """spec: list of (start_frame, end_frame, midi). Returns midi_f/active/rms."""
    midi_f = np.full(n, np.nan)
    for a, b, m in spec:
        midi_f[a:b] = m + np.random.default_rng(a).normal(0, 0.08, b - a)
    active = np.isfinite(midi_f)
    rms_db = np.where(active, -6.0, -80.0)
    return midi_f, active, rms_db


def test_segment_simple_line():
    spec = [(10, 40, 40), (45, 70, 43), (75, 120, 28)]
    midi_f, _active, rms = _track(spec, 130)
    notes = s2m.segment_by_onsets(midi_f, rms, [10, 45, 75], HOP)
    assert [n.pitch for n in notes] == [40, 43, 28]
    assert abs(notes[0].start - 0.10) < 1e-6 and abs(notes[2].end - 1.20) < 0.04


def test_repeated_notes_split_on_onsets():
    # one continuous voiced run at E1, but three attacks
    midi_f, _active, rms = _track([(0, 90, 28)], 100)
    notes = s2m.segment_by_onsets(midi_f, rms, [0, 30, 60], HOP)
    assert [n.pitch for n in notes] == [28, 28, 28]
    assert [round(n.start, 2) for n in notes] == [0.0, 0.3, 0.6]


def test_legato_pitch_change_without_attack_is_a_new_note():
    # hammer-on: 40 → 42 with no new onset
    midi_f, _active, rms = _track([(0, 30, 40), (30, 60, 42)], 70)
    notes = s2m.segment_by_onsets(midi_f, rms, [0], HOP)
    assert [n.pitch for n in notes] == [40, 42]
    assert abs(notes[1].start - 0.30) < 0.03


def test_dying_string_drift_and_slide_tail_are_ignored():
    midi_f, _active, rms = _track([(0, 15, 40)], 60)
    midi_f[15:40] = np.linspace(40, 37.2, 25)   # pitch sags as the note decays
    rms[15:40] = -12
    notes = s2m.segment_by_onsets(midi_f, rms, [0], HOP)
    assert [n.pitch for n in notes] == [40]


def test_unpitched_attack_is_skipped():
    midi_f, _active, rms = _track([(0, 30, 33)], 80)
    rms[50:60] = -6  # loud, but no pitch: a dead/slap noise
    notes = s2m.segment_by_onsets(midi_f, rms, [0, 50], HOP)
    assert [n.pitch for n in notes] == [33]


def test_tuning_offset_is_removed():
    midi_f, _active, rms = _track([(0, 200, 40)], 200)
    midi_f = midi_f + 0.42          # a recording 42 cents sharp
    assert abs(s2m.estimate_tuning(midi_f, rms) - 0.42) < 0.05
    notes = s2m.segment_by_onsets(midi_f, rms, [0], HOP)
    assert [n.pitch for n in notes] == [40]


def test_pick_onsets_ignores_small_wiggles():
    env = np.zeros(200) + 0.05
    env[20], env[50], env[80], env[83] = 1.0, 0.12, 0.8, 0.7   # 83: too close to 80
    rms = np.full(200, -6.0)
    assert s2m.pick_onsets(env, rms, HOP, rel=0.3) == [20, 80]
    rms[120:] = -80.0
    env[150] = 1.0
    assert 150 not in s2m.pick_onsets(env, rms, HOP, rel=0.3)   # silence → not an attack


def test_make_monophonic_prefers_lowest():
    N = s2m.Note
    notes = [N(0.0, 0.5, 40), N(0.01, 0.5, 52), N(0.4, 0.9, 43)]
    out = s2m.make_monophonic(notes)
    assert [n.pitch for n in out] == [40, 43]
    assert out[0].end <= out[1].start + 1e-9


def test_quantize_fits_grid_phase():
    # 16ths at 120 BPM, starting 40 ms after a beat tracker's (late) grid
    period = 0.5
    beats = np.arange(0.04, 8, period)
    truth = [0.0 + k * 0.125 for k in range(0, 40, 2)]
    jitter = np.random.default_rng(3).normal(0, 0.012, len(truth))
    notes = [s2m.Note(t + j, t + 0.1 + j, 40) for t, j in zip(truth, jitter)]
    q = s2m.quantize_notes(notes, beats, 120.0, 16)
    assert len(q) == len(notes)
    # every note lands in its own 16th slot (a 40 ms-late grid alone would
    # push some of them a slot off), and all share one grid
    assert max(abs(n.start - t) for n, t in zip(q, truth)) < 0.02
    offs = [n.start - t for n, t in zip(q, truth)]
    assert max(offs) - min(offs) < 0.005


def test_estimate_tempo_folds_double_time(monkeypatch):
    librosa = pytest.importorskip("librosa")
    monkeypatch.setattr(librosa.beat, "beat_track",
                        lambda **kw: (np.array([200.0]), np.arange(0, 10, 0.3)))
    bpm, beats = s2m.estimate_tempo(np.zeros(10), 22050)
    assert bpm == 100.0 and np.allclose(np.diff(beats), 0.6)


def test_write_midi_roundtrip(tmp_path):
    mido = pytest.importorskip("mido")
    notes = [s2m.Note(0.0, 0.5, 40, 100), s2m.Note(0.5, 1.0, 40, 90), s2m.Note(1.0, 1.25, 47, 80)]
    path = s2m.write_midi(notes, tmp_path / "b.mid", bpm=120)
    mid = mido.MidiFile(str(path))
    track = mid.tracks[1]
    assert any(m.type == "program_change" and m.program == 33 for m in track)
    ons = [m for m in track if m.type == "note_on"]
    assert [m.note for m in ons] == [40, 40, 47]
    # same-tick note_off of the 1st note comes before the 2nd note_on
    seq = [(m.type, m.note) for m in track if m.type in ("note_on", "note_off")]
    assert seq[:3] == [("note_on", 40), ("note_off", 40), ("note_on", 40)]
    assert abs(mid.length - 1.25) < 0.01


# ── transcription on synthetic audio ────────────────────────────────────────

def _pluck_line(sr=44100, bpm=110, pat=None):
    rng = np.random.default_rng(0)
    beat = 60 / bpm
    pat = pat or [(28, .5), (28, .5), (40, .5), (31, .5), (33, 1), (35, .5), (36, .5), (38, 1), (43, .5), (45, .5)]
    t, truth = 0.3, []
    y = np.zeros(int((0.3 + sum(d for _, d in pat) * beat + 1) * sr))
    for m, d in pat:
        f = 440 * 2 ** ((m - 69) / 12)
        dur = d * beat * 0.9
        tt = np.arange(int(dur * sr)) / sr
        env = np.exp(-tt * 4) * np.minimum(1, tt / 0.004) * np.minimum(1, (dur - tt) / 0.01)
        w = sum(np.sin(2 * np.pi * f * k * tt + rng.uniform(0, 6)) / k ** 1.3 for k in range(1, 9))
        i = int(t * sr)
        y[i:i + tt.size] += 0.4 * env * w
        truth.append((t, m))
        t += d * beat
    return y, sr, truth


def test_transcribe_pyin_synthetic_bass():
    pytest.importorskip("librosa")
    y, sr, truth = _pluck_line()
    notes = s2m.transcribe_pyin(y, sr)
    hits = sum(1 for t, m in truth if any(abs(n.start - t) < 0.05 and n.pitch == m for n in notes))
    assert hits / len(truth) >= 0.9, [(round(n.start, 2), n.pitch) for n in notes]
    assert len(notes) <= len(truth) + 2


def test_pipeline_skip_separation(tmp_path):
    pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("mido")
    y, sr, truth = _pluck_line()
    src = tmp_path / "bass.wav"
    sf.write(str(src), y, sr)
    events = []
    res = s2m.run_pipeline(src, tmp_path / "out",
                           s2m.Options(skip_separation=True, bpm=110, quantize=16),
                           progress=lambda f, st, m: events.append(f))
    assert res.midi == {"bass": "bass.mid"} and res.bpm == 110
    data = json.loads((tmp_path / "out" / "bass.notes.json").read_text())
    assert len(data["notes"]) >= len(truth) - 1
    # string/fret tab for the default 4-string tuning
    assert res.tabs == {"bass": "bass.tab.txt"} and data["tuning"]["strings"] == [28, 33, 38, 43]
    # … and the Rocksmith XML next to it (needs FeedBack's lib/gp2rs.py)
    if not any(w.startswith("Rocksmith XML not written") for w in res.warnings):
        assert res.rsxml == {"bass": "bass_RS2.xml"} and (tmp_path / "out" / "bass_RS2.xml").is_file()
    assert all({"s", "f"} <= set(n) for n in data["notes"])
    assert all(data["tuning"]["strings"][n["s"]] + n["f"] == n["m"] for n in data["notes"])
    assert (tmp_path / "out" / "bass.tab.txt").read_text().startswith("Tuning:")
    assert events[-1] == 1.0 and events == sorted(events)


def test_cli_check_json():
    import subprocess
    r = subprocess.run([sys.executable, str(PLUGIN_DIR / "stems2midi.py"), "--check", "--json-progress"],
                       capture_output=True, text=True, timeout=120)
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["event"] == "check" and "can_separate" in info


# ── routes ──────────────────────────────────────────────────────────────────

def test_parse_options_sanitises():
    import routes
    o = routes.parse_options({"model": "evil; rm -rf", "stems": "bass,../x,guitar", "midi": "none",
                              "quantize": "7", "bpm": "9999", "device": "cuda", "backing": "false"})
    assert o["model"] == "htdemucs_6s" and o["stems"] == ["bass", "guitar"]
    assert o["midi"] == [] and o["quantize"] == 0 and o["bpm"] is None
    assert o["device"] == "cuda" and o["backing"] is False
    cmd = routes.build_command("python", Path("in.mp3"), Path("out"), o)
    assert "--no-backing" in cmd and cmd[cmd.index("--midi") + 1] == "none"


def test_tuning_option():
    import routes
    assert routes.parse_options({})["tuning"] == "auto"
    o = routes.parse_options({"tuning": "-2,0,0,0"})
    cmd = routes.build_command("python", Path("in.mp3"), Path("out"), o)
    assert "--tuning=-2,0,0,0" in cmd
    assert routes.parse_options({"tuning": "none"})["tuning"] == "none"
    assert routes.parse_options({"tuning": "B0 E1 A1 D2 G2"})["tuning"] == "B0 E1 A1 D2 G2"
    for bad in ("E1 A1; rm -rf /", "Q1 A1 D2 G2", "--model x"):
        with pytest.raises(ValueError):
            routes.parse_options({"tuning": bad})


def test_lyrics_option():
    import routes
    o = routes.parse_options({"lyrics": "Linha um\r\nlinha <b>dois</b>\x07", "language": "PT"})
    assert o["lyrics"] == "Linha um\nlinha bdois/b" and o["language"] == "pt"
    assert routes.parse_options({"lyrics": "   ", "language": "xx"})["lyrics"] == ""
    assert routes.parse_options({"language": "--model"})["language"] == ""
    o["lyrics_file"] = "/jobs/x/lyrics.input.txt"
    cmd = routes.build_command("python", Path("a.mp3"), Path("o"), o)
    assert "--lyrics=/jobs/x/lyrics.input.txt" in cmd and "--language=pt" in cmd


def test_notes_from_option():
    import routes
    o = routes.parse_options({})
    o["notes_from"] = "/jobs/x/notes_from.musicxml"
    assert "--notes-from=/jobs/x/notes_from.musicxml" in routes.build_command("python", Path("a.mp3"), Path("o"), o)
    o = routes.parse_options({"notes_track": "2"})
    o["notes_from"] = "/jobs/x/notes_from.gp5"
    cmd = routes.build_command("python", Path("a.mp3"), Path("o"), o)
    assert "--notes-from=/jobs/x/notes_from.gp5" in cmd and "--track=2" in cmd
    assert ".gp5" in routes.NOTES_EXTS


def test_guitar_pro_notes_keep_strings_and_legato(tmp_path):
    """A Guitar Pro tab lined up with a (synthetic) stem keeps the tab's strings,
    slap and hammer-ons."""
    gp = pytest.importorskip("guitarpro")
    pytest.importorskip("librosa")
    song = gp.Song()
    song.tempo = 120
    tr = song.tracks[0]
    tr.name, tr.channel.instrument = "Bass", 33
    tr.strings = [gp.GuitarString(i + 1, v) for i, v in enumerate((43, 38, 33, 28))]
    for _ in range(5):
        song.newMeasure()
    # A1 on the A string open (s1 f0) vs E string fret 5: the tab says fret 5 on E
    pattern = [(5, 4), (7, 4), (5, 3), (3, 3)]
    for k, m in enumerate(tr.measures):
        v = m.voices[0]
        for j, (fret, st) in enumerate(pattern):
            b = gp.Beat(v, duration=gp.Duration(value=4))
            b.status = gp.BeatStatus.normal
            n = gp.Note(b, value=fret, string=st, type=gp.NoteType.normal)
            if j == 1:
                n.effect.hammer = True
            b.notes.append(n)
            v.beats.append(b)
    p = tmp_path / "t.gp5"
    gp.write(song, str(p))
    sr = 22050
    y = np.zeros(int(sr * 16), np.float32)
    opens = {4: 28, 3: 33}
    for bar in range(6):
        for j, (fret, st) in enumerate(pattern):
            t0 = 2.0 + bar * 2.0 + j * 0.5
            f = 440 * 2 ** ((opens[st] + fret - 69) / 12)
            tt = np.arange(int(sr * 0.45)) / sr
            tone = sum(np.sin(2 * np.pi * f * h * tt) / h for h in (1, 2, 3)) * np.exp(-tt * 5)
            y[int(t0 * sr):int(t0 * sr) + len(tone)] += 0.4 * tone.astype(np.float32)
    notes, info = s2m.notes_from_file(p, y, sr)
    assert info["gp"]["tuning"] == [28, 33, 38, 43] and abs(info["offset"] - 2.0) < 0.05
    assert len(notes) == 24
    assert [n.string for n in notes[:4]] == [0, 0, 1, 1]
    assert [n.legato for n in notes[:4]] == [False, True, False, False]
    fr = s2m._fretting().assign_frets(notes, info["gp"]["tuning"],
                                      allowed=lambda src: {src.string} if src.string is not None else None)
    assert [(n.string, n.fret) for n in fr.notes[:4]] == [(0, 5), (0, 7), (1, 5), (1, 3)]


def test_cli_lyrics_on_a_result_folder(tmp_path):
    """stems2midi.py <folder> --lyrics: only the lyrics (no bass needed)."""
    sf = pytest.importorskip("soundfile")
    sr = 16000
    y = np.zeros(sr * 12, np.float32)
    tt = np.arange(int(sr * 0.3)) / sr
    t = 1.0
    for n in (3, 3):
        for k in range(n):
            s = int(t * sr)
            y[s:s + len(tt)] += 0.3 * np.sin(2 * np.pi * 220 * 2 ** (k / 12) * tt)
            t += 0.36
        t += 2.0
    sf.write(tmp_path / "vocals.wav", y, sr)
    (tmp_path / "letra.txt").write_text("um dois três\nmar sol céu\n", encoding="utf-8")
    rc = s2m.main([str(tmp_path), "--lyrics", str(tmp_path / "letra.txt"), "--lyrics-aligner", "simple",
                   "--shift", "5", "--title", "T"])
    assert rc == 0
    us = (tmp_path / "vocals.ultrastar.txt").read_text(encoding="utf-8")
    assert "#MP3:guitar.ogg" in us and "#TITLE:T" in us and "#LANGUAGE:Portuguese" in us
    assert (tmp_path / "vocals_RS2.xml").is_file()
    root = ET.parse(tmp_path / "vocals_RS2.xml").getroot()
    times = [float(v.get("time")) for v in root.iter("vocal")]
    assert len(times) == 6 and abs(times[0] - 6.0) < 0.1 and abs(times[3] - (6.0 + 3 * 0.36 + 2.0)) < 0.15


def test_slap_option():
    import routes
    assert routes.parse_options({})["slap"] is None
    o = routes.parse_options({"slap": "0:32-1:05, 130-160"})
    assert "--slap=0:32-1:05, 130-160" in routes.build_command("python", Path("a.mp3"), Path("o"), o)
    for bad in ("0:32-1:05; rm -rf /", "1:00-0:10"):
        with pytest.raises(ValueError):
            routes.parse_options({"slap": bad})
    assert routes.parse_options({"slap": " Auto "})["slap"] == "auto"


def test_pipeline_slap_auto(tmp_path):
    pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    y, sr, _truth = _pluck_line()
    src = tmp_path / "bass.wav"
    sf.write(str(src), y, sr)
    res = s2m.run_pipeline(src, tmp_path / "o", s2m.Options(skip_separation=True, bpm=110, slap="auto"))
    assert any("guessed automatically" in w for w in res.warnings)
    data = json.loads((tmp_path / "o" / "bass.notes.json").read_text())
    if "slap_sections" in data:          # whatever it guessed must be a valid section list
        assert data["slap_auto"] is True
        assert s2m._fretting().parse_sections(data["slap_sections"])


def test_pipeline_slap_all(tmp_path):
    pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    y, sr, _truth = _pluck_line()
    src = tmp_path / "bass.wav"
    sf.write(str(src), y, sr)
    s2m.run_pipeline(src, tmp_path / "o", s2m.Options(skip_separation=True, bpm=110, slap="all"))
    data = json.loads((tmp_path / "o" / "bass.notes.json").read_text())
    techs = {n.get("tech") for n in data["notes"]}
    assert "slap" in techs and data["tab_stats"]["slap"] > 0
    for n in data["notes"]:
        if n.get("tech"):
            assert n["tech"] == ("slap" if n["m"] <= 40 else "pop")


def test_cli_accepts_negative_offsets():
    args = s2m.build_parser().parse_args(["x.wav", "--tuning=-2,0,0,0"])
    assert args.tuning == "-2,0,0,0"


def test_safe_filename_and_python_path(tmp_path):
    import routes
    assert routes.safe_filename("../../etc/pa ss<wd>.MP3") == "pa ss_wd_.mp3"
    assert routes.valid_python_path("")
    fake = tmp_path / "notpython.exe"
    fake.write_text("x")
    assert not routes.valid_python_path(str(fake))
    assert routes.valid_python_path(sys.executable)


FAKE_CLI = textwrap.dedent('''
    import json, sys, time, pathlib
    if "--check" in sys.argv:
        print(json.dumps({"event": "check", "can_separate": True, "can_transcribe": True})); sys.exit()
    out = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
    out.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        print(json.dumps({"event": "progress", "pct": i / 3, "stage": "separate", "msg": "x"}), flush=True)
        time.sleep(0.05)
    (out / "bass.wav").write_bytes(b"RIFF....")
    (out / "bass.mid").write_bytes(b"MThd")
    (out / "bass.notes.json").write_text('{"notes": []}')
    print(json.dumps({"event": "done", "out_dir": str(out), "stems": {"bass": "bass.wav"},
                      "midi": {"bass": "bass.mid"}, "notes": {"bass": "bass.notes.json"},
                      "bpm": 100, "device": "cpu", "model": "htdemucs_6s", "elapsed": 0.1, "warnings": []}))
''')


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import logging
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import routes
    fake = tmp_path / "fake_cli.py"
    fake.write_text(FAKE_CLI)
    monkeypatch.setattr(routes, "CLI", fake)
    app = FastAPI()
    routes.setup(app, {"config_dir": tmp_path / "cfg", "log": logging.getLogger("test")})
    return TestClient(app)


def _wait(client, jid, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/plugins/stems2midi/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_job_lifecycle(client):
    base = "/api/plugins/stems2midi"
    r = client.post(f"{base}/jobs", files={"file": ("My Song.mp3", b"ID3fake", "audio/mpeg")},
                    data={"stems": "bass", "midi": "bass"})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    assert "input_path" not in r.json()
    job = _wait(client, jid)
    assert job["status"] == "done", job
    assert job["result"]["midi"] == {"bass": "bass.mid"}

    f = client.get(f"{base}/jobs/{jid}/files/bass.mid?download=1")
    assert f.status_code == 200 and f.content == b"MThd"
    assert "My%20Song%20-%20bass.mid" in f.headers["content-disposition"]
    assert client.get(f"{base}/jobs/{jid}/files/..%2F..%2Fjob.json").status_code in (400, 404)
    assert client.get(f"{base}/jobs/{jid}/files/job.json").status_code == 404
    assert client.get(f"{base}/jobs/{jid}/rocksmith.xml").status_code == 422   # no tab in the fake output
    z = client.get(f"{base}/jobs/{jid}/zip")
    assert z.status_code == 200 and z.content[:2] == b"PK"

    assert [j["id"] for j in client.get(f"{base}/jobs").json()["jobs"]] == [jid]
    assert client.delete(f"{base}/jobs/{jid}").status_code == 200
    assert client.get(f"{base}/jobs/{jid}").status_code == 404


def test_rejects_non_audio_and_bad_ids(client):
    base = "/api/plugins/stems2midi"
    r = client.post(f"{base}/jobs", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
    assert client.get(f"{base}/jobs/../../etc").status_code == 404
    assert client.post(f"{base}/settings", json={"python": "/bin/sh"}).status_code == 400
    r = client.post(f"{base}/jobs", files={"file": ("a.mp3", b"ID3", "audio/mpeg")}, data={"tuning": "X9"})
    assert r.status_code == 400 and "tuning" in r.json()["error"].lower()


def test_rocksmith_xml_route(client):
    import rsexport
    try:
        rsexport.load_gp2rs()
    except RuntimeError as e:
        pytest.skip(str(e))
    base = "/api/plugins/stems2midi"
    r = client.post(f"{base}/jobs", files={"file": ("Delfins - Canção.mp3", b"ID3fake", "audio/mpeg")},
                    data={"stems": "bass", "midi": "bass"})
    jid = r.json()["id"]
    assert _wait(client, jid)["status"] == "done"
    out = client.app.state.stems2midi.root / jid / "out"
    notes = [{"t": 1 + i * 0.5, "d": 0.4, "m": 33 + (i % 2) * 7, "v": 90, "s": 1, "f": (i % 2) * 7} for i in range(16)]
    (out / "bass.notes.json").write_text(json.dumps({
        "version": 2, "bpm": 120, "duration": 10, "beats": [1 + i * 0.5 for i in range(16)],
        "tuning": {"strings": [28, 33, 38, 43], "offsets": [0, 0, 0, 0]},
        "anchors": [{"t": 1, "f": 5, "w": 4}], "notes": notes}))
    x = client.get(f"{base}/jobs/{jid}/rocksmith.xml", params={"artist": "Delfins", "title": "Canção <x>", "year": "19x6"})
    assert x.status_code == 200, x.text
    assert "attachment" in x.headers["content-disposition"]
    root = ET.fromstring(x.content)
    assert root.findtext("artistName") == "Delfins" and root.findtext("title") == "Cancao x"
    assert root.find("albumYear") is None and len(root.findall(".//note")) == 16
    assert b"<albumName></albumName>" in x.content


def test_pipeline_auto_tuning_picks_five_string_for_low_b(tmp_path):
    pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    pat = [(23, .5), (23, .5), (26, .5), (28, .5)] * 4 + [(33, .5), (35, .5)]   # B0 D1 E1 riff
    y, sr, _truth = _pluck_line(pat=pat)
    src = tmp_path / "bass.wav"
    sf.write(str(src), y, sr)
    res = s2m.run_pipeline(src, tmp_path / "o", s2m.Options(skip_separation=True, bpm=110))
    data = json.loads((tmp_path / "o" / "bass.notes.json").read_text())
    assert data["tuning"]["strings"] == [23, 28, 33, 38, 43], res.warnings
    assert any("Tuning (auto): B standard (5)" in w for w in res.warnings)
    # with a fixed 4-string tuning it only suggests
    res4 = s2m.run_pipeline(src, tmp_path / "o4", s2m.Options(skip_separation=True, bpm=110, tuning="4"))
    assert any("5-string in B standard" in w for w in res4.warnings)


def test_meta_from_filename_and_command():
    import routes
    assert routes.meta_from_filename("Delfins - Canção de Engate.mp3") == {"artist": "Delfins", "title": "Canção de Engate"}
    assert routes.meta_from_filename("x<y>.mp3") == {"title": "xy"}
    o = routes.parse_options({"album": "O Caminho", "year": "1996"})
    o["meta"]["artist"] = "-rm"
    cmd = routes.build_command("python", Path("a.mp3"), Path("o"), o)
    assert "--album=O Caminho" in cmd and "--year=1996" in cmd and "--artist=-rm" in cmd


def test_cli_folder_mode_writes_only_the_xml(tmp_path, capsys):
    import rsexport
    try:
        rsexport.load_gp2rs()
    except RuntimeError as e:
        pytest.skip(str(e))
    notes = [{"t": 1 + i * 0.5, "d": 0.4, "m": 33, "v": 90, "s": 1, "f": 0} for i in range(12)]
    (tmp_path / "bass.notes.json").write_text(json.dumps({
        "bpm": 120, "duration": 8, "beats": [], "source": "Band - Song.mp3",
        "tuning": {"strings": [28, 33, 38, 43], "offsets": [0, 0, 0, 0]}, "anchors": [], "notes": notes}))
    assert s2m.main([str(tmp_path), "--album", "LP"]) == 0
    root = ET.parse(tmp_path / "bass_RS2.xml").getroot()
    assert root.findtext("artistName") == "Band" and root.findtext("albumName") == "LP"


def test_find_offset_lead_in_silence():
    pytest.importorskip("librosa")
    y, sr, _ = _pluck_line(sr=11025, pat=[(28, .5), (33, .25), (35, .25), (40, 1), (31, .5), (36, .5)] * 6)
    lead = 4.6
    target = np.concatenate([np.zeros(int(lead * sr)), y]) + 0.001 * np.random.default_rng(1).standard_normal(len(y) + int(lead * sr))
    off, conf = s2m.find_offset(y, target, sr)
    assert abs(off - lead) < 0.02 and conf > 1


def test_rocksmith_xml_post_with_shift(client):
    import rsexport
    try:
        rsexport.load_gp2rs()
    except RuntimeError as e:
        pytest.skip(str(e))
    base = "/api/plugins/stems2midi"
    jid = client.post(f"{base}/jobs", files={"file": ("Band - Song.mp3", b"ID3fake", "audio/mpeg")},
                      data={"stems": "bass", "midi": "bass"}).json()["id"]
    assert _wait(client, jid)["status"] == "done"
    out = client.app.state.stems2midi.root / jid / "out"
    notes = [{"t": 1 + i * 0.5, "d": 0.4, "m": 33, "v": 90, "s": 1, "f": 0} for i in range(16)]
    (out / "bass.notes.json").write_text(json.dumps({
        "bpm": 120, "duration": 10, "beats": [1 + i * 0.5 for i in range(16)],
        "tuning": {"strings": [28, 33, 38, 43], "offsets": [0, 0, 0, 0]}, "anchors": [], "notes": notes}))
    r = client.post(f"{base}/jobs/{jid}/rocksmith.xml", data={"shift": "5", "album": "LP"})
    assert r.status_code == 200, r.text
    root = ET.fromstring(r.content)
    assert root.findtext("artistName") == "Band" and root.findtext("albumName") == "LP"
    assert abs(float(root.find(".//note").get("time")) - 5.985) < 0.002
    assert r.headers["x-stems2midi-shift"] == "+5.000"
    assert client.post(f"{base}/jobs/{jid}/rocksmith.xml", data={"shift": "99"}).status_code == 400


def test_merge_weak_repeats_keeps_real_replucks():
    hop = 0.01
    env = np.zeros(400)
    for t, h in ((0.0, 10), (1.0, 0.5), (2.0, 9), (3.0, 10)):   # 1.0 s: a wobble, not a pluck
        env[int(t / hop)] = h
    env[1::7] += 0.2                                              # small background peaks
    N = s2m.Note
    notes = [N(0.0, 1.0, 40), N(1.0, 2.0, 40), N(2.0, 3.0, 40), N(3.0, 3.5, 43)]
    out = s2m.merge_weak_repeats(notes, env, hop, 0.25)
    assert [(n.start, n.end, n.pitch) for n in out] == [(0.0, 2.0, 40), (2.0, 3.0, 40), (3.0, 3.5, 43)]
    assert s2m.merge_weak_repeats(notes, env, hop, 0) == notes


def test_parts_and_slap_names_options():
    import routes
    o = routes.parse_options({"parts": "verse=0:21-0:42, chorus=0:42-1:05", "slap": "Verse, chorus"})
    assert o["parts"] == "verse=0:21-0:42, chorus=0:42-1:05" and o["slap"] == "verse,chorus"
    cmd = routes.build_command("python", Path("a.mp3"), Path("o"), o)
    assert "--parts=verse=0:21-0:42, chorus=0:42-1:05" in cmd and "--slap=verse,chorus" in cmd
    for bad in ({"parts": "refrao=0:10-0:20"}, {"parts": "verse=0:20"}, {"slap": "refrao"}):
        with pytest.raises(ValueError):
            routes.parse_options(bad)
    assert routes.parse_options({"parts": " AUTO "})["parts"] == "auto"
    assert s2m.slap_part_names("verse, Chorus") == ["verse", "chorus"]
    assert s2m.slap_part_names("all") == [] and s2m.slap_part_names("0:10-0:20") == []


def test_cli_slap_names_need_parts(capsys):
    with pytest.raises(SystemExit):
        s2m.main(["x.wav", "--slap", "verse"])
    with pytest.raises(SystemExit):
        s2m.main(["x.wav", "--parts", "verse=0:10-0:20", "--slap", "chorus"])
    assert "not in --parts" in capsys.readouterr().err


def test_pipeline_parts_and_slap_by_name(tmp_path):
    pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    import rsexport
    try:
        rsexport.load_gp2rs()
    except RuntimeError as e:
        pytest.skip(str(e))
    riff = [(28, .5), (28, .5), (40, .5), (31, .5), (33, .5), (35, .5), (36, .5), (38, .5)]   # 1 bar at 120
    other = [(43, 1), (45, 1), (47, 1), (45, 1)]                                              # 1 bar
    y, sr, _ = _pluck_line(bpm=120, pat=riff + other + riff + other)
    src = tmp_path / "Band - Song.wav"
    sf.write(str(src), y, sr)
    res = s2m.run_pipeline(src, tmp_path / "o", s2m.Options(
        skip_separation=True, bpm=120, parts="verse=0:00.3-0:02.3", slap="verse", rs_shift=5.0))
    assert any(w.startswith("Song parts:") for w in res.warnings), res.warnings
    data = json.loads((tmp_path / "o" / "bass.notes.json").read_text())
    assert data["parts"] == "verse=0:00.3-0:02.3"
    techs = [n.get("tech") for n in data["notes"]]
    assert techs.count("slap") + techs.count("pop") >= 8          # both verses slapped …
    late = [n for n in data["notes"] if 2.4 < n["t"] < 4.2 or 6.4 < n["t"] < 8.2]
    assert not any(n.get("tech") for n in late)                    # … the part between isn't
    r = ET.parse(tmp_path / "o" / "bass_RS2.xml").getroot()
    secs = [(s.get("name"), float(s.get("startTime"))) for s in r.iter("section")]
    assert [n for n, _t in secs].count("verse") == 2 and all(t >= 5 for _n, t in secs)


def test_merge_notes_and_bar_spec():
    N = s2m.Note
    auto = [N(start=i * 0.5, end=i * 0.5 + 0.4, pitch=40) for i in range(16)]       # 2 s bars at 120 BPM
    other = [N(start=i * 0.25, end=i * 0.25 + 0.2, pitch=28) for i in range(32)]
    bars = [(k + 1, k * 2.0, k * 2.0 + 2.0) for k in range(4)]
    r = s2m.parse_bar_spec("2-3", bars)
    assert r == [(2.0, 6.0)]
    assert s2m.parse_bar_spec("0:02-0:04", bars, shift=1.0) == [(2.0, 4.0)]      # song (MP3) time
    with pytest.raises(ValueError):
        s2m.parse_bar_spec("9-12", bars)
    m = s2m.merge_notes(auto, other, r)
    inside = [n for n in m if 2.0 <= n.start < 6.0]
    assert inside and all(n.pitch == 28 for n in inside)
    assert all(n.pitch == 40 for n in m if n.start < 1.9 or n.start >= 6.0)
    assert s2m._compact_ranges([3, 4, 5, 9, 11, 12]) == "3-5, 9, 11-12"


def test_notes_use_option():
    import routes
    o = routes.parse_options({"notes_use": "12-30, 1:02-1:30"})
    o["notes_from"] = "/j/n.musicxml"
    cmd = routes.build_command("python", Path("a.mp3"), Path("o"), o)
    assert "--use=12-30, 1:02-1:30" in cmd
    assert routes.parse_options({"notes_use": "doubtful"})["notes_use"] == "doubtful"
    assert routes.parse_options({})["notes_use"] == "all"
    with pytest.raises(ValueError):
        routes.parse_options({"notes_use": "rm -rf /"})


def test_only_option():
    import routes
    o = routes.parse_options({"only": "2:50-5:15"})
    assert "--only=2:50-5:15" in routes.build_command("python", Path("a.mp3"), Path("o"), o)
    assert routes.parse_options({})["only"] == ""
    with pytest.raises(ValueError):
        routes.parse_options({"only": "; rm"})


def test_double_triggers_and_sustained_repeats():
    N = s2m.Note
    notes = [N(1.0, 1.08, 31), N(1.1, 1.6, 31), N(1.6, 2.0, 31), N(2.5, 2.6, 33), N(2.62, 3.0, 31)]
    m = s2m.merge_double_triggers(notes)
    assert [(round(n.start, 2), round(n.end, 2), n.pitch) for n in m][:2] == [(1.0, 1.6, 31), (1.6, 2.0, 31)]
    assert len(m) == 4                      # different pitches are kept
    hop = 0.01
    env = np.zeros(400)
    for f in (100, 160, 250):
        env[f] = 1.0                        # every boundary has a strong onset …
    rise = np.full(400, -10.0)              # … but the level never rises
    kept = s2m.merge_weak_repeats([N(1.0, 1.6, 31), N(1.6, 2.4, 31)], env, hop, rise_curve=rise, min_rise_db=4.0)
    assert len(kept) == 1 and kept[0].end == 2.4
    rise[160:165] = 0.0                     # a real re-pluck: +10 dB
    kept = s2m.merge_weak_repeats([N(1.0, 1.6, 31), N(1.6, 2.4, 31)], env, hop, rise_curve=rise, min_rise_db=4.0)
    assert len(kept) == 2
