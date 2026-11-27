"""Notes from another transcriber (AnthemScore, Guitar Pro, MuseScore…) lined
up with the song — ``stems2midi.py … --notes-from FILE``.

Some recordings are hard for the built-in pitch tracker: when the bass has
almost no fundamental (thin 80s/90s mixes) it locks on a harmonic or hears
no note at all. A neural transcriber such as AnthemScore copes better, but
its MusicXML / MIDI is written on a fixed tempo grid that slowly drifts away
from the real song. This module takes those notes (pitches as they are) and
puts them on the song's own time:

1. a global fit — tempo ratio and offset — maximising the notes that land on
   an attack of the bass stem (±50 ms);
   (bass lines repeat, so a fit one or two beats off scores almost as well:
   the candidates are told apart by how well the stem sounds the pitches);
2. a local warp — the median distance to the nearest attack, smoothed over
   ±8 s — that follows the band's tempo drift;
3. each note start is snapped to the attack it sits on (±60 ms), so the
   chart starts notes where the bass actually plays.

The rest of the tool then works as usual: beat map, strings and frets,
slap/pop, song parts, phrases, Rocksmith XML, review.

Accepted files: MusicXML (``.musicxml``, ``.xml``, ``.mxl``), MIDI
(``.mid``, ``.midi``) and Guitar Pro (``.gp3`` ``.gp4`` ``.gp5``, and ``.gpx``
``.gp`` when FeedBack's ``gp2rs_gpx`` is there). For MusicXML the first part
is read; for MIDI every note (the bass usually is the only track); for Guitar
Pro the bass track (or ``track=``), played as performed (repeats, voltas,
D.S./D.C. — FeedBack's ``gp2rs`` reader), with its strings, frets, tuning,
slap/pop and hammer-ons/pull-offs/slides, which the tool keeps.

Hand-made tabs are often written at another tempo than the recording
(double: 187 BPM for a 96 BPM song); the fit also tries half / double / 3:2
when the file's length says so.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GP_EXTS = {".gp3", ".gp4", ".gp5", ".gpx", ".gp"}
EXTS = {".musicxml", ".xml", ".mxl", ".mid", ".midi"} | GP_EXTS


# ── reading ──────────────────────────────────────────────────────────────

def _benchmark():
    import importlib.util
    import sys
    mod = sys.modules.get("plugin_stems2midi_benchmark")
    if mod is None:
        spec = importlib.util.spec_from_file_location("plugin_stems2midi_benchmark", HERE / "benchmark.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plugin_stems2midi_benchmark"] = mod
        spec.loader.exec_module(mod)
    return mod


def _mxl_xml(path: Path) -> bytes:
    """The score inside a compressed MusicXML (.mxl)."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(z.read("META-INF/container.xml"))
            full = next(e.get("full-path") for e in root.iter() if e.tag.endswith("rootfile"))
            return z.read(full)
        except Exception:  # noqa: BLE001
            cand = [n for n in names if n.lower().endswith((".xml", ".musicxml")) and not n.startswith("META-INF")]
            if not cand:
                raise ValueError(f"no score inside {path.name}")
            return z.read(cand[0])


def read_midi(path: Path) -> list[tuple[float, float, int]]:
    import mido
    mf = mido.MidiFile(str(path))
    t, on, out = 0.0, {}, []
    for msg in mf:                      # iterating a MidiFile gives delta times in seconds
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on.setdefault((msg.channel, msg.note), []).append(t)
        elif msg.type in ("note_off", "note_on"):
            starts = on.get((msg.channel, msg.note))
            if starts:
                s = starts.pop(0)
                out.append((s, max(0.02, t - s), int(msg.note)))
    return sorted(out)


# ── Guitar Pro (FeedBack's gp2rs reader) ────────────────────────────────

def _gp2rs():
    """FeedBack's gp2rs with the real PyGuitarPro (rsexport may have loaded it
    with a stand-in, enough for writing XML but not for reading .gp files)."""
    try:
        import guitarpro
    except ImportError as e:
        raise ValueError("reading Guitar Pro files needs PyGuitarPro: "
                         "python -m pip install pyguitarpro (ATUALIZAR.bat installs it)") from e
    import importlib.util
    import sys
    mod = sys.modules.get("plugin_stems2midi_rsexport")
    if mod is None:
        spec = importlib.util.spec_from_file_location("plugin_stems2midi_rsexport", HERE / "rsexport.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plugin_stems2midi_rsexport"] = mod
        spec.loader.exec_module(mod)
    gp2rs = mod.load_gp2rs()
    if getattr(gp2rs, "guitarpro", None) is not guitarpro:
        gp2rs.guitarpro = guitarpro
    return gp2rs, mod


def _gp2rs_gpx():
    """FeedBack's reader for Guitar Pro 6–8 (.gpx / .gp)."""
    import importlib
    import sys
    gp2rs, rsx = _gp2rs()
    try:
        return importlib.import_module("gp2rs_gpx")
    except ImportError:
        pass
    for d in rsx._lib_dirs():
        if (d / "gp2rs_gpx.py").is_file():
            if str(d) not in sys.path:
                sys.path.append(str(d))
            try:
                return importlib.import_module("gp2rs_gpx")
            except ImportError as e:
                raise ValueError(f"Guitar Pro 6–8 reader (gp2rs_gpx) could not be loaded: {e}") from e
    raise ValueError("Guitar Pro 6–8 files (.gpx/.gp) need FeedBack's lib/gp2rs_gpx.py — or save the "
                     "file as .gp5 in Guitar Pro / TuxGuitar")


def _is_bass_name(name: str) -> bool:
    low = (name or "").lower()
    return any(k in low for k in ("bass", "baixo", "bajo", "basse", "contrabaixo"))


def _parse_gp(path: Path):
    import guitarpro
    if not Path(path).is_file():
        raise ValueError(f"no file {path}")
    try:
        return guitarpro.parse(str(path))
    except Exception as e:  # noqa: BLE001 — PyGuitarPro raises many kinds
        raise ValueError(f"{Path(path).name}: not a Guitar Pro 3–5 file that can be read ({e}) — "
                         "open it in Guitar Pro / TuxGuitar and save it as .gp5") from e


_OTHER = ("string", "violin", "viola", "cello", "vocal", "voice", "choir", "guitar", "guitarra", "piano",
          "key", "synth", "pad", "organ", "horn", "brass", "trumpet", "sax", "flute", "harp", "drum",
          "percussion", "bateria", "voz", "teclado", "cordas")


def _is_other_name(name: str) -> bool:
    low = (name or "").lower()
    return any(k in low for k in _OTHER)


def gp_tracks(path: str | Path) -> list[dict]:
    """[{"index", "name", "strings", "is_bass", "notes"}] of a Guitar Pro file."""
    path = Path(path)
    if path.suffix.lower() in (".gpx", ".gp"):
        gx = _gp2rs_gpx()
        try:
            root = gx._load_gpif(str(path))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"{path.name}: not a Guitar Pro 6–8 file that can be read ({e})") from e
        out = []
        for t, info in zip(gx._gpif_tracks(root), gx.list_tracks(str(path))):
            out.append({"index": info["index"], "name": info["name"], "strings": info["strings"],
                        "is_bass": bool(info["is_bass"]) and not info["is_drums"], "notes": info["notes"],
                        "tuning": sorted(t.get("string_pitches") or [])})
        return out
    gp2rs, _ = _gp2rs()
    song = _parse_gp(path)
    out = []
    for i, t in enumerate(song.tracks):
        drums = bool(getattr(t, "isPercussionTrack", False))
        n = sum(len(b.notes) for m in t.measures for v in m.voices for b in v.beats)
        out.append({"index": i, "name": t.name.strip(), "strings": len(t.strings),
                    "is_bass": (not drums) and bool(t.strings) and gp2rs._is_bass_track(t), "notes": n,
                    "tuning": sorted(s.value for s in t.strings),
                    "capo": int(getattr(t, "offset", 0) or 0)})
    return out


def pick_track(tracks: list[dict], track: str | int | None = None) -> dict:
    """The bass track: ``track`` (number from 1 as Guitar Pro shows it, or part
    of the name) or the first bass track with notes (a bass name breaks ties)."""
    if track not in (None, ""):
        key = str(track).strip()
        if key.isdigit():
            k = int(key) - 1
            if 0 <= k < len(tracks):
                return tracks[k]
            raise ValueError(f"track {key}: the file has {len(tracks)} track(s)")
        hit = [t for t in tracks if key.lower() in t["name"].lower()]
        if not hit:
            raise ValueError(f"no track named like {key!r} (tracks: {', '.join(t['name'] for t in tracks)})")
        return hit[0]
    # a low tuning alone isn't enough: MIDI-made files put strings / synths on "bass" tunings
    bass = [t for t in tracks if t["is_bass"] and t["notes"]
            and (_is_bass_name(t["name"]) or not _is_other_name(t["name"]))]
    if not bass:
        bass = [t for t in tracks if _is_bass_name(t["name"]) and t["notes"]]
    if not bass:
        raise ValueError("no bass track found (tracks: " + ", ".join(
            f"{t['index'] + 1} {t['name']}" for t in tracks) + ") — choose one with --track")
    bass.sort(key=lambda t: (not _is_bass_name(t["name"]), t["index"]))
    return bass[0]


def _gp_slaps(song, track_index: int) -> dict:
    """{(measure number, 1/16 position in the bar, Guitar Pro string, fret): "slap"|"pop"}
    — FeedBack's GP3–5 reader doesn't carry slap/pop."""
    import guitarpro
    out = {}
    track = song.tracks[track_index]
    for m in track.measures:
        mstart = m.header.start
        for v in m.voices:
            for b in v.beats:
                fx = b.effect.slapEffect
                if fx not in (guitarpro.SlapEffect.slapping, guitarpro.SlapEffect.popping):
                    continue
                pos16 = int(round((b.start - mstart) / 240))
                for n in b.notes:
                    out[(m.header.number, pos16, n.string, n.value)] = \
                        "slap" if fx == guitarpro.SlapEffect.slapping else "pop"
    return out


def _slap_key(t: float, s: int, f: int, ebeats: list, n_strings: int):
    """The _gp_slaps key of a note at ``t`` (XML time) on Rocksmith string ``s``."""
    downs = [(i, bt, m) for i, (bt, m) in enumerate(ebeats) if m >= 0]
    cur = None
    for d in downs:
        if d[1] <= t + 0.02:
            cur = d
        else:
            break
    if cur is None:
        return None
    i0, t0, num = cur
    times = [bt for bt, _m in ebeats]
    idx = np.arange(len(times), dtype=float)
    pos = float(np.interp(t, times, idx)) - i0
    return (num, int(round(pos * 4)), n_strings - s, f)


def read_gp(path: str | Path, track: str | int | None = None) -> tuple[list[tuple], dict]:
    """Bass notes of a Guitar Pro file as performed (repeats expanded):
    [(start_s, dur_s, midi, {"s", "f", "tech"?, "lg"?, "dead"?})] in the file's time
    (dead notes have no pitch: kept for lining up, then left out), and
    {"track", "tracks", "tuning" (open strings low → high), "muted"}."""
    import tempfile
    import xml.etree.ElementTree as ET
    path = Path(path)
    tracks = gp_tracks(path)
    tr = pick_track(tracks, track)
    single_pass = False
    slaps = {}
    if path.suffix.lower() in (".gpx", ".gp"):
        gx = _gp2rs_gpx()
        mbars = list(gx._load_gpif(str(path)).find("MasterBars") or [])
        # FeedBack's GP6–8 reader plays repeats once: the file would be shorter than the song
        single_pass = any(mb.find("Repeat") is not None or mb.find("AlternateEndings") is not None
                          for mb in mbars)
        with tempfile.TemporaryDirectory() as tmp:
            files = gx.convert_file(str(path), tmp, [tr["index"]], 0.0, {tr["index"]: "Bass"})
            if not files:
                raise ValueError(f"{path.name}: track {tr['name']!r} gave no notes")
            xml = Path(files[0]).read_text(encoding="utf-8")
    else:
        gp2rs, _ = _gp2rs()
        gsong = _parse_gp(path)
        xml = gp2rs.convert_track(gsong, tr["index"], 0.0, "Bass")
        slaps = _gp_slaps(gsong, tr["index"])
    root = ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    opens = list(tr["tuning"])
    capo = int(tr.get("capo", 0))
    levels = list(root.iter("level"))
    if not levels or not opens:
        raise ValueError(f"no notes in {path.name} ({tr['name']})")
    lv = max(levels, key=lambda l: len(l.findall("notes/note")) + len(l.findall("chords/chord")))
    raw = []                       # (time, sustain, string, fret, element)
    for n in lv.findall("notes/note"):
        raw.append((float(n.get("time")), float(n.get("sustain") or 0), n))
    for c in lv.findall("chords/chord"):     # bass: keep the lowest string of a chord / double stop
        cn = c.findall("chordNote")
        if cn:
            low = min(cn, key=lambda x: int(x.get("string", 0)))
            raw.append((float(c.get("time")), float(low.get("sustain") or 0), low))
    raw.sort(key=lambda r: r[0])
    ebeats = [(float(e.get("time")), int(e.get("measure", -1))) for e in root.iter("ebeat")]
    out, muted, slide_to = [], 0, False
    for k, (t, sus, el) in enumerate(raw):
        s, f = int(el.get("string", 0)), int(el.get("fret", 0))
        if not 0 <= s < len(opens):
            continue
        if el.get("mute") == "1":              # dead note: no pitch, but its attack helps the fit
            muted += 1
            slide_to = False
            out.append((round(t, 4), 0.05, opens[s] + capo + f, {"dead": 1}))
            continue
        nxt = raw[k + 1][0] if k + 1 < len(raw) else t + max(sus, 0.25)
        d = sus if sus > 0.02 else max(0.05, min(nxt - t, 0.5))
        extra = {"s": s, "f": f + capo}
        if slaps and not el.get("slap") == "1" and not el.get("pluck") == "1":
            tech = slaps.get(_slap_key(t, s, f, ebeats, len(opens)))
            if tech:
                extra["tech"] = tech
        if el.get("slap") == "1":
            extra["tech"] = "slap"
        elif el.get("pluck") == "1":
            extra["tech"] = "pop"
        if el.get("hammerOn") == "1" or el.get("pullOff") == "1" or slide_to:
            extra["lg"] = 1
        slide_to = int(el.get("slideTo", -1) or -1) >= 0
        out.append((round(t, 4), round(d, 4), opens[s] + capo + f, extra))
    if not out:
        raise ValueError(f"no notes in {path.name} ({tr['name']})")
    # a bass tabbed on a guitar-tuned track (E2 A2 D3 G3 …) sounds an octave lower
    octave = -12 if min(opens) >= 36 and (tr["is_bass"] or _is_bass_name(tr["name"])) else 0
    if octave:
        out = [(t, d, m + octave, x) for t, d, m, x in out]
        opens = [o + octave for o in opens]
    # its strings/frets are kept only on a bass tuning (4–6 strings, top ≤ C3)
    bass_tuning = 4 <= len(opens) <= 6 and max(opens) <= 48 and opens == sorted(set(opens))
    if not bass_tuning:
        out = [(t, d, m, {k: v for k, v in x.items() if k not in ("s", "f")}) for t, d, m, x in out]
    beats = [(float(e.get("time")), int(e.get("measure", -1))) for e in root.iter("ebeat")]
    marks = [(str(e.get("name") or ""), float(e.get("startTime") or 0)) for e in root.iter("section")
             if (e.get("name") or "default") != "default"]
    return out, {"beats": beats, "sections": marks, "track": tr["name"], "track_no": tr["index"] + 1,
                 "tracks": [f"{t['index'] + 1} {t['name']}" + (" (bass)" if t["is_bass"] else "") for t in tracks],
                 "tuning": opens if bass_tuning else None, "octave": octave, "muted": muted, "capo": capo,
                 "repeats_once": single_pass}


def read_notes(path: str | Path, track: str | int | None = None) -> list[tuple]:
    """(start_s, dur_s, midi[, extra]) in the file's own time (Guitar Pro:
    ``extra`` = string/fret/technique — see :func:`read_gp`)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in EXTS:
        raise ValueError(f"{path.name}: use MusicXML (.musicxml/.xml/.mxl), MIDI (.mid) "
                         "or Guitar Pro (.gp3/.gp4/.gp5/.gpx/.gp)")
    if ext in GP_EXTS:
        return read_gp(path, track)[0]
    if ext in (".mid", ".midi"):
        notes = read_midi(path)
    else:
        src: Path | io.BytesIO = path
        if ext == ".mxl":
            import tempfile
            data = _mxl_xml(path)
            with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as fh:
                fh.write(data)
            src = Path(fh.name)
        try:
            notes = _benchmark().parse_musicxml(src)
        finally:
            if ext == ".mxl":
                Path(src).unlink(missing_ok=True)
    if not notes:
        raise ValueError(f"no notes in {path.name}")
    return notes


# ── lining up ────────────────────────────────────────────────────────────

def _hits(x: np.ndarray, onsets: np.ndarray, tol: float) -> int:
    i = np.clip(np.searchsorted(onsets, x), 1, len(onsets) - 1)
    d = np.minimum(np.abs(onsets[i] - x), np.abs(onsets[i - 1] - x))
    return int(np.sum(d < tol))


def _nearest(x: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """Signed distance from each x to its nearest onset."""
    i = np.clip(np.searchsorted(onsets, x), 1, len(onsets) - 1)
    a, b = onsets[i - 1] - x, onsets[i] - x
    return np.where(np.abs(a) < np.abs(b), a, b)


TEMPO_FACTORS = (0.5, 2 / 3, 1.5, 2.0)    # a file written at half / double (or 3:2) the song's tempo


def ratio_centers(times: np.ndarray, onsets: np.ndarray) -> list[float]:
    """Tempo ratios to search around: 1, and — when the file lasts much
    longer or shorter than the stem plays — the half / double / 3:2 factor
    nearest to that, and that length ratio itself (tabs are often written at
    double tempo: 187 BPM for a 96 BPM song; or at any tempo: 100 for 57.5)."""
    span_f = float(times.max() - times.min())
    span_s = float(onsets.max() - onsets.min())
    if span_f <= 1.0 or span_s <= 1.0:
        return [1.0]
    est = span_s / span_f
    if abs(np.log(est)) < np.log(1.25):
        return [1.0]
    near = min(TEMPO_FACTORS, key=lambda c: abs(np.log(est / c)))
    out = [1.0, near]
    if abs(np.log(est / near)) > np.log(1.06) and 0.3 < est < 3.5:
        out.append(round(float(est), 3))
    return out


WIDE_BELOW = 0.5            # fewer notes than this on an attack → also try every tempo ratio …
WIDE_RANGE = (0.4, 2.6)     # … in this range


def global_fit(times: np.ndarray, onsets: np.ndarray, max_ratio: float = 0.08,
               max_offset: float = 60.0, pitches: np.ndarray | None = None,
               pitch_score=None, centers=(1.0,), wide: bool = False) -> tuple[float, float, int]:
    """(ratio, offset, hits): song_time ≈ ratio × file_time + offset.

    Repetitive bass lines fit almost as well one or two beats off. Every
    (ratio, offset) within 90 % of the best hit count is a candidate; when
    ``pitch_score(song_times, pitches)`` is given (how well the stem's sound
    agrees with those pitches at those times), it picks among them."""
    span = max(1.0, float(times.max() - times.min()))
    step = max(0.0002, 0.02 / span)                 # ≈ 20 ms over the whole file
    sample = times if len(times) <= 400 else times[np.linspace(0, len(times) - 1, 400).astype(int)]
    cands: dict = {}

    def scan(ratios):
        for r in ratios:
            d = (onsets[None, :] - r * sample[:, None]).ravel()
            d = d[np.abs(d) <= max_offset]
            if not d.size:
                continue
            h, e = np.histogram(d, bins=np.arange(-max_offset, max_offset + 0.02, 0.02))
            for k in np.argsort(-h)[:12]:          # one peak per beat in repetitive music
                off = round(float(e[k] + 0.01), 2)
                cands[(round(float(r), 5), off)] = _hits(r * times + off, onsets, 0.08)  # drift-tolerant

    for c in centers:
        scan(np.arange(c * (1 - max_ratio), c * (1 + max_ratio) + 1e-9, step * 4 * c))
    if wide and max(cands.values(), default=0) < WIDE_BELOW * len(times):
        # the tab's tempo can be anything (written by ear at a round number): sweep them all
        c = WIDE_RANGE[0]
        while c <= WIDE_RANGE[1]:
            scan(np.arange(c * 0.985, c * 1.015 + 1e-9, step * 4 * c))
            c *= 1.03
    top = sorted(cands.items(), key=lambda kv: -kv[1])[:20]
    for (r0, _o), _n in top:                      # finer ratios around the best few
        scan(np.arange(r0 - step * 4, r0 + step * 4 + 1e-9, step))
    best_n = max(cands.values())
    pool = [(k, n) for k, n in cands.items() if n >= 0.9 * best_n]
    # merge near-identical candidates (same alignment within 30 ms everywhere)
    pool.sort(key=lambda kv: -kv[1])
    distinct: list = []
    for (r, o), n in pool:
        if all(abs((r - r2) * span) + abs(o - o2) > 0.08 for (r2, o2), _ in distinct):
            distinct.append(((r, o), n))
        if len(distinct) >= 12:
            break
    if pitch_score is not None and pitches is not None and len(distinct) > 1:
        scored = [(pitch_score(r * times + o, pitches) + 0.5 * n / best_n, (r, o), n) for (r, o), n in distinct]
        _s, (r, o), n = max(scored)
        return r, o, n
    (r, o), n = distinct[0]
    return r, o, n


SEG_NOTES = 16           # notes per segment of the coarse re-fit
SEG_RANGE = 1.0          # ± seconds a segment may move from the global fit
SEG_JUMP = 1.5           # cost (in hits) of a segment moving away from the one before …
SEG_PER_S = 4.0          # … plus this per second moved


def segment_offsets(x: np.ndarray, onsets: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Coarse local re-fit after the global one: the file and the band part
    ways now and then (a beat held longer, a bar added) by more than the
    smooth warp follows. Notes in segments of ``SEG_NOTES``; each segment
    gets the shift (± ``SEG_RANGE`` s) that puts most notes on an attack,
    with a cost for moving away from the previous segment's shift (Viterbi),
    so repetitive bass lines don't jump a beat. Returns a shift per note."""
    n = len(x)
    if n < 2 * SEG_NOTES:
        return np.zeros(n)
    order = np.argsort(x)
    xs = x[order]
    bounds = list(range(0, n, SEG_NOTES))
    shifts = np.round(np.arange(-SEG_RANGE, SEG_RANGE + 1e-9, 0.01), 3)
    gain = np.zeros((len(bounds), len(shifts)))
    for k, a in enumerate(bounds):
        seg = xs[a:a + SEG_NOTES]
        for j, sh in enumerate(shifts):
            gain[k, j] = _hits(seg + sh, onsets, tol)
    move = SEG_JUMP * (np.abs(shifts[:, None] - shifts[None, :]) > 0.025) + \
        SEG_PER_S * np.abs(shifts[:, None] - shifts[None, :])
    score = gain[0] - SEG_PER_S * np.abs(shifts) * 0.5        # start near the global fit
    back = np.zeros((len(bounds), len(shifts)), int)
    for k in range(1, len(bounds)):
        tot = score[:, None] - move                            # [from, to]
        back[k] = np.argmax(tot, axis=0)
        score = tot[back[k], np.arange(len(shifts))] + gain[k]
    path = [int(np.argmax(score))]
    for k in range(len(bounds) - 1, 0, -1):
        path.append(int(back[k][path[-1]]))
    path.reverse()
    seg_shift = shifts[path]
    centers = np.array([xs[a:a + SEG_NOTES].mean() for a in bounds])
    # piecewise: each note takes its segment's shift (interpolated across boundaries within 1 s)
    per = np.empty(n)
    for k, a in enumerate(bounds):
        per[a:a + SEG_NOTES] = seg_shift[k]
    out = np.empty(n)
    out[order] = per
    return out


def warp(times: np.ndarray, onsets: np.ndarray, ratio: float, offset: float,
         window: float = 8.0, tol: float = 0.12, rounds: int = 3) -> np.ndarray:
    """Song times of ``times``: the global fit, a coarse re-fit per segment
    (:func:`segment_offsets`) and a smooth local correction."""
    return refine(ratio * times + offset, onsets, window, tol, rounds)


def refine(x: np.ndarray, onsets: np.ndarray, window: float = 8.0, tol: float = 0.12,
           rounds: int = 3) -> np.ndarray:
    """A first map ``x`` of the notes on the song → segment re-fit + smooth warp."""
    x = x + segment_offsets(x, onsets)
    corr = np.zeros_like(x)
    for _ in range(rounds):
        d = _nearest(x + corr, onsets)
        ok = np.abs(d) < tol
        new = corr.copy()
        for i, v in enumerate(x + corr):
            w = ok & (np.abs(x + corr - v) < window)
            if w.sum() >= 4:
                new[i] = corr[i] + float(np.median(d[w]))
        corr = new
    return x + corr


# ── when one tempo ratio doesn't fit: follow the notes themselves ─────────

SEQ_ATTACK = 0.5      # global fit: fewer notes than this on an attack …
SEQ_PITCH = 0.8       # … or the stem sounding their pitches less than this (right fits: 0.9–2.2) → try it


def sequence_pairs(ft: np.ndarray, fpc: np.ndarray, rt: np.ndarray, rpc: np.ndarray,
                   match: int = 2, miss: int = -1, gap: int = -1) -> np.ndarray:
    """Line up two note sequences by pitch class (the file's and a quick
    transcription of the stem): semi-global alignment, free gaps at both ends.
    Returns the matched pairs [(file time, song time)], in order."""
    n, m = len(ft), len(rt)
    S = np.zeros((n + 1, m + 1), np.int32)
    B = np.zeros((n + 1, m + 1), np.int8)         # 0 diagonal, 1 up (file note skipped), 2 left
    k = np.arange(m + 1)
    for i in range(1, n + 1):
        mt = np.where(rpc == fpc[i - 1], match, miss)
        diag = S[i - 1, :-1] + mt
        up = S[i - 1, 1:] + gap
        row = np.maximum(diag, up)                   # best ending at (i, j) without a left gap
        bk = np.where(diag >= up, 0, 1).astype(np.int8)
        cand = np.empty(m + 1, np.int64)
        cand[0] = 0                                  # free start in the song
        cand[1:] = row
        # left gaps (song notes skipped): best[j] = max over k ≤ j of cand[k] + gap·(j − k)
        best = np.maximum.accumulate(cand - gap * k) + gap * k
        S[i] = best
        B[i] = np.where(best == cand, np.r_[1, bk], 2)
    i, j = n, int(np.argmax(S[n]))
    pairs = []
    while i > 0 and j > 0:
        b = B[i, j]
        if b == 0:
            if fpc[i - 1] == rpc[j - 1]:
                pairs.append((ft[i - 1], rt[j - 1]))
            i, j = i - 1, j - 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    return np.array(pairs[::-1], float).reshape(-1, 2)


def map_from_pairs(t: np.ndarray, pairs: np.ndarray, win: int = 9) -> np.ndarray:
    """File times → song times through matched pairs: stray matches dropped
    (far from the running median of their neighbours' offset), then linear
    interpolation (and extrapolation with the overall slope) between them."""
    if len(pairs) < 8:
        raise ValueError("too few matches")
    a, b = pairs[:, 0], pairs[:, 1]
    # local slope-free check: offset b − r·a against its neighbours
    r = float(np.polyfit(a, b, 1)[0])
    off = b - r * a
    h = win // 2
    med = np.array([np.median(off[max(0, k - h):k + h + 1]) for k in range(len(off))])
    keep = np.abs(off - med) < 0.15
    a, b = a[keep], b[keep]
    # strictly increasing in both
    ok = np.r_[True, (np.diff(a) > 0) & (np.diff(b) > 0)]
    a, b = a[ok], b[ok]
    x = np.interp(t, a, b)
    x = np.where(t < a[0], b[0] + r * (t - a[0]), x)
    x = np.where(t > a[-1], b[-1] + r * (t - a[-1]), x)
    return x


def _quality(x: np.ndarray, pitches: np.ndarray, onsets: np.ndarray, pitch_score=None) -> tuple[float, float]:
    """(share of notes on an attack, how much the stem sounds their pitches)."""
    att = _hits(x, onsets, 0.05) / max(1, len(x))
    return att, (float(pitch_score(x, pitches)) if pitch_score is not None else 1.0)


def line_up(notes: list[tuple[float, float, int]], onsets: np.ndarray,
            snap: float = 0.06, pitch_score=None, ref_notes=None) -> tuple[list[tuple[float, float, int]], dict]:
    """File notes → song notes [(start, dur, midi)] + a report. ``ref_notes``:
    [(time, midi)] of a quick transcription of the stem, or a function giving
    them — used when no single tempo ratio fits (the file's tempo changes
    where the band's doesn't, or the other way round)."""
    onsets = np.sort(np.asarray(onsets, float))
    if len(onsets) < 8:
        raise ValueError("too few attacks in the bass stem to line the notes up")
    t = np.array([n[0] for n in notes])
    e = np.array([n[0] + n[1] for n in notes])
    P = np.array([n[2] for n in notes])
    ratio, offset, hits0 = global_fit(t, onsets, pitches=P,
                                      pitch_score=pitch_score, centers=ratio_centers(t, onsets), wide=True)
    start = warp(t, onsets, ratio, offset)
    method = "tempo"
    att, pit = _quality(start, P, onsets, pitch_score)
    q = att + pit
    far = float(onsets.max()) + 5.0
    if (att < SEQ_ATTACK or pit < SEQ_PITCH) and far > 60.0:
        # the file may start much earlier / later than the song (a live version, a tab of one part)
        r2, o2, _h = global_fit(t, onsets, pitches=P, max_offset=far, pitch_score=pitch_score,
                                centers=ratio_centers(t, onsets), wide=True)
        s2 = warp(t, onsets, r2, o2)
        a2, p2 = _quality(s2, P, onsets, pitch_score)
        if a2 + p2 > q + 0.1:
            ratio, offset, start, att, pit, q = r2, o2, s2, a2, p2, a2 + p2
    if (att < SEQ_ATTACK or pit < SEQ_PITCH) and ref_notes is not None:
        ref = ref_notes() if callable(ref_notes) else ref_notes
        if ref is not None and len(ref) >= 16:
            rt = np.array([r_[0] for r_ in ref], float)
            rpc = np.array([int(r_[1]) % 12 for r_ in ref])
            order0 = np.argsort(t)
            try:
                x = map_from_pairs(t, sequence_pairs(t[order0], P[order0] % 12, rt, rpc))
                start2 = refine(x, onsets)
                q2 = sum(_quality(start2, P, onsets, pitch_score))
                if q2 > q + 0.1:
                    start, q, method = start2, q2, "notes"
                    ratio = float(np.polyfit(t, start, 1)[0])
                    offset = float(np.median(start - ratio * t))
            except ValueError:
                pass
    # the end of a note follows the same map (interpolated on the note starts)
    order = np.argsort(t)
    end = np.interp(e, t[order], start[order], left=None, right=None)
    end = np.where(e > t.max(), start.max() + ratio * (e - t.max()), end)
    # snap starts to the attack they sit on (each attack used once)
    d = _nearest(start, onsets)
    snapped = 0
    used = set()
    out = []
    for i in np.argsort(np.abs(d)):
        s = float(start[i])
        if abs(d[i]) < snap:
            o = round(float(s + d[i]), 4)
            if o not in used:
                used.add(o)
                s = o
                snapped += 1
        out.append((s, max(0.03, float(end[i] - start[i])), int(notes[i][2]), int(i)))
    out.sort()
    # no overlaps longer than the next note's start
    res, src = [], []
    for k, (s, dur, m, i) in enumerate(out):
        nxt = out[k + 1][0] if k + 1 < len(out) else None
        if nxt is not None and s + dur > nxt and nxt > s:
            dur = max(0.03, nxt - s)
        res.append((round(s, 4), round(dur, 4), m))
        src.append(i)
    info = {"ratio": round(ratio, 5), "offset": round(offset, 3), "notes": len(res), "method": method,
            "quality": round(float(q), 2),
            "on_attack": snapped, "share_on_attack": round(snapped / max(1, len(res)), 3),
            "tempo_note": (f"the file's tempo is {abs(1 - ratio) * 100:.1f} % "
                           f"{'faster' if ratio > 1 else 'slower'} than the song")
            if abs(1 - ratio) >= 0.002 else ""}
    if abs(np.log(ratio)) > np.log(1.25):
        names = {2.0: "double time", 0.5: "half time", 1.5: "3:2", 2 / 3: "2:3"}
        near = min(names, key=lambda c: abs(np.log(ratio / c)))
        info["tempo_note"] = (f"the file's tempo is {'on average ' if method == 'notes' else ''}"
                              f"{ratio:.2f}× the song's"
                              + (f" ({names[near]})" if abs(np.log(ratio / near)) < np.log(1.06)
                                 and method != "notes" else ""))
    info["src"] = src                      # index in ``notes`` of each returned note
    info["unmatched"] = unmatched_stretches([r[0] for r in res], onsets)
    return res, info


def unmatched_stretches(starts, onsets: np.ndarray, win: int = 16, min_share: float = 0.25,
                        tol: float = 0.06) -> list[tuple[float, float]]:
    """Song time ranges where the lined-up notes mostly miss the stem's
    attacks (≥ ``win`` notes in a row with < ``min_share`` on an attack):
    there the file probably doesn't follow the recording (a part missing,
    repeated or played differently)."""
    x = np.asarray(starts, float)
    if len(x) < win or len(onsets) < 2:
        return []
    hit = np.abs(_nearest(x, np.sort(np.asarray(onsets, float)))) < tol
    share = np.convolve(hit.astype(float), np.ones(win) / win, mode="valid")
    bad = share < min_share
    out: list[list[float]] = []
    for i in np.flatnonzero(bad):
        a, b = float(x[i]), float(x[i + win - 1])
        if out and a <= out[-1][1] + 1.0:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(round(a, 1), round(b, 1)) for a, b in out]


def stem_pitch_score(y: np.ndarray, sr: int):
    """A ``pitch_score(times, pitches)`` for :func:`line_up`: how much the
    stem sounds each pitch (or its harmonics — the fundamental may be missing)
    right after the given times, compared with the notes a tone around it."""
    import librosa
    if sr != 11025:
        y = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=11025)
        sr = 11025
    hop = 128
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.midi_to_hz(21), n_bins=72))
    D = librosa.amplitude_to_db(C + 1e-9, ref=np.max)           # rows: MIDI 21 … 92

    def harm(m, f0, f1):
        rows = [m - 21 + k for k in (0, 12, 19, 24) if 0 <= m - 21 + k < D.shape[0]]
        return float(np.max(D[rows, f0:f1].mean(axis=1))) if rows else -80.0

    def score(times, pitches):
        vals = []
        for t, m in zip(times, pitches):
            f0 = int(t * sr / hop) + 1
            f1 = f0 + 8
            if f0 < 0 or f1 >= D.shape[1]:
                continue
            here = harm(int(m), f0, f1)
            around = np.mean([harm(int(m) + k, f0, f1) for k in (-2, -1, 1, 2)])
            vals.append(here - around)
        return float(np.mean(vals)) / 6.0 if vals else 0.0      # ≈ 0…1
    return score


def trim_to_sound(notes: list[tuple[float, float, int]], y: np.ndarray, sr: int,
                  drop_db: float = 10.0, floor_db: float = -42.0,
                  min_dur: float = 0.06) -> tuple[list[tuple[float, float, int]], int]:
    """End each note where the stem actually stops sounding: the level falls
    ``drop_db`` under the note's own peak, or under ``floor_db`` (song
    level). Notation programs write notes legato, filling every rest, so
    without this a chart would hold notes through the silences. Returns the
    notes and how many were shortened."""
    import librosa
    if sr != 11025:
        y = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=11025)
        sr = 11025
    hop = 128
    rms = librosa.feature.rms(y=np.asarray(y, np.float32), frame_length=512, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
    db = np.convolve(db, np.ones(3) / 3, mode="same")          # 35 ms smoothing
    fps = sr / hop
    out, cut = [], 0
    for t, d, m in notes:
        f0 = int(t * fps)
        f1 = min(len(db), int((t + d) * fps) + 1)
        if f1 - f0 < 3:
            out.append((t, d, m))
            continue
        peak = float(db[f0:min(f1, f0 + int(0.12 * fps) + 1)].max())
        thr = max(peak - drop_db, floor_db)
        start_look = f0 + int(min_dur * fps)
        below = np.flatnonzero(db[start_look:f1] < thr)
        if below.size:
            e = (start_look + int(below[0])) / fps
            if e < t + d - 0.03:
                out.append((t, round(max(min_dur, e - t), 4), m))
                cut += 1
                continue
        out.append((t, d, m))
    return out, cut
