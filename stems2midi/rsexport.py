"""Rocksmith 2014 bass arrangement XML from a stems2midi transcription.

stems2midi writes ``bass_RS2.xml`` by itself after every transcription with a
tab. To redo it for an earlier result (any Python 3.9+, no extra packages):

    python plugins/stems2midi/rsexport.py "song_stems" --artist "Band" --title "Song" --year 1998

The XML is written with FeedBack's own writer (``lib/gp2rs.py``, the one the
Guitar Pro import uses) and then completed with what a transcription adds:

* **beat map** — the beats tracked on the full mix, smoothed (local linear fit
  over ±8 beats, so the tracker's jitter doesn't make the tempo wobble) and
  re-phased to the bass attacks; 4/4 with the bar start on the beat where the
  bass hits hardest/lowest. With a fixed ``--bpm`` it is a constant grid;
* **sections/phrases** — COUNT, then a ``riff`` section every 8 bars (or the
  song parts), ``noguitar`` where the bass rests, END after the last note;
  phrases (what Riff Repeater loops) every 4 bars inside them, the same notes
  → the same phrase name (``phrase_bars``);
* **notes** — string/fret from the tab, sustain on notes ≥ 0.25 s, hammer-on /
  pull-off on legato notes that stay on the same string, ``slap="1"`` /
  ``pluck="1"`` (pop) from the slap marking;
* anchors (hand positions) from the fretting, arrangement properties, tuning.

One difficulty level only: DLC Builder generates the others (DD). Import the
file into EOF (File → Import → Rocksmith) to correct notes and the beat map,
then build the psarc with DLC Builder as usual.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


SUSTAIN_MIN_S = 0.25
# The detected attack lands a little after where charters place the note:
# on 12 charted songs the chart notes sit a median 16 ms earlier than our
# onsets, so the export moves everything this much earlier.
ADVANCE_S = 0.015
LEGATO_GAP_S = 0.06
STANDARD_BASS = (28, 33, 38, 43, 48)      # E1 A1 D2 G2 (C3) — RS offsets are relative to these


# ─────────────────────────────────────────────────────────────────────────────
#  FeedBack's writer
# ─────────────────────────────────────────────────────────────────────────────

def _lib_dirs() -> list[Path]:
    """Where FeedBack's lib/gp2rs.py may be: the FeedBack checkout this plugin
    sits in (plugins/stems2midi → ../../lib), $FEEDBACK_DIR/lib, the current
    folder, and a copy shipped next to this file (installer: vendor/)."""
    here = Path(__file__).resolve().parent
    dirs = [here.parents[1] / "lib", here.parent / "lib"]
    env = os.environ.get("FEEDBACK_DIR")
    if env:
        dirs.append(Path(env) / "lib")
    dirs += [Path.cwd() / "lib", here / "vendor"]
    return [d for d in dirs if (d / "gp2rs.py").is_file()]


def load_gp2rs():
    """Import FeedBack's ``gp2rs`` (only its XML writer is used).

    gp2rs imports PyGuitarPro at module level for its Guitar Pro reader; when
    the Python running this (e.g. the plugin's ML environment) doesn't have it,
    a stand-in module is used just for the import — the writer never touches it.
    """
    if "gp2rs" in sys.modules:
        return sys.modules["gp2rs"]
    try:
        import gp2rs  # inside the FeedBack server, lib/ is on sys.path
        return gp2rs
    except ImportError:
        pass
    dirs = _lib_dirs()
    if not dirs:
        raise RuntimeError("Rocksmith export needs FeedBack's lib/gp2rs.py: keep the plugin inside "
                           "FeedBack's plugins/ folder or set FEEDBACK_DIR to the FeedBack folder.")
    import importlib.util
    spec = importlib.util.spec_from_file_location("gp2rs", dirs[0] / "gp2rs.py")
    mod = importlib.util.module_from_spec(spec)
    stub = None
    try:
        import guitarpro  # noqa: F401
    except ImportError:
        import types

        class _Any:  # annotations like guitarpro.Song resolve to this
            pass
        stub = types.ModuleType("guitarpro")
        stub.__getattr__ = lambda name: _Any
        sys.modules["guitarpro"] = stub
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"FeedBack's gp2rs.py ({dirs[0]}) could not be loaded: {e}") from e
    finally:
        if stub is not None and sys.modules.get("guitarpro") is stub:
            del sys.modules["guitarpro"]
    sys.modules["gp2rs"] = mod
    return mod


# ─────────────────────────────────────────────────────────────────────────────
#  Beat map (pure Python: runs in any Python FeedBack or the engine uses)
# ─────────────────────────────────────────────────────────────────────────────

def _median(xs) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _diffs(xs) -> list[float]:
    return [b - a for a, b in zip(xs, xs[1:])]


def _regularize(beats) -> list[float]:
    """Drop doubled beats and fill dropped ones so every step is ~one period."""
    beats = [float(b) for b in beats]
    med = _median(_diffs(beats))
    out = [beats[0]]
    for t in beats[1:]:
        gap = t - out[-1]
        if gap < 0.6 * med:
            continue
        k = max(1, int(round(gap / med)))
        base = out[-1]
        out.extend(base + gap * j / k for j in range(1, k))
        out.append(t)
    return out


def _smooth(beats, win: int = 8) -> list[float]:
    """Each beat replaced by a straight-line fit over its ±win neighbours."""
    n = len(beats)
    out = []
    for k in range(n):
        lo, hi = max(0, k - win), min(n, k + win + 1)
        xs = range(lo, hi)
        ys = beats[lo:hi]
        mx, my = sum(xs) / len(ys), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        out.append(my + slope * (k - mx))
    return out


def _nearest(grid, t) -> float:
    i = bisect.bisect_left(grid, t)
    cands = [grid[j] for j in (i - 1, i) if 0 <= j < len(grid)]
    return min(cands, key=lambda g: abs(g - t))


def _phase_to_onsets(grid, onsets) -> list[float]:
    """Shift the grid by the median offset of the attacks that sit near a beat."""
    if len(onsets) < 8 or len(grid) < 2:
        return list(grid)
    per = _median(_diffs(grid))
    d = [t - _nearest(grid, t) for t in onsets]
    d = [x for x in d if abs(x) < per / 4]
    shift = _median(d) if len(d) >= 8 else 0.0
    return [g + shift for g in grid]


def _extend(grid, start: float, end: float) -> list[float]:
    per_head = _median(_diffs(grid[:9])) if len(grid) > 2 else 0.5
    per_tail = _median(_diffs(grid[-9:])) if len(grid) > 2 else 0.5
    head = []
    t = grid[0] - per_head
    while t >= start - 1e-9:
        head.append(t)
        t -= per_head
    tail = []
    t = grid[-1] + per_tail
    while t <= end + per_tail:
        tail.append(t)
        t += per_tail
    return head[::-1] + list(grid) + tail


def downbeat_phase(grid, notes: list[dict], meter: int = 4) -> int:
    """Index (mod ``meter``) of the bar's first beat: where the bass lands
    loudest, lowest and longest."""
    if len(grid) < 2:
        return 0
    per = _median(_diffs(grid))
    score = [0.0] * meter
    for n in notes:
        i = bisect.bisect_left(grid, n["t"])
        i = min((j for j in (i - 1, i) if 0 <= j < len(grid)), key=lambda j: abs(grid[j] - n["t"]))
        if abs(grid[i] - n["t"]) < per / 6:
            score[i % meter] += (n.get("v", 90) / 127) * (1 + max(0, 45 - n["m"]) / 12) * min(n["d"], per)
    return max(range(meter), key=lambda k: score[k])


def make_beat_map(tracked, bpm: float, notes: list[dict], duration: float,
                  meter: int = 4) -> list[tuple[float, int]]:
    """[(time, measure_number or -1)] covering 0 … end of the song."""
    onsets = sorted(n["t"] for n in notes)
    end = max(duration, max((n["t"] + n["d"] for n in notes), default=0.0))
    tracked = sorted(float(t) for t in (tracked or []))
    if len(tracked) >= 8:
        grid = _smooth(_regularize(tracked))
    else:
        per = 60.0 / bpm
        first = onsets[0] if onsets else 0.0
        grid = [first % per + k * per for k in range(int((max(end, first) + per) / per) + 1)]
    grid = _phase_to_onsets(grid, onsets)
    grid = [t for t in _extend(grid, 0.0, end) if t >= 0]
    phase = downbeat_phase(grid, notes, meter)
    out, bar = [], 0
    for i, t in enumerate(grid):
        if i % meter == phase:
            bar += 1
            out.append((round(t, 6), bar))
        else:
            out.append((round(t, 6), -1))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Sections / phrases
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Part:
    name: str      # riff | noguitar
    start: float
    end: float


def make_parts(beat_map, notes: list[dict], bars_per_phrase: int = 8) -> list[Part]:
    downs = [t for t, m in beat_map if m > 0]
    if not notes or not downs:
        return []
    first = min(n["t"] for n in notes)
    last = max(n["t"] + n["d"] for n in notes)
    starts = [t for t in downs if t <= first + 1e-6]
    i0 = downs.index(starts[-1]) if starts else 0
    bounds = downs[i0::bars_per_phrase]
    parts: list[Part] = []
    for a, b in zip(bounds, bounds[1:] + [float("inf")]):
        if a > last:
            break
        has = any(a - 1e-6 <= n["t"] < b - 1e-6 for n in notes)
        name = "riff" if has else "noguitar"
        if parts and parts[-1].name == name == "noguitar":
            parts[-1].end = b
        else:
            parts.append(Part(name, a, b))
    # END: first downbeat after the last note
    end_t = next((t for t in downs if t >= last - 1e-6), beat_map[-1][0])
    if parts:
        parts[-1].end = end_t
    parts.append(Part("noguitar", end_t, end_t))
    return parts


# ─────────────────────────────────────────────────────────────────────────────
#  Export
# ─────────────────────────────────────────────────────────────────────────────

def _offsets(payload: dict) -> list[int]:
    tun = payload.get("tuning") or {}
    if tun.get("offsets") is not None:
        return list(tun["offsets"])
    strings = tun.get("strings") or list(STANDARD_BASS[:4])
    base = (23,) + STANDARD_BASS[:4] if len(strings) == 5 and strings[0] < 26 else STANDARD_BASS
    return [s - b for s, b in zip(strings, base)]


RS_MAX_FRET = 24
# Rhythm (guitar) path an octave down (A220, centOffset −1200): E1 A1 D2 G2 B2 E3
GUITAR_DOWN = (28, 33, 38, 43, 47, 52)
RHYTHM_XML = "bass5_rhythm_RS2.xml"


def top_string_notes(payload: dict) -> int:
    """Notes on the 5th (top) string of a 5-string tab."""
    tun = payload.get("tuning") or {}
    if len(tun.get("strings") or []) < 5:
        return 0
    return sum(1 for n in payload.get("notes") or [] if int(n.get("s", 0)) >= 4)


def wants_rhythm5(payload: dict, mode: str = "auto") -> bool:
    """Also write the 5-string part as a Rhythm arrangement? ``auto``: when the
    top string carries many notes (≥ 10 and ≥ 5 %) — a 5-string part a
    4-string chart would twist."""
    mode = (mode or "auto").lower()
    if mode in ("bass", "4", "no", "off"):
        return False
    k = top_string_notes(payload)
    if mode in ("rhythm", "both", "yes", "on"):
        return k > 0 or len((payload.get("tuning") or {}).get("strings") or []) >= 5
    n = len(payload.get("notes") or [])
    return k >= 10 and k >= 0.05 * max(1, n)


def rhythm5_payload(payload: dict) -> dict:
    """A 5-string bass tab as a Rocksmith Rhythm (6-string guitar path) one
    octave down — A220, centOffset −1200 — so a 5-string bass plays it string
    for string: B E A D G, the unused 6th string also on G — offsets
    −5 −5 −5 −5 −4 −9, as in 5-string CDLC (e.g. Tears for Fears' "Raoul and
    the Kings of Spain"). Slap/pop are dropped (not on the guitar path)."""
    tun = payload.get("tuning") or {}
    strings = list(tun.get("strings") or [])[:5]
    six = strings + [strings[-1]]
    notes = [{k: v for k, v in n.items() if k != "tech"} for n in payload.get("notes") or []]
    return dict(payload, notes=notes, cent_offset=-1200, rhythm5=True,
                tuning={**tun, "strings": six, "offsets": [a - b for a, b in zip(six, GUITAR_DOWN)]})


def four_string_for_rocksmith(payload: dict) -> tuple[dict, str]:
    """Rocksmith 2014 has only 4-string bass. A 5-string tab is charted the way
    custom-song authors do it: the 4 lowest strings (B E A D for B standard →
    offsets −5 −5 −5 −5, "B Standard" in Rocksmith/DLC Builder), and what was
    played on the top string moves to the string below, a fourth higher up the
    neck. Returns (payload, message); a 4-string payload comes back as is."""
    tun = payload.get("tuning") or {}
    strings = list(tun.get("strings") or [])
    if len(strings) <= 4:
        return payload, ""
    low4 = strings[:4]
    step = strings[4] - strings[3]
    moved, high = 0, 0
    notes = []
    for n in payload.get("notes") or []:
        n = dict(n)
        if int(n.get("s", 0)) >= 4:
            n["s"], n["f"] = 3, int(n["f"]) + step * (int(n["s"]) - 3)
            moved += 1
            if n["f"] > RS_MAX_FRET:
                high += 1
        notes.append(n)
    offs, cents = rs_tuning(low4)
    out = dict(payload, notes=notes, anchors=simple_anchors(notes), cent_offset=cents,
               tuning={**tun, "strings": low4, "offsets": offs})
    label = "B standard (B E A D)" if low4 == [23, 28, 33, 38] else \
        "tuning (MIDI " + " ".join(str(m) for m in low4) + ")"
    msg = (f"5-string tab → Rocksmith 4-string {label}: offsets "
           f"{' '.join(f'{o:+d}' for o in offs)}"
           + (f", {cents} cents" if cents else "") + "; "
           f"{moved} note(s) from the top string moved to the string below (+{step} frets)"
           + (f"; {high} above fret {RS_MAX_FRET}" if high else "") + ".")
    return out, msg


def rs_tuning(strings4: list[int]) -> tuple[list[int], int]:
    """Rocksmith offsets (vs E A D G) + cent offset for 4 open strings. Very low
    tunings are written an octave up with −1200 cents — B standard = +7 +7 +7
    +7 / −1200, as in the 5-string bass CDLC (e.g. Miley Cyrus' Midnight Sky)."""
    offs = [a - b for a, b in zip(strings4, STANDARD_BASS)]
    if offs and max(offs) <= -5:
        return [o + 12 for o in offs], -1200
    return offs, 0


def simple_anchors(notes: list[dict]) -> list[dict]:
    """Fret-hand positions (4-fret windows) following the fretted notes."""
    out, a = [], None
    for n in sorted(notes, key=lambda n: n["t"]):
        f = int(n.get("f", 0))
        if f <= 0:
            continue
        if a is None or f < a or f > a + 3:
            a = max(1, min(f if a is None or f < a else f - 3, RS_MAX_FRET - 3))
            out.append({"t": n["t"], "f": a, "w": 4})
    return out


def export_xml(payload: dict, *, title: str = "", artist: str = "", album: str = "",
               year: str = "", duration: float | None = None, meter: int = 4,
               bars_per_phrase: int = 8, arrangement: str = "Bass",
               advance: float = ADVANCE_S, shift: float = 0.0) -> tuple[str, list[str]]:
    """Return (xml_text, warnings). ``shift`` (seconds) moves everything later,
    e.g. by the silence EOF puts at the start of its guitar.ogg."""
    gp2rs = load_gp2rs()
    warnings: list[str] = []
    if arrangement == "Bass":
        payload, msg = four_string_for_rocksmith(payload)
        if msg:
            warnings.append(msg)
    shift = float(shift or 0.0)
    move = shift - advance
    notes = sorted(({**n, "t": max(0.0, n["t"] + move)} for n in payload.get("notes") or []),
                   key=lambda n: n["t"])
    if not notes or "s" not in notes[0]:
        raise ValueError("this transcription has no strings/frets — run it with a bass tuning (--tuning)")
    offsets = _offsets(payload)
    bpm = float(payload.get("bpm") or 120.0)
    duration = float(duration or payload.get("duration") or 0.0) + max(0.0, shift)
    if notes and notes[0]["t"] <= 0.0 and shift < 0:
        warnings.append("The shift moves notes before the start of the audio.")
    if not payload.get("beats"):
        warnings.append("No tracked beats in this transcription: constant-tempo beat map at "
                        f"{bpm:g} BPM — check it in EOF.")
    tracked = [t + shift for t in payload.get("beats") or [] if t + shift >= 0]
    beat_map = make_beat_map(tracked, bpm, notes, duration, meter)
    song_len = max(duration, beat_map[-1][0])
    parts = make_parts(beat_map, notes, bars_per_phrase)

    # notes
    rs_notes, last_on = [], {}
    for k, n in enumerate(notes):
        nxt = notes[k + 1]["t"] if k + 1 < len(notes) else n["t"] + n["d"]
        d = min(n["d"], max(0.0, nxt - n["t"]))
        rn = gp2rs.RsNote(time=round(n["t"], 3), string=int(n["s"]), fret=int(n["f"]),
                          sustain=round(d, 3) if d >= SUSTAIN_MIN_S else 0.0)
        prev = last_on.get(rn.string)
        if n.get("lg") and prev is not None and n["t"] - (prev["t"] + prev["d"]) < LEGATO_GAP_S \
                and prev is notes[k - 1]:
            if rn.fret > prev["f"]:
                rn.hammer_on = True
            elif rn.fret < prev["f"]:
                rn.pull_off = True
        last_on[rn.string] = n
        rs_notes.append(rn)

    anchors = [gp2rs.RsAnchor(time=round(max(0.0, a["t"] + move), 3), fret=max(1, int(a["f"])), width=int(a.get("w", 4)))
               for a in payload.get("anchors") or []]
    if not anchors:
        anchors = [gp2rs.RsAnchor(time=rs_notes[0].time, fret=1)]
    beats = [gp2rs.RsBeat(time=t, measure=m) for t, m in beat_map]
    counts: dict[str, int] = {}
    sections = []
    for p in parts:
        counts[p.name] = counts.get(p.name, 0) + 1
        sections.append(gp2rs.RsSection(name=p.name, time=round(p.start, 3), number=counts[p.name]))

    xml_text = gp2rs._build_xml(
        title, artist, album, str(year or ""), arrangement, offsets, len(offsets),
        round(song_len, 3), -beat_map[0][0], beats, sections, rs_notes, [], [], anchors,
        round(bpm, 3))
    return _finish(xml_text, notes, parts, beat_map, offsets,
                   int(payload.get("cent_offset") or 0), arrangement), warnings


def _finish(xml_text: str, notes: list[dict], parts: list[Part], beat_map, offsets,
            cent_offset: int = 0, arrangement: str = "Bass") -> str:
    """Add what gp2rs doesn't write: slap/pluck, COUNT/END phrases,
    arrangement properties."""
    root = ET.fromstring(xml_text)
    # EOF reads times as "seconds.milliseconds" — with more than 3 decimals the
    # digits are taken as milliseconds (0.314906 s → 314.9 s) and every note
    # ends up before the first beat and is dropped. Rocksmith/EOF use 3.
    for eb in root.iter("ebeat"):
        eb.set("time", f"{float(eb.get('time')):.3f}")
    sb = root.find("startBeat")
    if sb is not None and sb.text:
        sb.text = f"{float(sb.text):.3f}"
    # slap / pop
    for el, n in zip(root.find("levels/level/notes"), notes):
        if n.get("tech") == "slap":
            el.set("slap", "1")
        elif n.get("tech") == "pop":
            el.set("pluck", "1")
    # phrases: COUNT, one per part (riff1, riff2 … / noguitar), END
    phrases, iters = [], []
    t0 = beat_map[0][0]
    if not parts or parts[0].start > t0 + 1e-6:
        phrases.append("COUNT")
        iters.append((t0, 0))
    riffs = 0
    for i, p in enumerate(parts):
        if i == len(parts) - 1:
            name = "END"
        elif p.name == "riff":
            riffs += 1
            name = f"riff{riffs}"
        else:
            name = p.name
        if name not in phrases:
            phrases.append(name)
        iters.append((p.start, phrases.index(name)))
    for tag in ("phrases", "phraseIterations"):
        old = root.find(tag)
        idx = list(root).index(old)
        root.remove(old)
        new = ET.Element(tag, count=str(len(phrases) if tag == "phrases" else len(iters)))
        if tag == "phrases":
            for nm in phrases:
                ET.SubElement(new, "phrase", disparity="0", ignore="0", maxDifficulty="0",
                              name=nm, solo="0")
        else:
            for t, pid in iters:
                ET.SubElement(new, "phraseIteration", time=f"{t:.3f}", phraseId=str(pid))
        root.insert(idx, new)
    # header bits EOF / DLC Builder expect
    pos = list(root).index(root.find("arrangement")) + 1
    for tag, text in (("part", "1"), ("centOffset", str(int(cent_offset)))):
        if root.find(tag) is None:
            el = ET.Element(tag)
            el.text = text
            root.insert(pos, el)
            pos += 1
        elif tag == "centOffset":
            root.find(tag).text = text
    techs = {n.get("tech") for n in notes}
    legato = any(e.get("hammerOn") == "1" or e.get("pullOff") == "1"
                 for e in root.iter("note"))
    props = {
        "represent": "1", "bonusArr": "0", "standardTuning": "1" if not any(offsets) else "0",
        "nonStandardChords": "0", "barreChords": "0", "powerChords": "0", "dropDPower": "0",
        "openChords": "0", "fingerPicking": "0" if techs & {"slap", "pop"} else "1",
        "pickDirection": "0", "doubleStops": "0", "palmMutes": "0", "harmonics": "0",
        "pinchHarmonics": "0", "hopo": "1" if legato else "0", "tremolo": "0", "slides": "0",
        "unpitchedSlides": "0", "bends": "0", "tapping": "0", "vibrato": "0", "fretHandMutes": "0",
        "slapPop": "1" if techs & {"slap", "pop"} else "0", "twoFingerPicking": "0",
        "fifthsAndOctaves": "0", "syncopation": "0", "bassPick": "0",
        "sustain": "1" if any(float(e.get("sustain", 0)) > 0 for e in root.iter("note")) else "0",
        "pathLead": "0", "pathRhythm": "0", "pathBass": "1",
    }
    if arrangement != "Bass":                  # guitar path (5-string bass as Rhythm): a bonus arrangement
        props.update(pathBass="0", pathRhythm="1" if arrangement == "Rhythm" else "0",
                     pathLead="1" if arrangement == "Lead" else "0", slapPop="0", fingerPicking="1",
                     represent="0", bonusArr="1")
    ap = ET.Element("arrangementProperties", **props)
    root.insert(list(root).index(root.find("albumYear")) + 1, ap)
    if arrangement != "Bass" and root.find("tonebase") is None and root.find("ebeats") is not None:
        tb = ET.Element("tonebase")             # a bass tone on the guitar path
        tb.text = "bass"
        root.insert(list(root).index(root.find("ebeats")) + 1, tb)
    # an empty year is dropped (readers parse it as a number); empty names are
    # written as <tag></tag> like EOF does, not <tag />
    yr = root.find("albumYear")
    if yr is not None and not (yr.text or "").strip():
        root.remove(yr)
    return _serialize(root)


def _serialize(root: ET.Element) -> str:
    ET.indent(root, "  ")
    text = ET.tostring(root, encoding="unicode")
    text = re.sub(r"<(title|artistName|albumName) />", r"<\1></\1>", text)
    return "<?xml version='1.0' encoding='UTF-8'?>\n" + text + "\n"


def _chart_tools():
    """chart_tools.py next to this file, loaded by path (no sys.path changes
    inside the FeedBack server)."""
    import importlib.util
    mod = sys.modules.get("plugin_stems2midi_chart_tools")
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            "plugin_stems2midi_chart_tools", Path(__file__).resolve().parent / "chart_tools.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plugin_stems2midi_chart_tools"] = mod
        spec.loader.exec_module(mod)
    return mod


def beat_map_for(payload: dict, meter: int = 4, shift: float = 0.0) -> list[tuple[float, int]]:
    """The beat map export_xml() writes (times in the XML, i.e. + ``shift``)."""
    move = shift - ADVANCE_S
    notes = sorted(({**n, "t": max(0.0, n["t"] + move)} for n in payload.get("notes") or []),
                   key=lambda n: n["t"])
    tracked = [t + shift for t in payload.get("beats") or [] if t + shift >= 0]
    return make_beat_map(tracked, float(payload.get("bpm") or 120.0), notes,
                         float(payload.get("duration") or 0.0) + max(0.0, shift), meter)


def find_parts(payload: dict, parts: str) -> list[tuple[str, float, float]]:
    """Song parts ("verse=0:21-0:42, chorus=…": first occurrences, in the
    transcription's own time) → every occurrence [(name, start_s, end_s)],
    repeats found by their notes on the export's bar lines."""
    ct = _chart_tools()
    spec = ct.parse_parts(parts)
    root = ET.fromstring(export_xml(payload)[0].encode("utf-8"))
    found, beats, _notes = ct.find_structure(root, spec)
    bars = beats.bars
    t = lambda k: beats.t[bars[k]] if k < len(bars) else beats.t[-1]  # noqa: E731
    # the export moved notes ADVANCE_S earlier; give the ranges in note time
    return [(name, round(t(a) + ADVANCE_S - 0.03, 3), round(t(b) + ADVANCE_S - 0.03, 3))
            for name, a, b, _s, _p in found]


def apply_parts(xml_text: str, parts: str, shift: float = 0.0,
                slap_parts: str = "") -> tuple[str, list[str]]:
    """Write the song parts as Rocksmith sections/phrases into an exported XML
    (and mark slap/pop in the parts named in ``slap_parts``)."""
    ct = _chart_tools()
    spec = [(n, a + shift, b + shift) for n, a, b in ct.parse_parts(parts)]
    root = ET.fromstring(xml_text.encode("utf-8"))
    found, beats, notes = ct.find_structure(root, spec)
    merged = ct.write_structure(root, found, beats, notes)
    info = ", ".join(f"{n} {beats.t[beats.bars[a]] - shift:.1f}s" + ("" if p is None else f" ({s:.0%})")
                     for n, a, _b, s, p in found)
    msgs = [f"Song parts: {info}; sections: {', '.join(n for n, _t in merged)}."]
    names = {x.strip().lower() for x in (slap_parts or "").split(",") if x.strip()}
    if names:
        bars = beats.bars
        t = lambda k: beats.t[bars[k]] if k < len(bars) else beats.t[-1]  # noqa: E731
        c = ct.mark_slap(root, [(t(a) - 0.03, t(b) - 0.03) for n, a, b, _s, _p in found if n in names])
        msgs.append(f"Slap/pop in {', '.join(sorted(names))}: +{c['slap']} slap, +{c['pop']} pop.")
    return _serialize(root), msgs


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

XML_NAME = "bass_RS2.xml"      # written next to bass.notes.json
PHRASE_BARS = 4                # phrase (Riff Repeater unit) length in bars; sections stay longer


def guess_meta(name: str) -> dict:
    """"Artist - Title.mp3" → {"artist": "Artist", "title": "Title"}."""
    stem = Path(str(name or "")).stem.strip()
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return {"artist": artist.strip(), "title": title.strip()}
    return {"artist": "", "title": stem}


def render(payload: dict, fallback_name: str = "", **meta) -> tuple[str, list[str]]:
    """The full export of a transcription: metadata (given, stored, or from
    "Artist - Title" in the file name), shift and song parts."""
    guess = guess_meta(payload.get("source") or payload.get("title") or fallback_name)
    m = {k: (meta.get(k) or payload.get(k) or guess.get(k) or "") for k in ("artist", "title", "album", "year")}
    arrangement = meta.get("arrangement") or "Bass"
    if arrangement == "Rhythm":
        payload = rhythm5_payload(payload)
    xml_text, warns = export_xml(payload, title=m["title"], artist=m["artist"],
                                 album=m["album"], year=str(m["year"]), arrangement=arrangement,
                                 **{k: v for k, v in meta.items()
                                    if k in ("duration", "meter", "bars_per_phrase", "shift") and v is not None})
    parts = meta.get("parts") or payload.get("parts")
    if parts:
        xml_text, w = apply_parts(xml_text, parts, float(meta.get("shift") or 0.0),
                                  "" if arrangement != "Bass" else (meta.get("slap_parts") or ""))
        warns += w
    pb = meta.get("phrase_bars")
    if pb is None:
        pb = payload.get("phrase_bars", PHRASE_BARS)
    keep = meta.get("keep_accents")
    if keep is None:
        keep = payload.get("keep_accents", False)
    if int(pb or 0) > 0 or not keep:
        root = ET.fromstring(xml_text.encode("utf-8"))
        if int(pb or 0) > 0:
            _chart_tools().split_phrases(root, int(pb))
        if not keep:            # EOF / Rocksmith / DLC Builder: plain letters only
            _chart_tools().plain_metadata(root)
        xml_text = _serialize(root)
    return xml_text, warns


def write_xml(notes_path, out_path=None, **meta) -> tuple[Path, list[str]]:
    """Export ``bass.notes.json`` (or a result folder holding it) to
    ``bass_RS2.xml`` beside it. Missing artist/title come from what the
    transcription stored (the input file name)."""
    src = Path(notes_path)
    if src.is_dir():
        src = src / "bass.notes.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    five = meta.pop("five_string", None) or payload.get("five_string") or "auto"
    xml_text, warns = render(payload, fallback_name=src.parent.name, **meta)
    out = Path(out_path) if out_path else src.parent / XML_NAME
    out.write_text(xml_text, encoding="utf-8")
    if wants_rhythm5(payload, five):
        try:
            rx_text, _w = render(payload, fallback_name=src.parent.name, **{**meta, "arrangement": "Rhythm"})
            (out.parent / RHYTHM_XML).write_text(rx_text, encoding="utf-8")
            warns.append(f"5-string part ({top_string_notes(payload)} notes on the top string) also written as "
                         f"a Rhythm arrangement for a 5-string bass: {RHYTHM_XML} (B E A D G, A220 / "
                         f"−1200 cents) — add it in DLC Builder next to the bass.")
        except (ValueError, RuntimeError) as e:
            warns.append(f"{RHYTHM_XML} not written: {e}")
    return out, warns



def _cli_argv(argv):
    """Command-line arguments; paths with spaces / a trailing backslash from
    the installer's .bat files are rebuilt by s2m_config.cli_argv."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "plugin_stems2midi_config", Path(__file__).resolve().with_name("s2m_config.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.cli_argv(argv)
    except Exception:  # noqa: BLE001
        return argv

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="stems2midi result → Rocksmith 2014 bass XML")
    p.add_argument("notes", help="a stems2midi result folder, or its bass.notes.json (made with a bass tuning)")
    p.add_argument("-o", "--output", help=f"output XML (default: {XML_NAME} in the result folder)")
    p.add_argument("--title", default="")
    p.add_argument("--artist", default="")
    p.add_argument("--album", default="")
    p.add_argument("--year", default="")
    p.add_argument("--length", type=float, help="song length in seconds (default: from the transcription)")
    p.add_argument("--meter", type=int, default=4, help="beats per bar (default 4)")
    p.add_argument("--phrase-bars", type=int, default=None,
                   help="bars per phrase — what Riff Repeater loops (default 4; 0 = one phrase per section)")
    p.add_argument("--shift", type=float, default=0.0,
                   help="move everything this many seconds later — e.g. the silence EOF added at the "
                        "start of guitar.ogg when the transcription was made from the original MP3")
    a = p.parse_args(_cli_argv(argv))
    src = Path(a.notes)
    if not (src / "bass.notes.json").is_file() and not src.is_file():
        p.error(f"no bass.notes.json in {src}")
    try:
        out, warns = write_xml(src, a.output, title=a.title, artist=a.artist, album=a.album,
                               year=a.year, duration=a.length, meter=a.meter,
                               phrase_bars=a.phrase_bars, shift=a.shift)
    except (ValueError, RuntimeError) as e:
        p.error(str(e))
    for w in warns:
        print("warning:", w)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
