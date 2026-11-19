#!/usr/bin/env python3
"""chart_tools — edit a finished Rocksmith arrangement XML (e.g. EOF's
``PART REAL_BASS_RS2.xml``) without opening EOF.

Subcommands
───────────
slap   Mark slap/pop in the parts of the song you slap:
       notes with their own attack on the lower strings → slap,
       on the two highest strings → pop. Hammer-ons, pull-offs and slides
       stay plain (use --include-legato if your chart slaps those too).

           python chart_tools.py slap "PART REAL_BASS_RS2.xml" --sections "0:21-0:42, 1:26-1:38"
           python chart_tools.py slap song.xml --sections all --clear -o song_slap.xml

       Measured on charts made by hand (sections taken from the chart):
       Delfins slap F1 0.95 / pop 0.91, Dirty Loops 0.85 / 0.77,
       Primus 0.71 / 0.78 (Primus also slaps the D string).

structure  Name the song's parts once, find the repeats:

           python chart_tools.py structure song.xml --define "intro=0:00-0:21, verse=0:21-0:42, chorus=0:42-1:05"

       Give the first occurrence of each part (snapped to bar lines); every
       other stretch of bars with the same notes (≥ --min-match, default 60 %
       of notes at the same beat position and pitch) gets the same name.
       Writes Rocksmith sections (gaps become riff or noguitar) and phrases of
       at most 4 bars inside them (--phrase-bars; the same notes get the same
       phrase name). Options:
         --copy-tech   copy slap/pop (and hammer-on/pull-off) from the first
                       occurrence to its repeats — fix one verse in EOF, the
                       others follow
         --slap verse,chorus   mark slap/pop (string rule above) in every
                       occurrence of these parts

fourstring  5-string bass XML → Rocksmith 4-string (B standard as +7 ×4 / −1200 cents):

           python chart_tools.py fourstring bass_RS2.xml

dlcfix      an XML saved by EOF, ready for DLC Builder: hand-position
            changes moved off hammer-ons/pull-offs ("position shift into
            pull-off") and title/artist without accents (they name the .psarc):

           python chart_tools.py dlcfix "PART REAL_BASS_RS2.xml" --from-folder

rhythm5     the same part as a Rhythm arrangement an octave down (A220) a 5-string
            bass plays string for string (B E A D G):

           python chart_tools.py rhythm5 bass_RS2.xml

The output keeps every other attribute, all difficulty levels and EOF's
header comment; import it back in EOF (File → Import → Rocksmith) to review,
or take it straight to DLC Builder.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _fretting():
    """fretting.py next to this file, loaded by path under a private name (the
    FeedBack server imports this module too: no sys.path changes there)."""
    import importlib.util
    mod = sys.modules.get("plugin_stems2midi_fretting")
    if mod is None:
        spec = importlib.util.spec_from_file_location("plugin_stems2midi_fretting", HERE / "fretting.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plugin_stems2midi_fretting"] = mod
        spec.loader.exec_module(mod)
    return mod


parse_sections = _fretting().parse_sections

# Section names Rocksmith / EOF accept (EOF's predefined list)
RS_SECTIONS = ("noguitar intro preverse verse postvs modverse prechorus chorus postchorus modchorus hook "
               "prebrdg bridge postbrdg solo outro ambient breakdown interlude transition riff fadein fadeout "
               "buildup variation head modbridge melody vamp silence tapping").split()
_TECH = ("slap", "pluck", "hammerOn", "pullOff")

EOF_TIPS = """\
Back into EOF: select the PART REAL_BASS track FIRST, then File > Import >
Rocksmith (import replaces the active track; on another track it silently does
nothing). The notes land in difficulty 0 (first difficulty tab), not Expert —
fine to leave there.

By hand in EOF: select notes (drag or Ctrl+A) > Note > Rocksmith > Toggle >
Slap (Ctrl+Shift+S) / Pop (Ctrl+Shift+P); Edit > Selection > Conditional
deselect (Alt+D) keeps only notes on chosen strings ("lanes"); Beat >
Rocksmith > Place RS Section (Shift+S).

More: plugins/stems2midi/README.md ("Workflow with EOF")."""

_LEGATO = ("hammerOn", "pullOff", "slideTo", "slideUnpitchTo")


def _on(el, attr) -> bool:
    return el.get(attr) not in (None, "0", "-1")


def mark_slap(root: ET.Element, sections, include_legato: bool = False, clear: bool = False) -> dict:
    # string count from the notes (the <tuning> tag always has 6 slots)
    n_strings = 4
    for lv in root.iter("level"):
        for n in lv.iter("note"):
            n_strings = max(n_strings, int(n.get("string", 0)) + 1)
    n_low = max(1, n_strings - 2)
    counts = {"slap": 0, "pop": 0, "cleared": 0}
    for lv in root.iter("level"):
        notes = list(lv.iterfind("./notes/note"))
        for n in notes:
            if clear:
                for a in ("slap", "pluck"):
                    if _on(n, a):
                        counts["cleared"] += 1
                    n.attrib.pop(a, None)
            t = float(n.get("time", 0))
            if not any(a <= t < b for a, b in sections):
                continue
            if not include_legato and any(_on(n, a) for a in _LEGATO):
                continue
            if _on(n, "slap") or _on(n, "pluck"):
                continue  # already marked by hand
            if int(n.get("string", 0)) < n_low:
                n.set("slap", "1")
                counts["slap"] += 1
            else:
                n.set("pluck", "1")
                counts["pop"] += 1
    props = root.find("arrangementProperties")
    if props is not None and (counts["slap"] or counts["pop"]):
        props.set("slapPop", "1")
    return counts


def _load(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(str(path), parser)


def cmd_slap(args) -> int:
    src = Path(args.xml)
    try:
        sections = parse_sections(args.sections)
    except ValueError as e:
        print(f"--sections: {e}", file=sys.stderr)
        return 2
    if not sections:
        print("--sections is empty (use 'all' or ranges like 0:21-0:42)", file=sys.stderr)
        return 2
    tree = _load(src)
    counts = mark_slap(tree.getroot(), sections, args.include_legato, args.clear)
    out = Path(args.output) if args.output else src.with_name(src.stem + ".slap.xml")
    ET.indent(tree, space="  ")
    tree.write(str(out), encoding="UTF-8", xml_declaration=True)
    print(f"{out}: +{counts['slap']} slap, +{counts['pop']} pop"
          + (f", {counts['cleared']} old marks cleared" if args.clear else ""))
    return 0


# ── structure: name parts once, find the repeats ───────────────────────────

class Beats:
    """Beat-index ↔ seconds from <ebeats>; bar lines where measure ≥ 0."""

    def __init__(self, root: ET.Element):
        eb = list(root.iter("ebeat"))
        if len(eb) < 2:
            raise ValueError("the XML has no beat map (<ebeats>)")
        self.t = [float(e.get("time")) for e in eb]
        self.bars = [i for i, e in enumerate(eb) if e.get("measure") not in (None, "-1")]
        if not self.bars:
            self.bars = list(range(0, len(eb), 4))

    def beat(self, t: float) -> float:
        import bisect
        T = self.t
        i = min(max(bisect.bisect_right(T, t) - 1, 0), len(T) - 2)
        return i + (t - T[i]) / ((T[i + 1] - T[i]) or 1e-9)

    def time(self, b: float) -> float:
        i = min(max(int(b), 0), len(self.t) - 2)
        return self.t[i] + (b - i) * (self.t[i + 1] - self.t[i])

    def nearest_bar(self, t: float) -> int:
        """Index into self.bars of the bar line closest to t."""
        b = self.beat(t)
        return min(range(len(self.bars)), key=lambda k: abs(self.bars[k] - b))


def parse_parts(spec: str) -> list[tuple[str, float, float]]:
    """"verse=0:21-0:42, chorus=0:42-1:05" → [(name, start_s, end_s)]."""
    out = []
    for item in [x.strip() for x in str(spec or "").split(",") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"expected name=start-end, got {item!r}")
        name, rng = (x.strip() for x in item.split("=", 1))
        name = name.lower()
        if name not in RS_SECTIONS:
            raise ValueError(f"{name!r} is not a Rocksmith section name (e.g. {', '.join(RS_SECTIONS[1:9])}…)")
        (a, b), = parse_sections(rng)
        out.append((name, a, b))
    if not out:
        raise ValueError("no parts given")
    return out


def _pitch(n: ET.Element, base: list[int]) -> int:
    s = int(n.get("string", 0))
    return (base[s] if s < len(base) else 0) + int(n.get("fret", 0))


def _base_pitches(root: ET.Element) -> list[int]:
    tun = root.find("tuning")
    std = [28, 33, 38, 43, 48, 53]
    off = [int(tun.get(f"string{i}", 0)) if tun is not None else 0 for i in range(6)]
    return [a + b for a, b in zip(std, off)]


def _similarity(tpl, win, tol: float = 0.2):
    """tpl/win: [(rel_beat, pitch, elem)] → (score, [(tpl_elem, win_elem)])."""
    used, pairs = set(), []
    for r, p, e in tpl:
        best = None
        for j, (q, p2, e2) in enumerate(win):
            if j in used or p2 != p or abs(q - r) > tol:
                continue
            if best is None or abs(q - r) < abs(win[best][0] - r):
                best = j
        if best is not None:
            used.add(best)
            pairs.append((e, win[best][2]))
    total = len(tpl) + len(win)
    return (2 * len(pairs) / total if total else 0.0), pairs


def find_structure(root: ET.Element, parts, min_match: float = 0.6):
    """Occurrences [(name, bar_from, bar_to, score, pairs_with_first)] sorted
    by time; the given parts themselves have score 1 and pairs None."""
    beats = Beats(root)
    base = _base_pitches(root)
    lv = max(root.iter("level"), key=lambda L: len(L.findall("./notes/note")))
    notes = [(beats.beat(float(n.get("time"))), _pitch(n, base), n) for n in lv.iterfind("./notes/note")]
    bars = beats.bars

    def window(k0: int, k1: int):
        b0 = bars[k0]
        b1 = bars[k1] if k1 < len(bars) else b0 + (k1 - k0) * 4
        return [(b - b0, p, e) for b, p, e in notes if b0 - 0.2 <= b < b1 - 0.2]

    taken: set[int] = set()
    found = []
    templates = []
    for name, a, b in parts:
        k0, k1 = beats.nearest_bar(a), beats.nearest_bar(b)
        if k1 <= k0:
            raise ValueError(f"{name}: {a:.1f}-{b:.1f} s is shorter than a bar")
        templates.append((name, k0, k1, window(k0, k1)))
        found.append((name, k0, k1, 1.0, None))     # pairs None = the part as given
        taken.update(range(k0, k1))
    cands = []
    for name, k0, k1, tpl in templates:
        L = k1 - k0
        if not tpl:
            continue
        for k in range(0, len(bars) - L + 1):
            if any(x in taken for x in range(k, k + L)):
                continue
            score, pairs = _similarity(tpl, window(k, k + L))
            if score >= min_match:
                cands.append((score, name, k, k + L, pairs))
    for score, name, k, k2, pairs in sorted(cands, key=lambda c: -c[0]):
        if any(x in taken for x in range(k, k2)):
            continue
        taken.update(range(k, k2))
        found.append((name, k, k2, round(score, 2), pairs))
    return sorted(found, key=lambda f: f[1]), beats, notes


def write_structure(root: ET.Element, found, beats: Beats, notes) -> list[tuple[str, float]]:
    """Replace sections/phrases/phraseIterations; returns [(section, time)]."""
    bars = beats.bars
    t_bar = lambda k: beats.t[bars[k]] if k < len(bars) else beats.t[-1]  # noqa: E731
    note_beats = [b for b, _p, _e in notes]
    old_end = None
    phrases_el = root.find("phrases")
    if phrases_el is not None:
        names = [ph.get("name") for ph in phrases_el]
        for it in root.iter("phraseIteration"):
            pid = int(it.get("phraseId", -1))
            if 0 <= pid < len(names) and names[pid] == "END":
                old_end = float(it.get("time"))
    last_note = max((float(e.get("time")) + float(e.get("sustain", 0)) for _b, _p, e in notes), default=0.0)
    end_t = max(old_end or 0.0, t_bar(found[-1][2]) if found else 0.0, last_note)
    # timeline: found parts + gaps (riff if notes, else noguitar), from the first bar with notes
    first_k = beats.nearest_bar(min((float(e.get("time")) for _b, _p, e in notes), default=0.0))
    first_k = min([first_k] + [f[1] for f in found])
    segs, k = [], first_k
    for name, a, b, _s, _p in found:
        if a > k:
            has = any(bars[k] - 0.2 <= x < bars[a] - 0.2 for x in note_beats)
            segs.append(("riff" if has else "noguitar", t_bar(k)))
        segs.append((name, t_bar(a)))
        k = b
    if k < len(bars) and t_bar(k) < end_t - 0.05:
        has = any(x >= bars[k] - 0.2 for x in note_beats)
        segs.append(("riff" if has else "noguitar", t_bar(k)))
    # merge consecutive gaps of the same kind
    merged = []
    for nm, t in segs:
        if merged and nm == merged[-1][0] and nm in ("riff", "noguitar"):
            continue
        merged.append((nm, t))
    # sections
    for tag in ("sections", "phrases", "phraseIterations"):
        el = root.find(tag)
        if el is not None:
            root.remove(el)
    pos = list(root).index(root.find("ebeats")) + 1 if root.find("ebeats") is not None else len(root)
    sec = ET.Element("sections", count=str(len(merged) + 1))
    count: dict[str, int] = {}
    for nm, t in merged + [("noguitar", end_t)]:
        count[nm] = count.get(nm, 0) + 1
        ET.SubElement(sec, "section", name=nm, number=str(count[nm]), startTime=f"{t:.3f}")
    # phrases: COUNT, one per section occurrence (verse1, verse2 …), END
    ph_names, iters, idx = ["COUNT"], [(beats.t[0], 0)], {}
    for nm, t in merged:
        idx[nm] = idx.get(nm, 0) + 1
        label = f"{nm}{idx[nm]}" if nm != "noguitar" else "noguitar"
        if label not in ph_names:
            ph_names.append(label)
        iters.append((t, ph_names.index(label)))
    ph_names.append("END")
    iters.append((end_t, len(ph_names) - 1))
    if iters[1][0] <= iters[0][0] + 1e-3:
        iters.pop(0)                       # first part starts on the first beat: no COUNT
        ph_names_used = {i for _t, i in iters}
    else:
        ph_names_used = set(range(len(ph_names)))
    keep = [n for i, n in enumerate(ph_names) if i in ph_names_used]
    remap = {i: keep.index(n) for i, n in enumerate(ph_names) if i in ph_names_used}
    phr = ET.Element("phrases", count=str(len(keep)))
    for nm in keep:
        ET.SubElement(phr, "phrase", disparity="0", ignore="0", maxDifficulty="0", name=nm, solo="0")
    pit = ET.Element("phraseIterations", count=str(len(iters)))
    for t, i in iters:
        ET.SubElement(pit, "phraseIteration", time=f"{t:.3f}", phraseId=str(remap[i]))
    for el in (sec, pit, phr):            # insert so the order ends phrases, phraseIterations, sections
        root.insert(pos, el)
    return merged


# ── plain text for EOF / Rocksmith / DLC Builder ──────────────────────────

_ASCII_MAP = str.maketrans({"ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O",
                            "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "þ": "th", "ð": "d",
                            "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
                            "–": "-", "—": "-", "…": "...", "\u00a0": " "})
TEXT_TAGS = ("title", "artistName", "artistNameSort", "albumName", "albumNameSort",
             "songNameSort", "titleSort", "albumYear")


def to_ascii(text: str) -> str:
    """Accents off (ã → a, ç → c, é → e…), only plain ASCII left: EOF turns
    "Canção" into "CanAAo", and Rocksmith's font has no accented letters."""
    import unicodedata
    t = unicodedata.normalize("NFKD", str(text).translate(_ASCII_MAP))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return "".join(c for c in t if 32 <= ord(c) < 127)


def plain_metadata(root: ET.Element) -> int:
    """Song title / artist / album… written without accents. Returns how many changed."""
    n = 0
    for tag in TEXT_TAGS:
        el = root.find(tag)
        if el is not None and el.text and to_ascii(el.text) != el.text:
            el.text = to_ascii(el.text)
            n += 1
    return n


ANCHOR_MAX_WIDTH = 6           # widest fret-hand window we write (frets)
_T_EPS = 0.0015


def fix_legato_anchors(rs_notes, anchors, make_anchor):
    """No hand-position change on a hammer-on / pull-off (DLC Builder flags it
    as "position shift into pull-off", and the hand can't shift while the
    string is still ringing from the note before).

    For each run of legato notes (first picked note + its hammer-ons /
    pull-offs) that has an anchor change inside it, one anchor at the picked
    note covers the whole run; the hand position in force after the run is
    put back at the next note. A run too wide for one hand position loses the
    hammer-on/pull-off where the hand has to move. Returns (anchors, fixes)."""
    def legato(n):
        return bool(getattr(n, "hammer_on", False) or getattr(n, "pull_off", False))

    anchors = sorted(anchors, key=lambda a: a.time)

    def active(t):
        cur = None
        for a in anchors:
            if a.time <= t + _T_EPS:
                cur = a
            else:
                break
        return cur

    fixes, i = 0, 0
    while i < len(rs_notes):
        j = i
        while j + 1 < len(rs_notes) and legato(rs_notes[j + 1]):
            j += 1
        if j > i:
            run = rs_notes[i:j + 1]
            t0, t1 = run[0].time, run[-1].time
            inside = [a for a in anchors if t0 + _T_EPS < a.time <= t1 + _T_EPS]
            if inside:
                frets = [n.fret for n in run if n.fret > 0]
                lo, hi = (min(frets), max(frets)) if frets else (1, 1)
                after = active(t1)
                if hi - lo + 1 <= ANCHOR_MAX_WIDTH:
                    anchors = [a for a in anchors if a not in inside and abs(a.time - t0) > _T_EPS]
                    anchors.append(make_anchor(time=t0, fret=max(1, lo), width=max(4, hi - lo + 1)))
                    nxt = rs_notes[j + 1] if j + 1 < len(rs_notes) else None
                    if nxt is not None and after is not None and \
                            not any(t1 + _T_EPS < a.time <= nxt.time + _T_EPS for a in anchors):
                        # the notes after the run, until the next hand position
                        until = min((a.time for a in anchors if a.time > nxt.time + _T_EPS), default=1e9)
                        later = [n.fret for n in rs_notes[j + 1:] if n.time < until - _T_EPS and n.fret > 0]
                        w = max(4, hi - lo + 1)
                        if any(not max(1, lo) <= f < max(1, lo) + w for f in later):
                            anchors.append(make_anchor(time=nxt.time, fret=after.fret, width=after.width))
                    anchors.sort(key=lambda a: a.time)
                else:               # can't be one hand position: pick where the hand moves
                    for k in range(i + 1, j + 1):
                        n, p = rs_notes[k], rs_notes[k - 1]
                        if any(p.time + _T_EPS < a.time <= n.time + _T_EPS for a in anchors):
                            n.hammer_on = n.pull_off = False
                fixes += 1
        i = j + 1
    return anchors, fixes


class _Note:
    """A note (or chord) of an arrangement level, for fix_legato_anchors."""
    def __init__(self, el, chord=False):
        self.el, self.chord = el, chord
        self.time = float(el.get("time"))
        if chord:
            frets = [int(cn.get("fret", 0)) for cn in el.findall("chordNote")]
            self.fret = min([f for f in frets if f > 0], default=0)
            self.hammer_on = self.pull_off = False
        else:
            self.fret = int(el.get("fret", 0))
            self.hammer_on = el.get("hammerOn") == "1"
            self.pull_off = el.get("pullOff") == "1"


class _Anchor:
    def __init__(self, time, fret, width=4):
        self.time, self.fret, self.width = float(time), int(fret), int(width)


def dlc_anchors(root: ET.Element) -> int:
    """fix_legato_anchors on every difficulty level of an arrangement XML
    (e.g. saved by EOF). Returns how many runs were changed."""
    total = 0
    for level in root.iter("level"):
        notes_el, anchors_el = level.find("notes"), level.find("anchors")
        if notes_el is None or anchors_el is None or not len(anchors_el):
            continue
        notes = [_Note(e) for e in notes_el.findall("note")]
        chords_el = level.find("chords")
        if chords_el is not None:
            notes += [_Note(e, chord=True) for e in chords_el.findall("chord")]
        notes.sort(key=lambda n: n.time)
        old = [_Anchor(a.get("time"), a.get("fret"), float(a.get("width", 4))) for a in anchors_el.findall("anchor")]
        fixed, n = fix_legato_anchors(notes, old, _Anchor)
        if not n:
            continue
        total += n
        for nt in notes:
            if not nt.chord:
                for k, v in (("hammerOn", nt.hammer_on), ("pullOff", nt.pull_off)):
                    if nt.el.get(k) == "1" and not v:
                        nt.el.set(k, "0")
        for a in list(anchors_el):
            anchors_el.remove(a)
        for a in fixed:
            ET.SubElement(anchors_el, "anchor", time=f"{a.time:.3f}", fret=str(a.fret), width=f"{a.width:.3f}")
        anchors_el.set("count", str(len(fixed)))
    return total


def set_metadata(root: ET.Element, artist: str = "", title: str = "", album: str = "") -> None:
    """Artist / title / album and their sort fields (DLC Builder names the
    package from these)."""
    def put(tag, text):
        el = root.find(tag)
        if el is None:
            el = ET.SubElement(root, tag)
        el.text = text
    for tags, val in ((("artistName", "artistNameSort"), artist), (("title", "songNameSort", "titleSort"), title),
                      (("albumName", "albumNameSort"), album)):
        if val:
            for t in tags:
                if t == tags[0] or root.find(t) is not None:
                    put(t, val)


# ── 5-string → 4-string (Rocksmith has only 4-string bass) ─────────────────

FIVE_STRING_BASE = (23, 28, 33, 38, 43)      # B0 E1 A1 D2 G2 — a 5-string tab's offsets are vs these
FOUR_STRING_BASE = (28, 33, 38, 43)          # E1 A1 D2 G2 — Rocksmith's


GUITAR_DOWN = (28, 33, 38, 43, 47, 52)       # guitar an octave down (A220): E1 A1 D2 G2 B2 E3


def to_rhythm5(root: ET.Element) -> str:
    """A 5-string bass XML → Rocksmith Rhythm (guitar path, bonus arrangement)
    an octave down (A220, centOffset −1200) tuned B E A D G (6th string also G:
    −5 −5 −5 −5 −4 −9, as in 5-string CDLC), tonebase "bass", so a 5-string bass
    plays it string for string. Slap/pop removed (guitar path)."""
    tun = root.find("tuning")
    off = [int(tun.get(f"string{i}", 0)) if tun is not None else 0 for i in range(5)]
    strings = [b + o for b, o in zip(FIVE_STRING_BASE, off)]
    six = strings + [strings[4]]               # the unused 6th string on G too (as CDLC do)
    offs = [a - b for a, b in zip(six, GUITAR_DOWN)]
    if tun is None:
        tun = ET.SubElement(root, "tuning")
    tun.attrib.pop("stringCount", None)
    for i in range(6):
        tun.set(f"string{i}", str(offs[i]))
    co = root.find("centOffset")
    if co is None:
        co = ET.SubElement(root, "centOffset")
    co.text = "-1200"
    arr = root.find("arrangement")
    if arr is not None:
        arr.text = "Rhythm"
    props = root.find("arrangementProperties")
    if props is not None:
        props.set("pathBass", "0")
        props.set("pathRhythm", "1")
        props.set("pathLead", "0")
        props.set("slapPop", "0")
        props.set("standardTuning", "0")
        props.set("represent", "0")
        props.set("bonusArr", "1")
    tb = root.find("tonebase")
    if tb is None:
        tb = ET.Element("tonebase")
        eb = root.find("ebeats")
        root.insert(list(root).index(eb) + 1 if eb is not None else len(root), tb)
    tb.text = "bass"
    n_tech = 0
    for n in root.iter("note"):
        for a in ("slap", "pluck"):
            if n.get(a) not in (None, "-1", "0"):
                n.set(a, "-1")
                n_tech += 1
    return (f"Rhythm arrangement for a 5-string bass: A220 (centOffset -1200), offsets "
            f"{' '.join(f'{o:+d}' for o in offs)}" + (f"; {n_tech} slap/pop marks removed" if n_tech else "") + ".")


def to_four_string(root: ET.Element, max_fret: int = 24) -> str:
    """A 5-string bass XML → Rocksmith 4-string: the 4 lowest strings, notes on
    the top string moved to the string below (+5 frets), B standard written as
    +7 +7 +7 +7 / −1200 cents (as in 5-string CDLC). Returns a message, '' if
    the XML already was 4-string."""
    tun = root.find("tuning")
    notes = list(root.iter("note"))
    count = int(tun.get("stringCount", 0) or 0) if tun is not None else 0
    if count <= 4 and not any(int(n.get("string", 0)) >= 4 for n in notes):
        return ""
    off = [int(tun.get(f"string{i}", 0)) if tun is not None else 0 for i in range(5)]
    cents = int(float(root.findtext("centOffset") or 0))
    pitches = [b + o for b, o in zip(FIVE_STRING_BASE, off)]
    step = pitches[4] - pitches[3]
    moved = high = 0
    for n in notes:
        if int(n.get("string", 0)) >= 4:
            f = int(n.get("fret", 0)) + step
            n.set("string", "3")
            n.set("fret", str(f))
            moved += 1
            high += f > max_fret
    low4 = pitches[:4]
    offs = [a - b for a, b in zip(low4, FOUR_STRING_BASE)]
    if max(offs) <= -5:
        offs, cents = [o + 12 for o in offs], cents - 1200
    if tun is None:
        tun = ET.SubElement(root, "tuning")
    tun.attrib.pop("stringCount", None)
    for i in range(6):
        tun.set(f"string{i}", str(offs[i] if i < 4 else 0))
    co = root.find("centOffset")
    if co is None:
        co = ET.SubElement(root, "centOffset")
    co.text = str(cents)
    for lv in root.iter("level"):              # hand positions follow the new frets
        frets = [(float(n.get("time")), int(n.get("fret", 0))) for n in lv.iterfind("./notes/note")]
        anc_el = lv.find("anchors")
        if anc_el is None or not frets:
            continue
        for a in list(anc_el):
            anc_el.remove(a)
        a = None
        for t, f in sorted(frets):
            if f <= 0:
                continue
            if a is None or f < a or f > a + 3:
                a = max(1, min(f if a is None or f < a else f - 3, max_fret - 3))
                ET.SubElement(anc_el, "anchor", time=f"{t:.3f}", fret=str(a), width="4.000")
        anc_el.set("count", str(len(anc_el)))
    props = root.find("arrangementProperties")
    if props is not None:
        props.set("standardTuning", "0")
    return (f"5-string → 4-string: offsets {' '.join(f'{o:+d}' for o in offs)}, {cents} cents; "
            f"{moved} note(s) moved from the top string to the one below (+{step} frets)"
            + (f", {high} above fret {max_fret}" if high else "") + ".")


# ── phrases: shorter practice units inside the sections ────────────────────

KEEP_WHOLE = {"COUNT", "END", "noguitar"}


def _chunks(n: int, size: int) -> list[int]:
    """Bar counts of the phrases a section of ``n`` bars is cut into: ``size``
    bars each; a remainder under half a phrase joins the last one."""
    if size <= 0 or n <= size:
        return [n]
    out = [size] * (n // size)
    r = n - sum(out)
    if r:
        if r * 2 < size:
            out[-1] += r
        else:
            out.append(r)
    return out


def split_phrases(root: ET.Element, bars_per_phrase: int = 4, min_match: float = 0.85) -> int:
    """Cut every phrase longer than ``bars_per_phrase`` bars into phrases of
    that length (on bar lines; sections stay as they are). In Rocksmith a
    phrase is what Riff Repeater loops and what difficulty levels are built
    on, so shorter phrases = shorter bits to practise. Stretches with the same
    notes get the same phrase name (verse1, verse2, verse1…), as charters do.
    Returns the number of phrase iterations afterwards."""
    if bars_per_phrase <= 0:
        return len(root.findall("./phraseIterations/phraseIteration"))
    phr_el, it_el = root.find("phrases"), root.find("phraseIterations")
    if phr_el is None or it_el is None:
        return 0
    beats = Beats(root)
    bars = beats.bars
    t_bar = lambda k: beats.t[bars[k]] if k < len(bars) else beats.t[-1]  # noqa: E731
    base = _base_pitches(root)
    levels = list(root.iter("level"))
    lv = max(levels, key=lambda L: len(L.findall("./notes/note"))) if levels else None
    notes = [(beats.beat(float(n.get("time"))), _pitch(n, base)) for n in lv.iterfind("./notes/note")] if lv is not None else []
    old = [(float(it.get("time")), phr_el[int(it.get("phraseId"))]) for it in it_el]
    old.sort(key=lambda x: x[0])

    def window(k0, k1):
        b0, b1 = bars[k0], (bars[k1] if k1 < len(bars) else len(beats.t) - 1)
        return [(b - b0, p, None) for b, p in notes if b0 - 0.2 <= b < b1 - 0.2]

    names: list[str] = []          # phrase names in order
    attrs: dict[str, dict] = {}
    seen: dict[str, list] = {}     # base name → [(nbars, window, phrase name)]
    iters: list[tuple[float, str]] = []
    for i, (t, ph) in enumerate(old):
        nm = ph.get("name") or "riff"
        t_next = old[i + 1][0] if i + 1 < len(old) else None
        k0 = beats.nearest_bar(t)
        k1 = beats.nearest_bar(t_next) if t_next is not None else k0
        on_bar = abs(t_bar(k0) - t) < 0.06
        if nm in KEEP_WHOLE or t_next is None or not on_bar or k1 <= k0:
            pieces = [(t, k0, k1, None)]
        else:
            pieces, k = [], k0
            for n in _chunks(k1 - k0, bars_per_phrase):
                pieces.append((t_bar(k) if k > k0 else t, k, k + n, True))
                k += n
        stem = nm.rstrip("0123456789") or nm
        for tt, a, b, split in pieces:
            if not split:
                label = nm
            else:
                win = window(a, b)
                if not win:
                    label = "noguitar"
                else:
                    label = None
                    for nb, w, lab in seen.get(stem, []):
                        if nb == b - a and _similarity(w, win)[0] >= min_match:
                            label = lab
                            break
                    if label is None:
                        label = f"{stem}{len(seen.setdefault(stem, [])) + 1}"
                        seen[stem].append((b - a, win, label))
            if label not in attrs:
                attrs[label] = dict(ph.attrib, name=label) if label != "noguitar" or not split else \
                    {"disparity": "0", "ignore": "0", "maxDifficulty": "0", "name": "noguitar", "solo": "0"}
                names.append(label)
            iters.append((tt, label))
    # rewrite <phrases> and <phraseIterations> (keep their place)
    for tag in ("phrases", "phraseIterations"):
        el = root.find(tag)
        pos = list(root).index(el)
        root.remove(el)
        if tag == "phrases":
            new = ET.Element(tag, count=str(len(names)))
            for nm in names:
                ET.SubElement(new, "phrase", **attrs[nm])
        else:
            new = ET.Element(tag, count=str(len(iters)))
            for tt, nm in iters:
                ET.SubElement(new, "phraseIteration", time=f"{tt:.3f}", phraseId=str(names.index(nm)))
        new.tail = el.tail
        root.insert(pos, new)
    return len(iters)


def copy_tech(found, levels_root: ET.Element) -> int:
    """Copy slap/pop/hammer-on/pull-off from each part's first occurrence to
    the matched notes of its repeats. Returns the number of notes changed."""
    changed = 0
    for _name, _a, _b, score, pairs in found:
        if pairs is None:
            continue
        for src, dst in pairs:
            before = {a: dst.get(a) for a in _TECH}
            for a in _TECH:
                if _on(src, a):
                    dst.set(a, src.get(a))
                elif a in ("slap", "pluck"):
                    dst.attrib.pop(a, None)
            changed += before != {a: dst.get(a) for a in _TECH}
    props = levels_root.find("arrangementProperties")
    if props is not None and any(_on(n, "slap") or _on(n, "pluck") for n in levels_root.iter("note")):
        props.set("slapPop", "1")
    return changed


def cmd_structure(args) -> int:
    src = Path(args.xml)
    try:
        parts = parse_parts(args.define)
    except ValueError as e:
        print(f"--define: {e}", file=sys.stderr)
        return 2
    tree = _load(src)
    root = tree.getroot()
    try:
        found, beats, notes = find_structure(root, parts, args.min_match)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    merged = write_structure(root, found, beats, notes)
    n_phr = split_phrases(root, args.phrase_bars)
    if not args.keep_accents:
        plain_metadata(root)
    report = [f"{n}{'' if p is None else f' ({s:.0%})'} {beats.t[beats.bars[a]] // 60:.0f}:{beats.t[beats.bars[a]] % 60:04.1f}"
              for n, a, _b, s, p in found]
    msg = []
    if args.copy_tech:
        msg.append(f"{copy_tech(found, root)} notes got the first occurrence's slap/pop/hammer-on")
    if args.slap:
        names = {x.strip().lower() for x in args.slap.split(",") if x.strip()}
        bars = beats.bars
        ranges = [(beats.t[bars[a]] - 0.02, beats.t[bars[b]] if b < len(bars) else beats.t[-1])
                  for n, a, b, _s, _p in found if n in names]
        if ranges:
            c = mark_slap(root, ranges)
            msg.append(f"+{c['slap']} slap, +{c['pop']} pop in {', '.join(sorted(names))}")
    out = Path(args.output) if args.output else src.with_name(src.stem + ".structure.xml")
    ET.indent(tree, space="  ")
    tree.write(str(out), encoding="UTF-8", xml_declaration=True)
    print(f"{out}")
    print("  parts: " + " | ".join(report))
    print("  sections: " + ", ".join(n for n, _t in merged))
    print(f"  phrases: {n_phr} (a phrase every {args.phrase_bars} bars at most)" if args.phrase_bars > 0
          else f"  phrases: {n_phr} (one per section)")
    for m in msg:
        print("  " + m)
    return 0



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

def cmd_rhythm5(args) -> int:
    src = Path(args.xml)
    tree = _load(src)
    msg = to_rhythm5(tree.getroot())
    if not args.keep_accents:
        plain_metadata(tree.getroot())
    out = Path(args.output) if args.output else src.with_name(src.stem + ".rhythm.xml")
    ET.indent(tree, space="  ")
    tree.write(str(out), encoding="UTF-8", xml_declaration=True)
    print(out)
    print("  " + msg)
    return 0


def cmd_fourstring(args) -> int:
    src = Path(args.xml)
    tree = _load(src)
    msg = to_four_string(tree.getroot())
    if not msg:
        print("Already a 4-string arrangement — nothing to do.")
        return 0
    if not args.keep_accents:
        plain_metadata(tree.getroot())
    out = Path(args.output) if args.output else src.with_name(src.stem + ".4string.xml")
    ET.indent(tree, space="  ")
    tree.write(str(out), encoding="UTF-8", xml_declaration=True)
    print(out)
    print("  " + msg)
    return 0


def folder_meta(path: Path) -> dict:
    """"Artist - Title" from the folder holding the XML (EOF's Work folder)."""
    name = Path(path).resolve().parent.name
    if " - " in name:
        a, t = name.split(" - ", 1)
        return {"artist": a.strip(), "title": t.strip()}
    return {}


def cmd_dlcfix(args) -> int:
    src = Path(args.xml)
    tree = _load(src)
    root = tree.getroot()
    meta = folder_meta(src) if args.from_folder else {}
    meta.update({k: v for k, v in (("artist", args.artist), ("title", args.title), ("album", args.album)) if v})
    if not args.keep_accents:
        meta = {k: to_ascii(v) for k, v in meta.items()}
    set_metadata(root, **meta)
    changed = 0 if args.keep_accents else plain_metadata(root)
    n = dlc_anchors(root)
    out = Path(args.output) if args.output else src.with_name(src.stem + ".dlc.xml")
    ET.indent(tree, space="  ")
    tree.write(str(out), encoding="UTF-8", xml_declaration=True)
    print(out)
    print(f"  artist: {(root.findtext('artistName') or '').strip()!r}, title: {(root.findtext('title') or '').strip()!r}"
          + (f" ({changed} field(s) without accents)" if changed else ""))
    print(f"  {n} hand-position change(s) moved off hammer-ons/pull-offs" if n
          else "  no hand-position change on a hammer-on/pull-off")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Edit a Rocksmith arrangement XML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EOF_TIPS)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slap", help="mark slap/pop inside the slapped sections",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EOF_TIPS)
    s.add_argument("xml", help="arrangement XML (e.g. PART REAL_BASS_RS2.xml)")
    s.add_argument("--sections", required=True, help="'all' or ranges in seconds or m:ss, e.g. '0:21-0:42, 86-98'")
    s.add_argument("-o", "--output", help="output file (default: <name>.slap.xml next to the input)")
    s.add_argument("--include-legato", action="store_true", help="also mark hammer-ons/pull-offs/slides")
    s.add_argument("--clear", action="store_true", help="remove existing slap/pop marks first")
    s.set_defaults(func=cmd_slap)
    st = sub.add_parser("structure", help="name the parts once (verse=0:21-0:42 …); repeats are found by their notes",
                        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EOF_TIPS,
                        description="Give the first occurrence of each part; every other stretch of bars with "
                                    "the same notes gets the same name. Writes Rocksmith sections + phrases.")
    st.add_argument("xml", help="arrangement XML (e.g. exported by EOF, or bass_RS2.xml)")
    st.add_argument("--define", required=True,
                    help="first occurrence of each part, e.g. 'intro=0:00-0:21, verse=0:21-0:42, chorus=0:42-1:05'")
    st.add_argument("--min-match", type=float, default=0.6,
                    help="share of notes that must match for a repeat (default 0.6)")
    st.add_argument("--copy-tech", action="store_true",
                    help="copy slap/pop/hammer-on/pull-off from each first occurrence to its repeats")
    st.add_argument("--slap", help="mark slap/pop (string rule) in every occurrence of these parts, e.g. 'verse,chorus'")
    st.add_argument("--phrase-bars", type=int, default=4,
                    help="cut sections into phrases (what Riff Repeater loops) of at most N bars; "
                         "same notes → same phrase name (default 4; 0 = one phrase per section)")
    st.add_argument("--keep-accents", action="store_true",
                    help="keep accented letters in title/artist/album (default: written without accents)")
    st.add_argument("-o", "--output", help="output file (default: <name>.structure.xml next to the input)")
    st.set_defaults(func=cmd_structure)
    fs = sub.add_parser("fourstring", help="5-string bass XML → Rocksmith 4-string (B standard: +7 ×4, −1200 cents)",
                        description="Rocksmith 2014 has only 4-string bass (DLC Builder: 'Invalid strings used'). "
                                    "Keeps the 4 lowest strings, moves notes on the top string to the string "
                                    "below (+5 frets) and writes B standard as CDLC do (+7 +7 +7 +7, −1200 cents).")
    fs.add_argument("xml", help="arrangement XML with 5 strings")
    fs.add_argument("--keep-accents", action="store_true", help="keep accented letters in title/artist/album")
    fs.add_argument("-o", "--output", help="output file (default: <name>.4string.xml next to the input)")
    fs.set_defaults(func=cmd_fourstring)
    r5 = sub.add_parser("rhythm5", help="5-string bass XML → Rhythm arrangement an octave down (A220) for a 5-string bass",
                        description="Writes the 5-string part as a Rocksmith Rhythm (guitar) arrangement tuned "
                                    "B E A D G (+C) an octave down (centOffset -1200), so a 5-string bass plays it "
                                    "string for string. Add it next to the bass arrangement in DLC Builder.")
    r5.add_argument("xml", help="arrangement XML with 5 strings")
    r5.add_argument("--keep-accents", action="store_true", help="keep accented letters in title/artist/album")
    r5.add_argument("-o", "--output", help="output file (default: <name>.rhythm.xml next to the input)")
    r5.set_defaults(func=cmd_rhythm5)
    dx = sub.add_parser("dlcfix", help="get an XML saved by EOF ready for DLC Builder (anchors on pull-offs, names)",
                        description="DLC Builder: 'position shift into pull-off' — moves each hand-position change "
                                    "off hammer-ons/pull-offs to the picked note before them. Also writes "
                                    "title/artist/album (and their sort fields, which name the .psarc) without "
                                    "accents: EOF turns 'Pontapés' into 'PontapAs'.")
    dx.add_argument("xml", help="arrangement XML (e.g. PART REAL_BASS_RS2.xml saved by EOF)")
    dx.add_argument("--from-folder", action="store_true",
                    help="artist and title from the folder name ('Artist - Title'), e.g. EOF's Work folder")
    dx.add_argument("--artist", help="artist name")
    dx.add_argument("--title", help="song title")
    dx.add_argument("--album", help="album name")
    dx.add_argument("--keep-accents", action="store_true", help="keep accented letters")
    dx.add_argument("-o", "--output", help="output file (default: <name>.dlc.xml next to the input)")
    dx.set_defaults(func=cmd_dlcfix)
    args = p.parse_args(_cli_argv(argv))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
