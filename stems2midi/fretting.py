"""fretting — choose string & fret for every note of a bass line.

Given pitches (MIDI notes with timing) and a tuning, pick for each note the
string/fret a bassist would most plausibly use. Priorities, in order:

1. **Minimise hand jumps** — keep the fretting hand in one position as long
   as possible; when it must move, move as little as possible (and preferably
   during a rest rather than in the middle of a fast run).
2. **Stay at or below the 12th fret** — a note above it costs extra per fret.
3. Small tie-breakers: lower positions, avoid skipping strings, fretted notes
   slightly preferred over open strings, no wide 4-fret stretches in the first
   positions where frets are widest.

Model
─────
The hand is described by its *anchor*: the fret under the index finger. A
hand at anchor A reaches frets A..A+3 (one finger per fret, the Rocksmith
"anchor width 4"). Open strings need no finger, so they leave the hand where
it was. Every note (or group of simultaneous notes, e.g. a double stop) has a
few candidate fingerings — one per string the pitch fits on — and each
fretted fingering is compatible with a few anchors. The best sequence of
(fingering, anchor) states is found exactly with dynamic programming
(Viterbi): cost = Σ local cost (high frets, stretch…) + Σ transition cost
(anchor shift, string skip), so a slightly higher position is accepted now if
it avoids a jump later — something a note-by-note greedy choice can't do.

Complexity: O(N · S²) with S ≤ ~20 states per note → a few ms per hundred
notes in pure Python.

Tuning
──────
``parse_tuning`` accepts:
  * "4", "5" (low B), "5c" (high C), "6"
  * presets: "drop-d", "eb", "d-standard", "drop-c", "c-standard", "5-drop-a", …
  * note names, low→high or high→low: "D1 A1 D2 G2", "B0,E1,A1,D2,G2"
  * MIDI numbers: "26,33,38,43"
  * Rocksmith-style semitone offsets from standard: "-2,0,0,0"

Strings are numbered 0 = lowest (Rocksmith convention).
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Sequence

# ── tunings ─────────────────────────────────────────────────────────────────

STANDARD = {
    4: (28, 33, 38, 43),              # E1 A1 D2 G2
    5: (23, 28, 33, 38, 43),          # B0 E1 A1 D2 G2
    6: (23, 28, 33, 38, 43, 48),      # B0 E1 A1 D2 G2 C3
}

PRESETS = {
    "4": STANDARD[4], "bass-4": STANDARD[4], "standard": STANDARD[4], "e-standard": STANDARD[4],
    "5": STANDARD[5], "bass-5": STANDARD[5], "5-low-b": STANDARD[5],
    "5c": (28, 33, 38, 43, 48), "5-high-c": (28, 33, 38, 43, 48),
    "6": STANDARD[6], "bass-6": STANDARD[6],
    "drop-d": (26, 33, 38, 43),
    "eb": (27, 32, 37, 42), "eb-standard": (27, 32, 37, 42),
    "d-standard": (26, 31, 36, 41),
    "drop-c": (24, 31, 36, 41),
    "c-standard": (24, 29, 34, 39),
    "drop-b": (23, 30, 35, 40),
    "5-drop-a": (21, 28, 33, 38, 43),
    "5-a-standard": (21, 26, 31, 36, 41),
}

PRESET_LABELS = {
    "4": "E standard (4)", "bass-4": "E standard (4)", "standard": "E standard (4)",
    "e-standard": "E standard (4)", "5": "B standard (5)", "bass-5": "B standard (5)",
    "5-low-b": "B standard (5)", "5c": "E standard, high C (5)", "5-high-c": "E standard, high C (5)",
    "6": "Standard (6)", "bass-6": "Standard (6)", "drop-d": "Drop D", "eb": "E♭ standard",
    "eb-standard": "E♭ standard", "d-standard": "D standard", "drop-c": "Drop C",
    "c-standard": "C standard", "drop-b": "Drop B", "5-drop-a": "Drop A (5)",
    "5-a-standard": "A standard (5)",
}

_PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_to_midi(name: str) -> int:
    m = re.fullmatch(r"\s*([A-Ga-g])([#b♯♭]*)(-?\d+)\s*", name)
    if not m:
        raise ValueError(f"bad note name {name!r} (expected e.g. E1, Bb0, C#2)")
    pc = _PC[m.group(1).lower()] + m.group(2).count("#") + m.group(2).count("♯") \
        - m.group(2).count("b") - m.group(2).count("♭")
    return 12 * (int(m.group(3)) + 1) + pc


def midi_to_note(m: int) -> str:
    return f"{_NAMES[m % 12]}{m // 12 - 1}"


@dataclass(frozen=True)
class Tuning:
    strings: tuple[int, ...]      # open-string MIDI pitches, low → high
    name: str = ""

    @property
    def count(self) -> int:
        return len(self.strings)

    def offsets(self) -> list[int]:
        """Semitones from the standard tuning of the same string count
        (Rocksmith ``<tuning string0=…>`` style)."""
        std = STANDARD.get(self.count)
        if std is None:
            return [0] * self.count
        if self.count == 5 and self.strings[-1] - std[-1] >= 4:   # high-C 5-string
            std = PRESETS["5c"]
        return [s - r for s, r in zip(self.strings, std)]

    def label(self) -> str:
        return self.name or " ".join(midi_to_note(s) for s in self.strings)


def parse_tuning(spec: str | Sequence[int] | Tuning | None) -> Tuning:
    if isinstance(spec, Tuning):
        return spec
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return Tuning(STANDARD[4], "E standard (4)")
    if not isinstance(spec, str):
        vals = [int(v) for v in spec]
        return _validate(Tuning(tuple(vals), ""))
    key = spec.strip().lower().replace(" ", "-").replace("_", "-")
    if key in PRESETS:
        return Tuning(PRESETS[key], PRESET_LABELS.get(key, spec.strip()))
    tokens = [t for t in re.split(r"[\s,;/]+", spec.strip()) if t]
    if all(re.fullmatch(r"[+-]?\d+", t) for t in tokens):
        vals = [int(t) for t in tokens]
        if len(vals) in STANDARD and all(-12 <= v <= 12 for v in vals):
            # Rocksmith-style offsets from standard
            return _validate(Tuning(tuple(r + v for r, v in zip(STANDARD[len(vals)], vals)),
                                    "offsets " + ",".join(tokens)))
        return _validate(Tuning(tuple(vals), ""))
    vals = [note_to_midi(t) for t in tokens]
    return _validate(Tuning(tuple(vals), ""))


def _validate(t: Tuning) -> Tuning:
    s = list(t.strings)
    if len(s) < 3 or len(s) > 8:
        raise ValueError(f"expected 3–8 strings, got {len(s)}")
    if s == sorted(s, reverse=True) and s != sorted(s):
        s.reverse()  # given high → low (Guitar Pro order)
    if s != sorted(s) or len(set(s)) != len(s):
        raise ValueError(f"tuning must go strictly low→high or high→low: {t.strings}")
    if not (0 <= s[0] and s[-1] <= 80):
        raise ValueError(f"implausible bass tuning {s}")
    return Tuning(tuple(s), t.name)


# ── data ────────────────────────────────────────────────────────────────────

@dataclass
class TabNote:
    start: float
    end: float
    pitch: int            # sounding MIDI pitch (after any octave fix)
    string: int           # 0 = lowest string
    fret: int
    velocity: int = 90
    anchor: int | None = None     # index-finger fret of the hand position
    source_pitch: int | None = None  # original pitch if it had to be octave-shifted
    legato: bool = False              # no own attack (hammer-on / pull-off / slide target)
    tech: str | None = None           # "slap" | "pop" when assigned


@dataclass
class Anchor:
    time: float
    fret: int
    width: int = 4


@dataclass
class Weights:
    shift: float = 3.0           # any hand position change
    shift_per_fret: float = 1.0  # … plus per fret moved
    fast_run: float = 1.5        # multiplier when notes are < 150 ms apart
    rest_discount: float = 0.5   # multiplier when the hand has a rest ≥ 0.5 s to move
    # per fret above `high_fret`. Kept below the shift costs so that avoiding
    # jumps wins; on wide-range synthetic lines 0.5 halves the notes above the
    # 12th fret vs. ignoring it, for ~50 % more (small) shifts.
    high_per_fret: float = 0.5
    high_fret: int = 12
    per_fret: float = 0.02       # gentle pull towards low positions
    open_string: float = 0.3     # open string while the hand is at/above the 5th fret
    string_skip: float = 0.3     # per string skipped between consecutive notes
    low_stretch: float = 0.5     # using all 4 fingers in positions below the 5th fret
    reach: float = 0.05          # per fret between index finger and lowest fretted note


@dataclass
class FretResult:
    notes: list[TabNote]
    tuning: Tuning
    anchors: list[Anchor]
    warnings: list[str] = field(default_factory=list)
    cost: float = 0.0

    def stats(self) -> dict:
        return {
            "notes": len(self.notes),
            "shifts": max(0, len(self.anchors) - 1),
            "shift_frets": sum(abs(b.fret - a.fret) for a, b in zip(self.anchors, self.anchors[1:])),
            "above_12": sum(1 for n in self.notes if n.fret > 12),
            "open": sum(1 for n in self.notes if n.fret == 0),
            "max_fret": max((n.fret for n in self.notes), default=0),
        }


# ── core ────────────────────────────────────────────────────────────────────

HAND_SPAN = 4  # frets covered by one hand position (anchor .. anchor+3)


def _group(notes: Sequence, tol: float = 0.03) -> list[list]:
    """Group notes whose starts are within `tol` seconds (chords / double stops)."""
    out: list[list] = []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if out and n.start - out[-1][0].start <= tol:
            out[-1].append(n)
        else:
            out.append([n])
    return out


def _fit_pitch(p: int, tuning: Tuning, max_fret: int, mode: str) -> int | None:
    lo, hi = tuning.strings[0], tuning.strings[-1] + max_fret
    if lo <= p <= hi:
        return p
    if mode == "drop":
        return None
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p if lo <= p <= hi else None


def _fingerings(pitches: Sequence[int], tuning: Tuning, max_fret: int) -> list[tuple]:
    """All ways to play the pitches simultaneously: tuple of (string, fret),
    one per pitch, on distinct strings, fretted notes within one hand span."""
    per = [[(s, p - o) for s, o in enumerate(tuning.strings) if 0 <= p - o <= max_fret]
           for p in pitches]
    out = []
    for combo in itertools.product(*per):
        strings = [s for s, _ in combo]
        if len(set(strings)) != len(strings):
            continue
        fretted = [f for _, f in combo if f > 0]
        if fretted and max(fretted) - min(fretted) > HAND_SPAN - 1:
            continue
        out.append(combo)
    return out


def _anchors_for(fing: tuple, max_fret: int) -> list[int]:
    fretted = [f for _, f in fing if f > 0]
    lo = max(1, max(fretted) - (HAND_SPAN - 1))
    hi = min(fretted)
    return list(range(lo, min(hi, max_fret) + 1))


def _local_cost(fing: tuple, anchor: int | None, w: Weights) -> float:
    c = 0.0
    for _s, f in fing:
        if f > w.high_fret:
            c += w.high_per_fret * (f - w.high_fret)
        c += w.per_fret * f
    fretted = [f for _, f in fing if f > 0]
    if anchor is not None and fretted:
        if anchor < 5 and max(fretted) - anchor >= HAND_SPAN - 1:
            c += w.low_stretch
        c += w.reach * (min(fretted) - anchor)  # index finger on the lowest note if free
    return c


def assign_frets(notes: Sequence, tuning: str | Sequence[int] | Tuning | None = None,
                 max_fret: int = 24, out_of_range: str = "octave",
                 weights: Weights | None = None, allowed=None) -> FretResult:
    """Choose string/fret for each note.

    ``notes``: objects with ``start``, ``end``, ``pitch`` (and optional
    ``velocity``) — e.g. ``stems2midi.Note``. ``out_of_range``: "octave"
    moves unplayable notes by octaves into range, "drop" removes them.
    ``allowed``: optional ``f(note) -> set of string indexes | None`` to keep
    a note on certain strings (used for slap/pop); ignored for a note when
    no allowed string can play it.
    """
    tun = parse_tuning(tuning)
    w = weights or Weights()
    warnings: list[str] = []

    # 1) fit pitches into range, build simultaneous groups
    fixed = []
    moved = dropped = 0
    for n in notes:
        p = _fit_pitch(int(n.pitch), tun, max_fret, out_of_range)
        if p is None:
            dropped += 1
            continue
        if p != n.pitch:
            moved += 1
        fixed.append((n, p))
    if moved:
        warnings.append(f"{moved} note(s) outside the range of {tun.label()} were moved by octaves.")
    if dropped:
        warnings.append(f"{dropped} note(s) outside the range of {tun.label()} were dropped.")

    class _N:  # light wrapper so grouping works on fitted pitch
        __slots__ = ("src", "pitch", "start", "end")

        def __init__(self, src, p):
            self.src, self.pitch, self.start, self.end = src, p, float(src.start), float(src.end)

    groups = _group([_N(n, p) for n, p in fixed])

    # 2) states per group: (fingering, anchor, local_cost)
    layers: list[list[tuple]] = []
    pruned = 0
    for g in groups:
        g = list({x.pitch: x for x in g}.values())          # unison duplicates → one
        g.sort(key=lambda x: x.pitch)
        fings = _fingerings([x.pitch for x in g], tun, max_fret)
        while not fings and len(g) > 1:                     # impossible chord: drop top note
            g = g[:-1]
            pruned += 1
            fings = _fingerings([x.pitch for x in g], tun, max_fret)
        if allowed is not None:
            ok = [allowed(x.src) for x in g]
            keep = [f for f in fings if all(a is None or s in a for a, (s, _fr) in zip(ok, f))]
            fings = keep or fings
        states = []
        for f in fings:
            if any(fr > 0 for _, fr in f):
                for a in _anchors_for(f, max_fret):
                    states.append((f, a, _local_cost(f, a, w)))
            else:
                states.append((f, None, _local_cost(f, None, w)))  # open: hand stays
        layers.append((g, states))
    if pruned:
        warnings.append(f"{pruned} chord note(s) could not be fingered together and were dropped.")

    # 3) Viterbi over (fingering, anchor); an open-string state inherits the
    #    previous anchor, so it is expanded per incoming anchor.
    INF = math.inf
    prev_states: list[tuple] = []   # (fing, anchor, cost, backptr)
    history: list[list[tuple]] = []
    prev_group = None
    for li, (g, states) in enumerate(layers):
        cur: list[tuple] = []
        start = g[0].start
        if prev_group is None:
            for f, a, lc in states:
                cur.append((f, a, lc, None))
        else:
            prev_end = max(x.end for x in prev_group)
            ioi = start - prev_group[0].start
            gap = start - prev_end
            mult = 1.0
            if ioi < 0.15:
                mult = w.fast_run
            elif gap >= 0.5:
                mult = w.rest_discount
            # index best previous state per anchor for open-string expansion
            for f, a, lc in states:
                if a is None:
                    # hand stays: one state per distinct incoming anchor. An
                    # open string is free in the first positions but slightly
                    # discouraged when the hand sits higher up the neck.
                    best: dict = {}
                    for pi, (pf, pa, pc, _) in enumerate(prev_states):
                        c = pc + lc + _string_cost(pf, f, w)
                        if pa is not None and pa >= 5:
                            c += w.open_string * len(f)
                        if pa not in best or c < best[pa][0]:
                            best[pa] = (c, pi)
                    for pa, (c, pi) in best.items():
                        cur.append((f, pa, c, pi))
                    continue
                bc, bi = INF, -1
                for pi, (pf, pa, pc, _) in enumerate(prev_states):
                    c = pc + lc + _string_cost(pf, f, w)
                    if pa is not None and pa != a:
                        c += mult * (w.shift + w.shift_per_fret * abs(a - pa))
                    if c < bc:
                        bc, bi = c, pi
                cur.append((f, a, bc, bi))
        history.append(cur)
        prev_states = cur
        prev_group = g

    if not history:
        return FretResult([], tun, [], warnings, 0.0)

    # 4) backtrack
    idx = min(range(len(history[-1])), key=lambda i: history[-1][i][2])
    total = history[-1][idx][2]
    path = []
    for li in range(len(history) - 1, -1, -1):
        f, a, c, bp = history[li][idx]
        path.append((f, a))
        idx = bp if bp is not None else 0
    path.reverse()

    # 5) materialise notes + anchors (an all-open start gets anchor 1)
    out: list[TabNote] = []
    anchors: list[Anchor] = []
    first_anchor = next((a for _, a in path if a is not None), 1)
    for (g, _states), (f, a) in zip(layers, path):
        hand = a if a is not None else (anchors[-1].fret if anchors else first_anchor)
        if not anchors or anchors[-1].fret != hand:
            anchors.append(Anchor(time=g[0].start, fret=hand))
        for x, (s, fr) in zip(g, f):
            out.append(TabNote(start=x.start, end=x.end, pitch=x.pitch, string=s, fret=fr,
                               velocity=int(getattr(x.src, "velocity", 90)), anchor=hand,
                               source_pitch=(int(x.src.pitch) if x.src.pitch != x.pitch else None),
                               legato=bool(getattr(x.src, "legato", False)),
                               tech=getattr(x.src, "tech", None)))
    return FretResult(out, tun, anchors, warnings, round(total, 3))


def _string_cost(prev_f: tuple, f: tuple, w: Weights) -> float:
    if not prev_f or not f:
        return 0.0
    skip = abs(min(s for s, _ in f) - min(s for s, _ in prev_f)) - 1
    return w.string_skip * max(0, skip)


# ── tuning suggestion ───────────────────────────────────────────────────────

# A song "needs" lower strings when at least this many notes, and this share
# of all notes, sit below the lowest open string. Measured on 15 charted songs
# tracked down to B0: the one D-standard song (Korn) had 30 % of its notes in
# B0–D#1, Primus 17 %; the E-standard songs 0–6 % (octave slips / slides).
LOW_NOTES_MIN = 10
LOW_NOTES_SHARE = 0.08
LOWEST_TRACKED = 23          # B0: what the transcription listens down to for bass


def low_notes(pitches: Sequence[int], tuning) -> list[int]:
    """Pitches below the tuning's lowest open string (but not below B0)."""
    low = parse_tuning(tuning).strings[0]
    return [int(p) for p in pitches if LOWEST_TRACKED <= int(p) < low]


def suggest_tuning(pitches: Sequence[int], tuning) -> tuple[str | None, str]:
    """(preset to use instead or None, message). Suggests 5-string B standard
    (or, when nothing goes below D1, Drop D / D standard on a 4-string)."""
    tun = parse_tuning(tuning)
    pitches = list(pitches)
    below = low_notes(pitches, tun)
    if not pitches or not below:
        return None, ""
    share = len(below) / len(pitches)
    low_name = midi_to_note(tun.strings[0])
    msg = (f"{len(below)} note(s) ({share:.0%}) are below the lowest string ({low_name}), "
           f"down to {midi_to_note(min(below))}")
    if len(below) < LOW_NOTES_MIN or share < LOW_NOTES_SHARE:
        return None, msg + " — few enough to be slides or tracking slips."
    if tun.strings[0] <= PRESETS["5"][0]:
        return None, msg + "."
    alt = " (or Drop D / D standard on a 4-string)" if min(below) >= 26 and len(tun.strings) == 4 else ""
    return "5", msg + f": the song probably needs a 5-string in B standard{alt}."


# ── slap & pop ──────────────────────────────────────────────────────────────

def parse_sections(spec: str | None) -> list[tuple[float, float]]:
    """"all" → whole song; else comma-separated ranges in seconds or m:ss,
    e.g. "0:32-1:05, 130-162.5". Raises ValueError on bad input."""
    if not spec or not str(spec).strip():
        return []
    spec = str(spec).strip().lower()
    if spec in ("all", "yes", "true", "1"):
        return [(0.0, float("inf"))]

    def sec(x: str) -> float:
        x = x.strip()
        if ":" in x:
            m, s_ = x.split(":", 1)
            return int(m) * 60 + float(s_)
        return float(x)

    out = []
    for part in re.split(r"[;,]", spec):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"([\d:.]+)\s*-\s*([\d:.]+)", part)
        if not m:
            raise ValueError(f"bad section {part!r} (expected e.g. 0:32-1:05)")
        a, b = sec(m.group(1)), sec(m.group(2))
        if b <= a:
            raise ValueError(f"section {part!r} ends before it starts")
        out.append((a, b))
    return sorted(out)


# Pitch boundary between slap and pop, in semitones above the lower pop
# string's open note. Checked 0…5 on 11 charted slap songs (charters' notes
# and sections): +3 scores best (slap F1 0.85 / pop 0.78; +2: 0.80/0.78).
SLAP_SPLIT = 3


def apply_slap(res: FretResult, sections: Sequence[tuple[float, float]], max_fret: int = 24,
               weights: Weights | None = None) -> dict:
    """Mark slap/pop inside the given time ranges (the parts the player slaps).

    How slap bass is played: the thumb slaps the low strings (E/A, plus B on
    a 5-string), a finger pops the two highest (D/G). Inside a section, every
    note with its own attack up to E2 (2 semitones above the open string of
    the lower pop string — D2 in standard tuning) is a **slap**, F2 and up
    **pops**;
    hammer-ons/pull-offs/slides (no own attack) stay plain. The strings and
    frets are then chosen again with slaps kept on the low strings and pops
    on the high ones (when the note can be played there), so the hand
    positions (anchors) follow the new fingering.

    With the charters' notes and sections, 11 slap CDLCs score slap F1
    0.58–0.99 (most ≥ 0.9); see README. ``res`` is updated in place."""
    tun = res.tuning.strings
    n_low = max(1, len(tun) - 2)
    split = tun[n_low] + SLAP_SPLIT
    counts = {"slap": 0, "pop": 0}
    for n in res.notes:
        n.tech = None
        if n.legato or not any(a <= n.start < b for a, b in sections):
            continue
        n.tech = "slap" if n.pitch < split else "pop"
        counts[n.tech] += 1
    if not counts["slap"] and not counts["pop"]:
        return counts
    low, high = set(range(n_low)), set(range(n_low, len(tun)))

    def allowed(src):
        return low if src.tech == "slap" else high if src.tech == "pop" else None

    def refret(allowed_fn):
        new = assign_frets(res.notes, res.tuning, max_fret=max_fret, weights=weights, allowed=allowed_fn)
        if len(new.notes) != len(res.notes):
            return
        for old, nn in zip(res.notes, new.notes):
            nn.source_pitch = old.source_pitch
            nn.velocity = old.velocity
        res.notes[:] = new.notes
        res.anchors[:] = new.anchors
        res.cost = new.cost

    refret(allowed)

    # Octave pops: a pop an octave above a slap just played goes two strings
    # up and two frets higher — the octave shape (slap E-3 → pop D-5, slap
    # A-5 → pop G-7). Charters do this for 99 % of such pops (395/398 in 11
    # slap songs); the plain fretting got 84 %.
    want: dict[int, int] = {}
    for i, n in enumerate(res.notes):
        if n.tech != "pop":
            continue
        for j in range(i - 1, max(-1, i - 5), -1):
            m = res.notes[j]
            if n.start - m.start > OCTAVE_POP_WINDOW_S:
                break
            if m.tech == "slap" and n.pitch == m.pitch + 12:
                s2, f2 = m.string + 2, m.fret + 2
                if (s2 < len(tun) and tun[s2] - tun[m.string] == 10 and 0 < f2 <= max_fret
                        and (j not in want or want[j] == m.string)):
                    want[i], want[j] = s2, m.string
                break
    if want:
        for i, n in enumerate(res.notes):
            n._octave_string = want.get(i)
        refret(lambda src: ({src._octave_string} if getattr(src, "_octave_string", None) is not None
                            else allowed(src)))
        for n in res.notes:
            if hasattr(n, "_octave_string"):
                del n._octave_string
    return counts


OCTAVE_POP_WINDOW_S = 1.0


# ── output helpers ──────────────────────────────────────────────────────────

def to_ascii_tab(res: FretResult, bpm: float = 120.0, per_beat: int = 4,
                 beats_per_bar: int = 4, bars_per_line: int = 4, t0: float | None = None) -> str:
    """Plain-text tab for eyeballing the result (16th-note grid by default).
    Bars start at ``t0`` — by default the first note, which is usually a
    downbeat; pass the real first downbeat when it is known."""
    if not res.notes:
        return "(no notes)\n"
    if t0 is None:
        t0 = min(n.start for n in res.notes)
    step = 60.0 / bpm / per_beat
    cols_per_bar = per_beat * beats_per_bar
    n_cols = int(math.ceil((max(n.end for n in res.notes) - t0) / step)) + 1
    n_cols = int(math.ceil(n_cols / cols_per_bar) * cols_per_bar)
    grid = [[""] * n_cols for _ in res.tuning.strings]
    tech_row = [""] * n_cols
    for n in res.notes:
        c = max(0, int(round((n.start - t0) / step)))
        while c < n_cols and grid[n.string][c]:
            c += 1
        if c < n_cols:
            grid[n.string][c] = str(n.fret)
            if n.tech:
                tech_row[c] = {"slap": "S", "pop": "P"}.get(n.tech, "")
            elif n.legato:
                tech_row[c] = "h"
    names = [midi_to_note(p).rstrip("-0123456789") for p in res.tuning.strings]
    width = max(len(x) for x in names)
    lines = []
    bars = n_cols // cols_per_bar
    for b0 in range(0, bars, bars_per_line):
        for s in range(len(grid) - 1, -1, -1):  # highest string on top
            row = names[s].ljust(width) + "|"
            for b in range(b0, min(bars, b0 + bars_per_line)):
                for c in range(b * cols_per_bar, (b + 1) * cols_per_bar):
                    cell = grid[s][c]
                    colw = max(len(grid[k][c]) for k in range(len(grid))) or 1
                    row += (cell or "-").ljust(colw, "-") + "-"
                row += "|"
            lines.append(row)
        if any(tech_row):
            row = " " * width + " "
            for b in range(b0, min(bars, b0 + bars_per_line)):
                for c in range(b * cols_per_bar, (b + 1) * cols_per_bar):
                    colw = max(len(grid[k][c]) for k in range(len(grid))) or 1
                    row += (tech_row[c] or " ").ljust(colw) + " "
                row += " "
            lines.append(row.rstrip())
        lines.append("")
    head = f"Tuning: {res.tuning.label()}   Tempo: {bpm:g} BPM   (1 column = 1/{per_beat * 4} note)\n"
    if any(n.tech or n.legato for n in res.notes):
        head += "Under the tab: S = slap, P = pop, h = hammer-on/pull-off/slide (no own attack)\n"
    head += "\n"
    return head + "\n".join(lines)
