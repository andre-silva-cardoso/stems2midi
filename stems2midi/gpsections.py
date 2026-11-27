"""One example of each part, written in Guitar Pro, applied across the song
(``--sections-from``).

In the Guitar Pro file (bass track, .gp3/.gp4/.gp5; .gpx/.gp also read) you
write each *different* part once — a 2-bar riff is enough when the verse just
repeats it — and start each with a marker (Section → Marker) naming it:
intro, verse, chorus, bridge… (Portuguese names work too: refrão, verso,
ponte…). Order and tempo don't matter.

Each example is slid over the song beat by beat on the song's beat grid and
compared with the bass stem: do its notes fall on the stem's attacks, and does
the stem sound their pitches; are the stem's attacks there covered by it.
The best set of non-overlapping placements (dynamic programming over the
beats) is kept; inside a placement the example's notes — strings, frets,
slap/pop, hammer-ons — replace the built-in transcription, everywhere else
the transcription stays. A riff found 8 times in a row fills a verse; a
chorus with another ending can be written as a second "chorus" example and
the one that fits better wins. The tempo of the example doesn't matter (only
its beats); half / double time against the song's beat grid is tried.

The marker names become the Rocksmith sections (unless ``--parts`` is given).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

SR = 11025
HOP = 128
HIT_TOL = 0.07          # s: a note "on an attack" when an attack is this close
SAL_OK = 1.5            # dB over the neighbouring semitones: the stem sounds the pitch
SAL_SPREAD = 1.5
MIN_SCORE = 0.55        # placements scoring under this are not used
RECALL_W = 0.4          # weight of "covers the stem's attacks" in the score
RATIOS = (1.0, 0.5, 2.0)  # song beats per example beat

PT_NAMES = {"refrao": "chorus", "coro": "chorus", "verso": "verse", "estrofe": "verse", "ponte": "bridge",
            "introducao": "intro", "entrada": "intro", "final": "outro", "fim": "outro", "saida": "outro",
            "prerefrao": "prechorus", "posrefrao": "postchorus", "interludio": "interlude",
            "pre-chorus": "prechorus", "prechorus": "prechorus"}


@dataclass
class Example:
    name: str                       # Rocksmith section name
    label: str                      # the marker as written
    beats: int                      # length in the file's beats
    notes: list = field(default_factory=list)   # {"b", "db", "m", "s"?, "f"?, "tech"?, "lg"?, "dead"?}


def section_name(marker: str, allowed=None) -> str:
    """Marker → Rocksmith section name: "Verse 2" → verse, "Refrão" → chorus,
    unknown → riff."""
    t = unicodedata.normalize("NFKD", str(marker or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)          # "verse 2 (A)": the letter isn't the name
    t = re.sub(r"[#\d]+", " ", t)
    t = re.sub(r"[^a-z-]+", "", t)
    t = PT_NAMES.get(t, PT_NAMES.get(t.replace("-", ""), t.replace("-", "")))
    if allowed is not None and t not in allowed:
        return "riff"
    return t or "riff"


def _beat_pos(t, bt: np.ndarray) -> np.ndarray:
    """Time → position in beats on the beat times ``bt`` (extrapolated)."""
    t = np.asarray(t, float)
    idx = np.arange(len(bt), dtype=float)
    pos = np.interp(t, bt, idx)
    step0 = bt[1] - bt[0] if len(bt) > 1 else 0.5
    step1 = bt[-1] - bt[-2] if len(bt) > 1 else 0.5
    pos = np.where(t < bt[0], (t - bt[0]) / step0, pos)
    return np.where(t > bt[-1], len(bt) - 1 + (t - bt[-1]) / step1, pos)


def _beat_time(pos, bt: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, float)
    idx = np.arange(len(bt), dtype=float)
    t = np.interp(pos, idx, bt)
    step0 = bt[1] - bt[0] if len(bt) > 1 else 0.5
    step1 = bt[-1] - bt[-2] if len(bt) > 1 else 0.5
    t = np.where(pos < 0, bt[0] + pos * step0, t)
    return np.where(pos > len(bt) - 1, bt[-1] + (pos - len(bt) + 1) * step1, t)


def examples_from(raw: list[tuple], beats: list[tuple[float, int]], marks: list[tuple[str, float]],
                  allowed=None) -> list[Example]:
    """Split a Guitar Pro reading (notesimport.read_gp: notes in seconds, its
    ebeats and marker sections) into examples, in beats."""
    bt = np.array([b for b, _m in beats], float)
    downs = [i for i, (_b, m) in enumerate(beats) if m >= 0] or list(range(0, len(beats), 4))
    if len(bt) < 2:
        raise ValueError("the Guitar Pro file has no beats")
    end_t = max([bt[-1] + (bt[-1] - bt[-2])] + [r[0] + r[1] for r in raw])
    marks = sorted(marks, key=lambda x: x[1])
    starts = [(m, t) for m, t in marks]
    first_note = min(r[0] for r in raw) if raw else 0.0
    if not starts or first_note < starts[0][1] - 0.01:
        starts.insert(0, ("riff", min(first_note, starts[0][1] if starts else first_note)))
    out = []
    for i, (label, a) in enumerate(starts):
        b = starts[i + 1][1] if i + 1 < len(starts) else end_t
        pa, pb = float(_beat_pos(a, bt)), float(_beat_pos(b, bt))
        b0 = round(pa)
        length = int(round(pb - b0))
        # empty bars at the end of the block (the file's last bars, rests) are not part of it
        inside = [r for r in raw if a - 0.01 <= r[0] < b - 0.01]
        if inside:
            last = max(float(_beat_pos(r[0] + max(r[1], 0.01), bt)) for r in inside) - b0
            bar_ends = [k - b0 for k in downs if k - b0 >= last - 0.05] + [length]
            length = int(min(length, min(bar_ends)))
        notes = []
        for r in raw:
            t, d, m = r[0], r[1], r[2]
            if not (a - 0.01 <= t < b - 0.01):
                continue
            x = r[3] if len(r) > 3 else {}
            pos = float(_beat_pos(t, bt)) - b0
            db = float(_beat_pos(t + d, bt)) - b0 - pos
            notes.append({"b": round(pos, 4), "db": round(max(db, 0.05), 4), "m": int(m),
                          **{k: x[k] for k in ("s", "f", "tech", "lg", "dead") if k in x}})
        if length > 0 and any(not n.get("dead") for n in notes):
            out.append(Example(section_name(label, allowed), label or "riff", length, notes))
    if not out:
        raise ValueError("no notes in the Guitar Pro file's sections")
    return out


# ── evidence from the stem ──────────────────────────────────────────────

class Stem:
    """Attacks and pitch salience of the bass stem (mono)."""

    def __init__(self, y: np.ndarray, sr: int, onsets: np.ndarray | None = None):
        import librosa
        y = np.asarray(y, np.float32)
        if y.ndim > 1:
            y = y.mean(axis=0)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        self.duration = len(y) / SR
        if onsets is None:
            import stems2midi as _s2m          # noqa: PLC0415 — sibling module
            onsets = _s2m.stem_onsets(y, SR)
        self.onsets = np.sort(np.asarray(onsets, float))
        C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(21), n_bins=72))
        D = librosa.amplitude_to_db(C + 1e-9, ref=np.max)               # rows: MIDI 21 … 92
        # a pitch "sounds" when it or its harmonics stand out from the notes a tone around
        H = np.full((72, D.shape[1]), -80.0, np.float32)
        for m in range(72):
            rows = [m + k for k in (0, 12, 19, 24) if m + k < 72]
            H[m] = D[rows].max(axis=0)
        pad = np.pad(H, ((2, 2), (0, 0)), mode="edge")
        around = (pad[0:72] + pad[1:73] + pad[3:75] + pad[4:76]) / 4.0
        S = H - around
        # mean over the 8 frames (~90 ms) after each frame
        cs = np.cumsum(np.pad(S, ((0, 0), (1, 0))), axis=1)
        w = 8
        n = S.shape[1]
        hi = np.minimum(np.arange(n) + 1 + w, n)
        lo = np.minimum(np.arange(n) + 1, n)
        self.sal = (cs[:, hi] - cs[:, lo]) / np.maximum(hi - lo, 1)

    def hits(self, times: np.ndarray) -> np.ndarray:
        on = self.onsets
        if not len(on):
            return np.zeros(times.shape, bool)
        i = np.clip(np.searchsorted(on, times), 1, len(on) - 1)
        d = np.minimum(np.abs(on[i] - times), np.abs(on[i - 1] - times))
        return d <= HIT_TOL

    def salience(self, times: np.ndarray, midi: np.ndarray) -> np.ndarray:
        f = np.clip((times * SR / HOP).astype(int) + 1, 0, self.sal.shape[1] - 1)
        r = np.clip(midi.astype(int) - 21, 0, 71)
        return self.sal[r, f]


def song_grid(beats, bpm: float, duration: float, onsets: np.ndarray) -> np.ndarray:
    """Beat times covering the song: the tracked beats, extended at both ends;
    or (no tracked beats) a constant grid at ``bpm`` phased to the attacks."""
    b = np.asarray(beats if beats is not None else [], float)
    if len(b) >= 8:
        step = float(np.median(np.diff(b)))
        pre = np.arange(b[0] - step, -step, -step)[::-1]
        post = np.arange(b[-1] + step, duration + step, step)
        return np.concatenate([pre, b, post])
    step = 60.0 / max(bpm or 120.0, 1.0)
    best, phase = -1, 0.0
    for ph in np.arange(0, step, 0.01):
        grid = np.arange(ph, duration + step, step)
        if len(onsets):
            i = np.clip(np.searchsorted(grid, onsets), 1, len(grid) - 1)
            d = np.minimum(np.abs(grid[i] - onsets), np.abs(grid[i - 1] - onsets))
            n = int((d < 0.04).sum())
        else:
            n = 0
        if n > best:
            best, phase = n, ph
    return np.arange(phase, duration + step, step)


# ── placing the examples ────────────────────────────────────────────────

def score_starts(ex: Example, ratio: float, grid: np.ndarray, stem: Stem) -> np.ndarray:
    """Score (0..1) of the example placed at each song beat."""
    span = ex.beats * ratio
    n_start = int(len(grid) - np.ceil(span))
    if n_start <= 0:
        return np.zeros(0)
    ks = np.arange(n_start, dtype=float)[:, None]
    b = np.array([n["b"] for n in ex.notes], float)[None, :] * ratio
    m = np.array([n["m"] for n in ex.notes], float)
    dead = np.array([bool(n.get("dead")) for n in ex.notes])
    times = _beat_time(ks + b, grid)                                       # (starts, notes)
    hit = stem.hits(times)
    sal = stem.salience(times, np.broadcast_to(m, times.shape))
    pitch = 1.0 / (1.0 + np.exp(-(sal - SAL_OK) / SAL_SPREAD * 2))
    pitch[:, dead] = 1.0
    precision = (hit * (0.35 + 0.65 * pitch)).mean(axis=1)
    # the stem's attacks inside the placement that one of its notes covers
    t0 = _beat_time(ks[:, 0], grid)
    t1 = _beat_time(ks[:, 0] + span, grid)
    on = stem.onsets
    recall = np.ones(n_start)
    if len(on):
        count = np.searchsorted(on, t1 - 0.03) - np.searchsorted(on, t0 - 0.03)
        recall = np.where(count > 0, np.minimum(1.0, hit.sum(axis=1) / np.maximum(count, 1)), 1.0)
    # its notes must fall on attacks with the right pitch; covering the stem's other
    # attacks counts less (the stem also has ghost notes, drum bleed…)
    s = precision * (1 - RECALL_W + RECALL_W * recall)
    s[t1 > stem.duration + 0.5] = 0.0
    return s


@dataclass
class Placement:
    ex: int
    start: int            # song beat
    span: int             # song beats
    score: float


def choose(examples: list[Example], grid: np.ndarray, stem: Stem, min_score: float = MIN_SCORE,
           ratios=RATIOS) -> tuple[list[Placement], float, float]:
    """Best non-overlapping placements (one tempo ratio for the whole song):
    (placements, ratio, value = Σ score × beats)."""
    best_total, best_sel, best_ratio = -1.0, [], 1.0
    nb = len(grid)
    for r in ratios:
        if any(abs(e.beats * r - round(e.beats * r)) > 1e-6 for e in examples):
            continue
        scores = [score_starts(e, r, grid, stem) for e in examples]
        spans = [int(round(e.beats * r)) for e in examples]
        val = np.zeros(nb + 1)
        pick: list = [None] * (nb + 1)
        for k in range(nb - 1, -1, -1):
            val[k], pick[k] = val[k + 1], None
            for j, sc in enumerate(scores):
                if k < len(sc) and sc[k] >= min_score and k + spans[j] <= nb:
                    v = sc[k] * spans[j] + val[k + spans[j]]
                    if v > val[k] + 1e-9:
                        val[k], pick[k] = v, j
        sel, k = [], 0
        while k < nb:
            j = pick[k]
            if j is None:
                k += 1
                continue
            sel.append(Placement(j, k, spans[j], float(scores[j][k])))
            k += spans[j]
        if val[0] > best_total:
            best_total, best_sel, best_ratio = float(val[0]), sel, r
    return best_sel, best_ratio, max(best_total, 0.0)


def constant_grid(bpm: float, duration: float, onsets: np.ndarray) -> np.ndarray:
    return song_grid(None, bpm, duration, onsets)


def candidate_bpms(tracked_bpm: float, gp_bpm: float) -> list[float]:
    """Tempos to try for a constant grid: the tracked one and the file's, ×½ ×1 ×2."""
    out = []
    for b in (tracked_bpm, gp_bpm):
        for f in (0.5, 1.0, 2.0):
            v = (b or 0) * f
            if 50 <= v <= 240 and not any(abs(v / o - 1) < 0.02 for o in out):
                out.append(v)
    return out


def best_grid(examples: list[Example], stem: Stem, beats, bpm: float, gp_bpm: float,
              min_score: float = MIN_SCORE):
    """The beat grid the examples fit best: the tracked beats (with ½ / 2
    ratios) or a constant tempo near the tracked / the file's tempo (±6 %,
    then finer). Returns (grid, placements, ratio, value, label)."""
    best = None
    if beats is not None and len(beats) >= 8:
        g = song_grid(beats, bpm, stem.duration, stem.onsets)
        sel, r, v = choose(examples, g, stem, min_score)
        best = (g, sel, r, v, f"tracked beats ({bpm:.1f} BPM)")
    for base in candidate_bpms(bpm, gp_bpm):
        coarse = [base * f for f in np.arange(0.94, 1.0601, 0.01)]
        scored = []
        for b in coarse:
            g = constant_grid(b, stem.duration, stem.onsets)
            sel, r, v = choose(examples, g, stem, min_score, ratios=(1.0,))
            scored.append((v, b, g, sel, r))
        v0, b0 = max(scored, key=lambda x: x[0])[:2]
        for b in [b0 * f for f in np.arange(0.994, 1.0061, 0.002)]:
            g = constant_grid(b, stem.duration, stem.onsets)
            sel, r, v = choose(examples, g, stem, min_score, ratios=(1.0,))
            scored.append((v, b, g, sel, r))
        v, b, g, sel, r = max(scored, key=lambda x: x[0])
        if best is None or v > best[3] * 1.02:
            best = (g, sel, r, v, f"constant {b:.2f} BPM")
    return best


def apply(examples: list[Example], placements: list[Placement], ratio: float, grid: np.ndarray) -> dict:
    """Notes (song time) and occurrences of the chosen placements."""
    notes, ranges, occ = [], [], []
    for p in placements:
        ex = examples[p.ex]
        t0 = float(_beat_time(p.start, grid))
        t1 = float(_beat_time(p.start + p.span, grid))
        ranges.append((t0, t1))
        if occ and occ[-1]["ex"] == p.ex and abs(occ[-1]["end"] - t0) < 0.05:
            occ[-1]["end"] = t1                       # the same riff again: one occurrence
            occ[-1]["times"] += 1
        else:
            occ.append({"ex": p.ex, "name": ex.name, "label": ex.label, "start": t0, "end": t1, "times": 1})
        for n in ex.notes:
            if n.get("dead"):
                continue
            a = float(_beat_time(p.start + n["b"] * ratio, grid))
            b = float(_beat_time(p.start + (n["b"] + n["db"]) * ratio, grid))
            if a >= t1 - 0.01:
                continue
            notes.append({"t": a, "d": max(0.03, min(b, t1) - a), "m": n["m"],
                          **{k: n[k] for k in ("s", "tech", "lg") if k in n}})
    return {"notes": sorted(notes, key=lambda x: x["t"]), "ranges": ranges, "occurrences": occ}


def attack_offset(times, onsets: np.ndarray, tol: float = HIT_TOL) -> float:
    """Median distance from the placed notes to their attacks (0 when few hit)."""
    t = np.asarray(times, float)
    if len(t) < 8 or len(onsets) < 8:
        return 0.0
    i = np.clip(np.searchsorted(onsets, t), 1, len(onsets) - 1)
    a, b = onsets[i - 1] - t, onsets[i] - t
    d = np.where(np.abs(a) < np.abs(b), a, b)
    d = d[np.abs(d) <= tol]
    if len(d) < max(8, 0.3 * len(t)):
        return 0.0
    return round(float(np.median(d)), 4)


def parts_spec(occ: list[dict]) -> str:
    """--parts spec: the first occurrence of each section name."""
    seen, out = set(), []
    for o in occ:
        if o["name"] in seen:
            continue
        seen.add(o["name"])
        out.append(f"{o['name']}={o['start']:.2f}-{o['end']:.2f}")
    return ", ".join(out)


def _mmss(t: float) -> str:
    m, s = divmod(int(round(max(0.0, t))), 60)
    return f"{m}:{s:02d}"


def report(examples: list[Example], occ: list[dict], duration: float) -> str:
    covered = sum(o["end"] - o["start"] for o in occ)
    lines = [f"Examples placed over {covered / max(duration, 1e-9):.0%} of the song:"]
    for o in occ:
        times = f" ×{o['times']}" if o["times"] > 1 else ""
        lines.append(f"  {_mmss(o['start'])}-{_mmss(o['end'])}  {o['label']}{times}")
    unused = [e.label for i, e in enumerate(examples) if not any(o["ex"] == i for o in occ)]
    if unused:
        lines.append("  not found in the song: " + ", ".join(unused))
    return "\n".join(lines)


def place(raw, gp_info: dict, y: np.ndarray, sr: int, beats, bpm: float, allowed=None,
          onsets: np.ndarray | None = None, min_score: float = MIN_SCORE) -> dict:
    """Everything: examples from a read_gp result, placed on the song.
    Returns {"notes", "ranges", "occurrences", "examples", "ratio", "report", "parts"}."""
    examples = examples_from(raw, gp_info.get("beats") or [], gp_info.get("sections") or [], allowed)
    stem = Stem(y, sr, onsets)
    fb = np.array([b for b, _m in gp_info.get("beats") or []], float)
    gp_bpm = 60.0 / float(np.median(np.diff(fb))) if len(fb) > 2 else 0.0
    grid, sel, ratio, value, how = best_grid(examples, stem, beats, bpm, gp_bpm, min_score)
    res = apply(examples, sel, ratio, grid)
    res["grid"] = how
    shift = attack_offset([n["t"] for n in res["notes"]], stem.onsets)
    if shift:          # the beat grid sits a little before/after the attacks
        for n in res["notes"]:
            n["t"] = round(n["t"] + shift, 4)
    res["shift"] = shift
    res.update({"examples": examples, "ratio": ratio, "placements": sel,
                "report": report(examples, res["occurrences"], stem.duration),
                "parts": parts_spec(res["occurrences"])})
    return res
