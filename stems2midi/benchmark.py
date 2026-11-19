#!/usr/bin/env python3
"""benchmark — measure stems2midi against charts you already finished.

Each song folder (e.g. ``myCDLCs/Work/<Artist - Song>/``) should hold:

* the finished Rocksmith bass XML (``PART REAL_BASS_RS2.xml`` from EOF) — the
  ground truth, already synced to…
* …the audio EOF used (``guitar.ogg``; may carry EOF's lead silence — that's
  fine, the XML times are relative to this very file);
* optionally (with ``--baseline``) a raw AnthemScore MusicXML export
  (``*.musicxml.xml``) to compare against the current workflow.

For every song it runs (or reuses) the stems2midi pipeline on the audio and
reports: note precision/recall/F1 (onset ±50 ms, exact pitch), octave-only
mistakes, onset error, sustain agreement, string/fret agreement with your
own fingering, beat-tracking accuracy vs the chart's beat map, and the
techniques your chart uses (slap/pop/slides…) that the transcriber does not
yet detect.

    python benchmark.py "C:\\...\\myCDLCs\\Work"                   # separate + transcribe + score
    python benchmark.py WORK --only Delfins --model htdemucs_ft
    python benchmark.py WORK --reuse                              # re-score existing outputs
    python benchmark.py WORK --skip-separation                    # transcribe the full mix (baseline)

Results: ``<out>/report.md`` and ``<out>/report.json`` (default out =
``<WORK>/_stems2midi_benchmark``).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))



# ── ground truth: Rocksmith XML ─────────────────────────────────────────────

@dataclass
class GtNote:
    t: float
    sustain: float
    string: int
    fret: int
    pitch: int
    tech: tuple = ()


@dataclass
class Chart:
    notes: list[GtNote]
    beats: list[float]
    downbeats: list[float]
    strings: list[int]          # open-string MIDI pitches, low → high
    offset: float               # <offset> (negative lead-in), seconds
    start_beat: float
    tempo: float
    anchors: list[tuple] = field(default_factory=list)   # (time, fret)
    title: str = ""


_TECH_ATTRS = {
    "slap": "slap", "pluck": "pop", "hammerOn": "hammer-on", "pullOff": "pull-off",
    "palmMute": "palm mute", "mute": "fret-hand mute", "harmonic": "harmonic",
    "vibrato": "vibrato", "tremolo": "tremolo", "accent": "accent", "tap": "tap",
}


def _f(el, name, default=0.0):
    try:
        return float(el.get(name, default))
    except (TypeError, ValueError):
        return default


def _hardest_notes(root) -> tuple[list, list]:
    """Notes of the full-difficulty chart.

    Charts with dynamic difficulty (DLC Builder/EOF "DD") store each phrase
    at several levels; the real chart is, per phrase iteration, the notes of
    level = that phrase's maxDifficulty. Single-level charts: that level."""
    levels = {int(lv.get("difficulty", 0)): lv for lv in root.iter("level")}
    if not levels:
        return [], []

    def collect(lv, t0=-1e9, t1=1e9):
        out = [(n, _f(n, "time")) for n in lv.iterfind("./notes/note") if t0 <= _f(n, "time") < t1]
        for c in lv.iterfind("./chords/chord"):
            t = _f(c, "time")
            if t0 <= t < t1:
                out.extend((cn, t) for cn in c.iterfind("chordNote"))
        anc = [a for a in lv.iterfind("./anchors/anchor") if t0 <= _f(a, "time") < t1]
        return out, anc

    if len(levels) == 1:
        return collect(next(iter(levels.values())))
    phrases = [int(p.get("maxDifficulty", 0)) for p in root.iterfind("./phrases/phrase")]
    iters = [(_f(pi, "time"), int(pi.get("phraseId", 0))) for pi in root.iterfind("./phraseIterations/phraseIteration")]
    if not iters or not phrases:
        return collect(levels[max(levels)])
    raw, anchors = [], []
    avail = sorted(levels)
    for k, (t0, pid) in enumerate(iters):
        t1 = iters[k + 1][0] if k + 1 < len(iters) else 1e9
        want = phrases[pid] if 0 <= pid < len(phrases) else max(avail)
        lv = levels[max([d for d in avail if d <= want] or [avail[0]])]
        r, a = collect(lv, t0 if k else -1e9, t1)
        raw += r
        anchors += a
    return raw, anchors


def parse_rs_xml(path: str | Path) -> Chart:
    root = ET.parse(str(path)).getroot()
    tun = root.find("tuning")
    offs = [int(tun.get(f"string{i}", 0)) for i in range(6)] if tun is not None else [0] * 6
    cent = float(root.findtext("centOffset") or 0)
    capo = int(float(root.findtext("capo") or 0))
    raw, anchor_els = _hardest_notes(root)
    max_string = max((int(n.get("string", 0)) for n, _ in raw), default=3)
    n_strings = max(4, max_string + 1)
    std = {4: [28, 33, 38, 43], 5: [23, 28, 33, 38, 43], 6: [23, 28, 33, 38, 43, 48]}[min(6, n_strings)]
    strings = [p + o + capo + round(cent / 100) for p, o in zip(std, offs)]
    notes = []
    for n, t in raw:
        s, fr = int(n.get("string", 0)), int(n.get("fret", 0))
        tech = [lab for attr, lab in _TECH_ATTRS.items() if n.get(attr) not in (None, "0", "-1")]
        if n.get("slideTo") not in (None, "-1"):
            tech.append("slide")
        if n.get("slideUnpitchTo") not in (None, "-1"):
            tech.append("unpitched slide")
        if n.get("bend") not in (None, "0", "0.0"):
            tech.append("bend")
        notes.append(GtNote(t, _f(n, "sustain"), s, fr, strings[s] + fr, tuple(tech)))
    notes.sort(key=lambda x: (x.t, x.pitch))
    beats, downbeats = [], []
    for b in root.iterfind("./ebeats/ebeat"):
        t = _f(b, "time")
        beats.append(t)
        if b.get("measure") not in (None, "-1"):
            downbeats.append(t)
    anchors = [(_f(a, "time"), int(a.get("fret", 0))) for a in anchor_els]
    return Chart(notes, beats, downbeats, strings, float(root.findtext("offset") or 0),
                 float(root.findtext("startBeat") or 0), float(root.findtext("averageTempo") or 0),
                 anchors, (root.findtext("title") or "").strip())


# ── baseline: AnthemScore MusicXML ──────────────────────────────────────────

_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_musicxml(path: str | Path) -> list[tuple[float, float, int]]:
    """(start_s, dur_s, midi) for every sounding note of the first part.
    Handles divisions, tempo changes, chords, backup/forward, ties and
    octave-transposing clefs (bass is often written an octave up)."""
    root = ET.parse(str(path)).getroot()
    part = root.find("part")
    if part is None:
        return []
    divisions, tempo, octave_change = 1.0, 120.0, 0
    t_beats = 0.0          # position in quarter notes
    sec_at = [(0.0, 0.0, 120.0)]   # (beat, seconds, tempo) breakpoints
    out, open_ties = [], {}

    def to_sec(beat):
        b0, s0, tp = [x for x in sec_at if x[0] <= beat + 1e-9][-1]   # the latest change (ties: last)
        return s0 + (beat - b0) * 60.0 / tp

    for meas in part.iterfind("measure"):
        last_start = t_beats
        for el in meas:
            tag = el.tag
            if tag == "attributes":
                d = el.findtext("divisions")
                if d:
                    divisions = float(d)
                oc = el.findtext("./transpose/octave-change")
                if oc:
                    octave_change = int(oc)
            elif tag in ("direction", "sound"):
                snd = el if tag == "sound" else el.find(".//sound")
                bpm = snd.get("tempo") if snd is not None else None
                if not bpm and tag == "direction":      # only a metronome mark (AnthemScore)
                    pm = el.findtext(".//metronome/per-minute")
                    unit = el.findtext(".//metronome/beat-unit") or "quarter"
                    if pm:
                        try:
                            bpm = float(pm) * {"half": 2.0, "eighth": 0.5}.get(unit, 1.0)
                        except ValueError:
                            bpm = None
                if bpm:
                    tempo = float(bpm)
                    sec_at.append((t_beats, to_sec(t_beats), tempo))
            elif tag == "backup":
                t_beats -= float(el.findtext("duration") or 0) / divisions
            elif tag == "forward":
                t_beats += float(el.findtext("duration") or 0) / divisions
            elif tag == "note":
                if el.find("grace") is not None:
                    continue
                dur = float(el.findtext("duration") or 0) / divisions
                is_chord = el.find("chord") is not None
                start = last_start if is_chord else t_beats
                if not is_chord:
                    last_start = t_beats
                    t_beats += dur
                p = el.find("pitch")
                if p is None or el.find("rest") is not None:
                    continue
                midi = 12 * (int(p.findtext("octave")) + 1) + _STEP[p.findtext("step")] \
                    + int(float(p.findtext("alter") or 0)) + 12 * octave_change
                ties = {t.get("type") for t in el.iterfind("tie")}
                if "stop" in ties and midi in open_ties:
                    idx = open_ties.pop(midi)
                    s0, d0, m0 = out[idx]
                    out[idx] = (s0, to_sec(start + dur) - s0, m0)
                    if "start" in ties:
                        open_ties[midi] = idx
                    continue
                s = to_sec(start)
                out.append((s, to_sec(start + dur) - s, midi))
                if "start" in ties:
                    open_ties[midi] = len(out) - 1
    return sorted(out)


def best_offset(gt_t: list[float], pred_t: list[float], search: float = 20.0, tol: float = 0.05) -> float:
    """Time shift that makes the most predicted onsets land on chart onsets
    (AnthemScore exports start at 0 of the stem, the chart at EOF's audio)."""
    if not gt_t or not pred_t:
        return 0.0
    import bisect
    diffs = []
    for p in pred_t[:400]:
        lo = bisect.bisect_left(gt_t, p - search)
        hi = bisect.bisect_right(gt_t, p + search)
        diffs.extend(g - p for g in gt_t[lo:hi])
    if not diffs:
        return 0.0
    step = tol / 2
    hist: dict[int, int] = {}
    for d in diffs:
        k = round(d / step)
        hist[k] = hist.get(k, 0) + 1
    k = max(hist, key=lambda k: hist[k] + hist.get(k - 1, 0) + hist.get(k + 1, 0))
    near = [d for d in diffs if abs(d - k * step) <= step * 1.5]
    return statistics.median(near) if near else k * step


# ── scoring ─────────────────────────────────────────────────────────────────

def match(gt: list[tuple], pred: list[tuple], tol: float, same_pitch) -> list[tuple[int, int]]:
    """One-to-one onset matching. gt/pred items: (time, pitch, ...).
    Greedy on |Δt| among pairs that satisfy `same_pitch`."""
    import bisect
    pt = [p[0] for p in pred]
    cands = []
    for i, g in enumerate(gt):
        lo, hi = bisect.bisect_left(pt, g[0] - tol), bisect.bisect_right(pt, g[0] + tol)
        for j in range(lo, hi):
            if same_pitch(g[1], pred[j][1]):
                cands.append((abs(pred[j][0] - g[0]), i, j))
    cands.sort()
    used_g, used_p, pairs = set(), set(), []
    for _d, i, j in cands:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        pairs.append((i, j))
    return pairs


def align_sequences(gt: list[tuple], pred: list[tuple], max_dt: float = 0.3) -> list[tuple[int, int]]:
    """Order-preserving alignment of two note sequences (longest common
    subsequence with a time window).

    Independent of the chart's sync: a pair may match only if the pitches are
    equal and the onsets are within `max_dt`, and matches can't cross — so a
    chart that drifts ±100 ms or more is still compared note for note.
    gt/pred items: (time, pitch, ...), sorted by time. Returns (i, j) pairs."""
    import bisect
    import numpy as np
    n, m = len(gt), len(pred)
    if not n or not m:
        return []
    pt = [p[0] for p in pred]
    L = np.zeros((n + 1, m + 1), dtype=np.int32)
    ok = []
    for i in range(1, n + 1):
        t, pitch = gt[i - 1][0], gt[i - 1][1]
        lo, hi = bisect.bisect_left(pt, t - max_dt), bisect.bisect_right(pt, t + max_dt)
        js = [j + 1 for j in range(lo, hi) if pred[j][1] == pitch]
        ok.append(set(js))
        temp = L[i - 1].copy()
        for jj in js:
            v = L[i - 1, jj - 1] + 1
            if v > temp[jj]:
                temp[jj] = v
        L[i] = np.maximum.accumulate(temp)
    pairs, i, jj = [], n, m
    while i > 0 and jj > 0:
        if jj in ok[i - 1] and L[i, jj] == L[i - 1, jj - 1] + 1:
            pairs.append((i - 1, jj - 1))
            i, jj = i - 1, jj - 1
        elif L[i - 1, jj] == L[i, jj]:
            i -= 1
        else:
            jj -= 1
    pairs.reverse()
    return pairs


def _tech_scores(chart: Chart, gt: list, pred_sorted: list, pairs: list) -> dict | None:
    """Slap/pop agreement on correctly transcribed notes (aligned pairs)."""
    if not any(p.get("tech") for p in pred_sorted) and not any(
            t in n.tech for n in chart.notes for t in ("slap", "pop")):
        return None
    out = {}
    for lab in ("slap", "pop"):
        tp = fp = fn = 0
        for i, j in pairs:
            g = lab in chart.notes[gt[i][2]].tech
            p = pred_sorted[j].get("tech") == lab
            tp += g and p
            fp += p and not g
            fn += g and not p
        out[lab] = prf(tp, tp + fn, tp + fp)
    return out


def chart_slap_sections(chart: Chart, window: float = 4.0, min_frac: float = 0.4, gap: float = 2.0) -> str:
    """Slap sections as a user would mark them, derived from a chart (for
    evaluating apply_slap): stretches where ≥ `min_frac` of the notes within
    ±`window` s are slap/pop, merged across gaps < `gap` s."""
    notes = [n for n in chart.notes if "fret-hand mute" not in n.tech]
    flags = []
    for n in notes:
        near = [m for m in notes if abs(m.t - n.t) <= window]
        flags.append(sum(1 for m in near if "slap" in m.tech or "pop" in m.tech) / max(1, len(near)) >= min_frac)
    secs: list[list[float]] = []
    for n, f in zip(notes, flags):
        if not f:
            continue
        if secs and n.t - secs[-1][1] < gap:
            secs[-1][1] = n.t + 0.3
        else:
            secs.append([max(0.0, n.t - 0.2), n.t + 0.3])
    return ", ".join(f"{a:.2f}-{b:.2f}" for a, b in secs)


def prf(n_match: int, n_gt: int, n_pred: int) -> dict:
    p = n_match / n_pred if n_pred else 0.0
    r = n_match / n_gt if n_gt else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def beat_fmeasure(ref: list[float], est: list[float], tol: float = 0.07) -> float:
    if not ref or not est:
        return 0.0
    pairs = match([(t, 0) for t in ref], [(t, 0) for t in est], tol, lambda a, b: True)
    return prf(len(pairs), len(ref), len(est))["f1"]


def drift_curve(gt_t: list[float], pred_t: list[float], window: float = 6.0,
                search: float = 0.15) -> list[float]:
    """Local timing offset of the chart vs. the audio, per chart note.

    Hand-synced charts drift: on the test songs the chart sits 40–90 ms off
    the real attacks and the error changes along the song. Scoring at ±50 ms
    against such a chart would mostly measure the chart's sync, so the
    "drift-corrected" score first shifts each chart note by the median offset
    to the nearest predicted onset in a ±`window` s neighbourhood."""
    import bisect
    if not gt_t or not pred_t:
        return [0.0] * len(gt_t)
    near = []
    for t in gt_t:
        k = bisect.bisect_left(pred_t, t)
        best = None
        for j in (k - 1, k):
            if 0 <= j < len(pred_t) and abs(pred_t[j] - t) <= search:
                if best is None or abs(pred_t[j] - t) < abs(best):
                    best = pred_t[j] - t
        near.append(best)
    known = [(t, d) for t, d in zip(gt_t, near) if d is not None]
    glob = statistics.median([d for _, d in known]) if known else 0.0
    kt = [t for t, _ in known]
    out = []
    for t in gt_t:
        lo, hi = bisect.bisect_left(kt, t - window), bisect.bisect_right(kt, t + window)
        vals = [known[i][1] for i in range(lo, hi)]
        out.append(statistics.median(vals) if len(vals) >= 5 else glob)
    return out


def grid_points(beats: list[float], per_beat: int = 4) -> list[float]:
    """Subdivide the chart's beat map (ebeats) into a 1/16 grid."""
    pts = []
    for a, b in zip(beats, beats[1:]):
        pts.extend(a + (b - a) * k / per_beat for k in range(per_beat))
    if beats:
        pts.append(beats[-1])
    return pts


def snap_to(points: list[float], t: float) -> float:
    import bisect
    k = bisect.bisect_left(points, t)
    cand = [points[i] for i in (k - 1, k) if 0 <= i < len(points)]
    return min(cand, key=lambda x: abs(x - t)) if cand else t


def score(chart: Chart, pred: list[dict], tol: float = 0.05, sustain_min: float = 0.25,
          drift: bool = False, skip_mutes: bool = True, grid: bool = False) -> dict:
    """pred items: {"t", "d", "m", optional "s", "f"} (stems2midi notes.json).
    `skip_mutes`: fret-hand muted (dead) notes have no pitch — leave them out.
    `drift`: correct the chart's local sync error first (see drift_curve)."""
    if skip_mutes:
        chart = Chart([n for n in chart.notes if "fret-hand mute" not in n.tech], chart.beats,
                      chart.downbeats, chart.strings, chart.offset, chart.start_beat, chart.tempo,
                      chart.anchors, chart.title)
    if drift:
        pt_sorted = sorted(float(p["t"]) for p in pred)
        offs = drift_curve([n.t for n in chart.notes], pt_sorted)
        chart = Chart([GtNote(n.t + o, n.sustain, n.string, n.fret, n.pitch, n.tech)
                       for n, o in zip(chart.notes, offs)], chart.beats, chart.downbeats,
                      chart.strings, chart.offset, chart.start_beat, chart.tempo, chart.anchors, chart.title)
    if grid and chart.beats:
        # Charts are snapped to their beat grid (EOF grid snap / Guitar Pro
        # rhythm) while a player's real attacks wander ±30–60 ms around it.
        # Grid mode snaps both sides to the chart's own 1/16 grid and asks
        # for the same slot — what matters for a playable chart.
        pts = grid_points(chart.beats)
        chart = Chart([GtNote(snap_to(pts, n.t), n.sustain, n.string, n.fret, n.pitch, n.tech)
                       for n in chart.notes], chart.beats, chart.downbeats, chart.strings,
                      chart.offset, chart.start_beat, chart.tempo, chart.anchors, chart.title)
        pred = [dict(p, t=snap_to(pts, float(p["t"]))) for p in pred]
        tol = 1e-3
    gt = [(n.t, n.pitch, i) for i, n in enumerate(chart.notes)]
    pr = sorted(((float(p["t"]), int(p["m"]), j) for j, p in enumerate(pred)), key=lambda x: x[0])
    pred_sorted = [pred[x[2]] for x in pr]
    pr = [(x[0], x[1], k) for k, x in enumerate(pr)]
    exact = match(gt, pr, tol, lambda a, b: a == b)
    pclass = match(gt, pr, tol, lambda a, b: a % 12 == b % 12)
    any_p = match(gt, pr, tol, lambda a, b: True)
    aligned = align_sequences(gt, pr)
    res = {
        "techniques": _tech_scores(chart, gt, pred_sorted, aligned),
        "gt_notes": len(gt), "pred_notes": len(pr),
        "exact": prf(len(exact), len(gt), len(pr)),
        "aligned": prf(len(aligned), len(gt), len(pr)),
        "octave_tolerant": prf(len(pclass), len(gt), len(pr)),
        "onset_only": prf(len(any_p), len(gt), len(pr)),
    }
    # what went wrong on notes found at the right time
    errs = {"octave": 0, "semitone": 0, "other": 0}
    for i, j in any_p:
        d = pr[j][1] - gt[i][1]
        if d == 0:
            continue
        if d % 12 == 0:
            errs["octave"] += 1
        elif abs(d) == 1:
            errs["semitone"] += 1
        else:
            errs["other"] += 1
    res["pitch_errors_on_time_matched"] = errs
    dts = [pr[j][0] - gt[i][0] for i, j in exact]
    res["onset_error_ms"] = {
        "median": round(statistics.median(dts) * 1000, 1) if dts else None,
        "mean_abs": round(statistics.mean(abs(x) for x in dts) * 1000, 1) if dts else None,
    }
    # sustain: Rocksmith only shows a tail on longer notes
    agree = tot = 0
    long_err = []
    for i, j in exact:
        g = chart.notes[gt[i][2]]
        pd = float(pred_sorted[j].get("d", 0))
        tot += 1
        agree += (g.sustain > 0) == (pd >= sustain_min)
        if g.sustain > 0:
            long_err.append(abs(pd - g.sustain))
    res["sustain"] = {
        "long_short_agreement": round(agree / tot, 3) if tot else None,
        "median_abs_error_s_on_long_notes": round(statistics.median(long_err), 3) if long_err else None,
    }
    # string/fret agreement with the chart on exactly-matched notes
    if pred_sorted and "s" in pred_sorted[0]:
        same = sum(1 for i, j in exact
                   if (chart.notes[gt[i][2]].string, chart.notes[gt[i][2]].fret)
                   == (pred_sorted[j].get("s"), pred_sorted[j].get("f")))
        res["string_fret_agreement"] = round(same / len(exact), 3) if exact else None
    return res


def fretting_vs_chart(chart: Chart) -> dict:
    """Run fretting.assign_frets on the chart's own pitches/timing and compare
    with the fingering the charter chose."""
    import fretting

    @dataclass
    class N:
        start: float
        end: float
        pitch: int

    notes = [N(n.t, n.t + max(n.sustain, 0.1), n.pitch) for n in chart.notes]
    res = fretting.assign_frets(notes, chart.strings)
    by_key = {}
    for n in res.notes:
        by_key.setdefault((round(n.start, 3), n.pitch), (n.string, n.fret))
    same = sum(1 for n in chart.notes if by_key.get((round(n.t, 3), n.pitch)) == (n.string, n.fret))
    gt_frets = [n.fret for n in chart.notes]
    ours = [n.fret for n in sorted(res.notes, key=lambda n: n.start)]
    return {
        "agreement": round(same / len(chart.notes), 3) if chart.notes else None,
        # same simple hand model for both: move only when a note is out of reach
        "chart_shifts": _hand_shifts(gt_frets), "ours_shifts": _hand_shifts(ours),
        "chart_above_12": sum(1 for f in gt_frets if f > 12),
        "ours_above_12": sum(1 for f in ours if f > 12),
    }


def _hand_shifts(frets: list[int]) -> int:
    hand, shifts = None, 0
    for f in frets:
        if f == 0:
            continue
        if hand is None:
            hand = f
        elif not hand <= f <= hand + 3:
            hand = f if f < hand else f - 3
            shifts += 1
    return shifts


# ── driver ──────────────────────────────────────────────────────────────────

def find_song_dirs(work: Path, only: str | None) -> list[Path]:
    dirs = []
    for d in sorted(p for p in work.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if only and only.lower() not in d.name.lower():
            continue
        if _gt_xml(d) and _audio(d):
            dirs.append(d)
    return dirs


def _gt_xml(d: Path) -> Path | None:
    exact = d / "PART REAL_BASS_RS2.xml"
    if exact.is_file():
        return exact
    cands = sorted(d.glob("PART REAL_BASS*_RS2.xml"))
    return cands[0] if cands else None


def _audio(d: Path) -> Path | None:
    for name in ("guitar.ogg", "guitar.wav"):
        if (d / name).is_file():
            return d / name
    for pat in ("*.ogg", "*.wav", "*.flac", "*.mp3", "*.opus", "*.m4a"):
        c = sorted(d.glob(pat))
        if c:
            return c[0]
    return None


def lead_silence(audio: Path) -> float:
    import numpy as np
    import stems2midi as s2m
    y = s2m.load_audio(audio, 11025, channels=1)[0]
    if not y.size:
        return 0.0
    thr = max(1e-4, float(np.max(np.abs(y))) * 10 ** (-45 / 20))
    idx = np.flatnonzero(np.abs(y) > thr)
    return round(float(idx[0]) / 11025, 3) if idx.size else 0.0


def run(args) -> int:
    import stems2midi as s2m

    work = Path(args.work)
    out_root = Path(args.out) if args.out else work / "_stems2midi_benchmark"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in find_song_dirs(work, args.only):
        name = d.name
        print(f"\n=== {name}", flush=True)
        chart = parse_rs_xml(_gt_xml(d))
        audio = _audio(d)
        song_out = out_root / re.sub(r"[^\w.-]+", "_", name)
        notes_json = song_out / "bass.notes.json"
        row = {"song": name, "audio": audio.name, "chart_notes": len(chart.notes),
               "chart_tuning": chart.strings, "chart_offset": chart.offset,
               "chart_tempo": chart.tempo,
               "techniques": _tech_counts(chart)}
        t0 = time.time()
        if not (args.reuse and notes_json.is_file()):
            tuning = " ".join(str(p) for p in chart.strings)
            src, skip = audio, args.skip_separation
            if args.reuse_stems and (song_out / "bass.wav").is_file():
                src, skip = song_out / "bass.wav", True   # stem from an earlier run
            slap = args.slap
            if args.slap_from_chart:
                slap = chart_slap_sections(chart) or None
                row["slap_sections_from_chart"] = slap
            opts = s2m.Options(model=args.model, device=args.device, stems=("bass",), backing=False,
                               midi_stems=("bass",), tuning=tuning, repo=args.repo,
                               skip_separation=skip, shifts=args.shifts, slap=slap)
            last = [-1]

            def prog(f, st, msg):
                pct = int(f * 100)
                if pct // 10 != last[0]:
                    last[0] = pct // 10
                    print(f"  [{pct:3d}%] {msg}", flush=True)

            res = s2m.run_pipeline(src, song_out, opts, progress=prog)
            row["device"] = res.device
            row["runtime_s"] = round(time.time() - t0, 1)
        data = json.loads(notes_json.read_text(encoding="utf-8"))
        row["transcription"] = score(chart, data["notes"], tol=args.tol)
        offs = drift_curve([n.t for n in chart.notes], sorted(float(p["t"]) for p in data["notes"]))
        if offs:
            row["chart_sync_ms"] = {"median": round(statistics.median(offs) * 1000),
                                    "min": round(min(offs) * 1000), "max": round(max(offs) * 1000)}
        row["fretting_on_chart_pitches"] = fretting_vs_chart(chart)
        row["lead_silence_s"] = lead_silence(audio)
        if not args.no_beats:
            y = s2m.load_audio(audio, 22050, channels=1)[0]
            bpm, beats = s2m.estimate_tempo(y, 22050)
            row["beats"] = {"est_bpm": bpm, "chart_bpm": round(chart.tempo, 2),
                            "f_measure_70ms": beat_fmeasure(chart.beats, list(beats))}
        mx = sorted(list(d.glob("*.musicxml.xml")) + list(d.glob("*.musicxml"))) if args.baseline else []
        if mx:
            base = parse_musicxml(mx[0])
            from difflib import SequenceMatcher
            same = SequenceMatcher(None, [n.pitch for n in chart.notes], [b[2] for b in base],
                                   autojunk=False).ratio()
            if same > 0.95:
                # an export of the finished chart, not a raw AnthemScore run
                row["baseline_note"] = f"{mx[0].name}: same notes as the chart ({same:.0%}), skipped"
                mx, base = [], []
        if mx:
            off = best_offset([n.t for n in chart.notes], [b[0] for b in base])
            pred = [{"t": s + off, "d": dur, "m": m} for s, dur, m in base]
            row["anthemscore"] = {"file": mx[0].name, "aligned_offset_s": round(off, 3),
                                  **score(chart, pred, tol=args.tol)}
        rows.append(row)
        _print_row(row)
    (out_root / "report.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_root / "report.md").write_text(to_markdown(rows), encoding="utf-8")
    print(f"\nReport: {out_root / 'report.md'}")
    return 0


def _tech_counts(chart: Chart) -> dict:
    c: dict[str, int] = {}
    for n in chart.notes:
        for t in n.tech:
            c[t] = c.get(t, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def _print_row(r: dict):
    t = r["transcription"]
    print(f"  ours:   note-sequence F1 {t['aligned']['f1']:.2f} (P {t['aligned']['precision']:.2f}"
          f" R {t['aligned']['recall']:.2f}) · strict ±50 ms F1 {t['exact']['f1']:.2f}"
          f" · chart sync {r.get('chart_sync_ms')}")
    if "anthemscore" in r:
        a = r["anthemscore"]
        print(f"  Anthem: note-sequence F1 {a['aligned']['f1']:.2f} (P {a['aligned']['precision']:.2f}"
              f" R {a['aligned']['recall']:.2f}) · strict F1 {a['exact']['f1']:.2f}")
    tq = t.get("techniques")
    if tq:
        print(f"  slap F1 {tq['slap']['f1']:.2f} (P {tq['slap']['precision']:.2f} R {tq['slap']['recall']:.2f})"
              f" · pop F1 {tq['pop']['f1']:.2f} (P {tq['pop']['precision']:.2f} R {tq['pop']['recall']:.2f})")
    f = r["fretting_on_chart_pitches"]
    print(f"  fretting vs your tab: {f['agreement']:.0%} same string/fret · hand shifts yours "
          f"{f['chart_shifts']} / ours {f['ours_shifts']}")


def to_markdown(rows: list[dict]) -> str:
    L = ["# stems2midi benchmark", "",
         "Ground truth = your finished `PART REAL_BASS_RS2.xml` (fret-hand mutes excluded).",
         "**Note-sequence F1**: same pitch, same order, onset within ±0.3 s — independent of how well",
         "the chart itself is synced. **Strict F1**: onset within ±50 ms of the chart.", "",
         "| Song | Chart notes | Note-seq F1 | Precision | Recall | Strict F1 | Octave errors | Chart sync (ms) "
         "| AnthemScore note-seq F1 | Fretting = yours | Beat F |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t, f = r["transcription"], r["fretting_on_chart_pitches"]
        a = r.get("anthemscore")
        sync = r.get("chart_sync_ms") or {}
        L.append(" | ".join([
            f"| {r['song']}", str(t["gt_notes"]), f"{t['aligned']['f1']:.2f}",
            f"{t['aligned']['precision']:.2f}", f"{t['aligned']['recall']:.2f}", f"{t['exact']['f1']:.2f}",
            str(t["pitch_errors_on_time_matched"]["octave"]),
            f"{sync.get('min', '?')}…{sync.get('max', '?')}",
            f"{a['aligned']['f1']:.2f}" if a else "—",
            f"{f['agreement']:.0%}", str(r.get("beats", {}).get("f_measure_70ms", "—")),
        ]) + " |")
    tq_rows = [r for r in rows if r["transcription"].get("techniques")]
    if tq_rows:
        L += ["", "## Slap / pop (inside the slap sections)", "",
              "| Song | Sections | Slap F1 | Slap P / R | Pop F1 | Pop P / R |", "|---|---|---|---|---|---|"]
        for r in tq_rows:
            q = r["transcription"]["techniques"]
            L.append(f"| {r['song']} | {r.get('slap_sections_from_chart') or '—'} | {q['slap']['f1']:.2f} | "
                     f"{q['slap']['precision']:.2f} / {q['slap']['recall']:.2f} | {q['pop']['f1']:.2f} | "
                     f"{q['pop']['precision']:.2f} / {q['pop']['recall']:.2f} |")
    L += ["", "## Techniques in your charts", ""]
    for r in rows:
        L.append(f"- **{r['song']}**: " + (", ".join(f"{k} ×{v}" for k, v in r["techniques"].items()) or "none"))
    L += ["", "Full numbers: `report.json`."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("work", help="folder with one sub-folder per song")
    p.add_argument("--out", help="output folder (default: <work>/_stems2midi_benchmark)")
    p.add_argument("--only", help="only songs whose folder name contains this text")
    p.add_argument("--model", default="htdemucs_6s")
    p.add_argument("--repo", help="local Demucs weights folder (offline)")
    p.add_argument("--device", default="auto")
    p.add_argument("--shifts", type=int, default=1)
    p.add_argument("--skip-separation", action="store_true", help="transcribe the full mix (baseline)")
    p.add_argument("--reuse", action="store_true", help="re-score existing outputs, don't re-run")
    p.add_argument("--reuse-stems", action="store_true",
                   help="re-transcribe the bass.wav of an earlier run instead of separating again")
    p.add_argument("--tol", type=float, default=0.05, help="onset tolerance in seconds (default 0.05)")
    p.add_argument("--no-beats", action="store_true")
    p.add_argument("--slap", help="slap sections passed to the transcriber (e.g. 'all')")
    p.add_argument("--slap-from-chart", action="store_true",
                   help="take the slap sections from each chart (evaluates the slap/pop rule)")
    p.add_argument("--baseline", action="store_true",
                   help="also score a raw AnthemScore MusicXML export found in the song folder")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
