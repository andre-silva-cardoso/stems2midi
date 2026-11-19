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

Accepted files: MusicXML (``.musicxml``, ``.xml``, ``.mxl``) and MIDI
(``.mid``, ``.midi``). For MusicXML the first part is read; for MIDI every
note (the bass usually is the only track).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXTS = {".musicxml", ".xml", ".mxl", ".mid", ".midi"}


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


def read_notes(path: str | Path) -> list[tuple[float, float, int]]:
    """(start_s, dur_s, midi) in the file's own time."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in EXTS:
        raise ValueError(f"{path.name}: use MusicXML (.musicxml/.xml/.mxl) or MIDI (.mid)")
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


def global_fit(times: np.ndarray, onsets: np.ndarray, max_ratio: float = 0.08,
               max_offset: float = 60.0, pitches: np.ndarray | None = None,
               pitch_score=None) -> tuple[float, float, int]:
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

    scan(np.arange(1 - max_ratio, 1 + max_ratio + 1e-9, step * 4))
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


def warp(times: np.ndarray, onsets: np.ndarray, ratio: float, offset: float,
         window: float = 8.0, tol: float = 0.12, rounds: int = 3) -> np.ndarray:
    """Song times of ``times``: the global fit plus a smooth local correction."""
    x = ratio * times + offset
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


def line_up(notes: list[tuple[float, float, int]], onsets: np.ndarray,
            snap: float = 0.06, pitch_score=None) -> tuple[list[tuple[float, float, int]], dict]:
    """File notes → song notes [(start, dur, midi)] + a report."""
    onsets = np.sort(np.asarray(onsets, float))
    if len(onsets) < 8:
        raise ValueError("too few attacks in the bass stem to line the notes up")
    t = np.array([n[0] for n in notes])
    e = np.array([n[0] + n[1] for n in notes])
    ratio, offset, hits0 = global_fit(t, onsets, pitches=np.array([n[2] for n in notes]),
                                      pitch_score=pitch_score)
    start = warp(t, onsets, ratio, offset)
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
        out.append((s, max(0.03, float(end[i] - start[i])), int(notes[i][2])))
    out.sort()
    # no overlaps longer than the next note's start
    res = []
    for k, (s, dur, m) in enumerate(out):
        nxt = out[k + 1][0] if k + 1 < len(out) else None
        if nxt is not None and s + dur > nxt and nxt > s:
            dur = max(0.03, nxt - s)
        res.append((round(s, 4), round(dur, 4), m))
    info = {"ratio": round(ratio, 5), "offset": round(offset, 3), "notes": len(res),
            "on_attack": snapped, "share_on_attack": round(snapped / max(1, len(res)), 3),
            "tempo_note": (f"the file's tempo is {abs(1 - ratio) * 100:.1f} % "
                           f"{'faster' if ratio > 1 else 'slower'} than the song")
            if abs(1 - ratio) >= 0.002 else ""}
    return res, info


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
