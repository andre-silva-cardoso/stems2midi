"""Check a bass transcription against its stem, bar by bar.

For every bar it measures where the transcription and the sound disagree:

* **missed attacks** — clear attacks in the stem with no note starting there
  (pops and ghost notes the pitch tracker heard as unpitched, fast notes);
* **weak notes** — notes whose start has hardly any attack in the stem;
* **pitch support** — how much the stem's energy at the note's pitch stands
  out; and whether the octave below / above is much stronger (octave doubt);
* **uncovered sound** — time where the stem clearly plays but no note sounds;
* **level** — how loud the bass is in that bar (quiet bars are guesswork).

A small logistic model (``review_model.json``, trained on hand-made charts)
turns these into the chance that the bar has mistakes; bars are then marked
ok / check / bad, with the main reasons, so the review in EOF can start with
the bars most likely to be wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SR = 11025
HOP = 128
MODEL_PATH = Path(__file__).with_name("review_model.json")
RELATIVE_ABOVE = 0.5      # more bars than this "bad" → rank the bars within the song
FEATURES = ["missed_per_note", "missed", "weak_frac", "support", "low_octave_frac",
            "high_octave_frac", "level_db", "uncovered", "density", "pitch_spread",
            "class_mismatch", "octave_up_rise"]
REASONS = {
    "missed": "attacks with no note",
    "weak_frac": "notes without a clear attack",
    "support": "pitch not clearly in the stem",
    "low_octave_frac": "octave below sounds stronger",
    "high_octave_frac": "octave above sounds stronger",
    "level_db": "bass very quiet",
    "uncovered": "bass sounds where there is no note",
    "class_mismatch": "a different note sounds stronger",
    "octave_up_rise": "octave above starts with the note (pop / octave?)",
}


class Analysis:
    """Stem curves computed once per song."""

    def __init__(self, y: np.ndarray, sr: int):
        import librosa
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=0)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        self.env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, n_fft=512, fmax=2000)
        pk = np.flatnonzero((self.env[1:-1] >= self.env[:-2]) & (self.env[1:-1] > self.env[2:])) + 1
        self.peaks = pk
        self.typ = float(np.percentile(self.env[pk], 97)) if pk.size else 1.0
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
        self.rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
        self.song_db = float(np.median(self.rms_db[self.rms_db > -45])) if np.any(self.rms_db > -45) else -30.0
        C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(21), n_bins=60))
        self.cqt_db = librosa.amplitude_to_db(C + 1e-10, ref=np.max)   # rows: MIDI 21 … 80
        n = min(len(self.env), len(self.rms_db), self.cqt_db.shape[1])
        self.env, self.rms_db, self.cqt_db = self.env[:n], self.rms_db[:n], self.cqt_db[:, :n]
        self.attacks = pk[(pk < n) & (self.env[np.minimum(pk, n - 1)] >= 0.35 * self.typ)]
        self.attacks = np.array([a for a in self.attacks if self.rms_db[min(n - 1, a + 3)] > -40])

    def frame(self, t: float) -> int:
        return int(min(len(self.env) - 1, max(0, round(t * SR / HOP))))


def _note_checks(an: Analysis, n: dict) -> tuple:
    """(attack strength rel. to the song, pitch support dB, low-octave flag,
    high-octave flag, pitch-class mismatch flag, octave-above rise dB)."""
    f0 = an.frame(n["t"])
    f1 = max(f0 + 3, an.frame(n["t"] + min(n["d"], 0.15)))
    att = float(np.max(an.env[max(0, f0 - 2):f0 + 3])) / (an.typ + 1e-9)
    p = int(n.get("m0", n["m"])) - 21
    D = an.cqt_db[:, f0:f1]
    if not 0 <= p < D.shape[0] or D.shape[1] == 0:
        return att, 0.0, 0, 0, 0, 0.0
    col = D.mean(axis=1)
    others = np.delete(col, [x for x in (p - 12, p, p + 12) if 0 <= x < len(col)])
    support = float(col[p] - np.median(others))
    low = int(p - 12 >= 0 and col[p - 12] > col[p] + 3)
    high = int(p + 12 < len(col) and col[p + 12] > col[p] + 9)
    # strongest pitch class in the bass range (B0 … C4) vs the note's
    lo_b, hi_b = 2, min(len(col), 40)
    chroma = np.full(12, -120.0)
    for i in range(lo_b, hi_b):
        chroma[(i + 21) % 12] = max(chroma[(i + 21) % 12], col[i])
    mism = int(np.argmax(chroma) != (p + 21) % 12)
    up_rise = 0.0
    if p + 12 < an.cqt_db.shape[0] and f0 >= 6:
        pre = an.cqt_db[p + 12, f0 - 6:f0 - 1].mean()
        up_rise = float(an.cqt_db[p + 12, f0 + 1:f0 + 6].mean() - pre)
    return att, support, low, high, mism, up_rise


def bar_features(an: Analysis, notes: list[dict], bars: list[tuple[float, float]]) -> np.ndarray:
    starts = np.array(sorted(n["t"] for n in notes)) if notes else np.zeros(0)
    att_t = an.attacks * HOP / SR
    checks = [_note_checks(an, n) for n in notes]
    rows = []
    for a, b in bars:
        idx = [i for i, n in enumerate(notes) if a <= n["t"] < b]
        nn = len(idx)
        at = att_t[(att_t >= a) & (att_t < b)]
        missed = int(sum(1 for x in at if not starts.size or np.min(np.abs(starts - x)) > 0.07))
        fa, fb = an.frame(a), max(an.frame(a) + 1, an.frame(b))
        lvl = float(np.percentile(an.rms_db[fa:fb], 90)) - an.song_db
        # sounding but uncovered: frames ≥ 12 dB above the song's quiet floor with no note on
        on = np.zeros(fb - fa, bool)
        for n in notes:
            s, e = an.frame(n["t"]) - fa, an.frame(n["t"] + n["d"]) - fa
            if e > 0 and s < fb - fa:
                on[max(0, s):max(0, min(fb - fa, e + 2))] = True
        loud = an.rms_db[fa:fb] > an.song_db - 6
        uncovered = float(np.mean(loud & ~on)) if loud.size else 0.0
        if nn:
            c = np.array([checks[i] for i in idx])
            weak = float(np.mean(c[:, 0] < 0.12))
            support = float(np.median(c[:, 1]))
            low = float(np.mean(c[:, 2]))
            high = float(np.mean(c[:, 3]))
            spread = float(np.ptp([notes[i]["m"] for i in idx]))
            mism = float(np.mean(c[:, 4]))
            uprise = float(np.mean(c[:, 5]))
        else:
            weak, support, low, high, spread, mism, uprise = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        rows.append([missed / max(1, nn), missed, weak, support, low, high, lvl, uncovered,
                     nn / max(1e-3, b - a), spread, mism, uprise])
    return np.array(rows, float).reshape(len(rows), len(FEATURES))


def load_model(path: Path = MODEL_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score(F: np.ndarray, model: dict | None = None) -> np.ndarray:
    """Chance (0–1) that each bar has mistakes."""
    model = model or load_model()
    z = (F - np.array(model["mean"])) / np.array(model["scale"])
    return 1 / (1 + np.exp(-(z @ np.array(model["coef"]) + model["intercept"])))


# a reason is only named when the bar really shows it: high for these …
_HIGH_IS_BAD = {"missed", "weak_frac", "low_octave_frac", "high_octave_frac", "uncovered",
                "class_mismatch", "octave_up_rise"}
# … low for these
_LOW_IS_BAD = {"support", "level_db"}


def reasons(row: np.ndarray, model: dict) -> list[str]:
    """What looks wrong in this bar, most telling first."""
    z = (row - np.array(model["mean"])) / np.array(model["scale"])
    contrib = z * np.array(model["coef"])
    out = []
    for i in np.argsort(-np.abs(z)):
        name = FEATURES[i]
        if name not in REASONS or contrib[i] <= 0.1:
            continue
        if (name in _HIGH_IS_BAD and z[i] < 0.75) or (name in _LOW_IS_BAD and z[i] > -0.75):
            continue
        out.append(REASONS[name])
        if len(out) == 2:
            break
    return out


def review(y: np.ndarray, sr: int, notes: list[dict], bars: list[tuple[float, float]],
           model: dict | None = None) -> list[dict]:
    """[{t, end, bar, p, level: ok|check|bad, why: [...]}] — one per bar."""
    model = model or load_model()
    an = Analysis(y, sr)
    F = bar_features(an, notes, bars)
    P = score(F, model)
    lo, hi = model.get("check", 0.5), model.get("bad", 0.75)
    relative = len(P) >= 8 and float(np.mean(P >= hi)) > RELATIVE_ABOVE
    if relative:
        # most of the song looks wrong by the general standard (e.g. a stem with
        # almost no fundamental): rank the bars within the song instead
        hi, lo = float(np.quantile(P, 0.85)), float(np.quantile(P, 0.70))
    out = []
    for k, ((a, b), p, row) in enumerate(zip(bars, P, F)):
        level = "bad" if p >= hi else "check" if p >= lo else "ok"
        out.append({"t": round(a, 3), "end": round(b, 3), "bar": k + 1, "p": round(float(p), 3),
                    "level": level, "why": reasons(row, model) if level != "ok" else [],
                    **({"relative": True} if relative else {})})
    return out


def bars_from_beat_map(beat_map) -> list[tuple[float, float]]:
    downs = [t for t, m in beat_map if m > 0]
    end = beat_map[-1][0] if beat_map else 0.0
    return [(a, b) for a, b in zip(downs, downs[1:] + [end]) if b > a]


def report(bars: list[dict], shift: float = 0.0) -> str:
    """Plain-text list of the bars to check (times in the song, and in EOF
    when ``shift`` is the EOF lead-in)."""
    def mmss(x):
        m, s = divmod(max(0.0, x), 60)
        return f"{int(m)}:{s:04.1f}"
    bad = [b for b in bars if b["level"] != "ok"]
    n_bad = sum(b["level"] == "bad" for b in bars)
    lines = [f"Bars to review: {n_bad} likely wrong, {len(bad) - n_bad} worth a check, "
             f"{len(bars) - len(bad)} look fine (of {len(bars)})."]
    if any(b.get("relative") for b in bars):
        lines.append("(most bars look doubtful by the usual standard — the stem is hard to read, e.g. the "
                     "bass has almost no fundamental — so these are the worst bars of this song)")
    if shift:
        lines.append(f"(times in brackets: in EOF, +{shift:g} s lead-in)")
    for b in sorted(bad, key=lambda x: -x["p"]):
        when = mmss(b["t"]) + (f" [{mmss(b['t'] + shift)}]" if shift else "")
        lines.append(f"  bar {b['bar']:>3}  {when:<16} {b['level']:<5} {b['p']:.0%}  {', '.join(b['why'])}")
    return "\n".join(lines) + "\n"
