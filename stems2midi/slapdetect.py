"""Experimental: suggest the slapped parts of a bass stem.

Used by ``--slap auto``. For every transcribed note we measure the attack
(band energies, centroid, flatness, crest, onset and high-band flux relative to
the song); a small logistic model looks at ±4 s of context around each note and
says "this passage is slapped". Passages of at least four notes become sections,
which :func:`fretting.apply_slap` then turns into slap (low strings) / pop (high
strings).

The model (``slap_model.json``) was trained on 15 Rocksmith charts (11 with
slap/pop, 4 fingerstyle). Leave-one-song-out it found the slap parts well on
some songs (Katy Perry, Delfins, Dark Necessities ≈ 0.85–0.97 slap F1) and
missed them on others, and it marks 5–17 % of fingerstyle songs as slap. Treat
the result as a first draft of the section list and correct it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SR = 22050
HOP = 128
MODEL_PATH = Path(__file__).with_name("slap_model.json")
MIN_NOTES = 4          # shortest passage worth calling a section
MERGE_GAP_S = 2.0      # sections closer than this are joined


def _band_ratios(seg: np.ndarray):
    if seg.size < 64:
        return [0.0, 0.0, 0.0, 0.0], 0.0, 0.0
    w = seg * np.hanning(seg.size)
    S = np.abs(np.fft.rfft(w, 4096)) ** 2 + 1e-12
    f = np.fft.rfftfreq(4096, 1 / SR)
    tot = S.sum()
    bands = [(0, 300), (300, 1500), (1500, 5000), (5000, 11025)]
    r = [float(np.log10(S[(f >= a) & (f < b)].sum() / tot + 1e-9)) for a, b in bands]
    cen = float((S * f).sum() / tot)
    flat = float(np.exp(np.mean(np.log(S))) / np.mean(S))
    return r, float(np.log10(cen + 1)), float(np.log10(flat + 1e-12))


def note_features(y: np.ndarray, times, pitches, next_times=None) -> np.ndarray:
    """Per-note attack features (19 columns); ``y`` is mono at 22050 Hz."""
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, n_fft=1024)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP))
    fr = librosa.fft_frequencies(sr=SR, n_fft=1024)
    hi = np.log1p(S[fr >= 1500])
    hflux = np.maximum(0, np.diff(hi, axis=1, prepend=hi[:, :1])).sum(0)
    pk_env = np.median(env[env > np.percentile(env, 90)]) + 1e-9
    pk_h = np.median(hflux[hflux > np.percentile(hflux, 90)]) + 1e-9
    rows = []
    for k, (t, p) in enumerate(zip(times, pitches)):
        f0 = min(len(env) - 1, int(round(t * SR / HOP)))
        lo, hi_ = max(0, f0 - 5), min(len(env) - 1, f0 + 5)
        fa = lo + int(np.argmax(env[lo:hi_ + 1])) if hi_ > lo else f0
        s = int(fa * HOP)
        nt = next_times[k] if next_times is not None else fa * HOP / SR + 0.5
        a = y[max(0, s - int(0.005 * SR)): s + int(0.04 * SR)]
        body_end = min(s + int(0.16 * SR), int(nt * SR) - int(0.01 * SR))
        b = y[s + int(0.06 * SR): max(s + int(0.06 * SR) + 64, body_end)]
        ra, ca, fla = _band_ratios(a)
        rb, cb, flb = _band_ratios(b)
        crest = (np.log10((np.max(np.abs(a)) + 1e-6) / (np.sqrt(np.mean(b ** 2)) + 1e-6))
                 if a.size and b.size else 0.0)
        f0p = 440 * 2 ** ((p - 69) / 12)
        rows.append(ra + rb + [ca, cb, fla, flb, ca - np.log10(f0p), crest,
                               np.log10(env[fa] / pk_env + 1e-6), np.log10(hflux[fa] / pk_h + 1e-6),
                               ra[2] - rb[2], ra[3] - rb[3], p])
    return np.array(rows, float).reshape(len(rows), 19)


def window_features(X: np.ndarray, t: np.ndarray, w: float, idx) -> np.ndarray:
    out = []
    for ti in t:
        m = np.abs(t - ti) <= w
        Z = X[m][:, idx]
        out.append(np.concatenate([np.median(Z, 0), np.percentile(Z, 90, axis=0), [m.sum() / (2 * w)]]))
    return np.array(out)


def load_model(path: Path = MODEL_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slap_probability(y: np.ndarray, notes, model: dict | None = None) -> np.ndarray:
    """Smoothed per-note probability that the note sits in a slapped passage."""
    model = model or load_model()
    if not notes:
        return np.zeros(0)
    t = np.array([n.start for n in notes], float)
    nxt = list(t[1:]) + [t[-1] + 0.5]
    X = note_features(y, t, [n.pitch for n in notes], nxt)
    F = window_features(X, t, model["window_s"], model["feature_index"])
    z = (F - np.array(model["mean"])) / np.array(model["scale"])
    p = 1 / (1 + np.exp(-(z @ np.array(model["coef"]) + model["intercept"])))
    sm = model.get("smooth_s", 2.0)
    return np.array([np.mean(p[np.abs(t - ti) <= sm]) for ti in t])


def detect_sections(y: np.ndarray, sr: int, notes, threshold: float | None = None,
                    model: dict | None = None) -> list[tuple[float, float]]:
    """Slapped passages of a mono bass stem as [(start_s, end_s), …]."""
    if not notes:
        return []
    if sr != SR:
        import librosa
        y = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=SR)
    model = model or load_model()
    notes = sorted(notes, key=lambda n: n.start)
    prob = slap_probability(y, notes, model)
    on = prob > (model.get("threshold", 0.5) if threshold is None else threshold)
    return sections_from_mask(notes, on)


def sections_from_mask(notes, on) -> list[tuple[float, float]]:
    secs: list[tuple[float, float]] = []
    i = 0
    while i < len(notes):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(notes) and on[j + 1]:
            j += 1
        if j - i + 1 >= MIN_NOTES:
            a, b = max(0.0, notes[i].start - 0.05), notes[j].end + 0.05
            if secs and a - secs[-1][1] < MERGE_GAP_S:
                secs[-1] = (secs[-1][0], b)
            else:
                secs.append((a, b))
        i = j + 1
    return [(round(a, 2), round(b, 2)) for a, b in secs]


def format_sections(secs) -> str:
    def mmss(x):
        m, s = divmod(x, 60)
        return f"{int(m)}:{s:05.2f}"
    return ", ".join(f"{mmss(a)}-{mmss(b)}" for a, b in secs)
