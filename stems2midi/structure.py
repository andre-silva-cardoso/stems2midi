"""Find the parts of a song (intro, verse, chorus…) from the audio.

Used by ``--parts auto``. Bar by bar (bars from the same beat map as the
Rocksmith XML) it measures harmony (chroma, per beat) and sound (MFCC); a
recurrence graph links bars that sound alike anywhere in the song, a path
graph links neighbouring bars, and the leading eigenvectors of the graph's
Laplacian are clustered (McFee & Ellis, "Analyzing song structure with
spectral clustering", ISMIR 2014). Runs of the same cluster are the parts;
the same letter means the same music (A = A).

Names are a guess from the shape of the song: the repeated part with the
most energy → chorus, the other main repeated part → verse, a repeated part
right before the chorus → prechorus, a one-off at the start / end → intro /
outro, one in the middle → bridge, the rest → riff. Check them; the list is
printed so you can correct the names and run again with ``--parts "…"``.

Tested against 7 charted songs (charters' sections, consecutive equal names
merged): part boundaries within ±2.5 s F 0.57, same-part agreement
(pairwise F) 0.65.
"""

from __future__ import annotations

import numpy as np

SR = 22050
HOP = 512
LETTERS = "ABCDEFGHIJ"


def _bars(t: list[float], downs: list[int]) -> list[tuple[int, int]]:
    """(first beat, end beat) of each bar."""
    out = []
    for i, d in enumerate(downs):
        e = downs[i + 1] if i + 1 < len(downs) else min(len(t) - 1, d + 4)
        if e > d:
            out.append((d, e))
    return out


def bar_features(y: np.ndarray, t: list[float], bars, nb: int = 4) -> np.ndarray:
    import librosa
    C = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    M = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)
    R = librosa.feature.rms(y=y, hop_length=HOP)
    fr = np.clip(librosa.time_to_frames(np.asarray(t), sr=SR, hop_length=HOP), 0, C.shape[1] - 1)
    Cs = librosa.util.sync(C, fr, aggregate=np.median)
    Ms = librosa.util.sync(M, fr, aggregate=np.mean)
    Rs = librosa.util.sync(R, fr, aggregate=np.mean)
    F = []
    for d, e in bars:
        e = min(e, Cs.shape[1] - 1)
        ch = Cs[:, d:max(e, d + 1)]
        if ch.shape[1] < nb:
            ch = np.pad(ch, ((0, 0), (0, nb - ch.shape[1])), mode="edge")
        ch = ch[:, :nb]
        F.append(np.concatenate([ch.flatten("F") / (np.linalg.norm(ch) + 1e-9),
                                 Ms[1:, d:max(e, d + 1)].mean(1),
                                 [np.log(Rs[0, d:max(e, d + 1)].mean() + 1e-6)]]))
    return np.array(F)


def cluster_bars(F: np.ndarray, kmin: int = 3, kmax: int = 7, seed: int = 0) -> np.ndarray:
    """Cluster label per bar (Laplacian structural segmentation)."""
    import librosa
    import scipy.linalg
    import scipy.ndimage
    import scipy.sparse.csgraph
    from scipy.cluster.vq import kmeans2

    n = len(F)
    if n < 6:
        return np.zeros(n, int)
    nchroma = F.shape[1] - 13
    Xc = F[:, :nchroma]
    Xm = F[:, nchroma:-1]
    Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-9)
    R = librosa.segment.recurrence_matrix(Xc.T, width=2, mode="affinity", sym=True,
                                          k=max(3, int(2 * np.sqrt(n))))
    R = librosa.segment.timelag_filter(scipy.ndimage.median_filter)(R, size=(1, 3))
    path_d = np.sum(np.diff(Xm, axis=0) ** 2, axis=1)
    path = np.exp(-path_d / (np.median(path_d) + 1e-9))
    Rp = np.diag(path, k=1) + np.diag(path, k=-1)
    dp, dr = Rp.sum(1), R.sum(1)
    mu = dp.dot(dp + dr) / (np.sum((dp + dr) ** 2) + 1e-9)
    A = mu * R + (1 - mu) * Rp
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)
    evecs = scipy.ndimage.median_filter(evecs, size=(5, 1))
    cn = np.cumsum(evecs ** 2, axis=1) ** 0.5
    kmax = min(kmax, n - 1)
    gaps = np.diff(evals[:kmax + 1])
    k = kmin + int(np.argmax(gaps[kmin - 1:kmax])) if len(gaps) > kmin else min(kmin, n)
    X = evecs[:, :k] / (cn[:, k - 1:k] + 1e-9)
    rng = np.random.default_rng(seed)
    best, best_d = None, np.inf
    for _ in range(10):     # k-means restarts, keep the tightest
        cent, lab = kmeans2(X, k, minit="++", seed=rng)
        d = float(np.sum((X - cent[lab]) ** 2))
        if d < best_d:
            best, best_d = lab, d
    return best


def name_parts(segs: list[dict], energy: np.ndarray) -> None:
    """Fill seg["name"] with a guessed Rocksmith section name (in place)."""
    letters = sorted({s["letter"] for s in segs})
    count = {L: sum(s["letter"] == L for s in segs) for L in letters}
    bars = {L: sum(s["bars"] for s in segs if s["letter"] == L) for L in letters}
    loud = {L: float(np.mean([energy[s["b0"]:s["b1"]].mean() for s in segs if s["letter"] == L]))
            for L in letters}
    rep = [L for L in letters if count[L] >= 2]
    names: dict[str, str] = {}
    if rep:
        chorus = max(rep, key=lambda L: (loud[L], bars[L]))
        names[chorus] = "chorus"
        rest = [L for L in rep if L != chorus]
        if rest:
            verse = max(rest, key=lambda L: (bars[L], -min(s["b0"] for s in segs if s["letter"] == L)))
            names[verse] = "verse"
            for L in rest:
                if L in names:
                    continue
                before = sum(1 for i, s in enumerate(segs[:-1])
                             if s["letter"] == L and segs[i + 1]["letter"] == chorus)
                names[L] = "prechorus" if before >= max(1, count[L] - 1) else "riff"
    first_main = min((i for i, s in enumerate(segs) if s["letter"] in names), default=len(segs))
    last_main = max((i for i, s in enumerate(segs) if s["letter"] in names), default=-1)
    for i, s in enumerate(segs):
        L = s["letter"]
        if L in names:
            s["name"] = names[L]
        elif i < first_main:
            s["name"] = "intro"          # everything before the first verse/chorus
        elif i > last_main:
            s["name"] = "outro"          # everything after the last one
        elif count[L] == 1:
            s["name"] = "bridge"
        else:
            s["name"] = "riff"


def analyze(y: np.ndarray, sr: int, beat_times: list[float], downbeats: list[int]) -> list[dict]:
    """Song parts: [{letter, name, start, end, b0, b1, bars}] (times in the
    audio's own seconds). ``beat_times``/``downbeats``: the beat map (beat
    times and the indices of the bar starts)."""
    import librosa
    if sr != SR:
        y = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=SR)
    bars = _bars(beat_times, downbeats)
    if len(bars) < 6:
        return []
    F = bar_features(y, beat_times, bars)
    lab = cluster_bars(F)
    # letters in order of first appearance
    order: dict[int, str] = {}
    for c in lab:
        order.setdefault(int(c), LETTERS[len(order) % len(LETTERS)])
    segs = []
    for i, c in enumerate(lab):
        if i == 0 or c != lab[i - 1]:
            segs.append({"letter": order[int(c)], "b0": i, "b1": i + 1})
        else:
            segs[-1]["b1"] = i + 1
    # one-bar blips are merged into the part before them
    merged = []
    for s in segs:
        if merged and s["b1"] - s["b0"] < 2:
            merged[-1]["b1"] = s["b1"]
        else:
            merged.append(s)
    segs = []
    for s in merged:
        if segs and segs[-1]["letter"] == s["letter"]:
            segs[-1]["b1"] = s["b1"]
        else:
            segs.append(s)
    for s in segs:
        s["bars"] = s["b1"] - s["b0"]
        s["start"] = round(float(beat_times[bars[s["b0"]][0]]), 3)
        s["end"] = round(float(beat_times[bars[s["b1"] - 1][1]]), 3)
    name_parts(segs, F[:, -1])
    joined = []
    for s in segs:              # e.g. three different one-off bits before the first verse = one intro
        if joined and s["name"] in ("intro", "outro") and joined[-1]["name"] == s["name"]:
            joined[-1]["b1"], joined[-1]["end"] = s["b1"], s["end"]
            joined[-1]["bars"] = joined[-1]["b1"] - joined[-1]["b0"]
        else:
            joined.append(s)
    segs = joined
    relabel: dict[str, str] = {}
    for s in segs:              # letters again in order of appearance
        s["letter"] = relabel.setdefault(s["letter"], LETTERS[len(relabel) % len(LETTERS)])
    num: dict[str, int] = {}
    for s in segs:
        num[s["name"]] = num.get(s["name"], 0) + 1
        s["number"] = num[s["name"]]
    return segs


def mmss(x: float) -> str:
    m, s = divmod(max(0.0, x), 60)
    return f"{int(m)}:{s:05.2f}"


def parts_spec(segs: list[dict]) -> str:
    """Every part with its range — the format --parts takes."""
    return ", ".join(f"{s['name']}={mmss(s['start'])}-{mmss(s['end'])}" for s in segs)


def report(segs: list[dict]) -> str:
    return " · ".join(f"{s['name']} {s['number']} ({s['letter']}) {mmss(s['start'])[:-3]}" for s in segs)
