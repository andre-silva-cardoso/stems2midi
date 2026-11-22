"""Second-pass hints: what the first transcription says about the song, as
options worth a re-run (printed in the warnings and bass.review.txt).

Nothing here changes the transcription — each hint is a guess that can be
wrong for a given song, so the user decides. Thresholds were set on the
benchmark songs (Delfins, Manic, Xutos, Korn, Lenny Kravitz, RHCP, Katy
Perry, Bandeira) and the two songs that needed the options (Queen — Who
Wants to Live Forever, Delfins — A Bandeira): none of the hints fires on a
song that didn't need it.

* --sustained   many same-pitch notes back to back with no dip in level
                between them: one long note cut in pieces (Queen: 44 % of
                the notes; the others ≤ 17 %).
* --only        long stretches where the bass stem is ≥ 12 dB quieter than
                in the rest of the song but notes were still found there:
                other instruments leaking into the stem, the bass not
                playing (Queen: orchestra before 2:50).
* --bpm         tempo octave: a fast tempo with long notes cut in pieces
                (ballad heard twice as fast), or a slow tempo with notes on
                every 16th (rock heard half as fast — Xutos: 80.7 for 159).
* --model       the bass stem holds far less of the song's low end than the
                mix does: the separation put the bass elsewhere (a thin,
                high or mid-heavy bass ends up in "guitar"/"other" with the
                6-stem model — Delfins, Lugar ao Sol: bass stem 21 dB under
                the mix below 250 Hz). Try htdemucs_ft, or a stem from another
                separator with --skip-separation.
* --notes-from  the review finds most bars doubtful (the stem is hard to
                read; Bandeira: 84 %, its fundamental 9 dB under the
                harmonics): a dedicated transcriber (AnthemScore…) read in.
"""

from __future__ import annotations

import numpy as np

SR = 11025
HOP = 128

CHOP_GAP_S = 0.05          # back to back: next note starts < 50 ms after the previous ends
CHOP_DIP_DB = 1.5          # less dip than this in the level at the boundary → no new attack
CHOP_SHARE = 0.25          # such joins, as a share of all notes
CHOP_OF_PAIRS = 0.5        # … and of the same-pitch joins
CHOP_MIN = 20

QUIET_WIN_S = 4.0
QUIET_DROP_DB = 12.0       # quieter than the loud part of the stem by this much
QUIET_MERGE_S = 8.0        # quiet gaps shorter than this stay inside a range
QUIET_SHARE = 0.10         # notes in quiet stretches, as a share of all notes
QUIET_MIN_NOTES = 20
QUIET_MIN_S = 20.0

FAST_BPM = 135.0
SLOW_BPM = 85.0
SLOW_IOI_BEATS = 0.3       # median gap between notes ≤ this many beats → 16ths at that tempo

WEAK_STEM_DB = 12.0        # bass stem's low end (40-250 Hz) this much under the mix's → separation missed it
LOW_BAND = (40.0, 250.0)
BETTER_MODEL = "htdemucs_ft"

HARD_SHARE = 0.5           # review: bars "likely wrong" above this share → stem hard to read
WEAK_F0_DB = 6.0           # harmonics louder than the fundamental by this much (median)


def _mono(y, sr):
    import librosa
    y = np.asarray(y, np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def _mmss(t: float) -> str:
    m, s = divmod(int(round(max(0.0, t))), 60)
    return f"{m}:{s:02d}"


def chopped_notes(y: np.ndarray, notes: list[dict]) -> tuple[int, int]:
    """(joins with no level dip, same-pitch joins) between back-to-back notes."""
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=256, hop_length=HOP)[0]
    db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)

    def fr(t):
        return int(np.clip(round(t * SR / HOP), 0, len(db) - 1))

    pairs = flat = 0
    for a, b in zip(notes, notes[1:]):
        if a["m"] != b["m"] or b["t"] - (a["t"] + a["d"]) >= CHOP_GAP_S:
            continue
        pairs += 1
        t = b["t"]
        before = db[fr(t - 0.12):max(fr(t - 0.04), fr(t - 0.12) + 1)]
        at = db[fr(t - 0.04):fr(t + 0.01) + 1]
        if len(before) and len(at) and float(np.mean(before) - np.min(at)) < CHOP_DIP_DB:
            flat += 1
    return flat, pairs


def quiet_stretches(y: np.ndarray, notes: list[dict]) -> dict:
    """Where the stem is much quieter than its loud part. Returns
    {"loud": [(start, end)…], "quiet_notes": n, "quiet_s": seconds with notes}."""
    import librosa
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    dur = len(y) / SR
    nw = max(1, int(np.ceil(dur / QUIET_WIN_S)))
    lv = np.full(nw, -120.0)
    per = QUIET_WIN_S * SR / hop
    for i in range(nw):
        seg = db[int(i * per):int((i + 1) * per)]
        if len(seg):
            lv[i] = np.percentile(seg, 90)
    cnt = np.zeros(nw, int)
    for n in notes:
        cnt[min(nw - 1, int(n["t"] / QUIET_WIN_S))] += 1
    if not cnt.any():
        return {"loud": [], "quiet_notes": 0, "quiet_s": 0.0}
    ref = np.percentile(lv[cnt > 0], 90)
    loud = lv >= ref - QUIET_DROP_DB
    ranges: list[list[float]] = []
    for i in np.flatnonzero(loud):
        a, b = i * QUIET_WIN_S, min(dur, (i + 1) * QUIET_WIN_S)
        if ranges and a - ranges[-1][1] < QUIET_MERGE_S:
            ranges[-1][1] = b
        else:
            ranges.append([a, b])
    inside = np.zeros(nw, bool)
    for a, b in ranges:
        inside[int(a // QUIET_WIN_S):int(np.ceil(b / QUIET_WIN_S))] = True
    return {"loud": [(a, b) for a, b in ranges],
            "quiet_notes": int(cnt[~inside].sum()),
            "quiet_s": float(((~inside) & (cnt > 0)).sum() * QUIET_WIN_S)}


def harmonic_excess(y: np.ndarray, notes: list[dict]) -> float:
    """Median of (louder of 2nd/3rd harmonic − fundamental), dB, at the notes."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=HOP))
    fq = librosa.fft_frequencies(sr=SR, n_fft=4096)
    out = []
    for n in notes:
        f0 = 440.0 * 2 ** ((n["m"] - 69) / 12)
        i = int(np.clip(round((n["t"] + min(0.08, n["d"] / 2)) * SR / HOP), 0, S.shape[1] - 1))
        h = []
        for k in (1, 2, 3):
            j = int(np.argmin(np.abs(fq - f0 * k)))
            h.append(20 * np.log10(S[max(0, j - 1):j + 2, i].max() + 1e-9))
        out.append(max(h[1], h[2]) - h[0])
    return float(np.median(out)) if out else 0.0


def low_end_frames(y: np.ndarray) -> np.ndarray:
    """Energy of the 40-250 Hz band per ~0.09 s frame (mono, 11025 Hz)."""
    import librosa
    if len(y) < 4096:
        return np.zeros(1)
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    fq = librosa.fft_frequencies(sr=SR, n_fft=4096)
    band = (fq >= LOW_BAND[0]) & (fq < LOW_BAND[1])
    return S[band].sum(axis=0)


def low_end_gap(bass: np.ndarray, mix: np.ndarray, win_s: float = QUIET_WIN_S) -> tuple[float, float]:
    """(gap, overall): how many dB the bass stem's low end is under the mix's.
    ``gap`` is taken over the best quarter of the loud 4 s windows, so a bass
    that simply rests for half the song (orchestra intro) doesn't count."""
    eb, em = low_end_frames(bass), low_end_frames(mix)
    n = min(len(eb), len(em))
    eb, em = eb[:n], em[:n]
    per = max(1, int(round(win_s * SR / 1024)))
    nw = max(1, n // per)
    wb = np.array([eb[i * per:(i + 1) * per].mean() for i in range(nw)])
    wm = np.array([em[i * per:(i + 1) * per].mean() for i in range(nw)])
    db = lambda x: 10 * np.log10(x + 1e-12)          # noqa: E731
    loud = db(wm) >= db(wm.max()) - 20
    gaps = db(wm[loud]) - db(wb[loud])
    return float(np.percentile(gaps, 25)), float(db(em.mean()) - db(eb.mean()))


def weak_stem(stems: dict, sr: int, model: str = "", stem: str = "bass") -> dict | None:
    """Hint when the separation left the bass out of its stem: the stem's low
    end far under the mix's (the mix = the sum of the stems). Names the stem
    that got most of the low end. ``stems``: {name: audio (channels, n)}."""
    if stem not in stems or len(stems) < 2:
        return None
    mono = {k: _mono(v, sr) for k, v in stems.items()}
    mix = np.sum([mono[k][:min(len(x) for x in mono.values())] for k in mono], axis=0)
    gap, overall = low_end_gap(mono[stem], mix)
    if gap < WEAK_STEM_DB:
        return None
    gap = overall
    lows = {k: float(low_end_frames(v).mean()) for k, v in mono.items()}
    other = max((k for k in lows if k != stem), key=lambda k: lows[k])
    where = f'; most of it is in the "{other}" stem' if lows[other] > 2 * lows[stem] else ""
    if model != BETTER_MODEL:
        return {"option": "--model", "value": BETTER_MODEL, "source": True,
                "why": f"the bass stem has {gap:.0f} dB less low end (40-250 Hz) than the song{where} — the "
                       f"separation missed much of the bass (a thin or high bass is often taken for guitar). "
                       f"Separate again with the 4-stem fine-tuned model ({BETTER_MODEL}, ~4x slower), or "
                       f"transcribe a bass stem from another separator with --skip-separation"}
    return {"option": "--skip-separation", "value": "<bass stem from another separator>", "source": True,
            "why": f"the bass stem has {gap:.0f} dB less low end (40-250 Hz) than the song{where}, even with "
                   f"{BETTER_MODEL} — separate the bass with another tool and transcribe that file"}


def suggest(y: np.ndarray, sr: int, notes: list[dict], bpm: float, review: list[dict] | None = None,
            used: dict | None = None) -> list[dict]:
    """Options worth a re-run, as [{"option", "value", "why"}]. ``notes`` are
    {"t", "d", "m"} in song time; ``used`` the options this run had (a hint
    for an option already given is left out)."""
    used = used or {}
    notes = sorted((n for n in notes if "m" in n), key=lambda n: n["t"])
    if len(notes) < 20:
        return []
    y = _mono(y, sr)
    out: list[dict] = []

    flat, pairs = chopped_notes(y, notes)
    chopped = flat >= max(CHOP_MIN, CHOP_SHARE * len(notes)) and flat >= CHOP_OF_PAIRS * max(1, pairs)
    if chopped and not used.get("sustained"):
        out.append({"option": "--sustained", "value": "",
                    "why": f"{flat} notes start with no new attack, right after a note of the same pitch "
                           f"({flat / len(notes):.0%} of the notes): long held notes cut in pieces"})

    if not used.get("only") and not used.get("notes_from"):
        q = quiet_stretches(y, notes)
        if q["loud"] and q["quiet_notes"] >= max(QUIET_MIN_NOTES, QUIET_SHARE * len(notes)) \
                and q["quiet_s"] >= QUIET_MIN_S:
            spec = ", ".join(f"{_mmss(a)}-{_mmss(b)}" for a, b in q["loud"])
            out.append({"option": "--only", "value": spec,
                        "why": f"{q['quiet_notes']} notes ({q['quiet_notes'] / len(notes):.0%}) come from "
                               f"stretches where the bass stem is {QUIET_DROP_DB:g}+ dB quieter than in the rest "
                               f"of the song — other instruments leaking in? Listen to bass.wav: if the bass "
                               f"plays only in {spec}, keep just those"})

    if not used.get("bpm") and bpm > 0:
        iois = np.diff([n["t"] for n in notes])
        ioi_beats = float(np.median(iois)) * bpm / 60 if len(iois) else 1.0
        if bpm >= FAST_BPM and chopped:
            out.append({"option": "--bpm", "value": f"{bpm / 2:.1f}",
                        "why": f"tempo found {bpm:.1f} BPM with long held notes — a slow song is often "
                               f"heard twice as fast; if the bars in EOF are half as long as they should be"})
        elif bpm <= SLOW_BPM and ioi_beats <= SLOW_IOI_BEATS and not chopped:
            out.append({"option": "--bpm", "value": f"{bpm * 2:.1f}",
                        "why": f"tempo found {bpm:.1f} BPM with a note every {ioi_beats:.2f} beat — a fast "
                               f"song is often heard half as fast; if the bars in EOF are twice as long as "
                               f"they should be"})

    if review and not used.get("notes_from"):
        bad = sum(b.get("level") == "bad" for b in review) / len(review)
        relative = any(b.get("relative") for b in review)     # review ranked within the song: most bars bad
        if bad > HARD_SHARE or relative:
            share = "most of the bars look doubtful" if relative else f"{bad:.0%} of the bars look doubtful"
            hx = harmonic_excess(y, notes)
            weak = (f"; the fundamental is {hx:.0f} dB under its harmonics, so the pitch is hard to "
                    f"hear" if hx >= WEAK_F0_DB else "")
            out.append({"option": "--notes-from", "value": "<file.gp5|.musicxml>",
                        "why": f"{share}{weak} — a Guitar Pro tab of the song (.gp5…) or a dedicated "
                               f"transcriber (e.g. AnthemScore → MusicXML) may do better; its notes are lined "
                               f"up with the stem (add --use doubtful to take them only in the doubtful bars)"})
    return out


def _arg(s: dict) -> str:
    v = s.get("value") or ""
    if not v:
        return s["option"]
    return f'{s["option"]} "{v}"' if any(c in v for c in " ,<>") else f"{s['option']} {v}"


def command(suggestions: list[dict], folder: str = "", extra: str = "", source: str = "") -> str:
    """One re-run with every hint: the result folder + the options (no new
    separation — the bass stem in the folder is used). A hint that needs a new
    separation (``"source": True``, e.g. another model) runs on ``source``
    (the song file) instead, into the same folder."""
    import os
    tool = "stems2midi.bat" if os.name == "nt" else "stems2midi"
    skip = next((s for s in suggestions if s.get("option") == "--skip-separation"), None)
    if skip is not None:
        return f'{tool} "{skip.get("value")}" --skip-separation' + (f" {extra}" if extra else "")
    if any(s.get("source") for s in suggestions) and source:
        hints = suggestions
        args = " ".join([_arg(s) for s in hints] + ([extra] if extra else []))
        out = f' -o "{folder}"' if folder else ""
        return f'{tool} "{source}"{out} {args}'.strip()
    args = " ".join([_arg(s) for s in suggestions] + ([extra] if extra else []))
    return f'{tool} "{folder}" {args}'.strip() if folder else args


def report(suggestions: list[dict], rerun: str = "") -> str:
    """Text block for bass.review.txt (``rerun``: the command from ``command``)."""
    if not suggestions:
        return ""
    lines = ["", "Worth trying (from this first pass — each is a guess, check with your ears):"]
    for s in suggestions:
        lines.append(f"  {_arg(s)}")
        lines.append(f"      {s['why']}.")
    if any(s.get("source") for s in suggestions):
        lines.append("Run the song again (new separation):")
    else:
        lines.append("Re-run on this result folder (uses its bass stem, no new separation):")
    lines.append(f"  {rerun or command(suggestions, '<this folder>')}")
    return "\n".join(lines) + "\n"
