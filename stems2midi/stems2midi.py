#!/usr/bin/env python3
"""stems2midi — split a song into stems and transcribe the bass stem to MIDI.

Usable three ways:

* CLI:      python stems2midi.py song.mp3 [-o out_dir] [options]
* Library:  from stems2midi import run_pipeline
* Plugin:   plugins/stems2midi/routes.py spawns this file as a subprocess with
            ``--json-progress`` so the heavy ML stack (PyTorch / Demucs) runs in
            its own interpreter and never blocks or bloats the FeedBack server.

Pipeline
────────
1. Source separation with Demucs. The default ``htdemucs_6s`` model yields six
   stems: drums, bass, other, vocals, guitar, piano. 4-stem models
   (``htdemucs``, ``htdemucs_ft``) are cleaner for bass/drums/vocals but have
   no dedicated guitar stem (guitar ends up in "other").
2. Optional backing track = sum of every stem except the ones removed
   (by default "bass"), handy for play-along practice.
3. Stem → MIDI:
   * ``pyin``  (default for bass) — monophonic pitch tracking (librosa pYIN)
     + onset-aware note segmentation. Robust for bass lines, no extra deps.
   * ``basic-pitch`` — Spotify's polyphonic transcriber, if installed. Better
     for chords (guitar/piano); for bass it is reduced to a monophonic line.
4. MIDI written with mido: one track, GM program 33 (Electric Bass, finger),
   tempo = detected BPM (or ``--bpm``), optional beat-grid quantisation.

Heavy imports (torch, demucs, librosa, basic_pitch) are lazy so ``--check``
and the plugin's status probe stay fast and never crash on a missing package.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import logging
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

__version__ = "1.2.9"

log = logging.getLogger("feedBack.plugin.stems2midi.core")

ProgressCB = Optional[Callable[[float, str, str], None]]  # (fraction 0..1, stage, message)

MODELS = {
    "htdemucs_6s": "6 stems: drums, bass, other, vocals, guitar, piano",
    "htdemucs": "4 stems: drums, bass, other, vocals (fast)",
    "htdemucs_ft": "4 stems, fine-tuned (best bass/drums/vocals, ~4x slower)",
}
DEFAULT_MODEL = "htdemucs_6s"
ALL_STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")

# General MIDI programs (0-based) per stem, used for the transcribed track.
GM_PROGRAMS = {
    "bass": 33,     # Electric Bass (finger)
    "guitar": 29,   # Overdriven Guitar
    "piano": 0,     # Acoustic Grand Piano
    "vocals": 52,   # Choir Aahs
    "other": 81,    # Lead 2 (sawtooth)
}

# Default pitch range per stem (note names, librosa notation).
PITCH_RANGES = {
    "bass": ("B0", "G4"),     # 5-string low B .. 24th fret on G
    "guitar": ("D2", "E6"),
    "piano": ("A0", "C8"),
    "vocals": ("C2", "C6"),
    "other": ("C1", "C7"),
}

# Transcription tuning (pYIN at 11025 Hz, hop 128 ≈ 11.6 ms). Chosen by a
# sweep over synthetic bass lines (plucked 16ths, octave jumps, full mix).
PYIN_FRAME = 1024
PYIN_MAX_TRANSITION = 35.92   # librosa default; higher values hurt on mixes
ONSET_NFFT = 512
ONSET_REL = 0.3      # onset peak threshold, fraction of the local maximum (benchmark-tuned)
PYIN_BLOCK_S = 30.0  # progress granularity

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".opus", ".webm", ".mp4"}


# ─────────────────────────────────────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Note:
    start: float      # seconds
    end: float        # seconds
    pitch: int        # MIDI note number
    velocity: int = 90
    legato: bool = False      # started without its own attack (hammer-on/pull-off/slide)
    tech: str | None = None   # "slap" | "pop" (see apply_slap)
    string: int | None = None # string to play it on (0 = lowest), e.g. from a Guitar Pro tab

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Options:
    model: str = DEFAULT_MODEL
    device: str = "auto"                    # auto | cpu | cuda | mps
    stems: tuple = ("bass", "guitar")       # stems to write ("all" expands)
    backing: bool = True                    # write a mix without `backing_remove`
    backing_remove: tuple = ("bass",)
    midi_stems: tuple = ("bass",)           # stems to transcribe
    engine: str = "auto"                    # auto | pyin | basic-pitch
    quantize: int = 0                       # 0 = off, else grid per whole note (8, 16, 32)
    bpm: float | None = None                # override tempo
    beats: list | None = None               # beat times from an earlier run (re-transcribing a result folder)
    fmin: str | None = None                 # override pitch range low (note name)
    fmax: str | None = None                 # override pitch range high
    min_note_ms: float = 60.0
    shifts: int = 1                         # Demucs test-time augmentation
    overlap: float = 0.25
    audio_format: str = "wav"               # wav | flac
    skip_separation: bool = False           # input already is the stem to transcribe
    repo: str | None = None                 # local folder with Demucs weights (offline)
    tuning: str | None = "auto"             # bass tuning for the tab ("auto" = 4-string E or 5-string B; None = no tab)
    slap: str | None = None                 # slap sections: "all" or "0:32-1:05, 130-162"
    rs_xml: bool = True                     # also write bass_RS2.xml (Rocksmith 2014) when there is a tab
    meta: dict = field(default_factory=dict)  # artist/title/album/year for the XML (default: from the file name)
    rs_shift: float = 0.0                   # move the XML this many seconds later (EOF lead-in silence)
    parts: str | None = None                # song parts, first occurrences: "verse=0:21-0:42, chorus=0:42-1:05"
    review: bool = True                     # check the bass transcription against its stem, bar by bar
    notes_from: str | None = None           # MusicXML/MIDI of another transcriber (AnthemScore…) for the bass
    notes_use: str = "all"                  # where: all | doubtful (bars the review flags) | "12-30, 1:02-1:30"
    notes_track: str | None = None          # Guitar Pro: track number (from 1) or name (default: the bass track)
    sections_from: str | None = None        # Guitar Pro with one example of each part (markers) → placed on the song
    only: str | None = None                 # the bass guitar plays only here: "2:50-5:15" / "40-90" (bars)
    sustained: bool = False                 # ballad: long notes, re-plucks only with a clear level rise
    phrase_bars: int = 4                    # Rocksmith phrases (Riff Repeater units) every N bars; 0 = per section
    lyrics: str | None = None               # lyrics text → vocals.ultrastar.txt + vocals_RS2.xml
    language: str | None = None             # lyrics language (pt, en…; None = guess)
    lyrics_aligner: str = "auto"            # auto | mms | simple
    keep_accents: bool = False              # lyrics files: keep ã, ç… (default: plain letters)
    five_string: str = "auto"               # 5-string tab in Rocksmith: auto | bass (4-string B) | rhythm (also a Rhythm XML)
    max_fret: int = 24
    suggest: bool = True                    # look for options worth a re-run (suggest.py)
    charter: str = ""                       # DLC Builder project author (charter name, e.g. ALSRC)
    dlc_project: bool = True                # write a DLC Builder project (.rs2dlc) next to the XML
    fetch_meta: bool = False                # look up album art + lyrics (iTunes / LRCLIB) into the folder


@dataclass
class Result:
    out_dir: str
    stems: dict = field(default_factory=dict)     # name -> file
    midi: dict = field(default_factory=dict)      # stem -> .mid
    notes: dict = field(default_factory=dict)     # stem -> .notes.json
    tabs: dict = field(default_factory=dict)      # stem -> .tab.txt
    rsxml: dict = field(default_factory=dict)     # stem -> Rocksmith 2014 arrangement XML
    review: dict = field(default_factory=dict)    # stem -> bars to review (.review.txt)
    lyrics: dict = field(default_factory=dict)    # ultrastar / xml file names, aligner, counts
    tuning: list | None = None                    # open-string MIDI pitches used for the tab
    bpm: float | None = None
    device: str | None = None
    model: str | None = None
    elapsed: float = 0.0
    warnings: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)  # [{"option", "value", "why"}] worth a re-run
    rerun: str = ""                                   # the re-run command with them
    project: str = ""                                 # DLC Builder project (.rs2dlc)
    fetched: dict = field(default_factory=dict)       # cover / lyrics files looked up on the web


# ─────────────────────────────────────────────────────────────────────────────
#  Environment probe
# ─────────────────────────────────────────────────────────────────────────────

def _try_version(mod: str, dist: str | None = None):
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", None)
        if v is None:
            from importlib.metadata import version
            v = version(dist or mod)
        return str(v)
    except Exception:
        return None


def check_environment() -> dict:
    """Report which optional dependencies are importable and which device
    Demucs would use. Never raises."""
    info = {
        "stems2midi": __version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": np.__version__,
        "torch": _try_version("torch"),
        "demucs": _try_version("demucs"),
        "librosa": _try_version("librosa"),
        "soundfile": _try_version("soundfile"),
        "mido": _try_version("mido"),
        "torchaudio": _try_version("torchaudio"),
        "pyphen": _try_version("pyphen"),
        "guitarpro": _try_version("guitarpro", "PyGuitarPro"),
        "basic_pitch": None,
        "ffmpeg": shutil.which("ffmpeg"),
        "cuda": False,
        "gpu": None,
        "mps": False,
    }
    try:
        import importlib.util
        if importlib.util.find_spec("basic_pitch") is not None:
            from importlib.metadata import version
            info["basic_pitch"] = version("basic-pitch")
    except Exception:
        info["basic_pitch"] = None
    if info["torch"]:
        try:
            import torch
            info["cuda"] = bool(torch.cuda.is_available())
            if info["cuda"]:
                info["gpu"] = torch.cuda.get_device_name(0)
            info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception:
            pass
    info["can_separate"] = bool(info["torch"] and info["demucs"] and info["soundfile"])
    info["can_transcribe"] = bool(info["librosa"] and info["mido"])
    info["device"] = "cuda" if info["cuda"] else ("mps" if info["mps"] else "cpu")
    try:
        info["lyrics_mms"] = bool(info["torchaudio"]) and _sibling("lyrics").mms_available()
    except Exception:
        info["lyrics_mms"] = False
    return info


def resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
#  Audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str | Path, samplerate: int, channels: int = 2) -> np.ndarray:
    """Decode any common audio file to float32 ``(channels, samples)`` at
    ``samplerate``. Tries soundfile (libsndfile ≥1.1 reads MP3), then librosa
    (audioread / ffmpeg), then a direct ffmpeg pipe."""
    path = str(path)
    data, sr = None, None
    errors = []
    try:
        import soundfile as sf
        d, sr = sf.read(path, dtype="float32", always_2d=True)
        data = d.T
    except Exception as e:  # unsupported container (m4a, wma…)
        errors.append(f"soundfile: {e}")
    if data is None:
        try:
            import librosa
            d, sr = librosa.load(path, sr=None, mono=False)
            data = np.atleast_2d(d).astype(np.float32)
        except Exception as e:
            errors.append(f"librosa: {e}")
    if data is None and shutil.which("ffmpeg"):
        try:
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", str(channels),
                 "-ar", str(samplerate), "-"],
                capture_output=True, check=True,
            ).stdout
            data = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).T.copy()
            sr = samplerate
        except Exception as e:
            errors.append(f"ffmpeg: {e}")
    if data is None:
        raise RuntimeError("Could not decode audio file. Install ffmpeg for m4a/aac/wma. " + " | ".join(errors))

    # channel layout
    if data.shape[0] == 1 and channels == 2:
        data = np.repeat(data, 2, axis=0)
    elif data.shape[0] > channels:
        data = data[:channels] if channels > 1 else data.mean(axis=0, keepdims=True)
    elif channels == 1 and data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)

    if sr != samplerate:
        data = resample(data, sr, samplerate)
    return np.ascontiguousarray(data, dtype=np.float32)


def resample(data: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return data
    try:
        import soxr  # fast, ships with librosa
        return soxr.resample(data.T, sr_from, sr_to).T.astype(np.float32)
    except Exception:
        pass
    try:
        import julius, torch  # ships with demucs
        return julius.resample_frac(torch.from_numpy(data), sr_from, sr_to).numpy()
    except Exception:
        pass
    import librosa
    return librosa.resample(data, orig_sr=sr_from, target_sr=sr_to)


def write_audio(path: Path, data: np.ndarray, samplerate: int, fmt: str = "wav") -> Path:
    """Write ``(channels, samples)`` float audio. Rescales if it would clip."""
    import soundfile as sf
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0.999:
        data = data * (0.999 / peak)
    path = path.with_suffix("." + fmt)
    subtype = "PCM_24" if fmt in ("wav", "flac") else None
    sf.write(str(path), data.T, samplerate, subtype=subtype)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Source separation (Demucs)
# ─────────────────────────────────────────────────────────────────────────────

def _load_demucs_model(name: str, repo: str | None = None):
    from demucs.pretrained import get_model
    model = get_model(name, repo=Path(repo) if repo else None)
    model.eval()
    return model


def separate(input_path: str | Path, model_name: str = DEFAULT_MODEL, device: str = "auto",
             shifts: int = 1, overlap: float = 0.25, progress: ProgressCB = None,
             repo: str | None = None, _model=None) -> tuple[dict, int, str]:
    """Run Demucs on ``input_path``. Returns ``({stem: (ch, n) ndarray}, sr, device)``.

    ``_model`` lets tests inject an (untrained) model without downloading weights.
    """
    import inspect
    import torch
    from demucs.apply import BagOfModels, apply_model

    cb = progress or (lambda *_: None)
    cb(0.0, "model", f"Loading Demucs model '{model_name}' (first run downloads it)…")
    model = _model if _model is not None else _load_demucs_model(model_name, repo)
    sr = int(model.samplerate)
    n_ch = int(getattr(model, "audio_channels", 2))

    cb(0.0, "decode", "Decoding audio…")
    wav = load_audio(input_path, sr, channels=n_ch)
    ref = wav.mean(axis=0)
    mean, std = float(ref.mean()), float(ref.std()) or 1.0
    mix = torch.from_numpy((wav - mean) / std)[None]

    # ── progress accounting ──────────────────────────────────────────────
    n_models = len(model.models) if isinstance(model, BagOfModels) else 1
    if isinstance(model, BagOfModels):
        seg = min(float(getattr(m, "segment", 8.0) or 8.0) for m in model.models)
    else:
        seg = float(getattr(model, "segment", 8.0) or 8.0)
    stride = max(1, int((1 - overlap) * int(sr * seg)))
    n_chunks = max(1, math.ceil(mix.shape[-1] / stride))
    total = n_models * max(1, shifts) * n_chunks
    done = {"n": 0, "t": 0.0}

    def _tick():
        done["n"] += 1
        now = time.monotonic()
        if now - done["t"] > 0.4 or done["n"] >= total:
            done["t"] = now
            cb(min(1.0, done["n"] / total), "separate", f"Separating stems… ({done['n']}/{total})")

    kwargs = dict(shifts=shifts, split=True, overlap=overlap)
    supports_cb = "callback" in inspect.signature(apply_model).parameters
    patched = None
    if supports_cb:  # demucs ≥ 4.1
        kwargs["callback"] = lambda d: _tick() if d.get("state") == "end" else None
    else:            # demucs 4.0.x: hijack its tqdm progress bar
        import demucs.apply as dapply

        class _Bar:
            @staticmethod
            def tqdm(it, **_kw):
                for x in it:
                    yield x
                    _tick()

        patched = dapply.tqdm
        dapply.tqdm = _Bar
        kwargs["progress"] = True

    dev = resolve_device(device)
    cb(0.0, "separate", f"Separating stems on {dev.upper()}…")
    try:
        try:
            with torch.no_grad():
                out = apply_model(model, mix, device=dev, **kwargs)[0]
        except RuntimeError as e:  # CUDA OOM → retry on CPU
            if dev != "cpu" and "out of memory" in str(e).lower():
                log.warning("GPU out of memory, retrying on CPU")
                cb(0.0, "separate", "GPU out of memory — retrying on CPU…")
                torch.cuda.empty_cache()
                done["n"] = 0
                dev = "cpu"
                with torch.no_grad():
                    out = apply_model(model, mix, device="cpu", **kwargs)[0]
            else:
                raise
    finally:
        if patched is not None:
            import demucs.apply as dapply
            dapply.tqdm = patched

    out = out * std + mean
    stems = {name: out[i].cpu().numpy() for i, name in enumerate(model.sources)}
    return stems, sr, dev


# ─────────────────────────────────────────────────────────────────────────────
#  Monophonic transcription (pYIN + onset-aware segmentation)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tuning(midi_f: np.ndarray, rms_db: np.ndarray, gate_db: float = -30.0) -> float:
    """Global tuning offset in semitones (−0.5…0.5): circular mean of how far
    the loud voiced frames sit from equal temperament. Recordings are often
    10–30 cents off A440; without this, ±1-semitone flicker splits notes."""
    ok = np.isfinite(midi_f) & (rms_db > gate_db)
    if ok.sum() < 50:
        return 0.0
    frac = midi_f[ok] - np.round(midi_f[ok])
    return float(np.angle(np.mean(np.exp(2j * np.pi * frac))) / (2 * np.pi))


def pick_onsets(env: np.ndarray, rms_db: np.ndarray, hop_s: float, rel: float = 0.3,
                rel_typical: float = 0.5, rise_db: float = 3.0, win_s: float = 2.0,
                wait_s: float = 0.06, gate_db: float = -42.0, percentile: float = 90.0,
                rise_curve: np.ndarray | None = None) -> list[int]:
    """Attack frames from the onset-strength curve, two-tier.

    * **Strong** attacks: the peak reaches `rel` of the loudest attack within
      ±`win_s` and rises `rel` above its recent floor.
    * **Weaker** attacks (re-plucks of the same note right after a big one,
      fast repeated 8ths/16ths): the peak reaches `rel_typical` of the
      *typical* attack height nearby (`percentile` of the local peaks) and the
      level actually rises ≥ `rise_db` (measured on `rise_curve`, a short
      23 ms-frame level in dB, when given) — that rise is what separates a
      real re-pluck from flux wiggles inside a ringing note.

    Tuned on five hand-made charts (benchmark.py): mean note-sequence F1
    0.73 → 0.76, with fast repeated-note songs 0.64 → 0.81.
    """
    import bisect
    from scipy.ndimage import maximum_filter1d

    env = np.asarray(env, dtype=float)
    n = len(env)
    w = max(3, int(win_s / hop_s))
    en_max = env / (maximum_filter1d(env, size=2 * w + 1) + 1e-9)
    pk = np.flatnonzero((env[1:-1] >= env[:-2]) & (env[1:-1] >= env[2:])) + 1
    pv = env[pk]
    wait = max(1, int(round(wait_s / hop_s)))
    rw = 4
    peaks: list[int] = []
    for i in range(2, n - 2):
        v = env[i]
        if v < env[i - 1] or v < env[i + 1] or v < env[i - 2] or v < env[i + 2]:
            continue
        if rms_db[min(len(rms_db) - 1, i + 3)] < gate_db:
            continue
        ok = en_max[i] >= rel and en_max[i] - np.median(en_max[max(0, i - 12):i]) >= rel
        if not ok:
            lo, hi = bisect.bisect_left(pk, i - w), bisect.bisect_right(pk, i + w)
            typical = float(np.percentile(pv[lo:hi], percentile)) if hi > lo else float(env.max() or 1.0)
            if v / (typical + 1e-9) >= rel_typical:
                rc = rms_db if rise_curve is None else rise_curve
                before = float(np.min(rc[max(0, i - rw):i + 1]))
                after = float(np.max(rc[i:min(n, i + rw + 1)]))
                ok = after - before >= rise_db
        if not ok:
            continue
        if peaks and i - peaks[-1] < wait:
            if v > env[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return peaks


def segment_by_onsets(midi_f: np.ndarray, rms_db: np.ndarray, onsets: Sequence[int], hop_s: float,
                      min_note_s: float = 0.05, body_s: float = 0.11, release_db: float = 18.0,
                      legato_s: float = 0.09, tuning: float | None = None) -> list[Note]:
    """Onset-first note segmentation (one note per attack).

    For each attack: pitch = median of the tuning-corrected pitch track over
    the first ~110 ms (the fretted note — slides and the flat drift of a
    dying string come later and are ignored); end = next attack, a
    `release_db` drop from the attack level, or the string going unvoiced.
    A new pitch that holds ≥ `legato_s` without a new attack becomes its own
    note (hammer-on / pull-off / slide). Attacks with no pitch (dead/ghost
    notes) are skipped.

    Benchmarked against hand-made charts (benchmark.py): note-sequence F1
    0.73 → 0.81 and 0.70 → 0.80 on two songs vs. the earlier pitch-run
    segmentation (mean 0.77 over five songs).
    """
    from scipy.ndimage import median_filter

    n = midi_f.size
    tun = estimate_tuning(midi_f, rms_db) if tuning is None else tuning
    mc = midi_f - tun
    lab = np.where(np.isfinite(mc), np.round(mc), -1).astype(int)
    lab = median_filter(lab, size=5, mode="nearest")
    min_f = max(1, int(round(min_note_s / hop_s)))
    body = max(3, int(round(body_s / hop_s)))
    leg = max(2, int(round(legato_s / hop_s)))
    ref = float(np.nanmax(rms_db)) if rms_db.size else 0.0
    ons = sorted(set(int(o) for o in onsets if 0 <= int(o) < n))
    notes: list[Note] = []

    def add(a: int, b: int, p: int, vel: int, legato: bool = False):
        if b - a >= min_f:
            notes.append(Note(round(a * hop_s, 4), round(b * hop_s, 4), int(p), vel, legato))

    for k, s in enumerate(ons):
        lim = ons[k + 1] if k + 1 < len(ons) else min(n, s + int(4.0 / hop_s))
        if lim - s < min_f:
            continue
        pk = float(np.max(rms_db[s:min(lim, s + 8)]))
        e, unv = lim, 0
        for j in range(s + 3, lim):
            if rms_db[j] < pk - release_db:
                e = j
                break
            unv = unv + 1 if not np.isfinite(mc[j]) else 0
            if unv >= 4 and j - s > body:
                e = j - 3
                break
        b0, b1 = s + 2, min(e, s + body)
        vals = mc[b0:b1]
        vals = vals[np.isfinite(vals)]
        if vals.size < 3:
            vals = mc[s:e]
            vals = vals[np.isfinite(vals)]
            if vals.size < 3:
                continue  # unpitched attack: dead note / slap noise
        p = int(np.round(np.median(vals)))
        vel = int(np.clip(50 + (pk - (ref - 30)) / 30 * 65, 30, 120))
        cur_s, cur_p, cur_leg, j = s, p, False, b1
        while j < e:
            if lab[j] >= 0 and lab[j] != cur_p and rms_db[j] > pk - 10:
                r = j
                while r < e and lab[r] == lab[j]:
                    r += 1
                seg_v = mc[j:r]
                seg_v = seg_v[np.isfinite(seg_v)]
                # a real fretted note holds still; a glide/sag sweeps through
                stable = seg_v.size >= leg and float(np.ptp(seg_v)) < 0.6
                if r - j >= leg and stable and abs(int(lab[j]) - cur_p) <= 7:
                    add(cur_s, j, cur_p, vel, cur_leg)
                    cur_s, cur_p, cur_leg = j, int(lab[j]), True
                j = r
            else:
                j += 1
        add(cur_s, e, cur_p, vel, cur_leg)
    return notes


SUSTAINED_RISE_DB = 4.0   # --sustained: a same-pitch note is a re-pluck only if the level rises this much


def transcribe_pyin(y: np.ndarray, sr: int, fmin: str = "B0", fmax: str = "G4",
                    min_note_ms: float = 60.0, gate_db: float = -45.0,
                    progress: ProgressCB = None, sustained: bool = False) -> list[Note]:
    """Monophonic transcription of a mono signal ``y`` at ``sr``. ``sustained``
    (ballads, long notes over strings / pads): a same-pitch note only counts as
    a re-pluck when the level clearly rises, and double triggers are merged."""
    import librosa

    cb = progress or (lambda *_: None)
    target_sr = 11025  # bass fundamentals < 400 Hz → plenty of headroom, 2x faster
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    hop = 128
    # 93 ms analysis frame: long enough for low B (31 Hz), short enough not to
    # blend neighbouring notes (a longer frame hears E1+E2 as one E1).
    frame = PYIN_FRAME
    hop_s = hop / sr

    f_lo, f_hi = librosa.note_to_hz(fmin), librosa.note_to_hz(fmax)
    # pYIN cost is O(frames); process in ~30 s blocks so we can report progress
    block = int(PYIN_BLOCK_S * sr) // hop * hop
    ctx = frame // hop * hop  # real audio on both sides of a block, not padding
    f0_parts = []
    n_blocks = max(1, math.ceil(len(y) / block))
    for b in range(n_blocks):
        a = b * block
        lo_i = max(0, a - ctx)
        last = b == n_blocks - 1
        seg = y[lo_i: len(y) if last else a + block + ctx]
        f0, vflag, _vprob = librosa.pyin(
            seg, fmin=f_lo, fmax=f_hi, sr=sr, frame_length=frame, hop_length=hop,
            fill_na=np.nan, center=True,
            max_transition_rate=PYIN_MAX_TRANSITION,
        )
        f0 = np.where(vflag, f0, np.nan)[(a - lo_i) // hop:]
        f0_parts.append(f0 if last else f0[: block // hop])
        cb((b + 1) / n_blocks * 0.85, "transcribe", f"Tracking pitch… {int((b + 1) / n_blocks * 100)}%")
    f0 = np.concatenate(f0_parts)
    n = f0.size

    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0][:n]
    if rms.size < n:
        rms = np.pad(rms, (0, n - rms.size))
    rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)

    # short STFT for onsets: 16th notes at 140 BPM are only ~107 ms apart
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_fft=ONSET_NFFT, fmax=2000)[:n]
    if onset_env.size < n:
        onset_env = np.pad(onset_env, (0, n - onset_env.size))
    # short-frame level (23 ms) to see the dip/rise of a re-plucked note
    rms_s = librosa.feature.rms(y=y, frame_length=256, hop_length=hop, center=True)[0][:n]
    if rms_s.size < n:
        rms_s = np.pad(rms_s, (0, n - rms_s.size))
    rise_curve = librosa.amplitude_to_db(rms_s + 1e-10, ref=np.max)
    onsets = pick_onsets(onset_env, rms_db, hop_s, rel=ONSET_REL, gate_db=gate_db, rise_curve=rise_curve)

    with np.errstate(divide="ignore", invalid="ignore"):
        midi_f = 69 + 12 * np.log2(f0 / 440.0)
    cb(0.95, "transcribe", "Segmenting notes…")
    notes = segment_by_onsets(midi_f, rms_db, onsets, hop_s, min_note_s=min_note_ms / 1000.0)
    if sustained:
        notes = merge_double_triggers(notes)
    return merge_weak_repeats(notes, onset_env, hop_s, REPEAT_MIN_ATTACK, rise_curve=rise_curve,
                              min_rise_db=SUSTAINED_RISE_DB if sustained else REPEAT_MIN_RISE_DB)


def merge_double_triggers(notes: list[Note], within: float = 0.15) -> list[Note]:
    """A very short note followed, at the same pitch, by another one right
    after (< ``within`` s from its start) is one note heard twice."""
    out: list[Note] = []
    for n in notes:
        if out and n.pitch == out[-1].pitch and out[-1].duration < 0.12 and n.start - out[-1].start < within:
            p = out[-1]
            out[-1] = Note(p.start, max(p.end, n.end), p.pitch, p.velocity, p.legato)
            continue
        out.append(n)
    return out


def stem_onsets(y: np.ndarray, sr: int, gate_db: float = -45.0) -> np.ndarray:
    """Attack times (s) of a mono stem — the same detector the transcription
    uses (``pick_onsets``)."""
    import librosa
    target_sr = 11025
    if sr != target_sr:
        y = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    hop = 128
    rms = librosa.feature.rms(y=y, frame_length=PYIN_FRAME, hop_length=hop, center=True)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_fft=ONSET_NFFT, fmax=2000)
    rs = librosa.feature.rms(y=y, frame_length=256, hop_length=hop, center=True)[0]
    rise = librosa.amplitude_to_db(rs + 1e-10, ref=np.max)
    n = min(len(env), len(rms_db), len(rise))
    on = pick_onsets(env[:n], rms_db[:n], hop / sr, rel=ONSET_REL, gate_db=gate_db, rise_curve=rise[:n])
    return np.asarray(on, float) * hop / sr


def notes_from_file(path: str | Path, y: np.ndarray, sr: int,
                    track: str | None = None) -> tuple[list[Note], dict]:
    """Notes of another transcriber's MusicXML / MIDI / Guitar Pro file, lined
    up with the stem ``y`` (see notesimport.py). A Guitar Pro tab also gives
    each note's string and slap/pop/hammer-on/pull-off/slide, and ``info``
    its track and tuning ("gp")."""
    ni = _sibling("notesimport")
    gp = None
    if Path(path).suffix.lower() in ni.GP_EXTS:
        raw, gp = ni.read_gp(path, track)
    else:
        raw = ni.read_notes(path)
    onsets = stem_onsets(y, sr)

    def ref_notes():            # only when no single tempo ratio fits: a quick transcription to follow
        return [(n.start, n.pitch) for n in transcribe_pyin(y, sr)]
    song, info = ni.line_up([r[:3] for r in raw], onsets, pitch_score=ni.stem_pitch_score(y, sr),
                            ref_notes=ref_notes)
    song, info["shortened"] = ni.trim_to_sound(song, y, sr)
    notes = []
    dur = len(y) / sr
    info["outside"] = 0
    for (t, d, m), i in zip(song, info["src"]):
        if t < 0 or t >= dur:            # before / after the audio (count-in, a longer ending)
            info["outside"] += 1
            continue
        x = raw[i][3] if len(raw[i]) > 3 else {}
        if x.get("dead"):
            continue
        notes.append(Note(start=t, end=t + d, pitch=m, tech=x.get("tech"), legato=bool(x.get("lg")),
                          string=x.get("s")))
    info["gp"] = gp
    return make_monophonic(notes, prefer="low"), info


def _mmss(t: float) -> str:
    m, sec = divmod(int(round(max(0.0, float(t)))), 60)
    return f"{m}:{sec:02d}"


def song_bars(notes: list[Note], beats, bpm: float, duration: float,
              shift: float = 0.0) -> list[tuple[int, float, float]]:
    """[(bar number as in the XML / EOF, start, end)] in the song's time — the
    bars of the Rocksmith beat map (an EOF lead-in adds bars before bar 1)."""
    rx = _sibling("rsexport")
    pl = {"notes": [{"t": n.start, "d": n.duration, "m": n.pitch} for n in notes],
          "beats": [float(b) for b in beats], "bpm": bpm, "duration": duration}
    downs = [t - shift for t, m in rx.beat_map_for(pl, shift=shift) if m > 0]
    return [(k + 1, a, b) for k, (a, b) in enumerate(zip(downs, downs[1:])) if b > a and b > 0]


def _compact_ranges(nums: list[int]) -> str:
    """[3, 4, 5, 9] → "3-5, 9"."""
    out, run = [], []
    for k in sorted(set(nums)):
        if run and k == run[-1] + 1:
            run.append(k)
        else:
            if run:
                out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = [k]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ", ".join(out)


def parse_bar_spec(spec: str, bars: list[tuple[int, float, float]],
                   shift: float = 0.0) -> list[tuple[float, float]]:
    """"12-30, 45" (bar numbers, as in EOF) or "1:02-1:30" (times in the song /
    MP3, as a player or a tab site gives them) → time ranges in the song.
    ``shift`` is kept for callers; times are not moved by it."""
    out = []
    by_num = {k: (a, b) for k, a, b in bars}
    for item in [x.strip() for x in str(spec or "").split(",") if x.strip()]:
        if ":" in item or "." in item:
            (a, b), = _fretting().parse_sections(item)
            out.append((a, b))
            continue
        lo, _, hi = item.partition("-")
        try:
            k0, k1 = int(lo), int(hi or lo)
        except ValueError:
            raise ValueError(f"not a bar range: {item!r} (use 12-30 or 1:02-1:30)")
        if k1 < k0 or k0 not in by_num:
            raise ValueError(f"bars {item}: the song has bars 1–{max(by_num) if by_num else 0}")
        out.append((by_num[k0][0], by_num.get(k1, by_num[max(by_num)])[1]))
    return out


def merge_notes(auto: list[Note], other: list[Note], ranges: list[tuple[float, float]]) -> list[Note]:
    """``auto`` notes outside ``ranges``, ``other`` notes inside them."""
    def inside(t):
        return any(a - 0.03 <= t < b - 0.03 for a, b in ranges)
    keep = [n for n in auto if not inside(n.start)] + [n for n in other if inside(n.start)]
    return make_monophonic(sorted(keep, key=lambda n: n.start), prefer="low")


# A same-pitch note that starts right where the previous one ends is kept as a
# re-pluck only if its attack reaches this share of the song's strong attacks
# (97th percentile of all onset-curve peaks). Quieter "attacks" are the wobble of a
# held note — e.g. a long note fading out at the end of a song, which was
# coming out as a string of short identical notes.
REPEAT_MIN_ATTACK = float(os.environ.get("STEMS2MIDI_REPEAT_MIN_ATTACK", 0.25))


REPEAT_MIN_RISE_DB = float(os.environ.get("STEMS2MIDI_REPEAT_MIN_RISE", 0.0))


def merge_weak_repeats(notes: list[Note], onset_env: np.ndarray, hop_s: float,
                       min_attack: float = REPEAT_MIN_ATTACK, gap_s: float = 0.03,
                       rise_curve: np.ndarray | None = None,
                       min_rise_db: float = REPEAT_MIN_RISE_DB) -> list[Note]:
    if len(notes) < 2 or min_attack <= 0 or not onset_env.size:
        return notes
    env = np.asarray(onset_env, dtype=float)
    pk = np.flatnonzero((env[1:-1] >= env[:-2]) & (env[1:-1] > env[2:])) + 1
    if not pk.size:
        return notes
    typical = float(np.percentile(env[pk], 97)) or 1.0
    out = [notes[0]]
    for n in notes[1:]:
        prev = out[-1]
        if n.pitch == prev.pitch and n.start - prev.end <= gap_s:
            f = int(round(n.start / hop_s))
            att = float(np.max(env[max(0, f - 2):f + 3])) if f < env.size else 0.0
            rise = np.inf
            if rise_curve is not None and min_rise_db > 0 and 4 <= f < len(rise_curve) - 4:
                rise = float(np.max(rise_curve[f:f + 5]) - np.min(rise_curve[f - 4:f + 1]))
            if att < min_attack * typical or rise < min_rise_db:
                out[-1] = Note(prev.start, max(prev.end, n.end), prev.pitch, prev.velocity, prev.legato)
                continue
        out.append(n)
    return out


def make_monophonic(notes: list[Note], prefer: str = "low") -> list[Note]:
    """Collapse overlapping notes into a single line. For bass, overlaps are
    usually octave/harmonic ghosts, so the lowest note wins by default."""
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: (n.start, n.pitch if prefer == "low" else -n.velocity))
    out: list[Note] = []
    for n in notes:
        if out and n.start < out[-1].end - 1e-3:
            last = out[-1]
            better = (n.pitch < last.pitch) if prefer == "low" else (n.velocity > last.velocity)
            if n.start - last.start < 0.05:  # practically simultaneous: pick one
                if better:
                    out[-1] = n
                continue
            if better or n.end > last.end:
                last.end = n.start  # truncate previous, new note takes over
                if last.duration < 0.03:
                    out.pop()
                out.append(n)
            continue
        out.append(n)
    return [n for n in out if n.duration > 0.02]


def transcribe_basic_pitch(path: str | Path, fmin: str, fmax: str, min_note_ms: float,
                           mono: bool, progress: ProgressCB = None) -> list[Note]:
    import librosa
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    cb = progress or (lambda *_: None)
    cb(0.1, "transcribe", "Running Basic Pitch…")
    _out, _midi, events = predict(
        str(path), ICASSP_2022_MODEL_PATH,
        minimum_note_length=min_note_ms,
        minimum_frequency=float(librosa.note_to_hz(fmin)),
        maximum_frequency=float(librosa.note_to_hz(fmax)),
        multiple_pitch_bends=False, melodia_trick=True,
    )
    notes = [Note(float(s), float(e), int(p), int(np.clip(round(a * 127), 20, 127)))
             for s, e, p, a, *_ in events]
    notes.sort(key=lambda n: (n.start, n.pitch))
    cb(0.95, "transcribe", f"Basic Pitch found {len(notes)} notes")
    return make_monophonic(notes) if mono else notes


# ─────────────────────────────────────────────────────────────────────────────
#  Tempo, quantisation, MIDI
# ─────────────────────────────────────────────────────────────────────────────

def align_shift(folder: str | Path, audio: str | Path, sr: int = 11025) -> float:
    """Shift (s) that puts a result folder's notes on ``audio``; prints it."""
    ref = result_audio(Path(folder), sr)
    if ref is None:
        raise ValueError(f"no stems (.wav/.flac) in {folder} to compare with")
    tgt = load_audio(audio, sr, channels=1)[0]
    off, conf = find_offset(ref, tgt, sr)
    print(f"Alignment to {Path(audio).name}: {off:+.3f} s (confidence {conf:g}"
          + (" — low, check the result" if conf < 1.03 else "") + ")")
    return off


def find_offset(ref: np.ndarray, target: np.ndarray, sr: int, max_shift_s: float = 30.0) -> tuple[float, float]:
    """Seconds by which the same music starts later in ``target`` than in
    ``ref`` (e.g. EOF's guitar.ogg with added lead-in vs. the original MP3),
    from the cross-correlation of their low-band onset curves. ``ref`` may be
    just the bass stem. Returns (offset_s, confidence: peak / next best peak)."""
    import librosa
    hop = 256

    def env(y):
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop, n_mels=24, fmax=300)
        e = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr, hop_length=hop)
        return (e - e.mean()) / (e.std() + 1e-9)

    a, b = env(ref), env(target)
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    xc = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    lag_max = int(max_shift_s * sr / hop)
    lags = np.r_[np.arange(0, lag_max + 1), np.arange(-lag_max, 0)]
    vals = np.r_[xc[:lag_max + 1], xc[n - lag_max:]]
    i = int(np.argmax(vals))
    k = lags[i]
    # sub-frame peak (parabola through the neighbours)
    l, c, r = (xc[(k - 1) % n], xc[k % n], xc[(k + 1) % n])
    frac = 0.5 * (l - r) / (l - 2 * c + r) if (l - 2 * c + r) != 0 else 0.0
    others = np.abs(lags - k) > int(0.2 * sr / hop)
    second = float(np.max(vals[others])) if others.any() else 0.0
    conf = float(vals[i] / second) if second > 0 else float("inf")
    coarse = (k + frac) * hop / sr
    # refine on a 4× finer onset curve, ±60 ms around the coarse lag
    hop = fine_hop = 64          # env() picks up the new hop
    a, b = env(ref), env(target)
    k0 = int(round(coarse * sr / fine_hop))
    span = int(0.06 * sr / fine_hop)
    best, best_k = -np.inf, k0
    for kk in range(k0 - span, k0 + span + 1):
        if kk >= 0:
            m = min(len(a), len(b) - kk)
            v = float(np.dot(a[:m], b[kk:kk + m])) / max(m, 1) if m > 0 else -np.inf
        else:
            m = min(len(a) + kk, len(b))
            v = float(np.dot(a[-kk:-kk + m], b[:m])) / max(m, 1) if m > 0 else -np.inf
        if v > best:
            best, best_k = v, kk
    return round(best_k * fine_hop / sr, 3), round(conf, 2)


def result_audio(folder: Path, sr: int) -> np.ndarray | None:
    """The music of a result folder, rebuilt from its stems (bass + backing
    when there is a backing track, else every stem saved)."""
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".wav", ".flac"))
    backing = [p for p in files if p.stem.startswith("backing_no_")]
    if backing:
        removed = backing[0].stem[len("backing_no_"):].split("_")
        files = [backing[0]] + [p for p in files if p.stem in removed]
    else:
        files = [p for p in files if not p.stem.startswith("backing")]
    if not files:
        return None
    parts = [load_audio(p, sr, channels=1)[0] for p in files]
    n = max(len(x) for x in parts)
    return np.sum([np.pad(x, (0, n - len(x))) for x in parts], axis=0)


def estimate_tempo(y: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """Return (bpm, beat_times). Falls back to 120 BPM on silence."""
    import librosa
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    if not np.isfinite(bpm) or bpm <= 0:
        return 120.0, np.array([])
    beats = np.asarray(beats, dtype=float)
    # beat trackers love double/half time; fold into the range most songs are
    # notated in (use --bpm when the song really is > 170 BPM)
    while bpm >= 170:
        bpm /= 2
        beats = beats[::2]
    while bpm < 60:
        bpm *= 2
        if beats.size > 1:
            mids = (beats[:-1] + beats[1:]) / 2
            beats = np.sort(np.concatenate([beats, mids]))
    return round(bpm, 2), beats


def quantize_notes(notes: list[Note], beats: np.ndarray, bpm: float, grid: int) -> list[Note]:
    """Snap note boundaries to a beat-synchronous grid. ``grid`` is the
    subdivision per whole note (16 → sixteenths). Following the tracked beats
    (not a fixed tempo) keeps drifting live recordings in sync."""
    if grid <= 0 or not notes:
        return notes
    per_beat = max(1, grid // 4)
    period = 60.0 / bpm
    if beats is None or len(beats) < 2:
        beats = np.arange(0.0, notes[-1].end + 2 * period, period)
    beats = np.asarray(beats, dtype=float)
    # extend the beat grid to cover the whole song at the local tempo
    head = np.arange(beats[0] - period, -period, -period)[::-1]
    tail_start = beats[-1] + period
    tail = np.arange(tail_start, max(tail_start, notes[-1].end + 2 * period), period)
    full = np.concatenate([head, beats, tail])
    grid_pts = np.concatenate([
        np.linspace(a, b, per_beat, endpoint=False) for a, b in zip(full[:-1], full[1:])
    ] + [full[-1:]])
    # Beat trackers report beats a few tens of ms late/early relative to the
    # actual attacks; a phase error near half a grid step snaps notes into the
    # wrong slot. Re-phase the grid so it best fits the detected note onsets.
    step = period / per_beat
    starts = np.array([n.start for n in notes])
    best, best_cost = 0.0, np.inf
    for shift in np.linspace(-step / 2, step / 2, 41):
        g = grid_pts + shift
        idx = np.clip(np.searchsorted(g, starts), 1, len(g) - 1)
        d = np.minimum(np.abs(starts - g[idx - 1]), np.abs(starts - g[idx]))
        cost = float(np.mean(np.minimum(d, step / 2)) + 0.05 * abs(shift))
        if cost < best_cost:
            best, best_cost = float(shift), cost
    grid_pts = grid_pts + best
    grid_pts = np.maximum(grid_pts[grid_pts > -step / 2], 0.0)

    def snap(t: float) -> int:
        return int(np.argmin(np.abs(grid_pts - t)))

    out: list[Note] = []
    for n in notes:
        a = snap(n.start)
        b = max(a + 1, snap(n.end))
        b = min(b, len(grid_pts) - 1)
        if b <= a:
            continue
        q = Note(float(grid_pts[a]), float(grid_pts[b]), n.pitch, n.velocity)
        if out and out[-1].pitch == q.pitch and abs(out[-1].start - q.start) < 1e-6:
            continue  # two fragments collapsed into one slot
        if out and out[-1].end > q.start:
            out[-1].end = q.start
        out.append(q)
    return [n for n in out if n.end > n.start]


def write_midi(notes: list[Note], path: Path, bpm: float = 120.0, program: int = 33,
               track_name: str = "Bass", channel: int = 0, ppq: int = 480) -> Path:
    import mido

    mid = mido.MidiFile(ticks_per_beat=ppq, type=1)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="stems2midi", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)

    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=track_name, time=0))
    tr.append(mido.Message("program_change", program=int(program), channel=channel, time=0))
    ticks_per_s = ppq * bpm / 60.0
    events = []
    for n in notes:
        on = int(round(n.start * ticks_per_s))
        off = max(on + 1, int(round(n.end * ticks_per_s)))
        p = int(np.clip(n.pitch, 0, 127))
        events.append((on, 1, "note_on", p, int(np.clip(n.velocity, 1, 127))))
        events.append((off, 0, "note_off", p, 0))
    events.sort(key=lambda e: (e[0], e[1]))  # note_off before note_on on the same tick
    last = 0
    for tick, _o, kind, p, v in events:
        tr.append(mido.Message(kind, note=p, velocity=v, channel=channel, time=tick - last))
        last = tick
    tr.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(tr)
    path = Path(path)
    mid.save(str(path))
    return path


def note_name(m: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[m % 12]}{m // 12 - 1}"


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _truthy(v) -> bool:
    """A flag that may come from stems2midi.ini as text ("yes", "no", "0"…)."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "yes", "true", "on", "sim", "s")
    return bool(v)


def slap_part_names(spec: str | None) -> list[str]:
    """"verse, chorus" (names of song parts) → ["verse", "chorus"]; [] for
    'all', 'auto' or time ranges."""
    if not spec:
        return []
    toks = [t.strip().lower() for t in str(spec).replace(";", ",").split(",") if t.strip()]
    if toks and all(t.isalpha() for t in toks) and toks not in (["all"], ["auto"]):
        return toks
    return []


def auto_parts(y: np.ndarray, sr: int, payload: dict) -> list[dict]:
    """Song parts from the audio (structure.py) on the XML's bar lines."""
    if y.ndim > 1:
        y = y.mean(axis=0)
    bm = _sibling("rsexport").beat_map_for(payload)
    t = [b for b, _m in bm]
    downs = [i for i, (_b, m) in enumerate(bm) if m > 0]
    return _sibling("structure").analyze(y, sr, t, downs)


def _tab_payload(fret, tuning, bpm, beats, duration) -> dict:
    """What rsexport needs from a fretting result (before notes.json exists)."""
    return {
        "bpm": bpm, "duration": float(duration), "beats": [float(b) for b in beats],
        "tuning": {"strings": list(tuning.strings), "offsets": tuning.offsets()},
        "anchors": [{"t": a.time, "f": a.fret, "w": a.width} for a in fret.anchors],
        "notes": [{"t": n.start, "d": n.end - n.start, "m": n.pitch, "v": n.velocity,
                   "s": n.string, "f": n.fret, **({"lg": 1} if n.legato else {})} for n in fret.notes],
    }


def _sibling(name: str):
    """Sibling module, importable however this file was launched (CLI,
    tools/ launcher via runpy, plugin subprocess, tests)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import importlib
    return importlib.import_module(name)


def _fretting():
    return _sibling("fretting")


def _expand_stems(stems: Iterable[str], available: Iterable[str]) -> list[str]:
    available = list(available)
    wanted = []
    for s in stems:
        s = s.strip().lower()
        if not s:
            continue
        if s == "all":
            return available
        wanted.append(s)
    return [s for s in available if s in wanted]


def pick_engine(stem: str, engine: str, has_basic_pitch: bool) -> str:
    if engine in ("pyin", "basic-pitch"):
        if engine == "basic-pitch" and not has_basic_pitch:
            raise RuntimeError("basic-pitch is not installed (pip install basic-pitch)")
        return engine
    # auto: monophonic tracker for bass/vocals, polyphonic for the rest
    if stem in ("bass", "vocals") or not has_basic_pitch:
        return "pyin"
    return "basic-pitch"


def _gp_tuning(gp: dict, tab_tuning, auto_tuning: bool, res: Result):
    """Track note + the Guitar Pro file's tuning for the tab (its strings kept).
    Returns (tab_tuning, gp_strings, auto_tuning)."""
    gp_strings = None
    res.warnings.append(
        f"Guitar Pro track {gp['track_no']} '{gp['track']}'"
        + (f" (of {len(gp['tracks'])}: {', '.join(gp['tracks'])}; --track to choose)"
           if len(gp["tracks"]) > 1 else "")
        + (" — written an octave up (guitar tuning), moved down." if gp.get("octave") else ".")
        + (f" {gp['muted']} dead note(s) left out." if gp.get("muted") else "")
        + (" Its repeat signs are read once (Guitar Pro 6–8 files): if the song plays them, "
           "save the tab as .gp5 (Guitar Pro / TuxGuitar) and use that." if gp.get("repeats_once")
           else ""))
    if gp.get("tuning") and (auto_tuning or (tab_tuning is not None
                                             and list(tab_tuning.strings) == gp["tuning"])):
        tab_tuning = _fretting().parse_tuning(gp["tuning"])
        named = [k for k, v in _fretting().PRESETS.items() if tuple(v) == tab_tuning.strings]
        if named:
            tab_tuning = _fretting().parse_tuning(named[0])
        gp_strings = tuple(gp["tuning"])
        auto_tuning = False
        res.warnings.append(f"Tuning from the Guitar Pro file: {tab_tuning.label()}; "
                            f"strings and frets as in the tab.")
    return tab_tuning, gp_strings, auto_tuning


def _add_hints(res: Result, hints: list[dict], out: Path, opts: Options, input_path: Path,
               payload: dict | None = None) -> None:
    sg = _sibling("suggest")
    res.suggestions = list(res.suggestions) + [h for h in hints if h not in res.suggestions]
    res.rerun = sg.command(res.suggestions, str(Path(out).resolve()),
                           f"--shift {opts.rs_shift:g}" if opts.rs_shift else "",
                           source=str(Path(input_path).resolve()))
    if payload is not None:
        payload["suggestions"] = res.suggestions
    for h in hints:
        res.warnings.append(f"Worth trying {sg._arg(h)}: {h['why']}.")
    new_sep = any(h.get("source") for h in res.suggestions)
    res.warnings = [w for w in res.warnings if not w.startswith(("Re-run on the result folder", "Run again:"))]
    res.warnings.append(("Run again: " if new_sep else "Re-run on the result folder (no new separation): ")
                        + res.rerun)


# ── DLC Builder project, song audio, cover / lyrics ──────────────────────

def song_audio_name(shift: float = 0.0) -> str:
    """song.ogg, or song_lead5s.ogg when the XML is moved for EOF's lead-in."""
    return "song.ogg" if not shift else f"song_lead{shift:g}s.ogg"


def write_song_audio(folder: Path, audio: np.ndarray, sr: int, shift: float = 0.0) -> str:
    """The song for DLC Builder, moved like the XML (silence added for a
    lead-in, cut for a negative shift). Ogg Vorbis (wav if that fails)."""
    import soundfile as sf
    a = np.atleast_2d(np.asarray(audio, np.float32))
    n = int(round(abs(shift) * sr))
    if shift > 0:
        a = np.concatenate([np.zeros((a.shape[0], n), np.float32), a], axis=1)
    elif shift < 0:
        a = a[:, n:]
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 0.999:
        a = a * (0.999 / peak)
    name = song_audio_name(shift)
    # in blocks: libsndfile's Vorbis encoder crashes the whole process (segfault)
    # when a long song is written in one call
    block = 32768
    try:
        with sf.SoundFile(str(folder / name), "w", sr, a.shape[0], format="OGG", subtype="VORBIS") as f:
            for i in range(0, a.shape[1], block):
                f.write(a[:, i:i + block].T)
    except Exception:  # noqa: BLE001 — libsndfile without Vorbis
        (folder / name).unlink(missing_ok=True)
        name = name[:-4] + ".wav"
        sf.write(str(folder / name), a.T, sr, subtype="PCM_16")
    return name


def song_audio_for(folder: Path, shift: float = 0.0, source: str | Path | None = None) -> tuple[str, str]:
    """(audio file for the project, note). Reuses an earlier song*.ogg made
    for this shift; else rebuilds it from the song file (next to the folder)
    or from the stems (bass + backing)."""
    folder = Path(folder)
    name = song_audio_name(shift)
    for cand in (name, name[:-4] + ".wav"):
        if (folder / cand).is_file():
            return cand, ""
    src = None
    if source:
        for p in (Path(source), folder.parent / Path(source).name, folder / Path(source).name):
            if p.is_file() and p.suffix.lower() not in (".json",):
                src = p
                break
    try:
        if src is not None:
            if src.suffix.lower() in _sibling("dlcproject").AUDIO_EXTS and not shift and src.parent == folder:
                return src.name, ""
            return write_song_audio(folder, load_audio(src, 44100, channels=2), 44100, shift), ""
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".wav", ".flac"))
        backing = [p for p in files if p.stem.startswith("backing_no_")]
        if backing:
            removed = backing[0].stem[len("backing_no_"):].split("_")
            parts = [backing[0]] + [p for p in files if p.stem in removed]
            if len(parts) == 1 + len(removed):
                data = [load_audio(p, 44100, channels=2) for p in parts]
                n = max(d.shape[1] for d in data)
                mix = np.sum([np.pad(d, ((0, 0), (0, n - d.shape[1]))) for d in data], axis=0)
                return write_song_audio(folder, mix, 44100, shift), ""
    except Exception as e:  # noqa: BLE001
        return "", f"song audio not made ({e}) — set it in DLC Builder"
    return "", "no song audio found for the project — set it in DLC Builder (wav/ogg)"


def write_dlc_project(folder: str | Path, charter: str = "", meta: dict | None = None, shift: float = 0.0,
                      mix: np.ndarray | None = None, sr: int = 44100,
                      source: str | Path | None = None) -> tuple[str, list[str]]:
    """Song audio + ``Artist_Title.rs2dlc`` in ``folder``; (file name, warnings)."""
    folder = Path(folder)
    warns = []
    if mix is not None:
        audio, note = write_song_audio(folder, mix, sr, shift), ""
    else:
        audio, note = song_audio_for(folder, shift, source)
    if note:
        warns.append(note)
    path = _sibling("dlcproject").write(folder, charter=charter, meta=meta, audio=audio)
    who = f" (author {charter})" if charter else " (author: DLC Builder's charter name — set charter in stems2midi.ini)"
    warns.append(f"DLC Builder project: {path.name}{who} — open it in DLC Builder, add tones, build.")
    return path.name, warns


def _audio_seconds(path: Path) -> float:
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001
        return 0.0


def _fetch_meta(folder: Path, input_path: Path, opts: Options, res: Result, log=None) -> None:
    """Cover + lyrics into ``folder``; fills missing album/year and, without
    --lyrics, uses the lyrics found."""
    mf = _sibling("metafetch")
    mf.set_version(__version__)
    guess = _sibling("rsexport").guess_meta(input_path.name)
    artist = opts.meta.get("artist") or guess.get("artist") or ""
    title = opts.meta.get("title") or guess.get("title") or ""
    got = mf.fetch(folder, artist, title, opts.meta.get("album") or "", _audio_seconds(input_path),
                   log=log or (lambda m: None))
    res.fetched = got
    for n in got["notes"]:
        res.warnings.append(f"Web lookup: {n}.")
    if got["album"] and not opts.meta.get("album"):
        opts.meta["album"] = got["album"]
    if got["year"] and not opts.meta.get("year"):
        opts.meta["year"] = got["year"]
    if got["lyrics"] and not opts.lyrics:
        text = (folder / got["lyrics"]).read_text(encoding="utf-8")
        if text.strip():
            opts.lyrics = text
            res.warnings.append(f"Lyrics from the web ({got['lyrics']}) lined up with the vocals — check them; "
                                f"edit the file and run on the folder with --lyrics to redo.")


def run_pipeline(input_path: str | Path, out_dir: str | Path | None = None,
                 opts: Options | None = None, progress: ProgressCB = None,
                 _model=None) -> Result:
    opts = opts or Options()
    t0 = time.monotonic()
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    out = Path(out_dir) if out_dir else input_path.with_name(input_path.stem + "_stems")
    out.mkdir(parents=True, exist_ok=True)
    res = Result(out_dir=str(out), model=None if opts.skip_separation else opts.model)

    # progress budget: separation dominates when it runs
    sep_w = 0.0 if opts.skip_separation else 0.75
    midi_w = 1.0 - sep_w - 0.05

    def report(frac, stage, msg):
        if progress:
            progress(max(0.0, min(1.0, frac)), stage, msg)

    if opts.fetch_meta:
        report(0.005, "fetch", "Looking up album art and lyrics…")
        _fetch_meta(out, input_path, opts, res, log=lambda m: report(0.005, "fetch", m))

    stems_audio: dict[str, np.ndarray] = {}
    sr = 44100
    weak_hint = None
    if opts.skip_separation:
        report(0.01, "decode", "Decoding audio…")
        name = (opts.midi_stems or ("bass",))[0]
        stems_audio[name] = load_audio(input_path, sr, channels=2)
        res.device = "n/a"
    else:
        stems_audio, sr, dev = separate(
            input_path, opts.model, opts.device, opts.shifts, opts.overlap,
            progress=lambda f, s, m: report(f * sep_w, s, m), repo=opts.repo, _model=_model,
        )
        res.device = dev

        report(sep_w, "write", "Writing stems…")
        wanted = list(opts.stems)
        if opts.lyrics and "vocals" not in wanted and "all" not in wanted:
            wanted.append("vocals")             # the lyrics are lined up with it (and you'll want it)
        for name in _expand_stems(wanted, stems_audio):
            res.stems[name] = write_audio(out / name, stems_audio[name], sr, opts.audio_format).name
        if opts.backing:
            removed = [s for s in opts.backing_remove if s in stems_audio]
            if removed:
                keep = [s for s in stems_audio if s not in removed]
                backing = np.sum([stems_audio[s] for s in keep], axis=0)
                fname = "backing_no_" + "_".join(removed)
                res.stems[fname] = write_audio(out / fname, backing, sr, opts.audio_format).name
        missing = [s for s in opts.midi_stems if s not in stems_audio]
        if missing:
            res.warnings.append(f"Model '{opts.model}' has no stem(s) {missing}; skipped for MIDI.")
        if opts.suggest and "bass" in opts.midi_stems:
            try:
                weak_hint = _sibling("suggest").weak_stem(stems_audio, sr, opts.model)
            except Exception as e:  # noqa: BLE001 — hints are extras
                res.warnings.append(f"Bass stem check not done: {e}")

    # ── tempo from the full mix (sum of stems == mix) ────────────────────
    midi_targets = [s for s in opts.midi_stems if s in stems_audio]
    base = sep_w + 0.05
    if midi_targets:
        import librosa
        mix_mono = np.sum([a for a in stems_audio.values()], axis=0).mean(axis=0)
        if opts.bpm:
            # fixed tempo: constant grid, phase is fitted to the notes later
            bpm, beats = float(opts.bpm), np.array([])
        elif opts.beats:
            # the beats an earlier run found in the full mix (a result folder re-run)
            beats = np.asarray(opts.beats, float)
            bpm = float(60.0 / np.median(np.diff(beats))) if len(beats) > 1 else 120.0
        else:
            report(base, "tempo", "Estimating tempo…")
            bpm, beats = estimate_tempo(librosa.resample(mix_mono, orig_sr=sr, target_sr=22050), 22050)
            if opts.skip_separation:
                res.warnings.append("Tempo was detected from the stem alone and may be off; "
                                    "set the BPM for reliable quantization.")
        res.bpm = bpm

        has_bp = check_environment()["basic_pitch"] is not None if opts.engine != "pyin" else False
        share = midi_w / len(midi_targets)
        for i, stem in enumerate(midi_targets):
            engine = pick_engine(stem, opts.engine, has_bp)
            lo, hi = PITCH_RANGES.get(stem, ("C1", "C7"))
            tab_tuning = None
            gp_strings = None          # tuning of a Guitar Pro file whose strings are kept
            sections_info = None       # --sections-from: where each example was placed
            auto_tuning = stem == "bass" and str(opts.tuning or "").strip().lower() == "auto"
            if stem == "bass" and opts.tuning:
                tab_tuning = _fretting().parse_tuning("4" if auto_tuning else opts.tuning)
                # listen down to B0 (or the lowest open string, if lower) so notes
                # below a 4-string's E can be noticed and a 5-string suggested
                lo = _fretting().midi_to_note(min(tab_tuning.strings[0], _fretting().LOWEST_TRACKED) - 1)
            lo, hi = opts.fmin or lo, opts.fmax or hi
            start = base + i * share

            def sub(f, s, m, _st=start):
                report(_st + f * share, s, f"[{stem}] {m}")

            sub(0.0, "transcribe", f"Transcribing with {engine}…")
            if stem == "bass" and opts.sections_from:
                engine = "pyin+sections"
                y_m = stems_audio[stem].mean(axis=0)
                sub(0.05, "transcribe", f"Reading the examples in {Path(opts.sections_from).name}…")
                raw, gp = _sibling("notesimport").read_gp(opts.sections_from, opts.notes_track)
                tab_tuning, gp_strings, auto_tuning = _gp_tuning(gp, tab_tuning, auto_tuning, res)
                auto = transcribe_pyin(y_m, sr, lo, hi, opts.min_note_ms,
                                       progress=lambda f, st, m: sub(0.1 + 0.6 * f, st, m),
                                       sustained=opts.sustained)
                sub(0.75, "transcribe", "Finding where each example is played…")
                gs_mod = _sibling("gpsections")
                placed = gs_mod.place(raw, gp, y_m, sr, beats, bpm, allowed=_sibling("chart_tools").RS_SECTIONS,
                                      onsets=stem_onsets(y_m, sr))
                ex_notes = [Note(start=n["t"], end=n["t"] + n["d"], pitch=n["m"], tech=n.get("tech"),
                                 legato=bool(n.get("lg")), string=n.get("s")) for n in placed["notes"]]
                notes = merge_notes(auto, ex_notes, placed["ranges"])
                sections_info = {"file": Path(opts.sections_from).name, "occurrences": [
                    {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}
                    for o in placed["occurrences"]], "ratio": placed["ratio"]}
                if placed["parts"] and not opts.parts:
                    opts.parts = placed["parts"]        # the markers become the Rocksmith sections
                res.warnings.append(f"{Path(opts.sections_from).name}: " + placed["report"].replace("\n", "; ")
                                    + " — the rest is the built-in transcription.")
                if not placed["occurrences"]:
                    res.warnings.append("None of the examples matched the bass stem: same song / version? Same "
                                        "tuning? Each example must start on its first beat (marker on that bar).")
            elif stem == "bass" and opts.notes_from:
                use = (opts.notes_use or "all").strip()
                engine = "file" if use.lower() == "all" else "pyin+file"
                sub(0.05, "transcribe", f"Lining up the notes of {Path(opts.notes_from).name}…")
                file_notes, imp = notes_from_file(opts.notes_from, stems_audio[stem].mean(axis=0), sr,
                                                  opts.notes_track)
                gp = imp.get("gp")
                if gp:
                    tab_tuning, gp_strings, auto_tuning = _gp_tuning(gp, tab_tuning, auto_tuning, res)
                if imp.get("unmatched"):
                    res.warnings.append(
                        "Where the file's notes mostly miss the bass attacks (a part missing, repeated or "
                        "played differently in the file? check in EOF): "
                        + ", ".join(f"{_mmss(a)}-{_mmss(b)}" for a, b in imp["unmatched"]) + ".")
                notes = file_notes
                if engine == "pyin+file":
                    auto = transcribe_pyin(stems_audio[stem].mean(axis=0), sr, lo, hi, opts.min_note_ms,
                                           progress=lambda f, st, m: sub(0.1 + 0.7 * f, st, m),
                                           sustained=opts.sustained)
                    dur = mix_mono.shape[-1] / sr
                    bars = song_bars(auto, beats, bpm, dur, opts.rs_shift)
                    if use.lower() == "doubtful":
                        sub(0.85, "review", "Finding the doubtful bars…")
                        rv = _sibling("review")
                        checked = rv.review(stems_audio[stem], sr,
                                            [{"t": n.start, "d": n.duration, "m": n.pitch} for n in auto],
                                            [(a, b) for _k, a, b in bars])
                        chosen = [(k, a, b) for (k, a, b), c in zip(bars, checked) if c["level"] != "ok"]
                    else:
                        ranges = parse_bar_spec(use, bars, opts.rs_shift)
                        chosen = [(k, a, b) for k, a, b in bars
                                  if any(x - 0.05 <= a and b <= y + 0.05 for x, y in ranges)] or \
                                 [(0, x, y) for x, y in ranges]
                    notes = merge_notes(auto, file_notes, [(a, b) for _k, a, b in chosen])
                    nums = [k for k, _a, _b in chosen if k]
                    merge_msg = (
                        f"Bars from {Path(opts.notes_from).name}: {len(chosen)} of {len(bars)}"
                        + (f" ({_compact_ranges(nums)})" if nums else "")
                        + "; the rest is the built-in transcription.")
                else:
                    merge_msg = ""
                res.warnings.append(
                    f"Notes from {Path(opts.notes_from).name}: {imp['notes']} lined up with the bass stem "
                    f"({imp['share_on_attack']:.0%} on an attack; offset {imp['offset']:+.2f} s; "
                    f"{imp.get('shortened', 0)} shortened to where the stem goes quiet"
                    + (f"; {imp['outside']} fall outside the audio, left out" if imp.get("outside") else "")
                    + ("; no single tempo ratio fitted — lined up note by note with a quick transcription "
                       "of the stem" if imp.get("method") == "notes" else "")
                    + (f"; {imp['tempo_note']}" if imp["tempo_note"] else "") + ").")
                if merge_msg:
                    res.warnings.append(merge_msg)
                if imp["share_on_attack"] < 0.4:
                    res.warnings.append("Few of those notes fall on an attack of the stem — is it the same "
                                        "song / version? Check the result in EOF.")
            elif engine == "pyin":
                notes = transcribe_pyin(stems_audio[stem].mean(axis=0), sr, lo, hi,
                                        opts.min_note_ms, progress=sub, sustained=opts.sustained)
            else:
                src = out / res.stems[stem] if stem in res.stems else None
                tmp = None
                if src is None:  # stem wasn't saved: write a temp file for basic-pitch
                    tmp = write_audio(out / f".tmp_{stem}", stems_audio[stem], sr, "wav")
                    src = tmp
                try:
                    notes = transcribe_basic_pitch(src, lo, hi, opts.min_note_ms,
                                                   mono=stem in ("bass", "vocals"), progress=sub)
                finally:
                    if tmp is not None:
                        tmp.unlink(missing_ok=True)
            pre_only = [{"t": n.start, "d": n.duration, "m": n.pitch} for n in notes]
            if stem == "bass" and opts.only:
                try:
                    keep = parse_bar_spec(opts.only, song_bars(notes, beats, bpm, mix_mono.shape[-1] / sr,
                                                               opts.rs_shift))
                    before = len(notes)
                    notes = [n for n in notes if any(a - 0.05 <= n.start < b for a, b in keep)]
                    res.warnings.append(f"Bass only in {opts.only}: {before - len(notes)} note(s) elsewhere "
                                        f"dropped (other low instruments in the bass stem).")
                except ValueError as e:
                    res.warnings.append(f"--only ignored: {e}")
            if opts.quantize:
                notes = quantize_notes(notes, beats, bpm, opts.quantize)

            midi_path = write_midi(notes, out / f"{stem}.mid", bpm=bpm,
                                   program=GM_PROGRAMS.get(stem, 0), track_name=stem.capitalize())
            res.midi[stem] = midi_path.name
            payload = {
                "version": 1, "stem": stem, "engine": engine, "bpm": bpm,
                "quantize": opts.quantize,
                "source": input_path.name,
                "phrase_bars": opts.phrase_bars,
                **({"keep_accents": True} if opts.keep_accents else {}),
                **({"only": opts.only} if opts.only else {}),
                **({"sustained": True} if opts.sustained else {}),
                **({"bpm_set": True} if opts.bpm else {}),
                **({"notes_from": Path(opts.notes_from).name} if opts.notes_from else {}),
                **({"sections_from": sections_info} if sections_info else {}),
                **({"notes_track": opts.notes_track} if opts.notes_track else {}),
                **({"five_string": opts.five_string} if opts.five_string != "auto" else {}),
                **({"parts": opts.parts} if opts.parts and opts.parts.strip().lower() != "auto" else {}),
                **{k: str(v) for k, v in (opts.meta or {}).items() if v and k in ("artist", "title", "album", "year")},
                "duration": round(float(mix_mono.shape[-1]) / sr, 3),
                "beats": [round(float(b), 4) for b in beats],
                "notes": [{"t": n.start, "d": round(n.duration, 4), "m": n.pitch, "v": n.velocity}
                          for n in notes],
            }
            if tab_tuning is not None and notes:
                better, why = _fretting().suggest_tuning([n.pitch for n in notes], tab_tuning)
                if auto_tuning:
                    if better:
                        tab_tuning = _fretting().parse_tuning(better)
                    res.warnings.append(f"Tuning (auto): {tab_tuning.label()}"
                                        + (f" — {why}" if better else "."))
                elif better:
                    res.warnings.append(f"{why[0].upper()}{why[1:]} Re-run with that tuning "
                                        f"(Bass tuning → 5-string · B E A D G, or --tuning 5); "
                                        f"for now they were moved up an octave.")
                sub(0.97, "tab", "Choosing strings and frets…")
                keep_strings = None
                if gp_strings and tuple(tab_tuning.strings) == gp_strings:
                    def keep_strings(src):          # the tab's own string for each of its notes
                        s_ = getattr(src, "string", None)
                        return {s_} if s_ is not None else None
                fret = _fretting().assign_frets(notes, tab_tuning, max_fret=opts.max_fret,
                                                allowed=keep_strings)
                res.warnings.extend(fret.warnings)
                res.tuning = list(tab_tuning.strings)
                slap_counts = None
                slap_spec = opts.slap
                if slap_spec and slap_spec.strip().lower() == "auto":
                    sub(0.98, "tab", "Looking for slapped parts…")
                    sd = _sibling("slapdetect")
                    secs = sd.detect_sections(stems_audio[stem].mean(axis=0), sr, notes)
                    slap_spec = sd.format_sections(secs)
                    res.warnings.append(
                        "Slap sections guessed automatically (experimental): "
                        + (slap_spec or "none found") + ". Check them and re-run with the corrected list.")
                part_ranges = []
                if str(opts.parts or "").strip().lower() == "auto":
                    sub(0.98, "tab", "Finding the song parts in the audio…")
                    try:
                        segs = auto_parts(mix_mono, sr, _tab_payload(fret, tab_tuning, bpm, beats,
                                                                     mix_mono.shape[-1] / sr))
                        opts = replace(opts, parts=_sibling("structure").parts_spec(segs) or None)
                        payload["parts_auto"] = segs
                        if opts.parts:
                            payload["parts"] = opts.parts
                        res.warnings.append("Song parts guessed from the audio (same letter = same music; "
                                            "check the names): " + _sibling("structure").report(segs))
                    except Exception as e:  # noqa: BLE001
                        opts = replace(opts, parts=None)
                        res.warnings.append(f"Song parts not found: {e}")
                if opts.parts:
                    sub(0.98, "tab", "Finding the song parts…")
                    try:
                        part_ranges = _sibling("rsexport").find_parts(
                            _tab_payload(fret, tab_tuning, bpm, beats, mix_mono.shape[-1] / sr), opts.parts)
                    except Exception as e:  # noqa: BLE001
                        res.warnings.append(f"Song parts not found: {e}")
                slap_names = slap_part_names(slap_spec)
                if slap_names:
                    slap_spec = ", ".join(f"{a:.2f}-{b:.2f}" for n, a, b in part_ranges if n in slap_names)
                    if not slap_spec:
                        res.warnings.append(f"Slap: no occurrence of {', '.join(slap_names)} "
                                            "(give them in Song parts).")
                if slap_spec:
                    slap_counts = _fretting().apply_slap(fret, _fretting().parse_sections(slap_spec))
                payload["version"] = 2
                payload["tuning"] = {"strings": list(tab_tuning.strings), "label": tab_tuning.label(),
                                     "offsets": tab_tuning.offsets()}
                payload["anchors"] = [{"t": round(a.time, 4), "f": a.fret, "w": a.width}
                                      for a in fret.anchors]
                payload["tab_stats"] = fret.stats()
                if slap_counts is not None:
                    payload["tab_stats"].update(slap_counts)
                    payload["slap_sections"] = slap_spec
                    if opts.slap.strip().lower() == "auto":
                        payload["slap_auto"] = True
                payload["notes"] = [
                    {"t": n.start, "d": round(n.end - n.start, 4), "m": n.pitch, "v": n.velocity,
                     "s": n.string, "f": n.fret, **({"m0": n.source_pitch} if n.source_pitch else {}),
                     **({"tech": n.tech} if n.tech else {}), **({"lg": 1} if n.legato else {})}
                    for n in fret.notes
                ]
                tab_path = out / f"{stem}.tab.txt"
                tab_path.write_text(_fretting().to_ascii_tab(fret, bpm=bpm), encoding="utf-8")
                res.tabs[stem] = tab_path.name
            if opts.review and stem == "bass" and tab_tuning is not None and payload["notes"]:
                sub(0.99, "review", "Checking the transcription against the stem…")
                try:
                    rv = _sibling("review")
                    bars = rv.bars_from_beat_map(_sibling("rsexport").beat_map_for(payload))
                    checked = rv.review(stems_audio[stem], sr, payload["notes"], bars)
                    if opts.rs_shift:   # number the bars as the XML / EOF does (lead-in adds bars)
                        xd = [t for t, m in _sibling("rsexport").beat_map_for(payload, shift=opts.rs_shift)
                              if m > 0]
                        for b in checked:
                            b["bar"] = 1 + int(np.argmin(np.abs(np.array(xd) - (b["t"] + opts.rs_shift))))
                    payload["review"] = checked
                    rpath = out / f"{stem}.review.txt"
                    rpath.write_text(rv.report(checked, opts.rs_shift), encoding="utf-8")
                    res.review[stem] = rpath.name
                    nb = sum(b["level"] == "bad" for b in checked)
                    nc = sum(b["level"] == "check" for b in checked)
                    rel = any(b.get("relative") for b in checked)
                    res.warnings.append(f"Review: {nb} bar(s) likely wrong, {nc} worth a check "
                                        f"(of {len(checked)}) — see {rpath.name}."
                                        + (" The stem is hard to read (most bars look doubtful): these are "
                                           "the worst bars of this song." if rel else ""))
                except Exception as e:  # noqa: BLE001
                    res.warnings.append(f"Review not done: {e}")
            if opts.suggest and stem == "bass" and payload["notes"]:
                try:
                    sg = _sibling("suggest")
                    hints = ([weak_hint] if weak_hint else []) + sg.suggest(
                        stems_audio[stem], sr, pre_only or payload["notes"], bpm, payload.get("review"),
                        used={"sustained": opts.sustained, "only": opts.only,
                              "bpm": opts.bpm, "notes_from": opts.notes_from or opts.sections_from})
                    weak_hint = None
                    # a 5-string bass is already suggested by the tuning check
                    if hints:
                        _add_hints(res, hints, out, opts, input_path, payload)
                        if stem in res.review:
                            rp = out / res.review[stem]
                            rp.write_text(rp.read_text(encoding="utf-8") + sg.report(hints, res.rerun),
                                          encoding="utf-8")
                except Exception as e:  # noqa: BLE001 — hints are extras
                    res.warnings.append(f"Option hints not computed: {e}")
            notes_path = out / f"{stem}.notes.json"
            notes_path.write_text(json.dumps(payload), encoding="utf-8")
            res.notes[stem] = notes_path.name
            if opts.rs_xml and tab_tuning is not None and notes:
                try:
                    xml_path, xml_warns = _sibling("rsexport").write_xml(notes_path, shift=opts.rs_shift)
                    if (out / _sibling("rsexport").RHYTHM_XML).is_file():
                        res.rsxml["bass5_rhythm"] = _sibling("rsexport").RHYTHM_XML
                    if opts.rs_shift:
                        xml_warns.append(f"Rocksmith XML moved {opts.rs_shift:+.3f} s (EOF lead-in; "
                                         "lead_silence / --shift).")
                    res.rsxml[stem] = xml_path.name
                    res.warnings.extend(xml_warns)
                except Exception as e:  # noqa: BLE001 — never fail the job over the extra file
                    res.warnings.append(f"Rocksmith XML not written: {e}")
            if not notes:
                res.warnings.append(f"No notes detected in the {stem} stem.")

    if opts.lyrics:
        voc = stems_audio.get("vocals")
        if voc is None and opts.skip_separation:
            voc = next(iter(stems_audio.values()), None)
        if voc is None:
            res.warnings.append(f"Lyrics not lined up: model '{opts.model}' has no vocals stem.")
        else:
            report(0.97, "lyrics", "Lining up the lyrics with the vocals…")
            try:
                info = _sibling("lyrics").process(
                    voc, sr, opts.lyrics, out, meta={**opts.meta, "audio": input_path.name},
                    shift=opts.rs_shift, lang=opts.language, aligner=opts.lyrics_aligner,
                    device=res.device if res.device in ("cuda", "mps") else "cpu",
                    log=lambda m: report(0.97, "lyrics", m), keep_accents=opts.keep_accents)
                res.lyrics = info
                if info.get("aligner") == "simple":
                    res.warnings.append("Lyrics lined up without the MMS model (simple aligner): line "
                                        "starts are usually right, words inside lines rough — check "
                                        "them in UltraStar Creator.")
            except Exception as e:  # noqa: BLE001 — never fail the job over the lyrics
                res.warnings.append(f"Lyrics not lined up: {e}")

    if weak_hint:                      # no notes to look at: the stem check alone
        _add_hints(res, [weak_hint], out, opts, input_path)

    if opts.dlc_project and (res.rsxml or res.lyrics):
        report(0.998, "project", "Writing the DLC Builder project…")
        mix = np.sum(list(stems_audio.values()), axis=0) if not opts.skip_separation and stems_audio else None
        try:
            # with --skip-separation the input is a stem, not the song: use an earlier
            # song*.ogg or the folder's stems (bass + backing) instead
            res.project, warns = write_dlc_project(out, opts.charter, opts.meta, opts.rs_shift, mix=mix, sr=sr,
                                                   source=None if opts.skip_separation else input_path)
            res.warnings.extend(warns)
        except Exception as e:  # noqa: BLE001 — never fail the job over the extra file
            res.warnings.append(f"DLC Builder project not written: {e}")

    res.elapsed = round(time.monotonic() - t0, 2)
    (out / "stems2midi.json").write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
    report(1.0, "done", "Done")
    return res


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _csv(v: str) -> tuple:
    return tuple(s.strip().lower() for s in v.split(",") if s.strip())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stems2midi",
        description="Split a song into stems (Demucs) and transcribe the bass stem to MIDI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  stems2midi song.mp3\n"
            "  stems2midi song.mp3 --stems all --quantize 16\n"
            "  stems2midi song.mp3 --model htdemucs_ft --stems bass --midi bass\n"
            "  stems2midi bass_only.wav --skip-separation --bpm 98\n"
            "  stems2midi \"Band - Song.mp3\" --slap \"0:21-0:42, 1:26-1:38\"\n"
            "  stems2midi \"Band - Song.mp3\" --parts \"verse=0:21-0:42, chorus=0:42-1:05\" --slap verse,chorus\n"
            "  stems2midi \"Band - Song.mp3\" --parts auto      (find the parts; check the guessed names)\n"
            "  stems2midi \"Band - Song_stems\" --align-to \"EOF project\\guitar.ogg\"   (redo only the XML)\n"
            "  stems2midi \"Band - Song.mp3\" --lyrics letra.txt   (+ UltraStar .txt and vocals XML)\n"
            "  stems2midi \"Band - Song_stems\" --notes-from anthemscore.musicxml   (notes from AnthemScore)\n"
            "  stems2midi \"Band - Song_stems\" --lyrics letra.txt (lyrics only, on an earlier result)\n"
            "  stems2midi --check\n"
            "\n"
            "Every run with a tab also writes bass_RS2.xml (Rocksmith 2014). In EOF: select the\n"
            "PART REAL_BASS track first, then File > Import > Rocksmith; the notes land in\n"
            "difficulty 0 (first tab), not Expert. Sections / slap-pop in bulk: chart_tools.py --help.\n"
            "Defaults (e.g. lead_silence = 5): stems2midi.ini — see README.md (\"Defaults\").\n"
            "More: README.md (\"Workflow with EOF\")."
        ),
    )
    p.add_argument("input", nargs="?", help="audio file (mp3, wav, flac, ogg, m4a…), or a result folder "
                                            "to (re)write only its Rocksmith XML")
    p.add_argument("-o", "--out", help="output folder (default: <input>_stems next to the file)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Demucs model: " + ", ".join(sorted(MODELS)) + " (default: %(default)s)")
    p.add_argument("--repo", help="local folder with Demucs model files (offline use)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--stems", default="bass,guitar", type=_csv,
                   help="stems to save, comma list or 'all' (default: bass,guitar)")
    p.add_argument("--no-backing", dest="backing", action="store_false",
                   help="don't write the backing track (mix without bass)")
    p.add_argument("--backing-remove", default="bass", type=_csv,
                   help="stems removed from the backing track (default: bass)")
    p.add_argument("--midi", default="bass", type=_csv,
                   help="stems to transcribe to MIDI (default: bass; 'none' to skip)")
    p.add_argument("--engine", default="auto", choices=["auto", "pyin", "basic-pitch"],
                   help="transcription engine (auto: pyin for bass/vocals, basic-pitch otherwise)")
    p.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8, 16, 32],
                   help="snap notes to a beat grid (16 = sixteenths; default off)")
    p.add_argument("--bpm", type=float, help="force the MIDI tempo instead of detecting it")
    p.add_argument("--tuning", default="auto",
                   help="bass tuning for the string/fret tab: auto, 4, 5, 5c, drop-d, eb, d-standard, "
                        "drop-c, note names 'D1 A1 D2 G2', MIDI '26,33,38,43' or offsets "
                        "'-2,0,0,0'; 'none' = no tab. auto (default) = 4-string E A D G unless "
                        "many notes go below E1, then 5-string B E A D G")
    p.add_argument("--max-fret", type=int, default=24, help="highest fret on the neck (default 24)")
    p.add_argument("--slap", help="slapped parts of the song: 'all', ranges like '0:32-1:05, 2:10-2:40', part "
                                  "names from --parts ('verse,chorus' = every verse and chorus), "
                                  "or 'auto' (experimental guess — check it); attacked notes there become "
                                  "slap (low strings) / pop (two highest strings)")
    p.add_argument("--artist", default="", help="artist for the Rocksmith XML (default: from 'Artist - Title.mp3')")
    p.add_argument("--title", default="", help="song title for the Rocksmith XML (default: from the file name)")
    p.add_argument("--album", default="", help="album for the Rocksmith XML")
    p.add_argument("--year", default="", help="year for the Rocksmith XML")
    p.add_argument("--shift", type=float, default=0.0,
                   help="Rocksmith XML: move everything this many seconds later (EOF lead-in silence; "
                        "default: lead_silence in stems2midi.ini, else 0)")
    p.add_argument("--parts", help="song parts, first occurrence of each: 'intro=0:00-0:21, verse=0:21-0:42, "
                                   "chorus=0:42-1:05' (Rocksmith section names; times in this audio file), "
                                   "or 'auto' (found in the audio, names guessed — check them). "
                                   "Repeats are found by their notes; the XML gets the sections. "
                                   "--slap then also takes part names: --slap verse,chorus")
    p.add_argument("--notes-from", metavar="FILE",
                   help="take the bass notes from another transcriber (AnthemScore, Guitar Pro, MuseScore…): "
                        "MusicXML (.musicxml/.xml/.mxl), MIDI or Guitar Pro (.gp3/.gp4/.gp5/.gpx/.gp — the "
                        "bass track as performed, with its strings/frets, tuning and slap/pop/legato). They are "
                        "lined up with the bass stem (tempo, half/double time and drift included) and go through "
                        "the rest of the tool (tab, slap, parts, XML)")
    p.add_argument("--use", dest="notes_use", default="all", metavar="WHERE",
                   help="with --notes-from: 'all' (default) — every note from the file; 'doubtful' — "
                        "the built-in transcription, with the bars the review flags taken from the file; "
                        "or bars / times: '12-30, 45-60' (bar numbers as in EOF) or '1:02-1:30'")
    p.add_argument("--only", metavar="WHERE",
                   help="the bass guitar plays only in these times / bars ('2:50-5:15', '40-90'); notes "
                        "elsewhere are dropped — for songs where orchestra, piano or synth bass fill the "
                        "bass stem before the bass comes in")
    p.add_argument("--sections-from", metavar="FILE.gp5",
                   help="Guitar Pro tab with ONE example of each part, each starting at a marker named like "
                        "the part (intro, verse, chorus…): every example is found in the song by the bass stem "
                        "and its notes (strings, frets, slap, hammer-ons) are used there; the markers become the "
                        "Rocksmith sections. Elsewhere the built-in transcription stays")
    p.add_argument("--track", dest="notes_track", metavar="N|NAME",
                   help="Guitar Pro file given to --notes-from: the track to use (number from 1, or part of its "
                        "name; default: the bass track)")
    p.add_argument("--no-suggest", action="store_true",
                   help="don't look for options worth a re-run (--sustained, --only, --bpm, --notes-from)")
    p.add_argument("--sustained", action="store_true",
                   help="ballads / long held notes (over strings, pads): a repeated note counts only when "
                        "it is clearly plucked again — no more one long note cut into many")
    p.add_argument("--phrase-bars", type=int, default=None,
                   help="Rocksmith XML: a phrase (what Riff Repeater loops) every N bars inside each "
                        "section/part (default 4; 0 = one phrase per section). Same notes → same phrase name")
    p.add_argument("--align-to", metavar="AUDIO",
                   help="Rocksmith XML: the audio the chart will use (e.g. EOF's guitar.ogg); the shift "
                        "is measured automatically")
    p.add_argument("--lyrics", metavar="TEXT_FILE",
                   help="lyrics (one sung line per line, repeats written out): lined up with the vocals "
                        "stem → vocals.ultrastar.txt (UltraStar Creator / EOF) + vocals_RS2.xml (Rocksmith)")
    p.add_argument("--language", default=None, help="lyrics language: pt, en, es… (default: guessed)")
    p.add_argument("--lyrics-aligner", default="auto", choices=["auto", "mms", "simple"],
                   help="mms = speech model (~1.2 GB download, once), simple = vocal attacks only; "
                        "auto (default) = mms when it can be loaded")
    p.add_argument("--rs-5string", default=None, choices=["auto", "bass", "rhythm"],
                   help="5-string tab in the Rocksmith XML: always a 4-string bass arrangement (B E A D, "
                        "+7 ×4 / −1200 cents); 'rhythm' also writes bass5_rhythm_RS2.xml — a Rhythm "
                        "arrangement an octave down (A220) a 5-string bass plays string for string; "
                        "'auto' (default) does that when the top string has many notes; 'bass' never")
    p.add_argument("--keep-accents", action="store_true",
                   help="keep accented letters in the lyrics files and the XML's title/artist/album "
                        "(default: ã → a, ç → c — what EOF, Rocksmith and DLC Builder take)")
    p.add_argument("--charter", default="",
                   help="your charter name for the DLC Builder project (author, DLC key), e.g. ALSRC "
                        "(stems2midi.ini: charter = ALSRC)")
    p.add_argument("--no-dlc-project", dest="dlc_project", action="store_false",
                   help="don't write the DLC Builder project (Artist_Title.rs2dlc) and song.ogg")
    p.add_argument("--fetch-meta", action="store_true",
                   help="look up the album art (iTunes / Cover Art Archive → cover.jpg) and the lyrics "
                        "(LRCLIB → lyrics.txt, lined up when --lyrics isn't given); also fills album/year")
    p.add_argument("--no-review", dest="review", action="store_false",
                   help="don't check the transcription against the stem (bass.review.txt)")
    p.add_argument("--no-rs-xml", dest="rs_xml", action="store_false",
                   help="don't write bass_RS2.xml (Rocksmith 2014 bass arrangement)")
    p.add_argument("--fmin", help="lowest note to detect, e.g. B0, E1 (default per stem)")
    p.add_argument("--fmax", help="highest note to detect, e.g. G4")
    p.add_argument("--min-note-ms", type=float, default=60.0)
    p.add_argument("--shifts", type=int, default=1,
                   help="Demucs random shifts (better quality, N× slower; default 1)")
    p.add_argument("--format", dest="audio_format", default="wav", choices=["wav", "flac"])
    p.add_argument("--skip-separation", action="store_true",
                   help="input is already an isolated stem: only transcribe it")
    p.add_argument("--json-progress", action="store_true",
                   help="emit one JSON object per line on stdout (used by the FeedBack plugin)")
    p.add_argument("--check", action="store_true", help="print the dependency/GPU report and exit")
    p.add_argument("--version", action="version", version=f"stems2midi {__version__}")
    return p



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

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        parser.set_defaults(**_sibling("s2m_config").cli_defaults())   # stems2midi.ini
    except Exception as e:  # noqa: BLE001 — a broken ini must not stop the tool
        log.warning("stems2midi.ini ignored: %s", e)
    args = parser.parse_args(_cli_argv(argv))
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)

    def emit(obj: dict):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    if args.check:
        info = check_environment()
        if args.json_progress:
            emit({"event": "check", **info})
        else:
            for k, v in info.items():
                print(f"{k:>15}: {v}")
        return 0

    if not args.input:
        build_parser().error("input file is required")

    meta = {"artist": args.artist, "title": args.title, "album": args.album, "year": args.year}
    lyrics_text = None
    if args.lyrics:
        try:
            lyrics_text = Path(args.lyrics).read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            lyrics_text = Path(args.lyrics).read_text(encoding="cp1252")
        except OSError as e:
            build_parser().error(f"--lyrics: {e}")
        if not lyrics_text.strip():
            build_parser().error("--lyrics: the file is empty")
    src = Path(args.input)
    use = str(args.notes_use or "all").strip()
    if use.lower() not in ("all", "doubtful") and not re.fullmatch(r"[\d:.,\s-]+", use):
        build_parser().error("--use: all, doubtful, bar numbers '12-30, 45' or times '1:02-1:30'")
    if args.only and not re.fullmatch(r"[\d:.,\s-]+", args.only):
        build_parser().error("--only: times '2:50-5:15' or bar numbers '40-90'")
    reuse_beats = None
    if args.sections_from:
        if args.notes_from:
            build_parser().error("--sections-from and --notes-from: use one of them")
        if not Path(args.sections_from).is_file():
            build_parser().error(f"--sections-from: no file {args.sections_from}")
        if Path(args.sections_from).suffix.lower() not in _sibling("notesimport").GP_EXTS:
            build_parser().error("--sections-from: a Guitar Pro file (.gp3/.gp4/.gp5/.gpx/.gp)")
    if args.notes_from:
        if not Path(args.notes_from).is_file():
            build_parser().error(f"--notes-from: no file {args.notes_from}")
        if Path(args.notes_from).suffix.lower() not in _sibling("notesimport").EXTS:
            build_parser().error("--notes-from: use MusicXML (.musicxml/.xml/.mxl), MIDI (.mid) "
                                 "or Guitar Pro (.gp3/.gp4/.gp5/.gpx/.gp)")
    # a result folder + an option that changes the notes: transcribe its bass stem again
    # (no new separation), keeping what the earlier run found (beats, slap, parts…)
    retranscribe = [o for o, v in (("--notes-from", args.notes_from), ("--sections-from", args.sections_from),
                                   ("--sustained", _truthy(args.sustained)),
                                   ("--bpm", args.bpm)) if v]
    if retranscribe and src.is_dir():
        stem = next((src / f"bass.{e}" for e in ("wav", "flac") if (src / f"bass.{e}").is_file()), None)
        if stem is None:
            build_parser().error(f"{retranscribe[0]}: no bass.wav / bass.flac in {src}")
        try:
            old = json.loads((src / "bass.notes.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        guess = _sibling("rsexport").guess_meta(old.get("source") or src.name.removesuffix("_stems"))
        for k in ("artist", "title", "album", "year"):
            if not meta.get(k):
                meta[k] = str(old.get(k) or guess.get(k) or "")
        # keep what the earlier run was given (slap sections, song parts)
        if not args.slap and old.get("slap_sections"):
            args.slap = str(old["slap_sections"])
            print(f"Slap sections from the earlier result: {args.slap}")
        if not args.parts and old.get("parts") and not args.sections_from:
            args.parts = str(old["parts"])
            print(f"Song parts from the earlier result: {args.parts}")
        if not args.only and old.get("only"):
            args.only = str(old["only"])
            print(f"Bass only in (from the earlier result): {args.only}")
        if not _truthy(args.sustained) and old.get("sustained"):
            args.sustained = True
            print("Long held notes (--sustained) as in the earlier result")
        if not args.bpm and old.get("bpm_set") and old.get("bpm"):
            args.bpm = float(old["bpm"])
            print(f"Tempo as in the earlier result: {args.bpm:g} BPM")
        if not args.bpm and len(old.get("beats") or []) > 1:
            reuse_beats = list(old["beats"])
            print(f"Beats from the earlier result ({len(reuse_beats)}; --bpm to set the tempo instead)")
        args.out = args.out or str(src)
        args.skip_separation = True
        args.input = str(stem)
        src = stem
    if src.is_dir() or src.name.endswith(".notes.json"):
        # an earlier result: only (re)write the Rocksmith XML
        folder = src if src.is_dir() else src.parent
        shift = args.shift
        if args.align_to:
            try:
                shift = align_shift(folder, args.align_to)
            except (ValueError, OSError) as e:
                build_parser().error(f"--align-to: {e}")
        lyr_mod = _sibling("lyrics")
        voc_path = lyr_mod.find_vocals(folder)
        audio_name = ""
        try:
            audio_name = Path(json.loads((folder / "bass.notes.json").read_text(encoding="utf-8"))
                              .get("source") or "").name
        except (OSError, ValueError):
            pass
        if _truthy(args.fetch_meta):
            mf = _sibling("metafetch")
            mf.set_version(__version__)
            guess = _sibling("rsexport").guess_meta(audio_name or folder.name.removesuffix("_stems"))
            got = mf.fetch(folder, meta.get("artist") or guess.get("artist") or "",
                           meta.get("title") or guess.get("title") or "", meta.get("album") or "", log=print)
            for n in got["notes"]:
                print(f"Web lookup: {n}")
            for k in ("album", "year"):
                if got[k] and not meta.get(k):
                    meta[k] = got[k]
            if got["lyrics"] and not lyrics_text and voc_path is not None \
                    and not (folder / lyr_mod.JSON_NAME).is_file():
                lyrics_text = (folder / got["lyrics"]).read_text(encoding="utf-8")
                print(f"Lyrics from the web ({got['lyrics']}) will be lined up — check them.")

        def project():
            if not _truthy(args.dlc_project):
                return
            try:
                name, warns = write_dlc_project(folder, args.charter, meta, shift, source=audio_name)
                for w in warns:
                    print(("" if w.startswith("DLC Builder project") else "warning: ") + w)
            except Exception as e:  # noqa: BLE001
                print(f"warning: DLC Builder project not written: {e}")
        if lyrics_text:
            if voc_path is None:
                build_parser().error("--lyrics: no vocals stem in this folder (run the song again with --lyrics)")
            y, vsr = load_audio(voc_path, 44100, channels=1)[0], 44100
            dev = resolve_device(args.device)
            try:
                info = lyr_mod.process(y, vsr, lyrics_text, folder, meta={**meta, "audio": audio_name},
                                       shift=shift, lang=args.language, aligner=args.lyrics_aligner,
                                       device=dev if dev in ("cuda", "mps") else "cpu",
                                       keep_accents=_truthy(args.keep_accents))
            except (ValueError, RuntimeError) as e:
                build_parser().error(f"lyrics: {e}")
            print(f"Lyrics: {info['syllables']} syllables, {info['lines']} lines ({info['aligner']} aligner)"
                  f" → {info['ultrastar']}, {info['xml']}")
        elif (folder / lyr_mod.JSON_NAME).is_file():
            info = lyr_mod.write_outputs(folder, {**meta, "audio": audio_name}, shift,
                                         keep_accents=_truthy(args.keep_accents))
            print(f"Lyrics written again → {info['ultrastar']}, {info['xml']}")
        if not (src.is_file() or (folder / "bass.notes.json").is_file()):
            if lyrics_text or (folder / lyr_mod.JSON_NAME).is_file():
                project()
                return 0
        parts = args.parts
        if str(parts or "").strip().lower() == "auto":
            y = result_audio(folder, 22050)
            if y is None:
                build_parser().error("--parts auto needs the stems (.wav/.flac) in the result folder")
            nj = src if src.is_file() else folder / "bass.notes.json"
            segs = auto_parts(y, 22050, json.loads(nj.read_text(encoding="utf-8")))
            parts = _sibling("structure").parts_spec(segs)
            print("Song parts guessed from the audio (same letter = same music; check the names):")
            print("  " + _sibling("structure").report(segs))
            print("To correct them, edit the names and run again with:")
            print(f'  --parts "{parts}"')
        nj = src if src.is_file() else folder / "bass.notes.json"
        if args.only and nj.is_file():                         # drop notes where the bass doesn't play
            pl = json.loads(nj.read_text(encoding="utf-8"))
            full = nj.with_name("bass.notes.full.json")
            if not full.is_file():
                full.write_text(json.dumps(pl), encoding="utf-8")    # the first transcription, kept
            else:
                pl["notes"] = json.loads(full.read_text(encoding="utf-8"))["notes"]
            ns = [Note(start=n["t"], end=n["t"] + n["d"], pitch=n["m"]) for n in pl["notes"]]
            try:
                keep = parse_bar_spec(args.only, song_bars(ns, pl.get("beats") or [], float(pl.get("bpm") or 120),
                                                           float(pl.get("duration") or 0), args.shift or 0.0))
            except ValueError as e:
                build_parser().error(f"--only: {e}")
            before = len(pl["notes"])
            pl["notes"] = [n for n in pl["notes"] if any(a - 0.05 <= n["t"] < b for a, b in keep)]
            pl["only"] = args.only
            nj.write_text(json.dumps(pl), encoding="utf-8")
            print(f"Bass only in {args.only}: {before - len(pl['notes'])} of {before} notes dropped "
                  f"(all kept in {full.name}).")
        if args.phrase_bars is not None and nj.is_file():     # remember it for later re-exports
            pl = json.loads(nj.read_text(encoding="utf-8"))
            pl["phrase_bars"] = max(0, args.phrase_bars)
            nj.write_text(json.dumps(pl, indent=1), encoding="utf-8")
        try:
            xml_path, warns = _sibling("rsexport").write_xml(
                src, shift=shift, parts=parts, slap_parts=",".join(slap_part_names(args.slap)),
                phrase_bars=args.phrase_bars, keep_accents=True if _truthy(args.keep_accents) else None,
                **({"five_string": args.rs_5string} if args.rs_5string else {}), **meta)
        except (ValueError, RuntimeError, OSError) as e:
            build_parser().error(f"Rocksmith XML: {e}")
        for w in warns:
            print(("" if w.startswith(("Song parts", "Slap/pop in")) else "warning: ") + w)
        print(f"Rocksmith XML: {xml_path}")
        project()
        return 0

    if args.model not in MODELS and not args.repo:
        build_parser().error(f"unknown model {args.model!r} (choose from {', '.join(sorted(MODELS))})")
    if str(args.tuning).lower() not in ("none", "auto"):
        try:
            _fretting().parse_tuning(args.tuning)
        except ValueError as e:
            build_parser().error(f"--tuning: {e}")
    auto = str(args.parts or "").strip().lower() == "auto"
    if args.parts and not auto:
        try:
            _sibling("chart_tools").parse_parts(args.parts)
        except ValueError as e:
            build_parser().error(f"--parts: {e}")
    if args.slap and slap_part_names(args.slap) and auto:
        pass                                   # names are checked once the parts are found
    elif args.slap and slap_part_names(args.slap):
        bad = [n for n in slap_part_names(args.slap)
               if n not in [x[0] for x in _sibling("chart_tools").parse_parts(args.parts or "riff=0-1")]]
        if bad and not args.parts:
            build_parser().error("--slap with part names (e.g. verse,chorus) needs --parts")
        if bad:
            build_parser().error(f"--slap: {', '.join(bad)} not in --parts")
    elif args.slap and args.slap.strip().lower() != "auto":
        try:
            _fretting().parse_sections(args.slap)
        except ValueError as e:
            build_parser().error(f"--slap: {e}")
    midi = tuple(s for s in args.midi if s != "none")
    opts = Options(
        model=args.model, device=args.device, stems=args.stems, backing=args.backing,
        backing_remove=args.backing_remove, midi_stems=midi, engine=args.engine,
        quantize=args.quantize, bpm=args.bpm, fmin=args.fmin, fmax=args.fmax,
        min_note_ms=args.min_note_ms, shifts=max(0, args.shifts),
        audio_format=args.audio_format, skip_separation=args.skip_separation, repo=args.repo,
        tuning=None if str(args.tuning).lower() == "none" else args.tuning, max_fret=args.max_fret,
        slap=args.slap, rs_xml=args.rs_xml, meta=meta, rs_shift=args.shift, parts=args.parts or None,
        beats=reuse_beats, suggest=not args.no_suggest, notes_track=args.notes_track,
        review=args.review, lyrics=lyrics_text, language=args.language,
        phrase_bars=4 if args.phrase_bars is None else max(0, args.phrase_bars),
        notes_from=args.notes_from, notes_use=args.notes_use, sections_from=args.sections_from, five_string=args.rs_5string or "auto",
        only=args.only, sustained=_truthy(args.sustained),
        lyrics_aligner=args.lyrics_aligner, keep_accents=_truthy(args.keep_accents),
        charter=str(args.charter or "").strip(), dlc_project=_truthy(args.dlc_project),
        fetch_meta=_truthy(args.fetch_meta),
    )

    last = {"pct": -1}

    def progress(frac: float, stage: str, msg: str):
        if args.json_progress:
            emit({"event": "progress", "pct": round(frac, 4), "stage": stage, "msg": msg})
            return
        pct = int(frac * 100)
        if pct != last["pct"] or stage in ("model", "write", "done"):
            last["pct"] = pct
            sys.stderr.write(f"\r[{pct:3d}%] {msg[:90]:<90}")
            sys.stderr.flush()
            if stage == "done":
                sys.stderr.write("\n")

    try:
        res = run_pipeline(args.input, args.out, opts, progress)
    except Exception as e:  # report cleanly to the plugin / user
        log.debug("pipeline failed", exc_info=True)
        if args.json_progress:
            emit({"event": "error", "error": f"{type(e).__name__}: {e}"})
        else:
            sys.stderr.write(f"\nerror: {type(e).__name__}: {e}\n")
        return 1

    if args.align_to and res.notes.get("bass") and res.rsxml.get("bass"):
        try:
            ref = load_audio(args.input, 11025, channels=1)[0]
            tgt = load_audio(args.align_to, 11025, channels=1)[0]
            shift, conf = find_offset(ref, tgt, 11025)
            res.warnings = [w for w in res.warnings if not w.startswith("Rocksmith XML moved")]
            res.warnings.append(f"Rocksmith XML aligned to {Path(args.align_to).name}: "
                                f"{shift:+.3f} s (confidence {conf:g})")
            _sibling("rsexport").write_xml(Path(res.out_dir) / res.notes["bass"], shift=shift, **meta)
            if res.lyrics:
                res.lyrics = _sibling("lyrics").write_outputs(
                    Path(res.out_dir), {**meta, "audio": Path(args.input).name}, shift,
                    keep_accents=opts.keep_accents)
            if res.project:     # the song audio moves with the XML
                res.project, _w = write_dlc_project(res.out_dir, opts.charter, opts.meta, shift,
                                                    source=None if opts.skip_separation else args.input)
        except Exception as e:  # noqa: BLE001
            res.warnings.append(f"Rocksmith XML alignment failed: {e}")

    if args.json_progress:
        emit({"event": "done", **asdict(res)})
    else:
        print(f"\nOutput: {res.out_dir}")
        for k, v in res.stems.items():
            print(f"  stem  {k:<18} {v}")
        for k, v in res.midi.items():
            print(f"  midi  {k:<18} {v}")
        for k, v in res.tabs.items():
            print(f"  tab   {k:<18} {v}")
        for k, v in res.rsxml.items():
            print(f"  RS    {k:<18} {v}  (Rocksmith 2014 XML: import in EOF or DLC Builder)")
        for k, v in res.review.items():
            print(f"  check {k:<18} {v}  (bars most likely to need fixing)")
        if res.lyrics:
            print(f"  lyrics {res.lyrics['ultrastar']} (UltraStar Creator / EOF), {res.lyrics['xml']} "
                  f"(Rocksmith vocals) — {res.lyrics['syllables']} syllables, {res.lyrics['aligner']} aligner")
        if res.project:
            print(f"  DLC   {'project':<18} {res.project}  (DLC Builder: open, add tones, build)")
        for k in ("cover", "lyrics", "lrc"):
            if res.fetched.get(k):
                print(f"  web   {k:<18} {res.fetched[k]}")
        if res.bpm:
            print(f"  tempo {res.bpm} BPM")
        for w in res.warnings:
            print(f"  warning: {w}")
        print(f"  ({res.elapsed:.1f}s on {res.device})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
