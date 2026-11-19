"""Lyrics → timed syllables, for UltraStar Creator, EOF and Rocksmith.

Give the lyrics as plain text (one sung line per line, every repeat written
out) and the song — or its vocals stem. The words are split into syllables,
lined up with the vocals and written as:

* ``vocals.ultrastar.txt`` — an UltraStar song file (``#MP3:guitar.ogg``
  when there's an EOF lead-in, the audio of the EOF project): open it in
  UltraStar Creator to fix what's off, or import it into EOF (File › Import ›
  Lyrics);
* ``vocals_RS2.xml`` — the Rocksmith 2014 vocals arrangement (like EOF's
  ``PART VOCALS_RS2.xml``), for DLC Builder;
* ``vocals.lyrics.json`` — the timed syllables (to write the files again).

Two ways of lining up:

* ``mms`` — a speech model that knows letters/sounds in 1100+ languages
  (torchaudio ``MMS_FA``, ~1.2 GB, downloaded once). Places every syllable.
* ``simple`` — no model: syllables are matched to the attacks and pauses of
  the vocals stem (line starts after pauses). Good for line starts, rougher
  inside fast lines.

``auto`` uses ``mms`` when torchaudio can load it, else ``simple``.

Syllables: write them yourself with ``-`` (``un-faith-ful``) or let the
hyphenation dictionaries (``pyphen``, pt / en / es / fr / de / it…) split
them; without ``pyphen`` a simple rule-based split is used.

Times are in the song's audio; ``--shift`` / ``lead_silence`` (stems2midi.ini)
moves them for EOF's guitar.ogg, like the bass XML.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
US_BPM = 400.0            # UltraStar file BPM: 1 beat = 60 / (4 × BPM) s = 37.5 ms
LANG_NAMES = {"pt": "Portuguese", "en": "English", "es": "Spanish", "fr": "French",
              "de": "German", "it": "Italian"}
PYPHEN_LANG = {"pt": "pt_PT", "en": "en_US", "es": "es", "fr": "fr", "de": "de_DE", "it": "it_IT"}


# ── text ─────────────────────────────────────────────────────────────────

@dataclass
class Syl:
    text: str              # as shown (punctuation kept)
    line: int
    word: int              # word index in the song
    first: bool            # first syllable of its word
    last: bool             # last syllable of its word
    t: float = 0.0
    end: float = 0.0
    pitch: int | None = None
    conf: float | None = None   # aligner's confidence (MMS)


@dataclass
class Lyrics:
    lines: list[list[list[str]]] = field(default_factory=list)   # line → word → syllables
    lang: str = "en"

    def syllables(self) -> list[Syl]:
        out, w = [], 0
        for li, line in enumerate(self.lines):
            for word in line:
                for k, s in enumerate(word):
                    out.append(Syl(s, li, w, k == 0, k == len(word) - 1))
                w += 1
        return out

    def words(self) -> list[tuple[int, str]]:
        """[(line, word as shown)]"""
        return [(li, "".join(word)) for li, line in enumerate(self.lines) for word in line]


_PT_HINTS = set("ãõçâêôáéíóúà")
_PT_WORDS = {"que", "não", "nao", "uma", "com", "para", "meu", "minha", "eu", "tu", "você", "és",
             "ela", "ele", "os", "as", "do", "da", "dos", "das", "em", "no", "na", "se", "mais",
             "quando", "sem", "ser", "estou", "é", "já", "também", "foi", "vou", "tudo", "nós"}
_EN_WORDS = {"the", "and", "you", "i", "to", "a", "of", "my", "is", "it", "in", "me", "your",
             "that", "on", "for", "be", "all", "this", "what", "with", "we", "so", "but", "am"}


def guess_language(text: str) -> str:
    words = re.findall(r"[\wÀ-ÿ']+", text.lower())
    pt = sum(w in _PT_WORDS for w in words) + 3 * sum(c in _PT_HINTS for c in text.lower())
    en = sum(w in _EN_WORDS for w in words)
    return "pt" if pt > en else "en"


_VOWELS = set("aeiouyáéíóúâêôãõàüèìòùäëïö")


def _diphthong_pt(a: str, b: str) -> bool:
    """Two Portuguese vowels in one syllable (ai, ei, ou, ui, ão, õe…)."""
    return ((a in "aeoáéóâêô" and b in "iu") or (a == "u" and b == "i") or (a == "i" and b == "u")
            or a + b in ("ão", "ãe", "õe", "ãi"))


def _split_rule(word: str, lang: str) -> list[str]:
    """Rule-based syllables (used when pyphen isn't installed)."""
    m = re.match(r"^([^\wÀ-ÿ]*)(.*?)([^\wÀ-ÿ]*)$", word)
    lead, w, trail = m.group(1), m.group(2), m.group(3)
    if not w:
        return [word]
    if lead or trail:
        parts = _split_rule(w, lang)
        parts[0], parts[-1] = lead + parts[0], parts[-1] + trail
        return parts
    low = w.lower()
    n = len(low)
    if n <= 2:
        return [w]
    iv = [c in _VOWELS for c in low]
    if lang == "pt":
        # u in qu/gu before e/i is not a vowel; i/u after a vowel form a diphthong
        for i in range(1, n - 1):
            if low[i] == "u" and low[i - 1] in "qg" and low[i + 1] in "eiéí":
                iv[i] = False
    else:
        # English & co.: a silent final e (life, time), -es / -ed after most consonants
        if low[-1] == "e" and not iv[-2] and any(iv[:-2]) and not (low[-2:] == "le" and not iv[-3]):
            iv[-1] = False
        elif low[-2:] in ("es", "ed") and n > 3 and not iv[-3] and any(iv[:-3]) and \
                not (low[-2:] == "ed" and low[-3] in "td") and \
                not (low[-2:] == "es" and (low[-3] in "sxz" or low[-4:-2] in ("ch", "sh"))):
            iv[-2] = False
        if low[0] == "y" and n > 1 and iv[1]:
            iv[0] = False       # you, yes: y is a consonant
    cuts = []
    i = 0
    nuclei = []                 # (start, end) of vowel groups
    while i < n:
        if iv[i]:
            j = i
            while j + 1 < n and iv[j + 1] and (lang != "pt" or _diphthong_pt(low[j], low[j + 1])):
                j += 1
            nuclei.append((i, j))
            i = j + 1
        else:
            i += 1
    for (a0, a1), (b0, _) in zip(nuclei, nuclei[1:]):
        cons = low[a1 + 1:b0]
        k = len(cons)
        if k == 0:
            cuts.append(b0)                       # hiatus
        elif k == 1:
            cuts.append(b0 - 1)
        else:
            pair = cons[-2:]
            together = {"ch", "lh", "nh", "qu", "gu", "bl", "br", "cl", "cr", "dr", "fl", "fr",
                        "gl", "gr", "pl", "pr", "tr", "vr", "th", "sh", "ph", "wh"}
            cuts.append(b0 - 2 if pair in together else b0 - 1)
    out, s = [], 0
    for c in cuts:
        if c > s:
            out.append(w[s:c])
            s = c
    out.append(w[s:])
    return [x for x in out if x]


_pyphen_cache: dict = {}


def syllabify(word: str, lang: str) -> list[str]:
    """Syllables of one word as shown (punctuation stays on the syllables)."""
    if "-" in word.strip("-"):
        parts = [p for p in word.split("-") if p]
        return parts or [word]
    core = re.sub(r"[^\wÀ-ÿ']", "", word)
    if len(core) <= 3:
        return [word]
    pieces = None
    try:
        import pyphen
        key = PYPHEN_LANG.get(lang, lang)
        if key not in _pyphen_cache:
            _pyphen_cache[key] = pyphen.Pyphen(lang=key, left=1, right=2)
        pieces = _pyphen_cache[key].inserted(word, hyphen="\x00").split("\x00")
    except Exception:
        pieces = None
    if not pieces or len(pieces) == 1:
        pieces = _split_rule(word, lang) if pieces is None else pieces
    # keep leading/trailing punctuation-only pieces glued to their neighbour
    out = []
    for p in pieces:
        if out and not re.search(r"[\wÀ-ÿ]", p):
            out[-1] += p
        elif out and not re.search(r"[\wÀ-ÿ]", out[-1]):
            out[-1] += p
        else:
            out.append(p)
    return out or [word]


_SKIP_LINE = re.compile(r"^\s*(\[.*\]|\(.*\)|\{.*\}|#.*)\s*$")


_US_NOTE = re.compile(r"^([:*FRG])\s*(-?\d+)\s+(\d+)\s+(-?\d+)\s?(.*)$")


def is_ultrastar(text: str) -> bool:
    rows = [r for r in text.replace("\r", "").split("\n") if r.strip()]
    notes = sum(bool(_US_NOTE.match(r)) for r in rows)
    return notes >= 3 and notes >= 0.5 * len(rows)


def _parse_ultrastar(text: str, lang: str) -> Lyrics:
    """An UltraStar .txt as the lyrics: its syllables, words (a space before
    or after a syllable ends the word) and lines (``-`` breaks) are kept;
    ``~`` (the same syllable held on another pitch) joins the one before."""
    lines: list[list[list[str]]] = [[]]
    word_open = False
    for raw in text.replace("\r", "").split("\n"):
        if raw.startswith("-"):
            if lines[-1]:
                lines.append([])
            word_open = False
            continue
        m = _US_NOTE.match(raw)
        if not m:
            continue
        syl = m.group(5)
        core = syl.strip()
        if not core or core.strip("~") == "":
            continue                                    # held / melisma
        if core.startswith("~"):
            core = core.lstrip("~")
        starts_word = syl.startswith(" ") or not word_open or not lines[-1]
        if starts_word:
            lines[-1].append([core])
        else:
            lines[-1][-1].append(core)
        word_open = not syl.endswith(" ")
    lines = [ln for ln in lines if ln]
    return Lyrics(lines, lang)


def parse_lyrics(text: str, lang: str | None = None) -> Lyrics:
    """Plain text → lines of words of syllables. Lines like ``[Chorus]`` or
    ``(x2)`` are ignored: write every sung line out. An UltraStar .txt is
    read as such (its syllables and lines kept)."""
    if is_ultrastar(text):
        words = " ".join(m.group(5) for m in map(_US_NOTE.match, text.split("\n")) if m)
        lang = (lang or "").lower()[:2] or guess_language(words)
        return _parse_ultrastar(text, lang)
    lang = (lang or "").lower()[:2] or guess_language(text)
    lines = []
    for raw in text.replace("\r", "").split("\n"):
        if not raw.strip() or _SKIP_LINE.match(raw):
            continue
        words = [w for w in raw.split() if re.search(r"[\wÀ-ÿ]", w)]
        if words:
            lines.append([syllabify(w, lang) for w in words])
    return Lyrics(lines, lang)


def normalize(s: str) -> str:
    """Letters a–z and ' only (what the aligner's alphabet knows)."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z']", "", s)


# ── audio ────────────────────────────────────────────────────────────────

ASR = 16000
HOP = 160                  # 10 ms


def _mono16k(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0) if y.shape[0] < y.shape[-1] else y.mean(axis=1)
    if sr != ASR:
        y = librosa.resample(y, orig_sr=sr, target_sr=ASR)
    return y


class Vocals:
    """Activity and attacks of the vocals stem (10 ms frames)."""

    def __init__(self, y: np.ndarray):
        import librosa
        self.y = y
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
        db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
        loud = db[db > -50]
        ref = float(np.percentile(loud, 70)) if loud.size else -20.0
        self.db = db
        self.active = db > ref - 18
        self.env = librosa.onset.onset_strength(y=y, sr=ASR, hop_length=HOP, n_fft=1024, fmax=4000)
        n = min(len(self.env), len(self.active))
        self.env, self.active, self.db = self.env[:n], self.active[:n], self.db[:n]
        self.n = n
        self.cum_sil = np.concatenate([[0], np.cumsum(~self.active)])

    def t(self, f):
        return np.asarray(f) * HOP / ASR

    def f(self, t):
        return np.clip(np.round(np.asarray(t) * ASR / HOP).astype(int), 0, self.n - 1)

    def events(self) -> tuple[np.ndarray, np.ndarray]:
        """Candidate syllable starts (frames) and their strength (0–1)."""
        e = self.env
        pk = np.flatnonzero((e[1:-1] >= e[:-2]) & (e[1:-1] > e[2:])) + 1
        typ = float(np.percentile(e[pk], 95)) if pk.size else 1.0
        strength = np.minimum(1.0, e[pk] / (typ + 1e-9))
        act_soon = np.array([self.active[p:p + 8].any() for p in pk], bool) if pk.size else np.zeros(0, bool)
        keep = act_soon & (strength > 0.08)
        pk, strength = pk[keep], strength[keep]
        # starts of sung stretches (after ≥ 150 ms quiet) are strong events
        a = self.active.astype(int)
        starts = np.flatnonzero(np.diff(a) == 1) + 1
        starts = [s for s in starts if s >= 15 and not self.active[s - 15:s].any()]
        ev = dict(zip(pk.tolist(), strength.tolist()))
        for s in starts:
            near = [p for p in ev if abs(p - s) <= 6]
            if near:
                for p in near:
                    ev[p] = max(ev[p], 0.9)
            else:
                ev[s] = 0.9
        fr = np.array(sorted(ev))
        # thin: at most one event per 70 ms (keep the strongest)
        out = []
        for p in fr:
            if out and p - out[-1] < 7:
                if ev[p] > ev[out[-1]]:
                    out[-1] = p
            else:
                out.append(p)
        fr = np.array(out, int)
        return fr, np.array([ev[p] for p in fr])


def phrases(voc: Vocals, min_gap: float = 0.25, min_len: float = 0.15,
            longest: float = 5.0) -> list[tuple[float, float]]:
    """Sung stretches: (start, end) s, split at pauses ≥ ``min_gap``."""
    a = voc.active.astype(int)
    d = np.diff(np.r_[0, a, 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out: list[list[float]] = []
    for x, y in zip(voc.t(st), voc.t(en)):
        if out and x - out[-1][1] < min_gap:
            out[-1][1] = y
        else:
            out.append([x, y])
    res = []
    for x, y in out:
        res.extend(_split_long(voc, x, y, longest))
    return [(float(x), float(y)) for x, y in res if y - x >= min_len]


def _split_long(voc: Vocals, a: float, b: float, longest: float = 5.0) -> list[tuple[float, float]]:
    """Long sung stretches (lines sung without a real pause) are cut at the
    deepest dip in level that lasts ≥ 80 ms."""
    if b - a <= longest:
        return [(a, b)]
    f0, f1 = voc.f(a), voc.f(b)
    db = voc.db[f0:f1]
    sm = np.convolve(db, np.ones(8) / 8, mode="same")
    lo, hi = int(1.2 / 0.01), len(sm) - int(1.2 / 0.01)
    if hi <= lo:
        return [(a, b)]
    k = lo + int(np.argmin(sm[lo:hi]))
    if sm[k] > np.percentile(sm, 60) - 6:      # no real dip: leave it
        return [(a, b)]
    c = float(voc.t(f0 + k))
    return _split_long(voc, a, c - 0.04, longest) + _split_long(voc, c + 0.04, b, longest)


def align_lines(ph: list[tuple[float, float]], nsyl: list[int], rate: float,
                max_ph: int = 5, max_lines: int = 3) -> list[tuple[int, int, int, int]]:
    """Which phrases each lyric line covers: per line (first phrase, last
    phrase, lines in its block, first line of its block). A block of
    1…``max_lines`` lines (sung without a pause) covers 1…``max_ph``
    consecutive phrases; phrases with no lyric line (ad-libs, backing vocals,
    bleed in a solo) are skipped at a cost. O(lines × phrases)."""
    L, P = len(nsyl), len(ph)
    INF = 1e18
    W_FIT, W_GAP, W_SKIP, W_PAIR = 2.0, 0.6, 1.0, 1.0
    skip = [W_SKIP * (0.5 + (b - a)) for a, b in ph]
    csung = np.concatenate([[0.0], np.cumsum([b - a for a, b in ph])])
    cnsyl = np.concatenate([[0], np.cumsum(nsyl)])

    def fit(q, p, n):
        sung = csung[p + 1] - csung[q]
        gaps = (ph[p][1] - ph[q][0]) - sung
        return W_FIT * np.log(max(0.2, sung) / (max(1, n) * rate)) ** 2 + W_GAP * max(0.0, gaps - 0.8)

    cost = np.full((L + 1, P + 1), INF)   # cost[l][p]: lines < l placed, phrases < p used or skipped
    back: dict = {}
    cost[0, 0] = 0.0
    for l in range(L + 1):
        for p in range(P + 1):
            c0 = cost[l, p]
            if c0 >= INF / 2:
                continue
            if p < P and c0 + skip[p] < cost[l, p + 1]:          # phrase p has no lyric
                cost[l, p + 1] = c0 + skip[p]
                back[(l, p + 1)] = ("skip",)
            if l == L:
                continue
            for e in range(p, min(P, p + max_ph)):              # block covers phrases p…e
                for k in range(1, min(max_lines, L - l) + 1):
                    c = c0 + fit(p, e, int(cnsyl[l + k] - cnsyl[l])) + W_PAIR * (k - 1)
                    if c < cost[l + k, e + 1]:
                        cost[l + k, e + 1] = c
                        back[(l + k, e + 1)] = ("place", l, p, k)
    if cost[L, P] >= INF / 2:
        raise ValueError("more lyric lines than sung phrases")
    out: list = [None] * L
    l, p = L, P
    while l > 0 or p > 0:
        bk = back[(l, p)]
        if bk[0] == "skip":
            p -= 1
            continue
        _, pl, q, k = bk
        for x in range(pl, l):
            out[x] = (q, p - 1, k, pl)
        l, p = pl, q
    return out


def _spread_lines(ph: list[tuple[float, float]], nsyl: list[int]) -> list[tuple[float, float]]:
    """Last resort: lines spread over the sung time in proportion to their
    syllables → [(start, end)] per line."""
    total = sum(b - a for a, b in ph) or 1.0
    cum, out = np.cumsum([0] + nsyl) / max(1, sum(nsyl)), []
    edges = np.concatenate([[0.0], np.cumsum([b - a for a, b in ph])])

    def at(x):                  # sung-time fraction → song time
        s = x * total
        k = int(np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(ph) - 1))
        return ph[k][0] + (s - edges[k])
    for i in range(len(nsyl)):
        out.append((at(cum[i]), at(min(1.0, cum[i + 1]) - 1e-6)))
    return out


def _place_in_span(voc: Vocals, fr: np.ndarray, st: np.ndarray, syls: list[Syl],
                   a: float, b: float, ph_starts: list[float]) -> None:
    """Syllables of one line (or two sharing a block) inside [a, b]."""
    n = len(syls)
    sel = np.flatnonzero((voc.t(fr) >= a - 0.08) & (voc.t(fr) < b - 0.05))
    T = voc.t(fr[sel]) if sel.size else np.zeros(0)
    w = st[sel] if sel.size else np.zeros(0)
    # phrase starts inside the block are strong candidates
    extra = [x for x in ph_starts if a - 0.01 <= x < b and (not T.size or np.min(np.abs(T - x)) > 0.05)]
    if extra:
        T = np.r_[T, extra]
        w = np.r_[w, [0.9] * len(extra)]
        o = np.argsort(T)
        T, w = T[o], w[o]
    for x in ph_starts:
        if a - 0.01 <= x < b:
            w[np.argmin(np.abs(T - x))] = 1.5
    E = len(T)
    if E < n:                              # not enough attacks: spread evenly
        for i, s in enumerate(syls):
            s.t = a + (b - a) * i / n
        return
    r = max(0.08, (b - a) / n)
    cw = np.concatenate([[0], np.cumsum(w)])
    INF = 1e18
    cost = np.full((n, E), INF)
    back = np.zeros((n, E), int)
    cost[0] = 3.0 * np.abs(T - a) - w + 0.5 * cw[:-1]
    for i in range(1, n):
        for j in range(i, E):
            k = np.arange(i - 1, j)
            dt = T[j] - T[k]
            sil = np.array([(voc.cum_sil[voc.f(T[j])] - voc.cum_sil[voc.f(T[x])]) * HOP / ASR for x in k])
            new_line = syls[i].line != syls[i - 1].line
            c = cost[i - 1, k] + 0.4 * np.log(np.maximum(dt, 0.04) / r) ** 2 \
                + (0.0 if new_line else 1.0 * np.maximum(0.0, sil - 0.35)) + 0.5 * (cw[j] - cw[k + 1])
            bb = int(np.argmin(c))
            cost[i, j] = c[bb] - w[j]
            back[i, j] = k[bb]
    j = int(np.argmin(cost[n - 1] + 0.5 * (cw[E] - cw[np.arange(E) + 1])))
    path = [j]
    for i in range(n - 1, 0, -1):
        j = back[i, j]
        path.append(j)
    for s, j in zip(syls, reversed(path)):
        s.t = float(T[j])


def align_simple(voc: Vocals, syls: list[Syl], log=print) -> None:
    """No model: lyric lines are matched to the sung phrases of the vocals
    stem (pauses), then each line's syllables to the attacks inside it."""
    nlines = max(s.line for s in syls) + 1
    nsyl = [sum(s.line == l for s in syls) for l in range(nlines)]
    fr, st = voc.events()
    blocks = None
    for min_gap, longest, max_lines in ((0.25, 5.0, 3), (0.12, 3.0, 6)):
        ph = phrases(voc, min_gap=min_gap, longest=longest)
        if not ph:
            raise ValueError("no singing found in the vocals stem")
        rate = sum(b - a for a, b in ph) / len(syls)
        try:
            for _ in range(2):                 # re-estimate the singing rate once
                blocks = align_lines(ph, nsyl, rate, max_lines=max_lines)
                used = sorted({k for q, p, _, _ in blocks for k in range(q, p + 1)})
                rate = max(0.12, sum(ph[k][1] - ph[k][0] for k in used) / len(syls))
            break
        except ValueError:
            blocks = None
    if blocks is None:
        log(f"Warning: {nlines} lyric lines but only {len(ph)} sung phrases found — lines spread "
            "evenly over the singing (check them; are these the lyrics of this song?).")
        for l, (a, b) in enumerate(_spread_lines(ph, nsyl)):
            group = [s for s in syls if s.line == l]
            _place_in_span(voc, fr, st, group, a, b, [a])
        _fill_ends(voc, syls)
        return
    done = set()
    for l, (q, p, k, first) in enumerate(blocks):
        if first in done:
            continue
        done.add(first)
        lines = set(range(first, first + k))
        group = [s for s in syls if s.line in lines]
        _place_in_span(voc, fr, st, group, ph[q][0], ph[p][1], [ph[x][0] for x in range(q, p + 1)])
    _fill_ends(voc, syls)


def _fill_ends(voc: Vocals, syls: list[Syl]) -> None:
    """End of each syllable: while the voice keeps sounding, up to the next one."""
    for i, s in enumerate(syls):
        nxt = syls[i + 1].t if i + 1 < len(syls) else s.t + 2.0
        f0, f1 = voc.f(s.t), voc.f(max(s.t + 0.05, nxt - 0.02))
        run = voc.active[f0:f1]
        stop = np.flatnonzero(~run)
        e = f0 + (stop[0] if stop.size else len(run))
        s.end = float(max(s.t + 0.05, min(nxt - 0.01, voc.t(e))))


# ── MMS forced alignment ─────────────────────────────────────────────────

def mms_problem() -> str:
    """Why the MMS aligner can't run here ('' = it can)."""
    try:
        import torch  # noqa: F401
    except Exception:
        return f"PyTorch is not installed in this Python ({sys.executable})"
    try:
        import torchaudio
        import torchaudio.functional as AF
    except Exception as e:
        return (f"torchaudio can't be loaded in {sys.executable} ({e}) — run ATUALIZAR.bat (installer) "
                f"to install it, or: \"{sys.executable}\" -m pip install "
                f"torchaudio=={torch.__version__.split('+')[0]}")
    del AF
    if not hasattr(torchaudio.pipelines, "MMS_FA"):
        return f"torchaudio {torchaudio.__version__} has no pipelines.MMS_FA (needs 2.1 or newer)"
    return ""


def mms_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchaudio
        return hasattr(torchaudio.pipelines, "MMS_FA")
    except Exception:
        return False


def load_mms(device: str = "cpu"):
    """torchaudio's MMS_FA aligner (weights downloaded once, ~1.2 GB, to the
    torch cache: %USERPROFILE%\\.cache\\torch\\hub\\checkpoints)."""
    import torchaudio
    return torchaudio.pipelines.MMS_FA.get_model(with_star=True).to(device)


def _mms_emissions(model, y: np.ndarray, device: str, chunk_s: float = 20.0, ctx_s: float = 2.0):
    """Frame log-probabilities for the whole song, computed in overlapping
    chunks (the model's attention would not fit a whole song at once).
    Returns (emissions [frames × tokens], seconds per frame)."""
    import torch
    n = len(y)
    chunk, ctx = int(chunk_s * ASR), int(ctx_s * ASR)
    outs = []
    with torch.inference_mode():
        for s in range(0, n, chunk):
            a, b = max(0, s - ctx), min(n, s + chunk + ctx)
            x = torch.from_numpy(np.ascontiguousarray(y[a:b])).float().unsqueeze(0).to(device)
            em, _ = model(x)
            em = torch.log_softmax(em.float(), dim=-1)[0].cpu()
            fps = em.shape[0] / ((b - a) / ASR)
            k0 = int(round((s - a) / ASR * fps))
            k1 = min(em.shape[0], k0 + int(round((min(n, s + chunk) - s) / ASR * fps)))
            outs.append(em[k0:k1])
    em = torch.cat(outs)
    return em, n / ASR / max(1, em.shape[0])


def ctc_align(logp: np.ndarray, tokens: list[int], blank: int = 0) -> list[tuple[int, int, float]]:
    """CTC forced alignment (Viterbi) in numpy: ``logp`` [frames × labels]
    log-probabilities, ``tokens`` the label sequence → one (start frame, end
    frame, mean probability) per token. Same result as torchaudio's
    ``forced_align`` + ``merge_tokens``, without its compiled extension (which
    must match the installed PyTorch exactly)."""
    logp = np.asarray(logp, dtype=np.float32)
    T = logp.shape[0]
    L = np.full(2 * len(tokens) + 1, blank, dtype=np.int64)
    L[1::2] = tokens
    N = len(L)
    if T < len(tokens):
        raise ValueError("audio too short for these lyrics")
    skip = np.zeros(N, bool)                  # s-2 → s allowed (different non-blank labels)
    skip[2:] = (L[2:] != blank) & (L[2:] != L[:-2])
    NEG = np.float32(-1e30)
    alpha = np.full(N, NEG, np.float32)
    alpha[0] = logp[0, L[0]]
    if N > 1:
        alpha[1] = logp[0, L[1]]
    back = np.zeros((T, N), np.int8)
    for t in range(1, T):
        a1 = np.r_[NEG, alpha[:-1]]
        a2 = np.where(skip, np.r_[NEG, NEG, alpha[:-2]], NEG)
        stack = np.stack([alpha, a1, a2])
        k = np.argmax(stack, axis=0)
        alpha = stack[k, np.arange(N)] + logp[t, L]
        back[t] = k
    s = int(N - 1 if N == 1 or alpha[N - 1] >= alpha[N - 2] else N - 2)
    path = np.empty(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    out = []
    for k, tok in enumerate(tokens):
        fr = np.flatnonzero(path == 2 * k + 1)
        if not fr.size:
            raise ValueError("alignment lost a letter")
        out.append((int(fr[0]), int(fr[-1]) + 1, float(np.exp(logp[fr, tok]).mean())))
    return out


SILENCE_PENALTY = 12.0     # log-prob taken from letters where the vocals stem is silent


def voice_mask(y16: np.ndarray, n_frames: int, sec: float, margin_db: float = 30.0,
               pad_s: float = 0.2) -> np.ndarray:
    """True for emission frames where the vocals stem has voice (lenient:
    30 dB under the song's typical singing level, widened by ±0.2 s)."""
    import librosa
    hop = 160
    rms = librosa.feature.rms(y=np.asarray(y16, np.float32), frame_length=1024, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
    loud = db[db > -50]
    ref = float(np.percentile(loud, 70)) if loud.size else -20.0
    act = db > ref - margin_db
    pad = max(1, int(pad_s * ASR / hop))
    act = np.convolve(act.astype(float), np.ones(2 * pad + 1), mode="same") > 0
    idx = np.clip((np.arange(n_frames) * sec * ASR / hop).astype(int), 0, len(act) - 1)
    return act[idx]


def mask_silence(logp: np.ndarray, voiced: np.ndarray, letters: list[int],
                 penalty: float = SILENCE_PENALTY) -> np.ndarray:
    """Letters can't be sung where the stem is silent: without this, the
    "anything" token lets a repeated chorus line jump into a guitar solo."""
    out = np.array(logp, dtype=np.float32, copy=True)
    cols = sorted(set(letters))
    if cols and voiced.size == out.shape[0] and voiced.mean() > 0.05:
        out[np.ix_(~voiced, cols)] -= penalty
    return out


def align_mms(y16: np.ndarray, syls: list[Syl], device: str = "cpu", model=None) -> None:
    """Letter-level CTC forced alignment (torchaudio MMS_FA). A ``*`` token
    ("anything") goes before, between and after the lyric lines, so ad-libs,
    backing vocals and solos don't drag the words. A syllable starts at its
    first letter and ends at its last."""
    import torch
    import torchaudio
    labels = torchaudio.pipelines.MMS_FA.get_dict(star=None)
    star = len(labels)
    if model is None:
        model = load_mms(device)
    tokens, owner = [star], [-1]
    for i, s in enumerate(syls):
        if i and s.line != syls[i - 1].line:
            tokens.append(star)
            owner.append(-1)
        for c in normalize(s.text):
            if c in labels:
                tokens.append(labels[c])
                owner.append(i)
    tokens.append(star)
    owner.append(-1)
    if len(tokens) <= 2:
        raise ValueError("no letters to align")
    em, sec = _mms_emissions(model, y16, device)
    if em.shape[-1] <= star:                     # model without the star column
        em = torch.cat([em, torch.full((em.shape[0], 1), -5.0)], dim=-1)
    if em.shape[0] < len(tokens) * 2:
        raise ValueError("song too short for these lyrics")
    logp = mask_silence(em.numpy(), voice_mask(y16, em.shape[0], sec),
                        [t for t in tokens if t != star])
    spans = ctc_align(logp, tokens, blank=0)
    t0: dict[int, float] = {}
    t1: dict[int, float] = {}
    conf: dict[int, list] = {}
    for (a, b, sc), o in zip(spans, owner):
        if o < 0:
            continue
        t0.setdefault(o, a * sec)
        t1[o] = b * sec
        conf.setdefault(o, []).append(sc)
    for i, s in enumerate(syls):
        if i in t0:
            s.t, s.end = float(t0[i]), float(max(t0[i] + 0.04, t1[i]))
            s.conf = float(np.mean(conf[i]))
        else:                                    # no letters (numbers, symbols)
            a = syls[i - 1].end if i else 0.0
            s.t, s.end = a, a + 0.1


def doubtful_lines(syls: list[Syl], n: int = 6, below: float = 0.06) -> list[tuple[int, float, float]]:
    """[(line, start, confidence)] of the lines the MMS aligner was least sure
    of (mean letter probability under ``below``), worst first."""
    by: dict[int, list] = {}
    for x in syls:
        if x.conf is not None:
            by.setdefault(x.line, []).append(x)
    rows = [(li, xs[0].t, float(np.mean([x.conf for x in xs]))) for li, xs in by.items()]
    return sorted([r for r in rows if r[2] < below], key=lambda r: r[2])[:n]


# ── pitch ────────────────────────────────────────────────────────────────

def add_pitch(y16: np.ndarray, syls: list[Syl]) -> None:
    """Median sung pitch (MIDI) of each syllable, from the vocals stem."""
    import librosa
    sr = 8000
    y = librosa.resample(y16, orig_sr=ASR, target_sr=sr)
    hop = 128
    f0, vf, _ = librosa.pyin(y, fmin=70, fmax=1000, sr=sr, frame_length=1024, hop_length=hop)
    midi = librosa.hz_to_midi(np.where(vf, f0, np.nan))
    for s in syls:
        a, b = int(s.t * sr / hop), int(max(s.t + 0.06, s.end) * sr / hop)
        m = midi[a:b + 1]
        m = m[np.isfinite(m)]
        s.pitch = int(round(float(np.median(m)))) if m.size else None
    known = [s.pitch for s in syls if s.pitch is not None]
    fill = int(np.median(known)) if known else 60
    last = fill
    for s in syls:
        if s.pitch is None:
            s.pitch = last
        last = s.pitch


# ── writers ──────────────────────────────────────────────────────────────

_ASCII_MAP = str.maketrans({"ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O",
                            "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "þ": "th", "ð": "d",
                            "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
                            "–": "-", "—": "-", "…": "...", "\u00a0": " "})


def to_ascii(text: str) -> str:
    """Accents off (ã → a, ç → c, é → e…) and only plain ASCII left — Rocksmith's
    lyric font, EOF and some UltraStar tools don't take the rest."""
    t = unicodedata.normalize("NFKD", str(text).translate(_ASCII_MAP))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return "".join(c for c in t if 32 <= ord(c) < 127)


def plain_syllables(syls: list[Syl]) -> list[Syl]:
    from dataclasses import replace as _replace
    return [_replace(s, text=to_ascii(s.text) or "?") for s in syls]


def ultrastar(syls: list[Syl], meta: dict, shift: float = 0.0, mp3: str = "guitar.ogg",
              bpm: float = US_BPM) -> str:
    """UltraStar .txt (times + ``shift`` s). Pitch 0 = C4 when not measured."""
    if not syls:
        raise ValueError("no syllables")
    beat = 60.0 / (4 * bpm)
    gap = round((syls[0].t + shift) * 1000)
    def b(t):
        return int(round(((t + shift) * 1000 - gap) / 1000 / beat))
    head = [f"#TITLE:{meta.get('title') or ''}", f"#ARTIST:{meta.get('artist') or ''}"]
    lang = meta.get("language")
    if lang:
        head.append(f"#LANGUAGE:{LANG_NAMES.get(lang, lang)}")
    for k in ("year", "genre"):
        if meta.get(k):
            head.append(f"#{k.upper()}:{meta[k]}")
    head += [f"#MP3:{mp3}", f"#BPM:{bpm:g}", f"#GAP:{gap}"]
    out = list(head)
    last_end = 0
    for i, s in enumerate(syls):
        if i and s.line != syls[i - 1].line:
            p = syls[i - 1]
            pe, ns = b(p.end), b(s.t)
            out.append(f"- {max(pe, min(ns - 1, pe + max(1, (ns - pe) // 2)))}")
        st = max(b(s.t), last_end)
        ln = max(1, b(s.end) - st)
        nxt = b(syls[i + 1].t) if i + 1 < len(syls) else st + ln + 1
        ln = max(1, min(ln, nxt - st))
        last_end = st + ln
        pitch = (s.pitch - 60) if s.pitch is not None else 0
        txt = s.text + (" " if s.last else "")
        out.append(f": {st} {ln} {pitch} {txt}")
    out.append("E")
    return "\r\n".join(out) + "\r\n"


def vocals_xml(syls: list[Syl], shift: float = 0.0) -> str:
    """Rocksmith 2014 vocals arrangement: ``-`` = the word goes on, ``+`` = end of line."""
    rows = []
    for i, s in enumerate(syls):
        end_line = i + 1 == len(syls) or syls[i + 1].line != s.line
        txt = s.text + ("-" if not s.last else "") + ("+" if end_line else "")
        ln = max(0.05, min(s.end, syls[i + 1].t if i + 1 < len(syls) else s.end) - s.t)
        rows.append(f'  <vocal time="{s.t + shift:.3f}" note="254" length="{ln:.3f}" '
                    f'lyric="{html.escape(txt, quote=True)}"/>')
    return ("<?xml version='1.0' encoding='UTF-8'?>\n"
            f'<vocals count="{len(rows)}">\n<!-- stems2midi lyrics -->\n'
            + "\n".join(rows) + "\n</vocals>\n")


def read_vocals_xml(path: Path) -> list[tuple[float, str]]:
    """(time, lyric) of a Rocksmith vocals XML (EOF's PART VOCALS_RS2.xml)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(str(path)).getroot()
    return [(float(v.get("time")), v.get("lyric") or "") for v in root.iter("vocal")]


def compare(syls: list[Syl], ref: list[tuple[float, str]], shift: float = 0.0) -> str:
    """How close the line and word starts are to a hand-made vocals XML.
    Words are matched by their letters (difflib), so a few differences in the
    lyrics or in the syllable split don't stop the comparison."""
    import difflib

    def words(items):            # [(t, word letters, line_start)]
        out, cur, new_line, new_word = [], None, True, True
        for t, lyr in items:
            core = lyr.rstrip("+")
            cont = core.endswith("-")
            core = core.rstrip("-")
            if new_word:
                cur = [t, "", new_line]
                out.append(cur)
            cur[1] += normalize(core)
            new_line = lyr.endswith("+")
            new_word = not cont
        return out
    ours = words([(s.t + shift, s.text + ("" if s.last else "-") +
                   ("+" if i + 1 == len(syls) or syls[i + 1].line != s.line else ""))
                  for i, s in enumerate(syls)])
    refs = words(ref)
    sm = difflib.SequenceMatcher(a=[w[1] for w in ours], b=[w[1] for w in refs], autojunk=False)
    pairs = [(ours[i + k], refs[j + k]) for i, j, n in sm.get_matching_blocks() for k in range(n)]
    lines = []
    if len(pairs) < 0.9 * max(len(ours), len(refs)):
        lines.append(f"(only {len(pairs)} of {len(refs)} reference words found in these lyrics — "
                     f"compared on those)")
    dt = np.array([o[0] - r[0] for o, r in pairs])
    if dt.size:
        med = float(np.median(dt))
        if abs(med) >= 0.3:
            lines.append(f"(these times are {abs(med):.2f} s {'later' if med > 0 else 'earlier'} than the reference "
                         f"overall — check the shift / lead-in; --align-to <EOF guitar.ogg> measures it)")
    for what, sel in (("line", lambda o, r: o[2] and r[2]), ("word", lambda o, r: True)):
        e = np.array([abs(o[0] - r[0]) for o, r in pairs if sel(o, r)])
        if not e.size:
            lines.append(f"{what} starts: nothing to compare")
            continue
        lines.append(f"{what} starts ({len(e)}): median error {np.median(e):.2f} s, "
                     f"within 0.15 s {np.mean(e < .15):.0%}, 0.3 s {np.mean(e < .3):.0%}, 1 s {np.mean(e < 1):.0%}")
        if dt.size and abs(float(np.median(dt))) >= 0.3:
            e2 = np.array([abs(o[0] - r[0] - float(np.median(dt))) for o, r in pairs if sel(o, r)])
            lines.append(f"   without that offset: median {np.median(e2):.2f} s, within 0.3 s {np.mean(e2 < .3):.0%}, "
                         f"1 s {np.mean(e2 < 1):.0%}")
    return "\n".join(lines)


# ── driver ───────────────────────────────────────────────────────────────

def run(y: np.ndarray, sr: int, text: str, lang: str | None = None, aligner: str = "auto",
        device: str = "cpu", pitch: bool = True, log=print) -> tuple[Lyrics, list[Syl], str]:
    lyr = parse_lyrics(text, lang)
    syls = lyr.syllables()
    if not syls:
        raise ValueError("no lyrics")
    y16 = _mono16k(y, sr)
    used = aligner
    if aligner in ("auto", "mms") and mms_available():
        try:
            log(f"Lining up {len(syls)} syllables with the MMS aligner…")
            align_mms(y16, syls, device=device)
            used = "mms"
        except Exception as e:
            if aligner == "mms":
                raise
            log(f"MMS aligner not available ({e}); using the simple one.")
            used = "simple"
    else:
        why = mms_problem() if aligner in ("auto", "mms") else ""
        if aligner == "mms":
            raise RuntimeError(f"the MMS aligner can't run: {why or 'not available'}")
        if why:
            log(f"MMS aligner not available: {why}.")
        used = "simple"
    voc = Vocals(y16)
    if used == "simple":
        log(f"Lining up {len(syls)} syllables with the vocals' attacks…")
        align_simple(voc, syls, log)
    if pitch:
        add_pitch(y16, syls)
    return lyr, syls, used


US_NAME = "vocals.ultrastar.txt"
JSON_NAME = "vocals.lyrics.json"
XML_FILE = "vocals_RS2.xml"


def save(out: Path, lyr: Lyrics, syls: list[Syl], used: str, text: str) -> Path:
    """Timed syllables (song time) → vocals.lyrics.json, so the files can be
    written again later with another shift / metadata."""
    import json
    from dataclasses import asdict
    path = Path(out) / JSON_NAME
    path.write_text(json.dumps({"lang": lyr.lang, "aligner": used, "text": text,
                                "syllables": [asdict(s) for s in syls]}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path


def write_outputs(out: Path, meta: dict | None = None, shift: float = 0.0,
                  mp3: str | None = None, keep_accents: bool = False) -> dict:
    """vocals.lyrics.json → vocals.ultrastar.txt + vocals_RS2.xml (times + shift).
    Accented letters are written without accents (ç → c) unless ``keep_accents``;
    the .json keeps the lyrics as given."""
    import json
    out = Path(out)
    data = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    syls = [Syl(**d) for d in data["syllables"]]
    meta = dict(meta or {})
    meta.setdefault("language", data.get("lang"))
    if mp3 is None:     # EOF's audio when there's a lead-in; else the song (not a stem / upload name)
        audio = Path(str(meta.get("audio") or "")).name
        plain = audio and not re.match(r"(?i)^(input|bass|vocals|guitar|drums|piano|other)\.", audio)
        mp3 = audio if (plain and not shift) else "guitar.ogg"
    if not keep_accents:
        syls = plain_syllables(syls)
        meta = {k: (to_ascii(v) if isinstance(v, str) and k in ("title", "artist", "album") else v)
                for k, v in meta.items()}
    (out / US_NAME).write_bytes(ultrastar(syls, meta, shift, mp3).encode("utf-8"))
    (out / XML_FILE).write_text(vocals_xml(syls, shift), encoding="utf-8")
    return {"ultrastar": US_NAME, "xml": XML_FILE, "json": JSON_NAME, "aligner": data.get("aligner"),
            "lines": len({s.line for s in syls}), "syllables": len(syls), "shift": shift}


def process(y: np.ndarray, sr: int, text: str, out: Path, meta: dict | None = None,
            shift: float = 0.0, lang: str | None = None, aligner: str = "auto",
            device: str = "cpu", pitch: bool = True, log=print, keep_accents: bool = False) -> dict:
    """Line the lyrics up with the vocals and write all the files to ``out``."""
    lyr, syls, used = run(y, sr, text, lang, aligner, device, pitch, log)
    save(out, lyr, syls, used, text)
    return write_outputs(out, meta, shift, keep_accents=keep_accents)


def find_vocals(src: Path) -> Path | None:
    if src.is_dir():
        for name in ("vocals.wav", "vocals.flac", "vocals.mp3"):
            if (src / name).is_file():
                return src / name
        return None
    return src



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

def main(argv=None) -> int:
    sys.path.insert(0, str(HERE))
    p = argparse.ArgumentParser(
        description="Line up lyrics with the vocals stem → UltraStar .txt + Rocksmith vocals XML. "
                    "(From a song: stems2midi.py song.mp3 --lyrics letra.txt does the separation too.)",
        epilog='Example: lyrics.py "Band - Song_stems" --lyrics letra.txt --artist Band --title Song')
    p.add_argument("source", help="result folder (<song>_stems with vocals.wav) or a vocals stem")
    p.add_argument("--lyrics", default=None,
                   help="text file: one sung line per line, every repeat written out; "
                        "without it, the files are only written again (other --shift / names)")
    p.add_argument("--language", default=None, help="pt, en, es, fr… (default: guessed)")
    p.add_argument("--aligner", choices=["auto", "mms", "simple"], default="auto")
    p.add_argument("--shift", type=float, default=None,
                   help="seconds added to every time (EOF lead-in; default: lead_silence in stems2midi.ini)")
    p.add_argument("--align-to", metavar="AUDIO", default=None,
                   help="the audio the lyrics will play with (EOF's guitar.ogg): the shift is measured from it")
    p.add_argument("--mp3", default=None, help="#MP3 in the UltraStar file (default guitar.ogg)")
    p.add_argument("--artist", default="")
    p.add_argument("--title", default="")
    p.add_argument("--no-pitch", dest="pitch", action="store_false", help="don't measure the sung pitch")
    p.add_argument("--keep-accents", action="store_true",
                   help="keep accented letters (ã, ç, é…) in the UltraStar / Rocksmith files "
                        "(default: written without accents — ã → a, ç → c)")
    p.add_argument("--device", default="auto")
    p.add_argument("-o", "--out", default=None, help="output folder (default: the source's folder)")
    p.add_argument("--compare", default=None, metavar="VOCALS_XML",
                   help="check the result against a hand-made vocals XML (same lyrics)")
    a = p.parse_args(_cli_argv(argv))
    try:
        import json as _json
        ver = _json.loads((HERE / "plugin.json").read_text(encoding="utf-8")).get("version", "?")
    except Exception:  # noqa: BLE001
        ver = "?"
    print(f"stems2midi {ver} — lyrics · Python: {sys.executable}")

    src = Path(a.source)
    out = Path(a.out) if a.out else (src if src.is_dir() else src.parent)
    out.mkdir(parents=True, exist_ok=True)
    if a.shift is None:
        try:
            import s2m_config
            a.shift = s2m_config.lead_silence()
        except Exception:
            a.shift = 0.0
    if a.align_to:
        folder = src if src.is_dir() else src.parent
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("plugin_stems2midi_core", HERE / "stems2midi.py")
            core = importlib.util.module_from_spec(spec)
            sys.modules["plugin_stems2midi_core"] = core
            spec.loader.exec_module(core)
            a.shift = core.align_shift(folder, a.align_to)      # prints the shift it found
        except Exception as e:  # noqa: BLE001
            p.error(f"--align-to: {e}")
    meta = {"artist": a.artist, "title": a.title}
    if not a.lyrics:
        if not (out / JSON_NAME).is_file():
            p.error("--lyrics is needed (no vocals.lyrics.json here yet)")
        info = write_outputs(out, meta, a.shift, a.mp3, keep_accents=a.keep_accents)
    else:
        voc_path = find_vocals(src)
        if voc_path is None:
            print(f"No vocals stem in {src}. Run stems2midi.py with --lyrics (or --stems vocals).",
                  file=sys.stderr)
            return 2
        device = a.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        import librosa
        y, sr = librosa.load(str(voc_path), sr=None, mono=True)
        text = Path(a.lyrics).read_text(encoding="utf-8-sig")
        lyr, syls, used = run(y, sr, text, a.language, a.aligner, device, a.pitch)
        save(out, lyr, syls, used, text)
        info = write_outputs(out, {**meta, "language": lyr.lang}, a.shift, a.mp3, keep_accents=a.keep_accents)
        doubt = doubtful_lines(syls)
        if doubt:
            print("Lines to check first (the aligner was least sure):")
            for li, t, c in doubt:
                words = " ".join(x.text for x in syls if x.line == li)
                print(f"  {int((t + a.shift) // 60)}:{(t + a.shift) % 60:04.1f}  {words[:50]}")
    print(f"{info['syllables']} syllables, {info['lines']} lines ({info['aligner']} aligner"
          f"{f', +{a.shift:g} s' if a.shift else ''})")
    print(f"UltraStar: {out / info['ultrastar']}\nRocksmith vocals: {out / info['xml']}")
    if a.compare:
        import json
        syls = [Syl(**d) for d in json.loads((out / JSON_NAME).read_text(encoding="utf-8"))["syllables"]]
        print(compare(syls, read_vocals_xml(Path(a.compare)), a.shift))
    return 0


if __name__ == "__main__":
    sys.exit(main())
