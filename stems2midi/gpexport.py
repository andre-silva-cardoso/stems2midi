"""The transcription as a Guitar Pro 5 file, on the Rocksmith XML's bars.

* ``bass.gp5`` — the whole song: bar N here is bar N in the XML / EOF (with
  the EOF lead-in, empty bars first), the tempo follows the song bar by bar,
  strings and frets as in the tab, slap/pop and hammer-ons/pull-offs, and a
  marker at each part found in the audio ("verse 2 (A)": same letter = same
  music). Fix it in Guitar Pro and give it back with ``--notes-from``.
* ``bass.examples.gp5`` — one example of each part (the bars the review
  trusts most), each starting at a marker with the part's name: fix those few
  bars and give the file back with ``--sections-from``.

Positions are rounded to 1/16 notes. Needs PyGuitarPro (installed with the
engine for --notes-from).
"""

from __future__ import annotations

import numpy as np

TICKS = 960                 # a quarter note
SIXTEENTHS = {16: (1, False), 12: (2, True), 8: (2, False), 6: (4, True), 4: (4, False),
              3: (8, True), 2: (8, False), 1: (16, False)}   # length in 1/16 → (GP duration value, dotted)


def _split(n16: int) -> list[int]:
    """A length in 1/16 notes as note values (largest first)."""
    out = []
    for v in sorted(SIXTEENTHS, reverse=True):
        while n16 >= v:
            out.append(v)
            n16 -= v
    return out


def bar_grid(beat_map: list[tuple[float, int]], meter: int = 4):
    """(beat times, bar numbers of each bar's first beat index) from the XML beat map."""
    t = np.array([b for b, _m in beat_map], float)
    downs = {i: m for i, (_b, m) in enumerate(beat_map) if m > 0}
    return t, downs


def notes_on_grid(notes: list[dict], beat_t: np.ndarray, downs: dict, meter: int = 4) -> dict:
    """{bar number: [(start16, len16, note), …]} — notes as 1/16 positions in their bar."""
    if len(beat_t) < 2:
        return {}
    idx = np.arange(len(beat_t), dtype=float)
    step = beat_t[-1] - beat_t[-2]
    bar_of_beat = {}
    for i0, m in downs.items():
        for k in range(meter):
            bar_of_beat[i0 + k] = (m, i0)

    def pos(t):
        return float(np.interp(t, beat_t, idx)) if t <= beat_t[-1] else len(beat_t) - 1 + (t - beat_t[-1]) / step

    out: dict[int, list] = {}
    for n in sorted(notes, key=lambda x: x["t"]):
        p16 = int(round(pos(n["t"]) * 4))
        e16 = int(round(pos(n["t"] + max(n.get("d", 0.1), 0.02)) * 4))
        beat = p16 // 4
        if beat not in bar_of_beat:
            continue                               # before the first bar line
        bar, i0 = bar_of_beat[beat]
        start = p16 - i0 * 4
        length = max(1, min(e16 - p16, meter * 4 - start))
        out.setdefault(bar, []).append([start, length, n])
    for bar, lst in out.items():               # one note per position (the lowest), no overlaps
        lst.sort(key=lambda x: (x[0], x[2]["m"]))
        kept = []
        for x in lst:
            if kept and kept[-1][0] == x[0]:
                continue
            if kept and (kept[-1][0] + kept[-1][1] > x[0] or x[2].get("lg")):
                kept[-1][1] = x[0] - kept[-1][0]       # a legato note: the one before rings into it
            kept.append(x)
        out[bar] = kept
    return out


def bar_tempos(beat_t: np.ndarray, downs: dict, meter: int = 4) -> dict:
    """{bar: whole BPM} so the bars land where the song's do: Guitar Pro
    tempos are whole numbers, so each bar's rounding error is carried into
    the next (no drift over the song)."""
    out = {}
    drift = 0.0                         # s the Guitar Pro bars are ahead of the song
    for i0, m in sorted(downs.items(), key=lambda x: x[1]):
        if i0 + meter < len(beat_t):
            length = beat_t[i0 + meter] - beat_t[i0]
        elif i0 + 1 < len(beat_t):
            length = (beat_t[i0 + 1] - beat_t[i0]) * meter
        else:
            continue
        want = length + drift           # make up for the earlier bars
        bpm = int(round(min(300.0, max(30.0, 60.0 * meter / max(want, 1e-3)))))
        drift = want - 60.0 * meter / bpm
        out[m] = bpm
    return out


def build_song(bars: list[int], grid_notes: dict, tempos: dict, tuning: list[int], markers: dict | None = None,
               meta: dict | None = None, meter: int = 4, follow_tempo: bool = True):
    """A PyGuitarPro Song with ``bars`` (XML bar numbers, in this order)."""
    import guitarpro as gp
    meta = meta or {}
    markers = markers or {}
    song = gp.Song()
    song.title = meta.get("title", "") or ""
    song.artist = meta.get("artist", "") or ""
    song.album = meta.get("album", "") or ""
    first_tempo = int(round(tempos.get(bars[0], 120.0))) if bars else 120
    song.tempo = max(30, min(300, first_tempo))
    track = song.tracks[0]
    track.name = "Bass"
    track.strings = [gp.GuitarString(number=i + 1, value=int(p)) for i, p in enumerate(sorted(tuning, reverse=True))]
    track.channel.instrument = 33                       # electric bass (finger)
    track.fretCount = 24
    ns = len(tuning)
    song.measureHeaders = []
    track.measures = []
    last_tempo = song.tempo
    for k, bar in enumerate(bars):
        start = TICKS + k * meter * TICKS
        header = gp.MeasureHeader(number=k + 1, start=start)
        header.timeSignature.numerator = meter
        header.timeSignature.denominator.value = 4
        if bar in markers:
            header.marker = gp.Marker(title=markers[bar])
        song.measureHeaders.append(header)
        measure = gp.Measure(track, header)
        track.measures.append(measure)
        voice = measure.voices[0]
        notes = grid_notes.get(bar, [])
        tempo = int(tempos.get(bar, last_tempo))
        tempo = max(30, min(300, tempo))
        change = follow_tempo and tempo != last_tempo
        cursor = 0
        prev_note = None
        events = []                                     # (start16, len16, note or None)
        for s16, l16, n in notes:
            if s16 > cursor:
                events.append((cursor, s16 - cursor, None))
            events.append((s16, l16, n))
            cursor = s16 + l16
        if cursor < meter * 4:
            events.append((cursor, meter * 4 - cursor, None))
        first_beat = True
        for s16, l16, n in events:
            parts = _split(l16)
            pos = s16
            for j, v in enumerate(parts):
                value, dotted = SIXTEENTHS[v]
                beat = gp.Beat(voice, duration=gp.Duration(value=value, isDotted=dotted),
                               start=start + pos * TICKS // 4)
                if n is None:
                    beat.status = gp.BeatStatus.rest
                else:
                    beat.status = gp.BeatStatus.normal
                    s = n.get("s")
                    f = n.get("f")
                    if s is None or f is None:
                        s, f = 0, max(0, int(n["m"]) - sorted(tuning)[0])
                    note = gp.Note(beat, value=int(f), string=ns - int(s),
                                   type=gp.NoteType.normal if j == 0 else gp.NoteType.tie)
                    beat.notes.append(note)
                    if j == 0:
                        if n.get("tech") == "slap":
                            beat.effect.slapEffect = gp.SlapEffect.slapping
                        elif n.get("tech") == "pop":
                            beat.effect.slapEffect = gp.SlapEffect.popping
                        if n.get("lg") and prev_note is not None:
                            prev_note.effect.hammer = True     # GP marks the note before the legato one
                        prev_note = note
                if first_beat and change:
                    beat.effect.mixTableChange = gp.MixTableChange(tempo=gp.MixTableItem(value=tempo, duration=0))
                    last_tempo = tempo
                first_beat = False
                voice.beats.append(beat)
                pos += v
            if n is None:
                prev_note = None if l16 >= 4 else prev_note
    return song


def export(path, payload: dict, beat_map: list[tuple[float, int]], shift: float = 0.0, markers: dict | None = None,
           bars: list[int] | None = None, meta: dict | None = None, meter: int = 4) -> dict:
    """Write ``payload``'s notes (song time) as GP5 on ``beat_map`` (XML time:
    song + ``shift``). ``bars``: which bars (default: 1 … last). Returns
    {"bars", "notes", "dropped"}."""
    import guitarpro as gp
    tuning = list((payload.get("tuning") or {}).get("strings") or [28, 33, 38, 43])
    notes = [{**n, "t": n["t"] + shift} for n in payload.get("notes") or [] if "m" in n]
    beat_t, downs = bar_grid(beat_map, meter)
    grid_notes = notes_on_grid(notes, beat_t, downs, meter)
    tempos = bar_tempos(beat_t, downs, meter)
    if bars is None:
        last = max(downs.values()) if downs else 1
        bars = list(range(1, last + 1))
    song = build_song(bars, grid_notes, tempos, tuning, markers, meta, meter)
    gp.write(song, str(path))
    placed = sum(len(grid_notes.get(b, [])) for b in bars)
    return {"bars": len(bars), "notes": placed, "dropped": len(notes) - sum(len(v) for v in grid_notes.values())}


def part_markers(segs: list[dict], bar_no) -> dict:
    """{bar: "verse 2 (A)"} for the whole-song file."""
    return {bar_no(s["b0"]): f"{s['name']} {s['number']} ({s['letter']})" for s in segs}
