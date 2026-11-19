"""Tests for fretting.py — string/fret assignment for bass."""

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fretting as fr  # noqa: E402


@dataclass
class N:
    start: float
    end: float
    pitch: int
    velocity: int = 90
    legato: bool = False


def seq(pitches, d=0.25):
    return [N(i * d, i * d + d * 0.9, p) for i, p in enumerate(pitches)]


def frets(res):
    return [(n.string, n.fret) for n in res.notes]


# ── tunings ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec,expected", [
    (None, (28, 33, 38, 43)),
    ("4", (28, 33, 38, 43)),
    ("5", (23, 28, 33, 38, 43)),
    ("5c", (28, 33, 38, 43, 48)),
    ("drop-d", (26, 33, 38, 43)),
    ("Drop D", (26, 33, 38, 43)),
    ("D1 A1 D2 G2", (26, 33, 38, 43)),
    ("G2 D2 A1 D1", (26, 33, 38, 43)),        # high → low accepted
    ("B0,E1,A1,D2,G2", (23, 28, 33, 38, 43)),
    ("Eb1 Ab1 Db2 Gb2", (27, 32, 37, 42)),
    ("-2,0,0,0", (26, 33, 38, 43)),            # Rocksmith offsets
    ("-1,-1,-1,-1,-1", (22, 27, 32, 37, 42)),
    ("26,33,38,43", (26, 33, 38, 43)),         # MIDI numbers
    ([28, 33, 38, 43], (28, 33, 38, 43)),
])
def test_parse_tuning(spec, expected):
    assert fr.parse_tuning(spec).strings == expected


@pytest.mark.parametrize("bad", ["Z1 A1", "E1 E1 A1 D2", "E1 A1", "E1 D2 A1 G2"])
def test_parse_tuning_rejects(bad):
    with pytest.raises(ValueError):
        fr.parse_tuning(bad)


def test_offsets():
    assert fr.parse_tuning("drop-d").offsets() == [-2, 0, 0, 0]
    assert fr.parse_tuning("5c").offsets() == [0, 0, 0, 0, 0]


# ── assignment ──────────────────────────────────────────────────────────────

def test_open_strings():
    res = fr.assign_frets(seq([28, 33, 38, 43]), "4")
    assert frets(res) == [(0, 0), (1, 0), (2, 0), (3, 0)]


def test_pentatonic_box_has_no_shifts():
    # A minor pentatonic up and down: should stay in one position
    res = fr.assign_frets(seq([33, 36, 38, 40, 43, 45, 43, 40, 38, 36, 33]), "4")
    st = res.stats()
    assert st["shifts"] == 0 and st["max_fret"] <= 12
    hand = res.anchors[0].fret
    assert all(n.fret == 0 or hand <= n.fret <= hand + 3 for n in res.notes)


def test_pitches_are_preserved():
    pitches = [28, 31, 33, 35, 36, 40, 43, 47, 50, 52]
    tun = fr.parse_tuning("4")
    res = fr.assign_frets(seq(pitches), tun)
    assert [tun.strings[n.string] + n.fret for n in res.notes] == pitches


def test_repeated_note_keeps_position():
    res = fr.assign_frets(seq([38, 38, 38, 38]), "4")
    assert len(set(frets(res))) == 1


def test_prefers_below_12th_fret():
    # A2 can be G-2, D-7, A-12 or E-17: with no context the low option wins
    res = fr.assign_frets(seq([45]), "4")
    assert res.notes[0].fret <= 7


def test_jumps_beat_low_frets():
    # a run that sits at 12–14 on G: staying up there beats hopping down
    res = fr.assign_frets(seq([55, 57, 55, 52, 57, 55], d=0.12), "4")
    assert res.stats()["shifts"] == 0


def test_dp_beats_greedy_on_shifts():
    random.seed(7)
    scale = [0, 2, 3, 5, 7, 8, 10]
    deg, t, line = 7, 0.0, []
    for _ in range(300):
        deg = max(0, min(13, deg + random.choice([-2, -1, -1, 0, 1, 1, 2, 4, -4])))
        line.append(N(t, t + 0.2, 28 + 12 * (deg // 7) + scale[deg % 7]))
        t += 0.25
    tun = fr.parse_tuning("4")
    res = fr.assign_frets(line, tun)

    def greedy(pick):
        """Note-by-note choice; the hand moves only when a note is out of reach."""
        hand, moves = None, 0
        for n in line:
            f = pick([n.pitch - o for o in tun.strings if 0 <= n.pitch - o <= 24], hand)
            if f == 0:
                continue
            if hand is None:
                hand = f
            elif not hand <= f <= hand + 3:
                hand, moves = (f if f < hand else f - 3), moves + 1
        return moves

    lowest = greedy(lambda c, h: min(c))
    nearest = greedy(lambda c, h: min(c) if h is None else min(
        c, key=lambda f: (0 if f == 0 or h <= f <= h + 3 else min(abs(f - h), abs(f - h - 3)), f)))
    shifts = res.stats()["shifts"]
    assert shifts <= lowest / 2 and shifts < nearest, (shifts, lowest, nearest)


def test_five_string_uses_low_b_and_four_string_moves_octave():
    line = seq([23, 26, 28, 31])
    r5 = fr.assign_frets(line, "5")
    assert r5.notes[0].string == 0 and r5.notes[0].fret == 0 and not r5.warnings
    r4 = fr.assign_frets(line, "4")
    assert r4.notes[0].pitch == 35 and r4.notes[0].source_pitch == 23
    assert r4.warnings
    r4d = fr.assign_frets(line, "4", out_of_range="drop")
    assert len(r4d.notes) == 2


def test_drop_d_low_d_is_open():
    res = fr.assign_frets(seq([26, 26, 33]), "drop-d")
    assert frets(res)[0] == (0, 0)


def test_double_stops_on_distinct_strings():
    notes = [N(0, 1, 40), N(0, 1, 47), N(1, 2, 38), N(1, 2, 45)]
    res = fr.assign_frets(notes, "4")
    for t in (0, 1):
        grp = [n for n in res.notes if n.start == t]
        assert len({n.string for n in grp}) == 2
        fretted = [n.fret for n in grp if n.fret]
        assert not fretted or max(fretted) - min(fretted) <= 3


def test_max_fret_respected():
    res = fr.assign_frets(seq([60, 62, 64]), "4", max_fret=20)
    assert all(n.fret <= 20 for n in res.notes)


def test_rest_makes_shift_cheaper_than_fast_run():
    w = fr.Weights()
    assert w.rest_discount < 1 < w.fast_run


def test_ascii_tab():
    res = fr.assign_frets(seq([28, 31, 33, 36]), "4")
    tab = fr.to_ascii_tab(res, bpm=60)
    lines = [l for l in tab.splitlines() if "|" in l]
    assert [l[0] for l in lines[:4]] == ["G", "D", "A", "E"]
    assert "3" in lines[3] and "0" in lines[3]


def test_large_input_is_fast():
    import time
    random.seed(3)
    line = [N(i * 0.2, i * 0.2 + 0.15, random.randint(23, 55)) for i in range(2000)]
    t0 = time.time()
    res = fr.assign_frets(line, "5")
    assert len(res.notes) == 2000 and time.time() - t0 < 5


# ── slap / pop ──────────────────────────────────────────────────────────────

def test_parse_sections():
    assert fr.parse_sections("all") == [(0.0, float("inf"))]
    assert fr.parse_sections("0:32-1:05, 130-162.5") == [(32.0, 65.0), (130.0, 162.5)]
    assert fr.parse_sections("") == []
    for bad in ("1:00-0:30", "abc", "10-"):
        with pytest.raises(ValueError):
            fr.parse_sections(bad)


def test_apply_slap_low_strings_slap_high_strings_pop():
    notes = seq([28, 41, 43, 33, 50])          # E1 F2 G2 A1 D3
    notes[3].legato = True                     # a hammer-on: stays plain
    res = fr.assign_frets(notes, "4")
    counts = fr.apply_slap(res, fr.parse_sections("0-0.9"))   # last note outside
    techs = [(n.pitch, n.string, n.tech) for n in res.notes]
    assert [t for _p, _s, t in techs] == ["slap", "pop", "pop", None, None]
    # slaps sit on the low strings, pops on the high ones
    assert techs[0][1] < 2 and techs[1][1] >= 2 and techs[2][1] >= 2
    assert all(res.tuning.strings[n.string] + n.fret == n.pitch for n in res.notes)
    assert counts == {"slap": 1, "pop": 2}
    tab = fr.to_ascii_tab(res, bpm=60)
    assert "S = slap" in tab and ("S" in tab.splitlines()[-2] or "P" in tab)


def test_apply_slap_split_is_above_e2():
    res = fr.assign_frets(seq([40, 41]), "4")
    fr.apply_slap(res, fr.parse_sections("all"))
    assert [n.tech for n in res.notes] == ["slap", "pop"]
    assert res.notes[0].string == 1 and res.notes[1].string >= 2      # E2 on A-7, F2 on D-3


def test_apply_slap_five_string_uses_three_low_strings():
    res = fr.assign_frets(seq([23, 28, 33, 38, 43]), "5")
    fr.apply_slap(res, fr.parse_sections("all"))
    assert [n.tech for n in res.notes] == ["slap", "slap", "slap", "slap", "pop"]
    assert [n.string for n in res.notes][:3] == [0, 1, 2]


# ── tuning suggestion ───────────────────────────────────────────────────────

def test_suggest_tuning():
    e_song = [28, 33, 38, 40, 43] * 40
    assert fr.suggest_tuning(e_song, "4")[0] is None
    few_low = e_song + [26] * 5                       # a few slips / slides
    assert fr.suggest_tuning(few_low, "4")[0] is None
    b_song = e_song + [23, 24, 26] * 10               # 30 notes, 13 %
    better, msg = fr.suggest_tuning(b_song, "4")
    assert better == "5" and "B standard" in msg and "Drop D" not in msg
    d_song = e_song + [26, 27] * 15
    better, msg = fr.suggest_tuning(d_song, "4")
    assert better == "5" and "Drop D" in msg
    assert fr.suggest_tuning(b_song, "5")[0] is None   # already a low B
    assert fr.low_notes([20, 23, 27, 28], "4") == [23, 27]


def test_octave_pop_uses_octave_shape():
    # slap G1 (E-3), pop G2 an octave up → D-5, not open G; slap D2 (A-5), pop D3 → G-7
    res = fr.assign_frets(seq([31, 43, 31, 38, 50, 38], d=0.2), "4")
    fr.apply_slap(res, fr.parse_sections("all"))
    got = [(n.tech, n.string, n.fret) for n in res.notes]
    assert got[0] == ("slap", 0, 3) and got[1] == ("pop", 2, 5)
    assert got[3] == ("slap", 1, 5) and got[4] == ("pop", 3, 7)
    for n in res.notes:
        assert res.tuning.strings[n.string] + n.fret == n.pitch
    assert not any(hasattr(n, "_octave_string") for n in res.notes)
