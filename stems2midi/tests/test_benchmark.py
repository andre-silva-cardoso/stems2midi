"""Tests for benchmark.py (scoring against Rocksmith XML charts)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark as b  # noqa: E402

XML = """<?xml version='1.0' encoding='UTF-8'?>
<song version="7">
  <title>Test</title><offset>-3.000</offset><startBeat>3.000</startBeat>
  <averageTempo>120</averageTempo>
  <tuning string0="-2" string1="0" string2="0" string3="0" string4="0" string5="0" />
  <capo>0</capo>
  <ebeats count="4"><ebeat time="3.0" measure="1"/><ebeat time="3.5"/><ebeat time="4.0"/><ebeat time="4.5"/></ebeats>
  <levels count="1"><level difficulty="0">
    <notes count="4">
      <note time="3.000" string="0" fret="0" sustain="0.400" />
      <note time="3.500" string="0" fret="3" slap="1" />
      <note time="4.000" string="1" fret="2" pluck="1" />
      <note time="4.500" string="2" fret="5" mute="1" />
    </notes>
    <chords count="1"><chord time="5.000" chordId="0">
      <chordNote string="1" fret="2"/><chordNote string="2" fret="2"/></chord></chords>
    <anchors count="1"><anchor time="3.0" fret="1" width="4"/></anchors>
  </level></levels>
</song>
"""


def _chart(tmp_path):
    p = tmp_path / "PART REAL_BASS_RS2.xml"
    p.write_text(XML, encoding="utf-8")
    return b.parse_rs_xml(p)


def test_parse_rs_xml(tmp_path):
    c = _chart(tmp_path)
    assert c.strings == [26, 33, 38, 43]                    # drop D
    assert [n.pitch for n in c.notes] == [26, 29, 35, 43, 35, 40]
    assert c.notes[0].sustain == 0.4 and c.offset == -3.0
    assert "slap" in c.notes[1].tech and "pop" in c.notes[2].tech
    assert "fret-hand mute" in c.notes[3].tech
    assert c.beats == [3.0, 3.5, 4.0, 4.5] and c.downbeats == [3.0]


def test_score_perfect_and_mutes_excluded(tmp_path):
    c = _chart(tmp_path)
    pred = [{"t": n.t + 0.01, "d": 0.3, "m": n.pitch} for n in c.notes if "fret-hand mute" not in n.tech]
    r = b.score(c, pred)
    assert r["gt_notes"] == 5 and r["exact"]["f1"] == 1.0 and r["aligned"]["f1"] == 1.0


def test_aligned_survives_chart_drift(tmp_path):
    c = _chart(tmp_path)
    pred = [{"t": n.t + 0.12, "d": 0.3, "m": n.pitch} for n in c.notes if "fret-hand mute" not in n.tech]
    r = b.score(c, pred)
    assert r["exact"]["f1"] == 0.0           # 120 ms off: strict says nothing matches
    assert r["aligned"]["f1"] == 1.0          # …but the notes are all there, in order
    assert b.score(c, pred, drift=True)["exact"]["f1"] == 1.0


def test_align_sequences_is_order_preserving():
    gt = [(0.0, 40), (0.5, 43), (1.0, 45), (1.5, 40)]
    pred = [(0.12, 40), (0.6, 43), (0.9, 47), (1.62, 40), (2.0, 45)]
    assert b.align_sequences(gt, pred) == [(0, 0), (1, 1), (3, 3)]


def test_fretting_vs_chart(tmp_path):
    c = _chart(tmp_path)
    f = b.fretting_vs_chart(c)
    assert 0 <= f["agreement"] <= 1 and f["chart_above_12"] == 0
