"""Tests for chart_tools.py (slap/pop marking in a Rocksmith XML)."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chart_tools as ct  # noqa: E402

XML = """<?xml version='1.0' encoding='UTF-8'?>
<song version="7">
<!-- EOF v1.8 -->
  <arrangementProperties slapPop="0" />
  <levels count="1"><level difficulty="0"><notes count="5">
    <note time="1.0" string="0" fret="3" />
    <note time="1.5" string="3" fret="5" />
    <note time="2.0" string="1" fret="5" hammerOn="1" />
    <note time="2.5" string="2" fret="2" pluck="1" />
    <note time="9.0" string="0" fret="0" />
  </notes></level></levels>
</song>
"""


def test_slap_cli(tmp_path):
    src = tmp_path / "PART REAL_BASS_RS2.xml"
    src.write_text(XML, encoding="utf-8")
    assert ct.main(["slap", str(src), "--sections", "0:00-0:05"]) == 0
    out = tmp_path / "PART REAL_BASS_RS2.slap.xml"
    text = out.read_text(encoding="utf-8")
    assert "<!-- EOF v1.8 -->" in text                      # header comment kept
    notes = ET.parse(str(out)).getroot().findall(".//note")
    attrs = [(n.get("slap"), n.get("pluck")) for n in notes]
    assert attrs[0] == ("1", None)          # low string → slap
    assert attrs[1] == (None, "1")          # high string → pop
    assert attrs[2] == (None, None)         # hammer-on stays plain
    assert attrs[3] == (None, "1")          # existing mark kept
    assert attrs[4] == (None, None)         # outside the section
    assert ET.parse(str(out)).getroot().find("arrangementProperties").get("slapPop") == "1"


def test_slap_clear_and_bad_sections(tmp_path):
    src = tmp_path / "a.xml"
    src.write_text(XML, encoding="utf-8")
    assert ct.main(["slap", str(src), "--sections", "9:00-9:10", "--clear", "-o", str(tmp_path / "b.xml")]) == 0
    notes = ET.parse(str(tmp_path / "b.xml")).getroot().findall(".//note")
    assert all(n.get("pluck") is None and n.get("slap") is None for n in notes)
    assert ct.main(["slap", str(src), "--sections", "nonsense"]) == 2


def _song_xml(tmp_path):
    """12 bars of 4/4 at 120 BPM (bar = 2 s). Riff A on bars 0-1, 4-5, 8-9;
    riff B on 2-3, 6-7; bars 10-11 empty. The first A has slap/pop marks."""
    beats = "".join(f'<ebeat time="{i * 0.5:.3f}" measure="{i // 4 + 1 if i % 4 == 0 else -1}" />' for i in range(48))
    A = [(0.0, 0, 3), (0.5, 2, 5), (1.0, 0, 3), (1.5, 2, 5), (2.0, 1, 5), (3.0, 3, 7)]
    B = [(0.0, 1, 0), (1.0, 1, 2), (2.0, 1, 3), (3.0, 1, 5)]
    notes = []
    for bar0, riff in ((0, A), (2, B), (4, A), (6, B), (8, A)):
        for dt, s, f in riff:
            tech = ""
            if bar0 == 0:
                tech = ' slap="1"' if s < 2 else ' pluck="1"'
            if bar0 == 8 and dt == 2.0:
                f, s = 7, 0          # one wrong note in the last A (still ≥ 60 % alike)
            notes.append(f'<note time="{bar0 * 2 + dt:.3f}" string="{s}" fret="{f}" sustain="0"{tech} />')
    xml = f"""<?xml version='1.0' encoding='UTF-8'?>
<song version="7">
  <arrangementProperties slapPop="1" />
  <tuning string0="0" string1="0" string2="0" string3="0" string4="0" string5="0" />
  <phrases count="2"><phrase name="COUNT" maxDifficulty="0" /><phrase name="END" maxDifficulty="0" /></phrases>
  <phraseIterations count="2"><phraseIteration time="0.000" phraseId="0" /><phraseIteration time="20.000" phraseId="1" /></phraseIterations>
  <ebeats count="48">{beats}</ebeats>
  <sections count="0" />
  <levels count="1"><level difficulty="0"><notes count="{len(notes)}">{''.join(notes)}</notes></level></levels>
</song>
"""
    p = tmp_path / "song.xml"
    p.write_text(xml, encoding="utf-8")
    return p


def test_structure_finds_repeats_and_copies_tech(tmp_path):
    src = _song_xml(tmp_path)
    out = tmp_path / "out.xml"
    assert ct.main(["structure", str(src), "--define", "verse=0:00-0:04, chorus=0:04-0:08",
                    "--copy-tech", "-o", str(out)]) == 0
    r = ET.parse(str(out)).getroot()
    secs = [(s.get("name"), s.get("number"), float(s.get("startTime"))) for s in r.iter("section")]
    assert secs[:5] == [("verse", "1", 0.0), ("chorus", "1", 4.0), ("verse", "2", 8.0),
                        ("chorus", "2", 12.0), ("verse", "3", 16.0)]
    assert secs[-1][0] == "noguitar"
    names = [p.get("name") for p in r.iter("phrase")]
    assert names[0] == "verse1" and names[-1] == "END" and "chorus2" not in names
    its = [(float(i.get("time")), names[int(i.get("phraseId"))]) for i in r.iter("phraseIteration")]
    assert its == sorted(its) and its[-1][1] == "END"
    # the same notes → the same phrase name (the repeated chorus is chorus1 again)
    assert [nm for _t, nm in its[:4]] == ["verse1", "chorus1", "verse1", "chorus1"]
    notes = r.findall(".//note")
    second_a = [n for n in notes if 8 <= float(n.get("time")) < 12]
    assert [(n.get("slap"), n.get("pluck")) for n in second_a] == [
        ("1", None), (None, "1"), ("1", None), (None, "1"), ("1", None), (None, "1")]
    third_a = {float(n.get("time")): n for n in notes if 16 <= float(n.get("time")) < 20}
    assert third_a[18.0].get("slap") is None          # the different note isn't touched
    assert third_a[16.0].get("slap") == "1"


def test_structure_slap_option_and_bad_names(tmp_path):
    src = _song_xml(tmp_path)
    out = tmp_path / "o.xml"
    assert ct.main(["structure", str(src), "--define", "chorus=0:04-0:08", "--slap", "chorus", "-o", str(out)]) == 0
    notes = ET.parse(str(out)).getroot().findall(".//note")
    b2 = [n for n in notes if 12 <= float(n.get("time")) < 16]
    assert all(n.get("slap") == "1" for n in b2)        # B riff is all on the A string
    assert ct.main(["structure", str(src), "--define", "refrao=0:04-0:08"]) == 2
    assert ct.main(["structure", str(src), "--define", "verse=0:04"]) == 2


def test_split_phrases_cuts_long_sections_and_names_repeats():
    import xml.etree.ElementTree as ET
    import chart_tools as ct
    beats = "".join(f'<ebeat time="{i * 0.5:.3f}" measure="{i // 4 + 1 if i % 4 == 0 else -1}"/>'
                    for i in range(4 * 21))
    # 16 bars of verse: bars 1-4 = 5-8 = A, 9-12 = B, 13-16 = A; then 4 bars riff
    notes = []
    for bar in range(16):
        pat = [0, 3] if (bar // 4) != 2 else [5, 7]
        for k, f in enumerate(pat):
            notes.append(f'<note time="{2.0 * bar + k:.3f}" string="0" fret="{f}" sustain="0"/>')
    xml = (f'<song><ebeats count="84">{beats}</ebeats>'
           '<phrases count="4"><phrase name="COUNT"/><phrase name="verse1"/><phrase name="riff1"/><phrase name="END"/></phrases>'
           '<phraseIterations count="4"><phraseIteration time="0.000" phraseId="0"/>'
           '<phraseIteration time="0.000" phraseId="1"/><phraseIteration time="32.000" phraseId="2"/>'
           '<phraseIteration time="40.000" phraseId="3"/></phraseIterations>'
           '<sections count="1"><section name="verse" number="1" startTime="0.000"/></sections>'
           f'<levels count="1"><level difficulty="0"><notes>{"".join(notes)}</notes></level></levels></song>')
    root = ET.fromstring(xml)
    n = ct.split_phrases(root, 4)
    names = [p.get("name") for p in root.find("phrases")]
    its = [(float(i.get("time")), names[int(i.get("phraseId"))]) for i in root.find("phraseIterations")]
    verse = [nm for t, nm in its if 0 <= t < 32 and nm.startswith("verse")]
    assert verse == ["verse1", "verse1", "verse2", "verse1"]
    assert [t for t, nm in its if nm.startswith("verse")] == [0.0, 8.0, 16.0, 24.0]
    assert its[-1][1] == "END" and len(root.find("sections")) == 1 and n == len(its)
    root2 = ET.fromstring(xml)
    ct.split_phrases(root2, 0)
    assert len(root2.find("phraseIterations")) == 4


def test_chunks():
    import chart_tools as ct
    assert ct._chunks(8, 4) == [4, 4] and ct._chunks(9, 4) == [4, 5] and ct._chunks(10, 4) == [4, 4, 2]
    assert ct._chunks(3, 4) == [3] and ct._chunks(7, 0) == [7]


def test_fourstring_converts_a_five_string_xml(tmp_path):
    xml = ('<song><title>Canção</title><centOffset>0</centOffset><arrangementProperties standardTuning="1"/>'
           '<tuning stringCount="5" string0="0" string1="0" string2="0" string3="0" string4="0" string5="0"/>'
           '<levels count="1"><level difficulty="0"><notes count="3">'
           '<note time="1.000" string="0" fret="0"/><note time="2.000" string="4" fret="2"/>'
           '<note time="3.000" string="3" fret="5"/></notes>'
           '<anchors count="1"><anchor time="1.000" fret="1" width="4.000"/></anchors></level></levels></song>')
    src = tmp_path / "bass_RS2.xml"
    src.write_text(xml, encoding="utf-8")
    assert ct.main(["fourstring", str(src)]) == 0
    r = ET.parse(str(tmp_path / "bass_RS2.4string.xml")).getroot()
    t = r.find("tuning")
    assert t.get("stringCount") is None and [t.get(f"string{i}") for i in range(6)] == ["7"] * 4 + ["0", "0"]
    assert r.findtext("centOffset") == "-1200" and r.findtext("title") == "Cancao"
    n = r.findall(".//note")
    assert (n[1].get("string"), n[1].get("fret")) == ("3", "7")
    assert all(int(a.get("fret")) >= 1 for a in r.iter("anchor"))
    assert ct.main(["fourstring", str(tmp_path / "bass_RS2.4string.xml")]) == 0     # already 4-string


def test_rhythm5_from_a_five_string_xml(tmp_path):
    xml = ('<song><title>T</title><arrangement>Bass</arrangement><centOffset>0</centOffset>'
           '<arrangementProperties pathBass="1" pathRhythm="0" slapPop="1"/>'
           '<tuning stringCount="5" string0="0" string1="0" string2="0" string3="0" string4="0" string5="0"/>'
           '<levels count="1"><level difficulty="0"><notes count="2">'
           '<note time="1.000" string="0" fret="0" slap="1"/><note time="2.000" string="4" fret="2" pluck="1"/>'
           '</notes></level></levels></song>')
    src = tmp_path / "bass_RS2.xml"
    src.write_text(xml, encoding="utf-8")
    assert ct.main(["rhythm5", str(src)]) == 0
    r = ET.parse(str(tmp_path / "bass_RS2.rhythm.xml")).getroot()
    assert r.findtext("arrangement") == "Rhythm" and r.findtext("centOffset") == "-1200"
    assert [r.find("tuning").get(f"string{i}") for i in range(6)] == ["-5", "-5", "-5", "-5", "-4", "-9"]
    n = r.findall(".//note")
    assert n[1].get("string") == "4" and n[1].get("fret") == "2" and n[0].get("slap") == "-1"
    props = r.find("arrangementProperties")
    assert props.get("pathRhythm") == "1" and props.get("bonusArr") == "1" and r.findtext("tonebase") == "bass"
