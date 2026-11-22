"""DLC Builder project (dlcproject.py) and its use by the CLI."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))

import dlcproject as dp  # noqa: E402

XML = """<?xml version='1.0' encoding='utf-8'?>
<song version="7"><title>{title}</title><arrangement>Bass</arrangement><centOffset>{cents}</centOffset>
<songLength>10</songLength><startBeat>0</startBeat><averageTempo>120</averageTempo>
<tuning string0="-2" string1="0" string2="0" string3="0" string4="0" string5="0" />
<artistName>{artist}</artistName><albumName>Ao Vivo</albumName><albumYear>1998</albumYear>
<arrangementProperties bassPick="1" /></song>"""


def _folder(tmp_path, title="Lugar ao Sol", artist="Delfins", cents=0):
    d = tmp_path / "song_stems"
    d.mkdir()
    (d / "bass_RS2.xml").write_text(XML.format(title=title, artist=artist, cents=cents), encoding="utf-8")
    return d


def test_key_and_sort_fields():
    assert dp.dlc_key("ALSRC", "Delfins", "Lugar ao Sol") == "ALSRCDelfinsLugarAoSol"
    assert dp.dlc_key("", "Sétima Legião", "Noutro Lugar") == "SetimaLegiaoNoutroLugar"
    assert dp.dlc_key("", "", "1979")[0].isalpha()
    assert dp.sort_value("The Cure") == "Cure" and dp.sort_value("Pontapés") == "Pontapes"
    assert dp.project_name("Sétima Legião", "Por Quem Não Esqueci") == "Setima-Legiao_Por-Quem-Nao-Esqueci.rs2dlc"


def test_project_from_folder(tmp_path):
    d = _folder(tmp_path, cents=-1200)
    (d / "vocals_RS2.xml").write_text("<vocals count='0' />", encoding="utf-8")
    (d / "cover.jpg").write_bytes(b"\xff\xd8\xff" + b"0" * 2000)
    p = dp.write(d, charter="ALSRC", meta={"year": "1996"}, audio="song.ogg")
    proj = json.loads(p.read_text(encoding="utf-8"))
    assert p.name == "Delfins_Lugar-ao-Sol.rs2dlc"
    assert proj["Author"] == "ALSRC" and proj["DLCKey"] == "ALSRCDelfinsLugarAoSol"
    assert proj["Title"] == {"Value": "Lugar ao Sol", "SortValue": "Lugar ao Sol"}
    assert proj["AlbumName"]["Value"] == "Ao Vivo" and proj["Year"] == 1996
    assert proj["AlbumArtFile"] == "cover.jpg" and proj["AudioFile"]["Path"] == "song.ogg"
    bass, voc = proj["Arrangements"]
    f = bass["Fields"][0]
    assert bass["Case"] == "Instrumental" and f["Name"] == 3 and f["RouteMask"] == 4
    assert f["Tuning"] == [-2, 0, 0, 0, 0, 0] and f["TuningPitch"] == 220.0 and f["BassPicked"] is True
    assert f["ID"] == f["PersistentID"] and 0 < f["MasterID"] < 2 ** 31
    assert voc["Case"] == "Vocals" and voc["Fields"][0]["XML"] == "vocals_RS2.xml"


def test_no_charter_leaves_author_to_dlc_builder(tmp_path):
    proj = dp.build(_folder(tmp_path))
    assert "Author" not in proj and proj["DLCKey"] == "DelfinsLugarAoSol"


def test_rewrite_keeps_what_was_set_in_dlc_builder(tmp_path):
    d = _folder(tmp_path)
    p = dp.write(d, charter="ALSRC", audio="song.ogg")
    proj = json.loads(p.read_text(encoding="utf-8"))
    f = proj["Arrangements"][0]["Fields"][0]
    f["BaseTone"], f["Tones"] = "Delfins_clean", ["Delfins_clean", "Delfins_drive"]
    proj["Tones"] = [{"Key": "Delfins_clean"}, {"Key": "Delfins_drive"}]
    proj["AudioPreviewFile"] = {"Path": "song_preview.wav", "Volume": -5}
    p.write_text(json.dumps(proj), encoding="utf-8")
    p2 = dp.write(d, charter="ALSRC", meta={"album": "Saber a Mar"}, audio="song_lead5s.ogg")
    new = json.loads(p2.read_text(encoding="utf-8"))
    nf = new["Arrangements"][0]["Fields"][0]
    assert p2 == p and len(new["Arrangements"]) == 1
    assert nf["ID"] == f["ID"] and nf["Tones"] == f["Tones"] and nf["BaseTone"] == "Delfins_clean"
    assert new["Tones"] == proj["Tones"] and new["AudioPreviewFile"]["Path"] == "song_preview.wav"
    assert new["AudioFile"]["Path"] == "song_lead5s.ogg" and new["AlbumName"]["Value"] == "Saber a Mar"


def test_no_xml_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        dp.build(tmp_path)


def test_song_audio_moved_like_the_xml(tmp_path):
    sf = pytest.importorskip("soundfile")
    import stems2midi as s2m
    a = np.random.default_rng(0).normal(0, 0.1, (2, 44100)).astype(np.float32)
    name = s2m.write_song_audio(tmp_path, a, 44100, shift=2.0)
    assert name in ("song_lead2s.ogg", "song_lead2s.wav")
    info = sf.info(str(tmp_path / name))
    assert abs(info.duration - 3.0) < 0.05
    # reused when asked again for the same shift; rebuilt from bass + backing otherwise
    assert s2m.song_audio_for(tmp_path, 2.0) == (name, "")
    sf.write(str(tmp_path / "bass.wav"), a.T, 44100)
    sf.write(str(tmp_path / "backing_no_bass.wav"), a.T, 44100)
    n0, note = s2m.song_audio_for(tmp_path, 0.0)
    assert n0.startswith("song.") and note == ""


def test_cli_folder_mode_writes_project_with_ini_charter(tmp_path, monkeypatch):
    pytest.importorskip("librosa")
    import stems2midi as s2m
    folder = tmp_path / "Delfins - Lugar ao Sol_stems"
    folder.mkdir()
    notes = [{"t": 0.5 * i, "d": 0.4, "m": 40 + i % 5, "v": 90, "s": 0, "f": i % 5} for i in range(16)]
    payload = {"version": 2, "source": "Delfins - Lugar ao Sol.mp3", "bpm": 120.0,
               "beats": [0.5 * i for i in range(20)], "duration": 10.0, "notes": notes,
               "tuning": {"strings": [28, 33, 38, 43], "label": "E standard", "offsets": [0, 0, 0, 0]}}
    (folder / "bass.notes.json").write_text(json.dumps(payload), encoding="utf-8")
    ini = tmp_path / "s2m.ini"
    ini.write_text("[defaults]\ncharter = ALSRC\n", encoding="utf-8")
    monkeypatch.setenv("STEMS2MIDI_CONFIG", str(ini))
    assert s2m.main([str(folder)]) == 0
    proj = json.loads((folder / "Delfins_Lugar-ao-Sol.rs2dlc").read_text(encoding="utf-8"))
    assert proj["Author"] == "ALSRC" and proj["Arrangements"][0]["Fields"][0]["XML"] == "bass_RS2.xml"
    assert s2m.main([str(folder), "--no-dlc-project", "--charter", "X"]) == 0
    assert json.loads((folder / "Delfins_Lugar-ao-Sol.rs2dlc").read_text(encoding="utf-8"))["Author"] == "ALSRC"
