"""Cover / lyrics lookup (metafetch.py) — the network is faked."""

import io
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metafetch as mf  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + b"x" * 5000

ITUNES = {"results": [
    {"artistName": "Delfins", "trackName": "Sou como um Rio", "collectionName": "Saber a Mar",
     "artworkUrl100": "https://is1.mzstatic.com/a/100x100bb.jpg", "releaseDate": "1996-01-01T08:00:00Z"},
    {"artistName": "Delfins", "trackName": "1 Lugar ao Sol (Remastered)", "collectionName": "Ao Vivo",
     "artworkUrl100": "https://is1.mzstatic.com/b/100x100bb.jpg", "releaseDate": "1998-05-01T08:00:00Z"},
]}
LRC = {"artistName": "Delfins", "trackName": "Lugar ao Sol", "instrumental": False, "duration": 349,
       "plainLyrics": "[Refrão]\nprimeira linha\nsegunda linha\n\n\n\nterceira linha",
       "syncedLyrics": "[00:12.30] primeira linha\n[00:15.00] segunda linha\n[00:20.00] terceira linha"}


@pytest.fixture
def web(monkeypatch):
    calls = []

    def get_json(url):
        calls.append(url)
        if "itunes.apple.com" in url:
            return ITUNES
        if "lrclib.net/api/get" in url:
            return None                     # 404 → search
        if "lrclib.net/api/search" in url:
            return [{**LRC, "artistName": "Outra Banda"}, LRC]
        if "musicbrainz" in url:
            return {"recordings": []}
        raise AssertionError(url)

    def get_bytes(url):
        calls.append(url)
        assert "1000x1000bb.jpg" in url and "/b/" in url
        return JPEG

    monkeypatch.setattr(mf, "get_json", get_json)
    monkeypatch.setattr(mf, "get_bytes", get_bytes)
    return calls


def test_names_match_despite_numbers_accents_and_tails():
    assert mf.similarity("1 lugar ao sol", "Lugar ao Sol (Remastered)") == 1.0
    assert mf.similarity("Setima Legiao", "Sétima Legião") == 1.0
    assert mf.score("Delfins", "Lugar ao Sol", "Xutos & Pontapés", "Lugar ao Sol") < mf.MIN_SCORE


def test_fetch_cover_lyrics_album_year(tmp_path, web):
    got = mf.fetch(tmp_path, "Delfins", "1 lugar ao sol", duration=348.9, log=lambda m: None)
    assert got["cover"] == "cover.jpg" and (tmp_path / "cover.jpg").read_bytes() == JPEG
    assert got["album"] == "Ao Vivo" and got["year"] == "1998"
    text = (tmp_path / "lyrics.txt").read_text(encoding="utf-8")
    assert text == "primeira linha\nsegunda linha\n\nterceira linha\n"
    assert got["lrc"] == "lyrics.lrc" and "[00:12.30]" in (tmp_path / "lyrics.lrc").read_text(encoding="utf-8")


def test_files_in_the_folder_are_kept(tmp_path, web):
    (tmp_path / "cover.png").write_bytes(b"\x89PNG mine")
    (tmp_path / "lyrics.txt").write_text("a minha letra\n", encoding="utf-8")
    got = mf.fetch(tmp_path, "Delfins", "Lugar ao Sol", log=lambda m: None)
    assert got["cover"] == "" and (tmp_path / "cover.png").read_bytes() == b"\x89PNG mine"
    assert (tmp_path / "lyrics.txt").read_text(encoding="utf-8") == "a minha letra\n"
    assert not any("mzstatic" in c for c in web)


def test_no_network_never_raises(tmp_path, monkeypatch):
    def down(url):
        raise urllib.error.URLError("proxy refused")
    monkeypatch.setattr(mf, "get_json", down)
    monkeypatch.setattr(mf, "get_bytes", down)
    got = mf.fetch(tmp_path, "Delfins", "Lugar ao Sol", log=lambda m: None)
    assert got["cover"] == got["lyrics"] == ""
    assert any("no connection" in n for n in got["notes"])


def test_wrong_artist_is_not_taken(tmp_path, monkeypatch):
    monkeypatch.setattr(mf, "get_json", lambda url: {"results": [ITUNES["results"][1]]} if "itunes" in url
                        else ([{**LRC, "artistName": "Outra Banda"}] if "search" in url else None))
    monkeypatch.setattr(mf, "get_bytes", lambda url: JPEG)
    got = mf.fetch(tmp_path, "Sétima Legião", "Lugar ao Sol", log=lambda m: None)
    assert got["cover"] == "" and got["lyrics"] == ""


def test_lrc_to_text():
    assert mf.lrc_to_text("[ar:X]\n[00:01.00] um\n[00:02.00]\n[00:03.50][01:03.50] dois") == "um\ndois"
