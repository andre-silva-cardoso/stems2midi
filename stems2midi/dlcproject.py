"""DLC Builder project (``.rs2dlc``) for a result folder.

Written next to ``bass_RS2.xml`` so the song opens in DLC Builder already
filled in: author (charter name), DLC key, artist / title / album / year,
the song audio, the album art and the arrangements found in the folder
(bass, 5-string Rhythm, vocals). Tones are left to DLC Builder (it makes the
base tone) or to you (Import tones).

The format is DLC Builder's project JSON ("Version" is the package version,
0.1 for a new project — DLC Builder puts it in the .psarc name; as DLC Builder
3.x saves it). Paths are relative to the project file, so the folder can be
moved. Only the standard library is used.
"""

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_VERSION = "0.1"      # the package version of a new project (you raise it in DLC Builder)
OLD_DEFAULT_VERSION = "2"    # what 1.2.6–1.3.4 wrote
AUDIO_EXTS = (".wav", ".ogg", ".wem")          # what DLC Builder takes as song audio
COVER_NAMES = ("cover.jpg", "cover.jpeg", "cover.png", "cover.dds")

# Rocksmith arrangement names / route masks (DLC Builder's enum values)
NAME_LEAD, NAME_RHYTHM, NAME_BASS = 0, 2, 3
ROUTE_LEAD, ROUTE_RHYTHM, ROUTE_BASS = 1, 2, 4


def ascii_text(text: str) -> str:
    """Accents off, plain ASCII only (DLC Builder names the package from the
    sort fields; EOF garbles anything else)."""
    t = unicodedata.normalize("NFKD", str(text or ""))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return "".join(c for c in t if 32 <= ord(c) < 127).strip()


def sort_value(text: str) -> str:
    """Sort field: plain letters, a leading "The " / "A " moved off."""
    t = ascii_text(text)
    t = re.sub(r"^(the|a|an)\s+", "", t, flags=re.I)
    return t


def title_sort(title: str, charter: str = "") -> str:
    """Title sort field: plain letters and the charter name at the end
    ("Lugar ao Sol ALSRC") — DLC Builder names the .psarc from the sort
    fields (Delfins_Lugar-ao-Sol-ALSRC_v1_p.psarc)."""
    t = sort_value(title)
    c = ascii_text(charter)
    if c and not t.lower().endswith(c.lower()):
        t = f"{t} {c}".strip()
    return t


def _words(text: str) -> str:
    return "".join(w[:1].upper() + w[1:] for w in re.findall(r"[A-Za-z0-9]+", ascii_text(text)))


def dlc_key(charter: str, artist: str, title: str) -> str:
    """Charter + artist + title in CamelCase letters/digits (as CDLC keys are:
    "DellKornWordUpV2"). Starts with a letter, at least 6 characters."""
    key = _words(charter) + _words(artist) + _words(title)
    if not key or not key[0].isalpha():
        key = "S2m" + key
    while len(key) < 6:
        key += "X"
    return key[:60]


def _year(v) -> int:
    m = re.search(r"\d{4}", str(v or ""))
    return int(m.group(0)) if m else 0


def _ids() -> dict:
    pid = str(uuid.uuid4())
    return {"ID": pid, "MasterID": random.randint(10_000_000, 2_147_483_000), "PersistentID": pid}


def read_arrangement(xml_path: Path) -> dict:
    """Title/artist/album, tuning offsets and A4 pitch of a Rocksmith XML."""
    root = ET.parse(xml_path).getroot()
    tun = root.find("tuning")
    offsets = [int(tun.get(f"string{i}", 0)) if tun is not None else 0 for i in range(6)]
    try:
        cents = float((root.findtext("centOffset") or "0").strip() or 0)
    except ValueError:
        cents = 0.0
    props = root.find("arrangementProperties")
    picked = props is not None and props.get("bassPick") == "1"
    return {
        "title": (root.findtext("title") or "").strip(),
        "artist": (root.findtext("artistName") or "").strip(),
        "album": (root.findtext("albumName") or "").strip(),
        "year": (root.findtext("albumYear") or "").strip(),
        "arrangement": (root.findtext("arrangement") or "").strip(),
        "tuning": offsets,
        "pitch": round(440.0 * 2 ** (cents / 1200.0), 3),
        "picked": picked,
    }


def instrumental(xml_name: str, info: dict, kind: str = "bass", priority: int = 0) -> dict:
    name, route, tone = {
        "bass": (NAME_BASS, ROUTE_BASS, "bass_base"),
        "rhythm": (NAME_RHYTHM, ROUTE_RHYTHM, "rhythm_base"),
        "lead": (NAME_LEAD, ROUTE_LEAD, "lead_base"),
    }[kind]
    ids = _ids()
    return {"Case": "Instrumental", "Fields": [{
        "ID": ids["ID"], "XML": xml_name, "Name": name, "RouteMask": route, "Priority": priority,
        "ScrollSpeed": 1.3, "BassPicked": bool(info.get("picked")) if kind == "bass" else False,
        "Tuning": list(info.get("tuning") or [0] * 6), "TuningPitch": info.get("pitch") or 440,
        "BaseTone": tone, "Tones": [], "MasterID": ids["MasterID"], "PersistentID": ids["PersistentID"],
    }]}


def vocals(xml_name: str) -> dict:
    ids = _ids()
    return {"Case": "Vocals", "Fields": [{
        "ID": ids["ID"], "XML": xml_name, "Japanese": False,
        "MasterID": ids["MasterID"], "PersistentID": ids["PersistentID"]}]}


def find_cover(folder: Path) -> str:
    for n in COVER_NAMES:
        if (folder / n).is_file():
            return n
    return ""


def project_name(artist: str, title: str) -> str:
    base = "_".join(p for p in (_dash(artist), _dash(title)) if p) or "song"
    return base + ".rs2dlc"


def _dash(text: str) -> str:
    return "-".join(re.findall(r"[A-Za-z0-9]+", ascii_text(text)))


# ── tones from DLC Builder's tone collection ─────────────────────────────
# DLC Builder keeps it in SQLite: ~/.rs2-dlcbuilder/tones/user.db (your "User"
# tab) and official.db; table Tones(Artist, Title, Name, BassTone, Description,
# Definition = the tone as JSON, the same shape as a project's "Tones" entries).

def tone_dbs() -> list[Path]:
    env = os.environ.get("STEMS2MIDI_TONE_DB")
    base = Path.home() / ".rs2-dlcbuilder" / "tones"
    return [Path(env)] if env else [base / "user.db", base / "official.db"]


def _norm(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", ascii_text(t).lower())


def list_tones(query: str = "", dbs: list[Path] | None = None, bass_only: bool = False) -> list[dict]:
    """Tones in the collection(s): [{"db", "artist", "title", "name", "bass", "label", "definition"}],
    filtered by the words of ``query`` (any order, in artist / title / name / description)."""
    import sqlite3
    words = _norm(query)
    out = []
    for db in dbs or tone_dbs():
        if not Path(db).is_file():
            continue
        try:
            con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
            rows = con.execute("SELECT artist, title, name, basstone, description, definition FROM tones").fetchall()
            con.close()
        except sqlite3.Error:
            continue
        for artist, title, name, bass, desc, definition in rows:
            label = " - ".join(x for x in (artist, title, name) if x) + (" - Bass" if bass else "")
            hay = set(_norm(" ".join(str(x or "") for x in (artist, title, name, desc, "bass" if bass else ""))))
            if words and not all(w in hay or any(h.startswith(w) for h in hay) for w in words):
                continue
            if bass_only and not bass:
                continue
            out.append({"db": Path(db).stem, "artist": artist or "", "title": title or "", "name": name or "",
                        "bass": bool(bass), "label": label, "definition": definition})
    return out


def find_tone(query: str, dbs: list[Path] | None = None) -> dict:
    """The collection tone named by ``query`` — "bass_base", "Queen 2", or as DLC
    Builder lists it: "Phil Collins - Do you Remember - Queen 2 - Bass". The
    user collection first; an exact name / label wins; else the only match
    (bass tones preferred). ValueError (with the candidates) otherwise."""
    q = " ".join(_norm(query))
    hits = list_tones(query, dbs)
    if not hits:
        raise ValueError(f"no tone matching {query!r} in DLC Builder's tone collection "
                         f"({', '.join(str(d) for d in (dbs or tone_dbs()))})")
    exact = [h for h in hits if " ".join(_norm(h["name"])) == q or " ".join(_norm(h["label"])) == q]
    pool = exact or hits
    for test in (lambda h: h["db"] == "user" and h["bass"], lambda h: h["db"] == "user", lambda h: h["bass"],
                 lambda h: True):
        sel = [h for h in pool if test(h)]
        if len(sel) == 1:
            return sel[0]
        if sel:
            pool = sel
            break
    raise ValueError(f"{len(pool)} tones match {query!r}: " + "; ".join(h["label"] for h in pool[:8])
                     + (" …" if len(pool) > 8 else "") + " — give more of the name")


def tone_entry(hit: dict) -> dict:
    """The project "Tones" entry for a collection tone (key = its name, as DLC
    Builder does when adding a tone from the collection)."""
    tone = json.loads(hit["definition"])
    key = re.sub(r"\s+", "_", hit["name"].strip()) or "bass_base"
    tone["Key"] = key
    tone["Name"] = key
    tone.pop("SortOrder", None)
    return tone


def build(folder: str | Path, *, charter: str = "", meta: dict | None = None, audio: str = "",
          volume: float = -7.0, tone: dict | None = None) -> dict:
    """The project dict for the arrangements in ``folder`` (``bass_RS2.xml``,
    ``bass5_rhythm_RS2.xml``, ``vocals_RS2.xml``). ``meta`` overrides what the
    bass XML says; ``audio`` is a file name in the folder (or a path)."""
    folder = Path(folder)
    meta = {k: v for k, v in (meta or {}).items() if v}
    arrs = []
    info = {}
    bass = folder / "bass_RS2.xml"
    if bass.is_file():
        info = read_arrangement(bass)
        arrs.append(instrumental(bass.name, info, "bass"))
    rhythm = folder / "bass5_rhythm_RS2.xml"
    if rhythm.is_file():
        arrs.append(instrumental(rhythm.name, read_arrangement(rhythm), "rhythm"))
    if (folder / "vocals_RS2.xml").is_file():
        arrs.append(vocals("vocals_RS2.xml"))
    if not arrs:
        raise ValueError(f"no Rocksmith XML (bass_RS2.xml / vocals_RS2.xml) in {folder}")
    artist = meta.get("artist") or info.get("artist") or ""
    title = meta.get("title") or info.get("title") or ""
    album = meta.get("album") or info.get("album") or ""
    year = _year(meta.get("year") or info.get("year"))
    proj = {"Version": PROJECT_VERSION}
    if charter.strip():
        proj["Author"] = charter.strip()
    proj.update({
        "DLCKey": dlc_key(charter, artist, title),
        "ArtistName": {"Value": artist, "SortValue": sort_value(artist)},
        "Title": {"Value": title, "SortValue": title_sort(title, charter)},
        "AlbumName": {"Value": album, "SortValue": sort_value(album)},
        "Year": year,
        "AlbumArtFile": find_cover(folder),
        "AudioFile": {"Path": audio, "Volume": volume},
        "AudioPreviewFile": {"Path": "", "Volume": volume},
        "IgnoredIssues": [],
        "Arrangements": arrs,
        "Tones": [],
    })
    if tone:                       # the base tone of the bass (and 5-string Rhythm) arrangement
        proj["Tones"] = [tone]
        for a in arrs:
            if a["Case"] == "Instrumental":
                a["Fields"][0]["BaseTone"] = tone["Key"]
    return proj


def write(folder: str | Path, *, charter: str = "", meta: dict | None = None, audio: str = "",
          replace_old: bool = True, tone: dict | None = None) -> Path:
    """Write the project into ``folder``. An earlier project written by this
    tool is kept up to date (its IDs and the tones / preview / volume you set
    in DLC Builder stay); returns its path."""
    folder = Path(folder)
    proj = build(folder, charter=charter, meta=meta, audio=audio, tone=tone)
    path = folder / project_name(proj["ArtistName"]["Value"], proj["Title"]["Value"])
    old_path = path if path.is_file() else next(iter(sorted(folder.glob("*.rs2dlc"))), None)
    if old_path is not None and replace_old:
        try:
            old = json.loads(old_path.read_text(encoding="utf-8-sig"))
            proj = merge(old, proj)
            path = old_path
        except (OSError, ValueError):
            pass
    path.write_text(json.dumps(proj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def merge(old: dict, new: dict) -> dict:
    """Refresh an existing project: metadata, audio and arrangement XMLs from
    ``new``; IDs, tones, preview, volumes, ignored issues and anything the
    user added (other arrangements) from ``old``."""
    out = dict(old)
    # the package version is yours (DLC Builder); only the old default is replaced
    out["Version"] = new["Version"] if str(old.get("Version") or OLD_DEFAULT_VERSION) == OLD_DEFAULT_VERSION \
        else old["Version"]
    for k in ("ArtistName", "Title", "AlbumName"):
        out[k] = new[k]
    if new.get("Author"):
        out["Author"] = new["Author"]
    if not old.get("DLCKey"):
        out["DLCKey"] = new["DLCKey"]
    if new.get("Year"):
        out["Year"] = new["Year"]
    if new.get("AlbumArtFile") and not old.get("AlbumArtFile"):
        out["AlbumArtFile"] = new["AlbumArtFile"]
    if new["AudioFile"].get("Path"):
        out["AudioFile"] = {**(old.get("AudioFile") or {}), "Path": new["AudioFile"]["Path"]}
    # a base tone asked for now replaces the one before; tones you added in DLC Builder stay
    if new.get("Tones"):
        t = new["Tones"][0]
        out["Tones"] = [x for x in (old.get("Tones") or []) if x.get("Key") != t["Key"]] + [t]
    arrs = [dict(a) for a in old.get("Arrangements") or []]
    for a in new["Arrangements"]:
        f = a["Fields"][0]
        same = next((o for o in arrs if o.get("Case") == a["Case"]
                     and Path(str(o["Fields"][0].get("XML", ""))).name == f["XML"]), None)
        if same is None:
            arrs.append(a)
            continue
        of = dict(same["Fields"][0])
        for k in ("Tuning", "TuningPitch", "BassPicked"):
            if k in f:
                of[k] = f[k]
        if new.get("Tones") and a["Case"] == "Instrumental":
            of["BaseTone"] = f["BaseTone"]
        same["Fields"] = [of]
    out["Arrangements"] = arrs
    return out
