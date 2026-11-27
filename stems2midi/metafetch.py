"""Album art and lyrics from free web services, saved in the song's folder.

* cover  — iTunes Search (artwork up to 1000×1000), else MusicBrainz +
  Cover Art Archive. Saved as ``cover.jpg`` (DLC Builder converts it).
* lyrics — LRCLIB (lrclib.net, open lyrics database, often with line
  timings). Saved as ``lyrics.txt`` (plain, one sung line per line — what
  ``--lyrics`` takes) and ``lyrics.lrc`` when timed lines exist.
* album / year — from the iTunes match when not given.

Genius has no lyrics API and its terms don't allow fetching its pages, so it
isn't used. Files already in the folder are never overwritten. Nothing here
can fail a run: no network, no match → a note and the run goes on. Standard
library only.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "stems2midi/{ver} (FeedBack plugin; https://github.com/andre-silva-cardoso/stems2midi)"
TIMEOUT = 15
COVER_NAME = "cover.jpg"
LYRICS_NAME = "lyrics.txt"
LRC_NAME = "lyrics.lrc"
COVER_SIZE = 1000            # iTunes artwork size asked for (px)
MIN_SCORE = 0.75             # artist and title must both match about this well

_version = "1.2.11"


def set_version(v: str) -> None:
    global _version
    _version = v


# ── network (patched in tests) ──────────────────────────────────────────

def _request(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT.format(ver=_version),
                                               "Accept": "application/json, image/*;q=0.9, */*;q=0.5"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:     # noqa: S310 — fixed https hosts
        return r.read()


def get_json(url: str):
    try:
        return json.loads(_request(url).decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def get_bytes(url: str) -> bytes:
    return _request(url)


# ── matching ────────────────────────────────────────────────────────────

def norm(s: str) -> str:
    """Lower case, no accents, no "(Remastered)"/"[Live]"/"- 2005 Remaster"
    tails, letters and digits only."""
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s*[(\[].*?[)\]]", " ", s)
    s = re.sub(r"\s+-\s+.*(remaster|version|live|edit|mix|mono|stereo).*$", " ", s)
    s = re.sub(r"^\d+\s*[-.]?\s+", "", s)           # "1 Lugar ao Sol" / "05 - Song"
    s = s.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", s)


def similarity(a: str, b: str) -> float:
    """0..1: 1 for equal names, high when one contains the other."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.75 + 0.25 * min(len(a), len(b)) / max(len(a), len(b))
    # common prefix / bigram overlap
    grams = lambda s: {s[i:i + 2] for i in range(len(s) - 1)} or {s}     # noqa: E731
    ga, gb = grams(a), grams(b)
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def score(artist: str, title: str, cand_artist: str, cand_title: str) -> float:
    sa = similarity(artist, cand_artist) if artist else 0.8
    st = similarity(title, cand_title)
    return min(sa, st)


# ── cover ───────────────────────────────────────────────────────────────

def itunes_search(artist: str, title: str, album: str = "") -> dict | None:
    """Best iTunes song match: {"artist", "title", "album", "year", "art"}."""
    term = " ".join(x for x in (artist, title) if x)
    url = "https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"term": term, "entity": "song", "limit": 25, "media": "music"})
    data = get_json(url) or {}
    best, best_s = None, 0.0
    for r in data.get("results") or []:
        s = score(artist, title, r.get("artistName", ""), r.get("trackName", ""))
        if album and similarity(album, r.get("collectionName", "")) >= 0.8:
            s += 0.05
        if s > best_s:
            best, best_s = r, s
    if best is None or best_s < MIN_SCORE:
        return None
    art = best.get("artworkUrl100") or best.get("artworkUrl60") or ""
    art = re.sub(r"/\d+x\d+(bb)?\.(jpg|png)$", f"/{COVER_SIZE}x{COVER_SIZE}bb.jpg", art)
    return {"artist": best.get("artistName", ""), "title": best.get("trackName", ""),
            "album": best.get("collectionName", ""), "year": str(best.get("releaseDate", ""))[:4],
            "art": art, "source": "iTunes"}


def musicbrainz_cover(artist: str, title: str, album: str = "") -> dict | None:
    """Release of the song on MusicBrainz with a front cover on the Cover Art Archive."""
    q = f'recording:"{title}"' + (f' AND artist:"{artist}"' if artist else "")
    if album:
        q += f' AND release:"{album}"'
    data = get_json("https://musicbrainz.org/ws/2/recording?" + urllib.parse.urlencode(
        {"query": q, "fmt": "json", "limit": 10})) or {}
    for rec in data.get("recordings") or []:
        credit = " ".join(c.get("name", "") for c in rec.get("artist-credit") or [])
        if score(artist, title, credit, rec.get("title", "")) < MIN_SCORE:
            continue
        for rel in rec.get("releases") or []:
            rid = rel.get("id")
            if rid:
                return {"artist": credit, "title": rec.get("title", ""), "album": rel.get("title", ""),
                        "year": str(rel.get("date", ""))[:4],
                        "art": f"https://coverartarchive.org/release/{rid}/front-500",
                        "source": "Cover Art Archive"}
    return None


# ── lyrics ──────────────────────────────────────────────────────────────

def lrclib(artist: str, title: str, album: str = "", duration: float = 0.0) -> dict | None:
    """{"plain", "synced", "artist", "title"} from LRCLIB, or None."""
    params = {"artist_name": artist, "track_name": title}
    if album:
        params["album_name"] = album
    if duration:
        params["duration"] = int(round(duration))
    hit = None
    if artist:
        try:
            hit = get_json("https://lrclib.net/api/get?" + urllib.parse.urlencode(params))
        except urllib.error.HTTPError:
            hit = None
        if hit and score(artist, title, hit.get("artistName", ""), hit.get("trackName", "")) < MIN_SCORE:
            hit = None
    if not hit:
        found = get_json("https://lrclib.net/api/search?" + urllib.parse.urlencode(
            {"track_name": title, **({"artist_name": artist} if artist else {})})) or []
        ranked = []
        for r in found:
            if r.get("instrumental") or not (r.get("plainLyrics") or r.get("syncedLyrics")):
                continue
            s = score(artist, title, r.get("artistName", ""), r.get("trackName", ""))
            if s < MIN_SCORE:
                continue
            closeness = -abs(float(r.get("duration") or 0) - duration) if duration else 0.0
            ranked.append((s, bool(r.get("syncedLyrics")), closeness, r))
        if ranked:
            hit = max(ranked, key=lambda x: x[:3])[3]
    if not hit or hit.get("instrumental"):
        return None
    plain = (hit.get("plainLyrics") or "").strip()
    synced = (hit.get("syncedLyrics") or "").strip()
    if not plain and synced:
        plain = lrc_to_text(synced)
    if not plain:
        return None
    return {"plain": plain, "synced": synced, "artist": hit.get("artistName", ""),
            "title": hit.get("trackName", ""), "source": "LRCLIB"}


_LRC_TAG = re.compile(r"\[\d+:\d+(?:[.:]\d+)?\]")


def lrc_to_text(lrc: str) -> str:
    """Timed LRC → plain lines (empty timed lines dropped)."""
    out = []
    for line in lrc.replace("\r", "").split("\n"):
        if re.match(r"^\[[a-z]+:", line, re.I):          # [ar:…] [ti:…] headers
            continue
        text = _LRC_TAG.sub("", line).strip()
        if text:
            out.append(text)
    return "\n".join(out)


def clean_lyrics(text: str) -> str:
    """One sung line per line: section headers ([Refrão], Chorus:) and blank
    runs out, the rest as given."""
    out = []
    for line in str(text).replace("\r", "").split("\n"):
        s = line.strip()
        if re.fullmatch(r"[\[(].*[\])]", s) or re.fullmatch(r"(?i)(chorus|refr[aã]o|verse|verso|bridge|ponte)"
                                                             r"\s*\d*\s*:?", s):
            continue
        out.append(s)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return text + "\n" if text else ""


# ── all together ────────────────────────────────────────────────────────

def fetch(folder: str | Path, artist: str, title: str, album: str = "", duration: float = 0.0,
          cover: bool = True, lyrics: bool = True, log=print) -> dict:
    """Fetch what's missing in ``folder``. Returns {"cover": file|"", "lyrics":
    file|"", "lrc": file|"", "album", "year", "notes": [str]}; never raises."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    res = {"cover": "", "lyrics": "", "lrc": "", "album": "", "year": "", "notes": []}
    if not title:
        res["notes"].append("cover/lyrics not looked up: no song title (name the file 'Artist - Title.mp3' "
                            "or give --title)")
        return res
    who = f"{artist} – {title}" if artist else title

    have_cover = any((folder / n).is_file() for n in ("cover.jpg", "cover.jpeg", "cover.png", "cover.dds"))
    if cover or not album:
        match = None
        for finder in (itunes_search, musicbrainz_cover):
            try:
                match = finder(artist, title, album)
            except Exception as e:  # noqa: BLE001 — network / service trouble
                res["notes"].append(f"{finder.__name__.split('_')[0]}: {_why(e)}")
                match = None
            if match:
                break
        if match:
            res["album"], res["year"] = match.get("album", ""), match.get("year", "")
            if cover and have_cover:
                res["notes"].append("cover: the folder already has one (kept)")
            elif cover and match.get("art"):
                try:
                    data = get_bytes(match["art"])
                    if len(data) > 1000 and data[:3] in (b"\xff\xd8\xff", b"\x89PN"):
                        name = COVER_NAME if data[:3] == b"\xff\xd8\xff" else "cover.png"
                        (folder / name).write_bytes(data)
                        res["cover"] = name
                        log(f"Album art: {match['album'] or who} ({match['source']}) → {name}")
                    else:
                        res["notes"].append("cover: the image wasn't a JPEG/PNG")
                except Exception as e:  # noqa: BLE001
                    res["notes"].append(f"cover: {_why(e)}")
        elif cover and not have_cover:
            res["notes"].append(f"cover: nothing found for {who}")

    if lyrics:
        if (folder / LYRICS_NAME).is_file():
            res["lyrics"] = LYRICS_NAME
            res["notes"].append(f"lyrics: {LYRICS_NAME} already in the folder (kept)")
        else:
            try:
                hit = lrclib(artist, title, album or res["album"], duration)
            except Exception as e:  # noqa: BLE001
                hit = None
                res["notes"].append(f"lyrics: LRCLIB {_why(e)}")
            if hit:
                text = clean_lyrics(hit["plain"])
                if text:
                    (folder / LYRICS_NAME).write_text(text, encoding="utf-8")
                    res["lyrics"] = LYRICS_NAME
                    if hit["synced"] and not (folder / LRC_NAME).is_file():
                        (folder / LRC_NAME).write_text(hit["synced"] + "\n", encoding="utf-8")
                        res["lrc"] = LRC_NAME
                    log(f"Lyrics: {hit['artist']} – {hit['title']} (LRCLIB) → {LYRICS_NAME}"
                        + (f" + {LRC_NAME}" if res["lrc"] else ""))
            elif not any(n.startswith("lyrics:") for n in res["notes"]):
                res["notes"].append(f"lyrics: nothing found for {who} on LRCLIB")
    return res


def _why(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code}"
    if isinstance(e, urllib.error.URLError):
        return f"no connection ({e.reason})"
    return f"{type(e).__name__}: {e}"
