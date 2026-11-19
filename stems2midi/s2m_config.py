"""User defaults for stems2midi — ``stems2midi.ini``.

Looked for (later files override earlier ones):

1. next to the installer folder / the plugin's parent  (``…/stems2midi.ini``)
2. in the plugin folder                                (``stems2midi/stems2midi.ini``)
3. per user: ``%APPDATA%\\stems2midi\\stems2midi.ini`` (Windows) or
   ``~/.config/stems2midi/stems2midi.ini``
4. the file named by ``STEMS2MIDI_CONFIG``

Example::

    [defaults]
    ; seconds of silence EOF adds at the start of guitar.ogg — the Rocksmith
    ; XML is moved by this much (same as --shift)
    lead_silence = 5
    tuning = auto
    model = htdemucs_6s
    language = pt            ; lyrics language (default: guessed)
    phrase_bars = 4          ; Rocksmith phrases every N bars (0 = one per section)

    [feedback]
    plugins_dir = C:\\Program Files\\feedback\\current\\resources\\slopsmith\\plugins

Only standard-library code: the FeedBack server, the engine and the
installer's Python can all read it.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILE_NAME = "stems2midi.ini"

# [defaults] key → command-line option (dest) of stems2midi.py
CLI_KEYS = {
    "lead_silence": "shift",
    "shift": "shift",
    "tuning": "tuning",
    "model": "model",
    "device": "device",
    "quantize": "quantize",
    "stems": "stems",
    "midi": "midi",
    "bpm": "bpm",
    "max_fret": "max_fret",
    "shifts": "shifts",
    "format": "audio_format",
    "slap": "slap",
    "parts": "parts",
    "language": "language",
    "phrase_bars": "phrase_bars",
    "lyrics_aligner": "lyrics_aligner",
    "keep_accents": "keep_accents",
    "rs_5string": "rs_5string",
    "sustained": "sustained",
}


def user_config_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "stems2midi" / FILE_NAME
    return Path.home() / ".config" / "stems2midi" / FILE_NAME


def config_paths() -> list[Path]:
    paths = [HERE.parent / FILE_NAME, HERE / FILE_NAME, user_config_path()]
    env = os.environ.get("STEMS2MIDI_CONFIG")
    if env:
        paths.append(Path(env))
    return paths


def load() -> configparser.ConfigParser:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None)
    for p in config_paths():
        try:
            if p.is_file():
                cp.read(p, encoding="utf-8-sig")
        except (OSError, configparser.Error):
            pass
    return cp


def cli_defaults(cp: configparser.ConfigParser | None = None) -> dict:
    """{argparse dest: value-as-string} from [defaults]; unknown keys ignored."""
    cp = cp or load()
    if not cp.has_section("defaults"):
        return {}
    out = {}
    for key, val in cp.items("defaults"):
        dest = CLI_KEYS.get(key.strip().lower())
        val = (val or "").strip().strip('"')
        if dest and val:
            out[dest] = val
    return out


def lead_silence(cp: configparser.ConfigParser | None = None) -> float:
    try:
        return float(cli_defaults(cp).get("shift", 0) or 0)
    except ValueError:
        return 0.0


# ── Windows command lines ───────────────────────────────────────────────────

RAW_ARGS_ENV = "S2M_RAW_ARGS"


def split_cmdline(raw: str) -> list[str]:
    """Split a command line the way the user meant it: ``"`` groups (and is
    dropped), whitespace outside quotes separates, a backslash is just a
    backslash. Unlike the C runtime, ``"C:\\My Song_stems\\"`` (a folder
    with a trailing backslash — what PowerShell's Tab completion gives) stays
    one argument instead of swallowing the rest of the line."""
    out, cur, quoted, has = [], [], False, False
    for ch in raw:
        if ch == '"':
            quoted, has = not quoted, True
        elif ch in " \t" and not quoted:
            if cur or has:
                out.append("".join(cur))
            cur, has = [], False
        else:
            cur.append(ch)
    if cur or has:
        out.append("".join(cur))
    return out


def cli_argv(argv=None):
    """The arguments for a command-line tool. When started from one of the
    installer's .bat files the raw command line comes in ``S2M_RAW_ARGS``
    (set by the .bat) and is split with :func:`split_cmdline`, so paths with
    spaces and a trailing backslash work. Otherwise ``argv`` / sys.argv."""
    if argv is not None:
        return argv
    import sys
    raw = os.environ.pop(RAW_ARGS_ENV, None)
    if raw and raw.strip():
        args = split_cmdline(raw.strip())
        if args:
            return args
    args = sys.argv[1:]
    # a quote inside an argument can only come from "…\" eaten by the C runtime
    # (best effort: the path is what came before the quote, the rest were options)
    if any('"' in a for a in args):
        fixed = []
        for a in args:
            if '"' in a:
                head, tail = a.split('"', 1)
                fixed.append(head)
                fixed.extend(tail.replace('"', " ").split())
            else:
                fixed.append(a)
        return fixed
    return args
