"""Tests for s2m_config.py (stems2midi.ini defaults)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import s2m_config as cfg  # noqa: E402
import stems2midi as s2m  # noqa: E402

INI = """\ufeff; comment
[defaults]
lead_silence = 5   ; EOF
tuning = 5
model = htdemucs_ft
unknown = 1

[feedback]
plugins_dir = C:\\\\Program Files\\\\feedback\\\\current\\\\resources\\\\slopsmith\\\\plugins
"""


def test_reads_user_file_and_maps_keys(tmp_path, monkeypatch):
    (tmp_path / "stems2midi").mkdir()
    (tmp_path / "stems2midi" / "stems2midi.ini").write_text(INI, encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.delenv("STEMS2MIDI_CONFIG", raising=False)
    d = cfg.cli_defaults()
    assert d == {"shift": "5", "tuning": "5", "model": "htdemucs_ft"}
    assert cfg.lead_silence() == 5.0
    assert cfg.load().get("feedback", "plugins_dir").endswith("plugins")


def test_env_file_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "none"))
    f = tmp_path / "x.ini"
    f.write_text("[defaults]\nlead_silence = 2.5\n", encoding="utf-8")
    monkeypatch.setenv("STEMS2MIDI_CONFIG", str(f))
    assert cfg.lead_silence() == 2.5


def test_cli_uses_ini_defaults(tmp_path, monkeypatch):
    f = tmp_path / "x.ini"
    f.write_text("[defaults]\nlead_silence = 5\ntuning = drop-d\n", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(tmp_path / "none"))
    monkeypatch.setenv("STEMS2MIDI_CONFIG", str(f))
    p = s2m.build_parser()
    p.set_defaults(**cfg.cli_defaults())
    a = p.parse_args(["song.mp3"])
    assert a.shift == 5.0 and a.tuning == "drop-d"
    assert p.parse_args(["song.mp3", "--shift", "0"]).shift == 0.0     # the command line wins


def test_no_ini_no_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "none"))
    monkeypatch.delenv("STEMS2MIDI_CONFIG", raising=False)
    monkeypatch.setattr(cfg, "HERE", tmp_path / "plugin")
    assert cfg.cli_defaults() == {} and cfg.lead_silence() == 0.0


def test_windows_command_line_with_spaces_and_trailing_backslash(monkeypatch):
    import s2m_config as c
    raw = r'".\Manic Street Preachers - Tristesse Durera_stems\" --lyrics "C:\My Docs\letra 1.txt" --compare "PART VOCALS_RS2.xml"'
    assert c.split_cmdline(raw) == [".\\Manic Street Preachers - Tristesse Durera_stems\\", "--lyrics",
                                    "C:\\My Docs\\letra 1.txt", "--compare", "PART VOCALS_RS2.xml"]
    assert c.split_cmdline('"Xutos & Pontapés.mp3" --parts "verse=0:21-0:42, chorus=0:42-1:05"') == [
        "Xutos & Pontapés.mp3", "--parts", "verse=0:21-0:42, chorus=0:42-1:05"]
    monkeypatch.setenv(c.RAW_ARGS_ENV, raw)
    assert c.cli_argv()[0].endswith("_stems\\")
    import os
    assert c.RAW_ARGS_ENV not in os.environ          # used once
    # what the C runtime makes of it without the .bat: best effort
    monkeypatch.setattr("sys.argv", ["x", '.\\A B" --lyrics letra.txt', "--no-pitch"])
    assert c.cli_argv() == [".\\A B", "--lyrics", "letra.txt", "--no-pitch"]
    assert c.cli_argv(["a b"]) == ["a b"]
