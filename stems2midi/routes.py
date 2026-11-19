"""stems2midi plugin — routes.py

Upload a song, split it into stems with Demucs and transcribe the bass stem to
MIDI. The heavy lifting runs in ``stems2midi.py`` as a **subprocess** (one job
at a time) so PyTorch never loads into the FeedBack server process, a crash or
OOM can't take the server down, and the ML stack can live in a separate Python
(e.g. a CUDA venv) configured in Settings.

Endpoints (all under /api/plugins/stems2midi):
    GET    /status                 dependency report, settings, install state
    POST   /settings               {"python": "<path or empty>"}
    POST   /install                pip-install requirements-ml.txt (background)
    GET    /jobs                   list jobs (newest first)
    POST   /jobs                   multipart: file + options → queued job
    GET    /jobs/{id}              job state
    POST   /jobs/{id}/cancel
    DELETE /jobs/{id}
    GET    /jobs/{id}/files/{name} download one output (stem / midi / notes)
    GET    /jobs/{id}/zip          every output as a zip
    POST   /jobs/{id}/lyrics       lyrics text → UltraStar .txt + Rocksmith vocals XML
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

PLUGIN_ID = "stems2midi"
PLUGIN_DIR = Path(__file__).resolve().parent
CLI = PLUGIN_DIR / "stems2midi.py"
REQ_ML = PLUGIN_DIR / "requirements-ml.txt"

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".wma",
              ".aiff", ".aif", ".opus", ".webm", ".mp4"}
MODELS = {"htdemucs_6s", "htdemucs", "htdemucs_ft"}
STEMS = {"drums", "bass", "other", "vocals", "guitar", "piano"}
ENGINES = {"auto", "pyin", "basic-pitch"}
DEVICES = {"auto", "cpu", "cuda", "mps"}
MAX_UPLOAD = 400 * 1024 * 1024
MAX_JOBS_KEPT = 30
MAX_LYRICS = 30000
LANGUAGES = {"pt", "en", "es", "fr", "de", "it"}
LYRICS_FILE = "lyrics.input.txt"
NOTES_EXTS = {".musicxml", ".xml", ".mxl", ".mid", ".midi", ".gp3", ".gp4", ".gp5", ".gpx", ".gp"}
MAX_NOTES_FILE = 20 * 1024 * 1024
_JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_FILE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_TUNING_RE = re.compile(r"^[A-Za-z0-9#♯♭ ,;/+\-]{1,48}$")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash on Windows


# ── pure helpers (unit-tested) ──────────────────────────────────────────────

def parse_options(form: dict) -> dict:
    """Validate the upload form into CLI-ready options. Unknown values fall
    back to defaults instead of erroring — the UI only sends known ones."""
    def _list(key, allowed, default):
        raw = str(form.get(key) or "").lower()
        vals = [v.strip() for v in raw.split(",") if v.strip() in allowed]
        return vals or list(default)

    model = str(form.get("model") or "htdemucs_6s")
    model = model if model in MODELS else "htdemucs_6s"
    engine = str(form.get("engine") or "auto")
    device = str(form.get("device") or "auto")
    try:
        quantize = int(form.get("quantize") or 0)
    except (TypeError, ValueError):
        quantize = 0
    try:
        bpm = float(form.get("bpm")) if form.get("bpm") not in (None, "") else None
    except (TypeError, ValueError):
        bpm = None
    if bpm is not None and not (20 <= bpm <= 400):
        bpm = None
    midi_raw = str(form.get("midi") or "bass").lower()
    return {
        "model": model,
        "stems": _list("stems", STEMS, ["bass", "guitar"]),
        "midi": [] if midi_raw == "none" else _list("midi", STEMS, ["bass"]),
        "backing": str(form.get("backing", "true")).lower() not in ("0", "false", "off", "no"),
        "engine": engine if engine in ENGINES else "auto",
        "device": device if device in DEVICES else "auto",
        "quantize": quantize if quantize in (0, 4, 8, 16, 32) else 0,
        "bpm": bpm,
        "format": "flac" if form.get("format") == "flac" else "wav",
        "skip_separation": str(form.get("skip_separation", "")).lower() in ("1", "true", "on", "yes"),
        "tuning": _tuning(form.get("tuning")),
        "slap": _slap(form.get("slap")),
        "parts": _parts(form.get("parts")),
        "meta": {k: clean_text(form.get(k), 4 if k == "year" else 120)
                 for k in ("artist", "title", "album", "year") if clean_text(form.get(k))},
        "lyrics": clean_lyrics(form.get("lyrics")),
        "notes_use": _notes_use(form.get("notes_use")),
        "notes_track": clean_text(form.get("notes_track"), 60),
        "only": _only(form.get("only")),
        "sustained": str(form.get("sustained", "")).lower() in ("1", "true", "on", "yes"),
        "language": _language(form.get("language")),
    }


def _only(v) -> str:
    """Where the bass guitar plays: times / bar numbers, or '' (everywhere)."""
    v = str(v or "").strip()
    if not v:
        return ""
    if re.fullmatch(r"[\d:.,\s-]{1,120}", v):
        return v
    raise ValueError("bass only in: times '2:50-5:15' or bars '40-90'")


def _notes_use(v) -> str:
    """all | doubtful | bar numbers / times ("12-30, 45", "1:02-1:30")."""
    v = str(v or "").strip()
    if not v or v.lower() == "all":
        return "all"
    if v.lower() == "doubtful":
        return "doubtful"
    if re.fullmatch(r"[\d:.,\s-]{1,120}", v):
        return v
    raise ValueError("notes: use all, doubtful, '12-30, 45' or '1:02-1:30'")


def clean_lyrics(v) -> str:
    """Pasted lyrics: lines kept, control characters and markup out."""
    s = str(v or "").replace("\r\n", "\n").replace("\r", "\n")[:MAX_LYRICS]
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f<>]", "", s)
    return s if s.strip() else ""


def _language(v) -> str:
    v = str(v or "").strip().lower()[:2]
    return v if v in LANGUAGES else ""


def clean_text(v, n: int = 120) -> str:
    """Free text for the XML metadata: no control characters or markup."""
    return re.sub(r"[\x00-\x1f\x7f<>&]", "", str(v or ""))[:n].strip()


def meta_from_filename(name: str) -> dict:
    """"Delfins - Canção de Engate.mp3" → artist/title (keeps accents)."""
    stem = clean_text(Path(str(name or "")).stem, 200)
    if " - " in stem:
        a, t = stem.split(" - ", 1)
        return {"artist": a.strip()[:120], "title": t.strip()[:120]}
    return {"title": stem[:120]} if stem else {}


def _slap(raw) -> str | None:
    """Slap sections ("all", "auto" or "0:32-1:05, 130-162"); ValueError if malformed."""
    t = str(raw or "").strip()
    if not t:
        return None
    if t.lower() == "auto":
        return "auto"
    if not re.fullmatch(r"[A-Za-z0-9:.,; \-]{1,200}", t):
        raise ValueError("slap sections: only times, '-', ',', 'all' or 'auto'")
    names = [x.strip().lower() for x in re.split(r"[,;]", t) if x.strip()]
    if names and all(x.isalpha() for x in names) and names not in (["all"], ["auto"]):
        bad = [x for x in names if x not in _chart_tools().RS_SECTIONS]
        if bad:
            raise ValueError(f"slap sections: {', '.join(bad)} is not a song part name")
        return ",".join(names)
    try:
        _fretting().parse_sections(t)
    except ValueError as e:
        raise ValueError(f"slap sections: {e}") from None
    return t


def _parts(raw) -> str | None:
    """Song parts "verse=0:21-0:42, chorus=0:42-1:05"; ValueError if malformed."""
    t = str(raw or "").strip()
    if not t:
        return None
    if t.lower() == "auto":
        return "auto"
    if not re.fullmatch(r"[A-Za-z0-9:.,;= \-]{1,600}", t):
        raise ValueError("song parts: only names, '=', times, '-' and ','")
    try:
        _chart_tools().parse_parts(t)
    except ValueError as e:
        raise ValueError(f"song parts: {e}") from None
    return t


def _tuning(raw) -> str:
    """Bass tuning spec for the tab; raises ValueError if it can't be parsed."""
    t = str(raw or "auto").strip()
    if t.lower() in ("none", "auto"):
        return t.lower()
    if not _TUNING_RE.match(t):
        raise ValueError("tuning: only note names, numbers, spaces and commas")
    try:
        _fretting().parse_tuning(t)  # same parser the worker uses
    except ValueError as e:
        raise ValueError(f"tuning: {e}") from None
    return t


def _fretting():
    """fretting.py is pure Python (no ML deps), so the server can validate
    tunings with it. Loaded by path under a private name (no sys.modules clash)."""
    import importlib.util
    mod = sys.modules.get("plugin_stems2midi_fretting")
    if mod is None:
        spec = importlib.util.spec_from_file_location("plugin_stems2midi_fretting", PLUGIN_DIR / "fretting.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plugin_stems2midi_fretting"] = mod
        spec.loader.exec_module(mod)
    return mod


def build_command(python: str, input_path: Path, out_dir: Path, opts: dict) -> list[str]:
    cmd = [python, "-u", str(CLI), str(input_path), "-o", str(out_dir), "--json-progress",
           "--model", opts["model"], "--device", opts["device"],
           "--stems", ",".join(opts["stems"]),
           "--midi", ",".join(opts["midi"]) or "none",
           "--engine", opts["engine"], "--quantize", str(opts["quantize"]),
           "--format", opts["format"],
           f"--tuning={opts.get('tuning') or '4'}"]  # '=' form: offsets like -2,0,0,0 start with '-'
    if opts.get("slap"):
        cmd.append(f"--slap={opts['slap']}")
    if opts.get("parts"):
        cmd.append(f"--parts={opts['parts']}")
    for k, v in (opts.get("meta") or {}).items():
        if k in ("artist", "title", "album", "year") and v:
            cmd.append(f"--{k}={v}")
    if not opts["backing"]:
        cmd.append("--no-backing")
    if opts.get("bpm"):
        cmd += ["--bpm", str(opts["bpm"])]
    if opts.get("skip_separation"):
        cmd.append("--skip-separation")
    if opts.get("only"):
        cmd.append(f"--only={opts['only']}")
    if opts.get("sustained"):
        cmd.append("--sustained")
    if opts.get("notes_from"):
        cmd.append(f"--notes-from={opts['notes_from']}")
        if opts.get("notes_use") and opts["notes_use"] != "all":
            cmd.append(f"--use={opts['notes_use']}")
        if opts.get("notes_track"):
            cmd.append(f"--track={opts['notes_track']}")
    if opts.get("lyrics_file"):
        cmd.append(f"--lyrics={opts['lyrics_file']}")
        if opts.get("language"):
            cmd.append(f"--language={opts['language']}")
    return cmd


def safe_filename(name: str) -> str:
    base = Path(name or "song").name
    stem = re.sub(r"[^A-Za-z0-9_. -]+", "_", Path(base).stem).strip(" .") or "song"
    return stem[:60] + Path(base).suffix.lower()


def valid_python_path(p: str) -> bool:
    """Only accept an existing file that looks like a Python interpreter —
    this path is executed, so don't let it point at arbitrary binaries."""
    if not p:
        return True  # empty = use the server's own interpreter
    path = Path(p)
    return path.is_file() and path.name.lower().startswith("python")


# ── job manager ─────────────────────────────────────────────────────────────

class JobManager:
    def __init__(self, root: Path, log, settings_path: Path, pip_target: Path | None):
        self.root = root
        self.log = log
        self.settings_path = settings_path
        self.pip_target = pip_target
        self.jobs: dict[str, dict] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.q: queue.Queue[str] = queue.Queue()
        self.check_cache: tuple[float, dict] | None = None
        self.install = {"running": False, "log": "", "ok": None}
        root.mkdir(parents=True, exist_ok=True)
        self._load()
        threading.Thread(target=self._worker, name="stems2midi-worker", daemon=True).start()

    # settings ------------------------------------------------------------
    def settings(self) -> dict:
        try:
            return json.loads(self.settings_path.read_text("utf-8"))
        except Exception:
            return {}

    def save_settings(self, data: dict):
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(data, indent=2), "utf-8")
        self.check_cache = None

    def python(self) -> str:
        p = os.environ.get("STEMS2MIDI_PYTHON") or self.settings().get("python") or ""
        return p if p and valid_python_path(p) else sys.executable

    def env(self) -> dict:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        # Deps installed by the Install button land in our private pylib dir
        # when using the server's own interpreter.
        if self.python() == sys.executable and self.pip_target:
            env["PYTHONPATH"] = os.pathsep.join(
                [str(self.pip_target)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        return env

    # persistence ---------------------------------------------------------
    def _job_dir(self, jid: str) -> Path:
        return self.root / jid

    def _save(self, job: dict):
        if job["id"] not in self.jobs:  # deleted meanwhile: don't resurrect its folder
            return
        d = self._job_dir(job["id"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(job, indent=2), "utf-8")

    def _load(self):
        for jf in sorted(self.root.glob("*/job.json")):
            try:
                job = json.loads(jf.read_text("utf-8"))
            except Exception:
                continue
            if job.get("status") in ("queued", "running"):
                job["status"] = "error"
                job["error"] = "Interrupted (FeedBack was restarted)"
                self._save(job)
            self.jobs[job["id"]] = job

    def public(self, job: dict) -> dict:
        return {k: v for k, v in job.items() if k not in ("input_path",)}

    def list(self) -> list[dict]:
        with self.lock:
            return [self.public(j) for j in sorted(self.jobs.values(), key=lambda j: -j["created"])]

    def get(self, jid: str) -> dict:
        if not _JOB_ID_RE.match(jid or "") or jid not in self.jobs:
            raise HTTPException(404, "Job not found")
        return self.jobs[jid]

    # lifecycle -----------------------------------------------------------
    def set_lyrics(self, jid: str, info: dict):
        """Record lyrics files made after the job finished (and drop the old zip)."""
        job = self.get(jid)
        with self.lock:
            if not isinstance(job.get("result"), dict):
                job["result"] = {}
            job["result"]["lyrics"] = info
            self._save(job)
        (self.root / jid / "all.zip").unlink(missing_ok=True)

    def create(self, filename: str, input_path: Path, opts: dict, jid: str) -> dict:
        job = {
            "id": jid, "filename": filename, "status": "queued", "pct": 0.0,
            "stage": "queued", "msg": "Waiting in queue…", "options": opts,
            "created": time.time(), "started": None, "finished": None,
            "result": None, "error": None, "input_path": str(input_path),
        }
        with self.lock:
            self.jobs[jid] = job
            self._save(job)
        self.q.put(jid)
        self._prune()
        return job

    def cancel(self, jid: str):
        job = self.get(jid)
        with self.lock:
            if job["status"] == "queued":
                job.update(status="cancelled", msg="Cancelled", finished=time.time())
                self._save(job)
                return
            if job["status"] != "running":
                return
            proc = self.procs.get(jid)
        job["cancel_requested"] = True
        if proc and proc.poll() is None:
            proc.terminate()

    def delete(self, jid: str):
        job = self.get(jid)
        if job["status"] in ("queued", "running"):
            self.cancel(jid)
            for _ in range(50):
                if self.jobs[jid]["status"] not in ("running",):
                    break
                time.sleep(0.1)
        with self.lock:
            self.jobs.pop(jid, None)
        shutil.rmtree(self._job_dir(jid), ignore_errors=True)

    def _prune(self):
        done = [j for j in sorted(self.jobs.values(), key=lambda j: j["created"])
                if j["status"] in ("done", "error", "cancelled")]
        for j in done[: max(0, len(self.jobs) - MAX_JOBS_KEPT)]:
            try:
                self.delete(j["id"])
            except HTTPException:
                pass

    def _worker(self):
        while True:
            jid = self.q.get()
            job = self.jobs.get(jid)
            if not job or job["status"] != "queued":
                continue
            try:
                self._run(job)
            except Exception as e:  # never kill the worker thread
                self.log.exception("stems2midi job %s crashed", jid)
                job.update(status="error", error=str(e), finished=time.time())
                self._save(job)

    def _run(self, job: dict):
        jid = job["id"]
        out_dir = self._job_dir(jid) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(self.python(), Path(job["input_path"]), out_dir, job["options"])
        self.log.info("stems2midi job %s: %s", jid, " ".join(cmd[1:4]))
        job.update(status="running", started=time.time(), msg="Starting…", stage="start")
        self._save(job)
        stderr_path = self._job_dir(jid) / "stderr.log"
        with open(stderr_path, "w", encoding="utf-8", errors="replace") as err:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8",
                errors="replace", env=self.env(), cwd=str(PLUGIN_DIR), creationflags=_NO_WINDOW,
            )
            with self.lock:
                self.procs[jid] = proc
            if job.get("cancel_requested"):
                proc.terminate()
            final = None
            last_save = 0.0
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                kind = ev.get("event")
                if kind == "progress":
                    job.update(pct=float(ev.get("pct", 0)), stage=ev.get("stage"), msg=ev.get("msg"))
                    if time.time() - last_save > 2:
                        self._save(job)
                        last_save = time.time()
                elif kind in ("done", "error"):
                    final = ev
            proc.wait()
            with self.lock:
                self.procs.pop(jid, None)

        job["finished"] = time.time()
        if job.get("cancel_requested"):
            job.update(status="cancelled", msg="Cancelled")
        elif final and final.get("event") == "done" and proc.returncode == 0:
            final.pop("event", None)
            final.pop("out_dir", None)
            self._ensure_rs_xml(out_dir, final, (job.get("options") or {}).get("meta") or {})
            job.update(status="done", pct=1.0, stage="done", msg="Done", result=final)
        else:
            tail = ""
            try:
                tail = stderr_path.read_text("utf-8", errors="replace")[-1500:]
            except OSError:
                pass
            msg = (final or {}).get("error") or f"Process exited with code {proc.returncode}"
            if "No module named" in tail or "ModuleNotFoundError" in msg:
                msg += " — install the engine (Engine panel → Install dependencies)."
            job.update(status="error", error=msg, log_tail=tail)
        job.pop("cancel_requested", None)
        self._save(job)
        # input no longer needed once finished; outputs stay
        try:
            Path(job["input_path"]).unlink(missing_ok=True)
        except OSError:
            pass

    def _ensure_rs_xml(self, out_dir: Path, result: dict, meta: dict):
        """The engine writes bass_RS2.xml itself; if it couldn't (e.g. an older
        engine), write it here with the server's own FeedBack writer."""
        notes = out_dir / "bass.notes.json"
        if (result.get("rsxml") or {}).get("bass") or not notes.is_file():
            return
        try:
            payload = json.loads(notes.read_text(encoding="utf-8"))
            if not payload.get("tuning"):
                return
            path, warns = _rsexport().write_xml(notes, shift=_config().lead_silence(), **meta)
            result.setdefault("rsxml", {})["bass"] = path.name
            result["warnings"] = [w for w in result.get("warnings", [])
                                  if not w.startswith("Rocksmith XML not written")] + warns
        except Exception as e:  # noqa: BLE001
            self.log.warning("stems2midi: Rocksmith XML failed: %s", e)

    # environment ----------------------------------------------------------
    def check(self, refresh: bool = False) -> dict:
        if self.check_cache and not refresh and time.time() - self.check_cache[0] < 120:
            return self.check_cache[1]
        info: dict = {"ok": False}
        try:
            r = subprocess.run([self.python(), str(CLI), "--check", "--json-progress"],
                               capture_output=True, text=True, timeout=120, env=self.env(),
                               creationflags=_NO_WINDOW)
            for line in r.stdout.splitlines():
                if line.startswith("{"):
                    info = json.loads(line)
                    info.pop("event", None)
                    info["ok"] = True
            if not info.get("ok"):
                info["error"] = (r.stderr or r.stdout or "no output")[-800:]
        except Exception as e:
            info["error"] = str(e)
        self.check_cache = (time.time(), info)
        return info

    def start_install(self):
        if self.install["running"]:
            return
        py = self.python()
        cmd = [py, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQ_ML)]
        if py == sys.executable and self.pip_target:
            self.pip_target.mkdir(parents=True, exist_ok=True)
            cmd[4:4] = ["--target", str(self.pip_target), "--upgrade"]
        self.install = {"running": True, "log": "$ " + " ".join(cmd) + "\n", "ok": None}

        def run():
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace",
                                     creationflags=_NO_WINDOW)
                for line in p.stdout:
                    self.install["log"] = (self.install["log"] + line)[-12000:]
                p.wait()
                self.install["ok"] = p.returncode == 0
            except Exception as e:
                self.install["log"] += f"\n{e}\n"
                self.install["ok"] = False
            finally:
                self.install["running"] = False
                self.check_cache = None

        threading.Thread(target=run, name="stems2midi-install", daemon=True).start()


# ── routes ──────────────────────────────────────────────────────────────────

def _config():
    """s2m_config.py (stdlib only), loaded by path like fretting.py."""
    import importlib.util
    mod = sys.modules.get("plugin_stems2midi_config")
    if mod is None:
        spec = importlib.util.spec_from_file_location("plugin_stems2midi_config", PLUGIN_DIR / "s2m_config.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plugin_stems2midi_config"] = mod
        spec.loader.exec_module(mod)
    return mod


def _config_info() -> dict:
    try:
        c = _config()
        return {"files": [str(p) for p in c.config_paths() if p.is_file()],
                "user_file": str(c.user_config_path()), "defaults": c.cli_defaults()}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _chart_tools():
    import importlib.util
    mod = sys.modules.get("plugin_stems2midi_chart_tools")
    if mod is None:
        spec = importlib.util.spec_from_file_location("plugin_stems2midi_chart_tools", PLUGIN_DIR / "chart_tools.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plugin_stems2midi_chart_tools"] = mod
        spec.loader.exec_module(mod)
    return mod


def _rsexport():
    """rsexport.py imports FeedBack's lib/gp2rs.py, which is importable in the
    server process. Loaded by path under a private name, like fretting.py."""
    import importlib.util
    mod = sys.modules.get("plugin_stems2midi_rsexport")
    if mod is None:
        spec = importlib.util.spec_from_file_location("plugin_stems2midi_rsexport", PLUGIN_DIR / "rsexport.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["plugin_stems2midi_rsexport"] = mod
        spec.loader.exec_module(mod)
    return mod


def _song_label(job: dict) -> str:
    """"Artist - Title" for download names (keeps accents), else the file name."""
    meta = (job.get("options") or {}).get("meta") or {}
    label = " - ".join(v for v in (meta.get("artist"), meta.get("title")) if v)
    label = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", label).strip(" .")
    return label or Path(job.get("filename") or "song").stem


def content_disposition(filename: str) -> str:
    """attachment header that survives non-ASCII names (RFC 6266/5987)."""
    from urllib.parse import quote
    ascii_name = filename.encode("ascii", "replace").decode().replace("?", "_").replace('"', "'")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def setup(app, context):
    log = context["log"]
    base = Path(context["config_dir"]) / PLUGIN_ID
    # Private package dir for the ML stack: only the worker subprocess sees it
    # (via PYTHONPATH), so torch/numpy/… can never shadow the server's own deps.
    mgr = JobManager(base / "jobs", log, base / "settings.json", base / "pylib")
    router = APIRouter(prefix=f"/api/plugins/{PLUGIN_ID}")
    app.state.stems2midi = mgr  # handy for tests / other plugins

    @router.get("/status")
    def status(refresh: int = 0):
        return {
            "python": mgr.python(),
            "server_python": sys.executable,
            "settings": mgr.settings(),
            "env_override": bool(os.environ.get("STEMS2MIDI_PYTHON")),
            "check": mgr.check(bool(refresh)),
            "install": {k: mgr.install[k] for k in ("running", "ok")} | {"log": mgr.install["log"][-4000:]},
            "config": _config_info(),
        }

    @router.post("/settings")
    async def save_settings(request: Request):
        body = await request.json()
        py = str(body.get("python") or "").strip().strip('"')
        if not valid_python_path(py):
            return JSONResponse({"error": "Not a Python interpreter path (must exist and be named python*)"}, 400)
        s = mgr.settings()
        s["python"] = py
        mgr.save_settings(s)
        return {"ok": True, "python": mgr.python()}

    @router.post("/install")
    def install():
        mgr.start_install()
        return {"ok": True}

    @router.get("/jobs")
    def jobs():
        return {"jobs": mgr.list()}

    @router.post("/jobs")
    async def create_job(request: Request):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "No file uploaded"}, 400)
        raw_name = getattr(upload, "filename", "") or "song.mp3"
        filename = safe_filename(raw_name)
        ext = Path(filename).suffix.lower()
        if ext not in AUDIO_EXTS:
            return JSONResponse({"error": f"Unsupported file type '{ext}'"}, 400)
        try:
            opts = parse_options({k: v for k, v in form.items() if k not in ("file", "notes_file")})
        except ValueError as e:
            return JSONResponse({"error": f"Bad option: {e}"}, 400)
        opts["meta"] = {**meta_from_filename(raw_name), **opts.get("meta", {})}
        jid = uuid.uuid4().hex[:12]
        jdir = mgr.root / jid
        jdir.mkdir(parents=True, exist_ok=True)
        dest = jdir / ("input" + ext)
        size = 0
        with open(dest, "wb") as f:
            while chunk := await upload.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    f.close()
                    shutil.rmtree(jdir, ignore_errors=True)
                    return JSONResponse({"error": "File too large (max 400 MB)"}, 413)
                f.write(chunk)
        nf = form.get("notes_file")
        if nf is not None and hasattr(nf, "read") and getattr(nf, "filename", ""):
            n_ext = Path(nf.filename).suffix.lower()
            if n_ext not in NOTES_EXTS:
                shutil.rmtree(jdir, ignore_errors=True)
                return JSONResponse({"error": "Notes file: use MusicXML (.musicxml/.xml/.mxl), MIDI (.mid) "
                                              "or Guitar Pro (.gp3/.gp4/.gp5/.gpx/.gp)"}, 400)
            npath = jdir / ("notes_from" + n_ext)
            data = await nf.read(MAX_NOTES_FILE + 1)
            if len(data) > MAX_NOTES_FILE:
                shutil.rmtree(jdir, ignore_errors=True)
                return JSONResponse({"error": "Notes file too large"}, 413)
            npath.write_bytes(data)
            opts["notes_from"] = str(npath)
            opts["notes_from_name"] = clean_text(Path(nf.filename).name)
        if opts.get("lyrics"):
            lpath = jdir / LYRICS_FILE
            lpath.write_text(opts["lyrics"], encoding="utf-8")
            opts["lyrics_file"] = str(lpath)
            opts["lyrics"] = True          # the text lives in the file, not in the job record
        job = mgr.create(filename, dest, opts, jid)
        return mgr.public(job)

    @router.get("/jobs/{jid}")
    def get_job(jid: str):
        return mgr.public(mgr.get(jid))

    @router.post("/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        mgr.cancel(jid)
        return {"ok": True}

    @router.delete("/jobs/{jid}")
    def delete_job(jid: str):
        mgr.delete(jid)
        return {"ok": True}

    def _out_dir(jid: str) -> Path:
        job = mgr.get(jid)
        if job["status"] != "done":
            raise HTTPException(409, "Job not finished")
        return (mgr.root / jid / "out").resolve()

    @router.get("/jobs/{jid}/files/{name}")
    def get_file(jid: str, name: str, download: int = 0):
        out = _out_dir(jid)
        if not _FILE_RE.match(name) or name.startswith("."):
            raise HTTPException(400, "Bad file name")
        path = (out / name).resolve()
        if path.parent != out or not path.is_file():
            raise HTTPException(404, "File not found")
        job = mgr.get(jid)
        if not download:
            return FileResponse(path, content_disposition_type="inline")
        return FileResponse(path, headers={"Content-Disposition": content_disposition(
            f"{_song_label(job)} - {name}")})

    @router.post("/jobs/{jid}/rocksmith.xml")
    async def post_rs_xml(jid: str, request: Request):
        """Re-export with other metadata, a shift, or aligned to an uploaded
        audio file (EOF's guitar.ogg) — the alignment runs in the engine."""
        out = _out_dir(jid)
        job = mgr.get(jid)
        form = await request.form()
        meta = {k: clean_text(form.get(k), 4 if k == "year" else 120) for k in ("artist", "title", "album", "year")}
        if meta["year"] and not re.fullmatch(r"\d{4}", meta["year"]):
            meta["year"] = ""
        meta = {k: v or ((job.get("options") or {}).get("meta") or {}).get(k, "") for k, v in meta.items()}
        raw_shift = str(form.get("shift") or "").strip()
        try:
            shift = float(raw_shift) if raw_shift else _config().lead_silence()
        except (TypeError, ValueError):
            shift = 0.0
        if not -60 <= shift <= 60:
            raise HTTPException(400, "shift must be between -60 and 60 s")
        try:
            pb = int(form.get("phrase_bars")) if str(form.get("phrase_bars") or "").strip() else None
        except (TypeError, ValueError):
            pb = None
        if pb is not None and not 0 <= pb <= 32:
            pb = None
        upload = form.get("align")
        has_upload = upload is not None and hasattr(upload, "read") and getattr(upload, "filename", "")
        if not (out / "bass.notes.json").is_file():
            raise HTTPException(404, "No bass transcription in this result")
        if has_upload:
            ext = Path(upload.filename).suffix.lower()
            if ext not in AUDIO_EXTS:
                raise HTTPException(400, f"Unsupported audio type '{ext}'")
            tmp = out.parent / ("align" + ext)
            cmd = [mgr.python(), "-u", str(CLI), str(out), f"--align-to={tmp}"]
            if pb is not None:
                cmd.append(f"--phrase-bars={pb}")
            cmd += [f"--{k}={v}" for k, v in meta.items() if v]
            try:
                size = 0
                with open(tmp, "wb") as f:
                    while chunk := await upload.read(1 << 20):
                        size += len(chunk)
                        if size > MAX_UPLOAD:
                            raise HTTPException(413, "File too large")
                        f.write(chunk)
                proc = await asyncio.to_thread(
                    subprocess.run, cmd, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", env=mgr.env(), cwd=str(PLUGIN_DIR), timeout=300,
                    creationflags=_NO_WINDOW)
            finally:
                tmp.unlink(missing_ok=True)
            if proc.returncode != 0:
                raise HTTPException(422, (proc.stderr or proc.stdout or "alignment failed").strip()[-800:])
            xml_bytes = (out / "bass_RS2.xml").read_bytes()
            m = re.search(r"Alignment to .*?: ([+-]?[\d.]+) s", proc.stdout or "")
            shift_hdr = m.group(1) if m else ""
        else:
            try:
                xml_text, _w = _rsexport().render(
                    json.loads((out / "bass.notes.json").read_text(encoding="utf-8")),
                    shift=shift, phrase_bars=pb, **meta)
            except (ValueError, RuntimeError) as e:
                raise HTTPException(422, str(e))
            xml_bytes, shift_hdr = xml_text.encode("utf-8"), f"{shift:+.3f}"
        return Response(xml_bytes, media_type="application/xml", headers={
            "Content-Disposition": content_disposition(f"{_song_label(job)} - bass_RS2.xml"),
            "X-Stems2midi-Shift": shift_hdr})

    @router.get("/jobs/{jid}/rocksmith.xml")
    def get_rs_xml(jid: str, title: str = "", artist: str = "", album: str = "", year: str = ""):
        out = _out_dir(jid)
        job = mgr.get(jid)
        src = out / "bass.notes.json"
        if not src.is_file():
            raise HTTPException(404, "No bass transcription in this result")
        meta = (job.get("options") or {}).get("meta") or {}
        year = clean_text(year, 4)
        year = year if re.fullmatch(r"\d{4}", year) else ""
        try:
            xml_text, _warns = _rsexport().render(
                json.loads(src.read_text(encoding="utf-8")),
                title=clean_text(title) or meta.get("title") or Path(job["filename"]).stem,
                artist=clean_text(artist) or meta.get("artist", ""),
                album=clean_text(album) or meta.get("album", ""), year=year or meta.get("year", ""),
                shift=_config().lead_silence())
        except (ValueError, RuntimeError) as e:
            raise HTTPException(422, str(e))
        name = f"{_song_label(job)} - bass_RS2.xml"
        return Response(xml_text.encode("utf-8"), media_type="application/xml",
                        headers={"Content-Disposition": content_disposition(name)})

    @router.post("/jobs/{jid}/lyrics")
    async def post_lyrics(jid: str, request: Request):
        """Line up pasted lyrics with the job's vocals stem (runs in the
        engine: the MMS aligner needs PyTorch) → UltraStar + vocals XML."""
        out = _out_dir(jid)
        job = mgr.get(jid)
        form = await request.form()
        text = clean_lyrics(form.get("lyrics"))
        if not text:
            raise HTTPException(400, "Paste the lyrics first")
        if not any((out / f"vocals.{e}").is_file() for e in ("wav", "flac")):
            raise HTTPException(409, "No vocals stem in this result — run the song again with the lyrics "
                                     "(or with the Vocals stem)")
        raw_shift = str(form.get("shift") or "").strip()
        try:
            shift = float(raw_shift) if raw_shift else _config().lead_silence()
        except (TypeError, ValueError):
            shift = 0.0
        if not -60 <= shift <= 60:
            raise HTTPException(400, "shift must be between -60 and 60 s")
        lpath = out.parent / LYRICS_FILE
        lpath.write_text(text, encoding="utf-8")
        meta = (job.get("options") or {}).get("meta") or {}
        cmd = [mgr.python(), "-u", str(CLI), str(out), f"--lyrics={lpath}", f"--shift={shift}"]
        lang = _language(form.get("language"))
        if lang:
            cmd.append(f"--language={lang}")
        cmd += [f"--{k}={v}" for k, v in meta.items() if k in ("artist", "title") and v]
        proc = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=mgr.env(), cwd=str(PLUGIN_DIR), timeout=1800,
            creationflags=_NO_WINDOW)
        if proc.returncode != 0 or not (out / "vocals.lyrics.json").is_file():
            raise HTTPException(422, (proc.stderr or proc.stdout or "lyrics failed").strip()[-800:])
        info = {"ultrastar": "vocals.ultrastar.txt", "xml": "vocals_RS2.xml", "shift": shift}
        try:
            d = json.loads((out / "vocals.lyrics.json").read_text(encoding="utf-8"))
            info.update(aligner=d.get("aligner"), syllables=len(d.get("syllables") or []),
                        lines=len({x.get("line") for x in d.get("syllables") or []}))
        except (OSError, ValueError):
            pass
        mgr.set_lyrics(jid, info)
        return {"ok": True, "lyrics": info, "log": (proc.stdout or "").strip()[-400:]}

    @router.get("/jobs/{jid}/zip")
    def get_zip(jid: str):
        out = _out_dir(jid)
        job = mgr.get(jid)
        zpath = out.parent / "all.zip"
        if not zpath.exists():
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for p in sorted(out.iterdir()):
                    if p.is_file() and not p.name.startswith("."):
                        z.write(p, p.name)
        return FileResponse(zpath, filename=f"{Path(job['filename']).stem} - stems2midi.zip")

    app.include_router(router)
    log.info("stems2midi ready (python: %s)", mgr.python())
