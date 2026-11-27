/* Stems → MIDI — screen.js
 * Plain JS, global scope, IIFE (FeedBack plugin conventions).
 * Upload → server queues a Demucs + transcription job (subprocess) → this
 * screen polls progress and renders stems (players + downloads) and a piano
 * roll of the MIDI with a WebAudio preview over the backing track.
 */
(function () {
'use strict';

const API = '/api/plugins/stems2midi';
const SCREEN_ID = 'plugin-stems2midi';
const LS = 'stems2midi.';
const STEM_LABELS = {
    bass: 'Bass', guitar: 'Guitar', drums: 'Drums', vocals: 'Vocals', piano: 'Piano',
    other: 'Other', backing_no_bass: 'Backing (no bass)',
};
const DL_ICON = '<svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path d="M10 3a1 1 0 011 1v7.59l2.3-2.3a1 1 0 111.4 1.42l-4 4a1 1 0 01-1.4 0l-4-4a1 1 0 111.4-1.42L9 11.6V4a1 1 0 011-1zM4 15a1 1 0 011 1v1h10v-1a1 1 0 112 0v2a1 1 0 01-1 1H4a1 1 0 01-1-1v-2a1 1 0 011-1z"/></svg>';

let root = null;
let status = null;          // last /status payload
let file = null;            // selected File
let pollTimer = null;
let active = false;         // our screen is showing
const rendered = new Map(); // job id -> signature of last render (avoid DOM churn)
const rolls = new Map();    // job id -> roll controller

// ── helpers ────────────────────────────────────────────────────────────
const $ = (sel) => root && root.querySelector(sel);
function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function store(key, val) {
    try {
        if (val === undefined) return localStorage.getItem(LS + key);
        localStorage.setItem(LS + key, val);
    } catch (_) { return null; }
    return null;
}
async function api(path, opts) {
    const res = await fetch(API + path, opts);
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok) throw new Error((body && (body.error || body.detail)) || ('HTTP ' + res.status));
    return body;
}
function fmtClock(sec) {
    sec = Math.max(0, Math.round(Number(sec) || 0));
    return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`;
}
function fmtDur(sec) {
    if (sec == null || !isFinite(sec)) return '';
    sec = Math.max(0, Math.round(sec));
    const m = Math.floor(sec / 60), s = sec % 60;
    return m ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`;
}
function fileUrl(jid, name, download) {
    return `${API}/jobs/${jid}/files/${encodeURIComponent(name)}${download ? '?download=1' : ''}`;
}
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const noteName = (m) => NOTE_NAMES[m % 12] + (Math.floor(m / 12) - 1);
// one colour per string, low → high (roll + tab legend)
const STRING_RGB = ['239,68,68', '234,179,8', '14,165,233', '34,197,94', '168,85,247', '249,115,22'];

// ── engine status ──────────────────────────────────────────────────────
async function loadStatus(refresh) {
    const pills = $('#s2m-pills');
    if (refresh && pills) pills.innerHTML = '<span class="s2m-dim">Checking…</span>';
    try {
        status = await api('/status' + (refresh ? '?refresh=1' : ''));
    } catch (e) {
        status = null;
        if (pills) pills.innerHTML = `<span class="s2m-pill s2m-p-off">Status unavailable: ${esc(e.message)}</span>`;
        return;
    }
    renderStatus();
    if (status.install && status.install.running) schedulePoll();
}

function renderStatus() {
    if (!root || !status) return;
    const c = status.check || {};
    const pill = (ok, label, cls) => `<span class="s2m-pill ${cls || (ok ? 's2m-p-on' : 's2m-p-off')}">${esc(label)}</span>`;
    const parts = [];
    if (!c.ok) {
        parts.push(pill(false, 'Python not runnable'));
    } else {
        if (c.stems2midi) parts.push(pill(true, 'stems2midi ' + c.stems2midi, 's2m-p-neutral'));
        parts.push(pill(true, 'Python ' + c.python));
        parts.push(pill(!!c.torch, c.torch ? 'PyTorch ' + c.torch : 'PyTorch missing'));
        parts.push(pill(!!c.demucs, c.demucs ? 'Demucs ' + c.demucs : 'Demucs missing'));
        parts.push(pill(!!c.librosa, c.librosa ? 'librosa ' + c.librosa : 'librosa missing'));
        parts.push(pill(!!c.lyrics_mms, c.lyrics_mms ? 'Lyrics: MMS aligner' : 'Lyrics: simple aligner (no torchaudio)', c.lyrics_mms ? '' : 's2m-p-neutral'));
        if (c.torch) {
            parts.push(c.cuda ? pill(true, 'GPU: ' + (c.gpu || 'CUDA'))
                : c.mps ? pill(true, 'GPU: Apple MPS')
                : pill(false, 'CPU only (slower)', 's2m-p-warn'));
        }
        parts.push(pill(!!c.basic_pitch, c.basic_pitch ? 'Basic Pitch ' + c.basic_pitch : 'Basic Pitch (optional)', c.basic_pitch ? 's2m-p-on' : 's2m-p-neutral'));
    }
    $('#s2m-pills').innerHTML = parts.join('');

    const hint = $('#s2m-status-hint');
    const ready = c.ok && c.can_separate && c.can_transcribe;
    hint.hidden = false;
    hint.classList.toggle('s2m-ok', !!ready);
    if (!c.ok) {
        hint.innerHTML = `Could not run <code>${esc(status.python)}</code>. ${esc(c.error || '')}`;
    } else if (ready) {
        hint.innerHTML = c.cuda
            ? 'Ready. A 4-minute song takes roughly 20–60 s on your GPU.'
            : 'Ready. Running on CPU: a 4-minute song takes roughly 3–8 minutes (htdemucs_ft ≈ 4× longer).';
    } else {
        hint.innerHTML = 'The separation engine (PyTorch + Demucs) is not installed for this Python. ' +
            'Use <strong>Install dependencies</strong> below, or install them into a separate venv and select it with the <em>Python…</em> button.';
    }

    // install panel
    const inst = status.install || {};
    const showInstall = !ready || inst.running || inst.ok === false;
    $('#s2m-install').hidden = !showInstall;
    const btn = $('#s2m-install-btn');
    btn.disabled = !!inst.running || !c.ok;
    btn.textContent = inst.running ? 'Installing…' : 'Install dependencies (~2 GB)';
    $('#s2m-install-state').textContent = inst.running
        ? 'Downloading and installing PyTorch, Demucs and librosa — this can take several minutes.'
        : inst.ok === true ? 'Installed. Re-checking…'
        : inst.ok === false ? 'Install failed — see the log. You can also install manually (README).'
        : 'One-time download into a private package folder used only by this plugin.';
    const logEl = $('#s2m-install-log');
    logEl.hidden = !inst.log;
    if (inst.log) {
        const atBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 8;
        logEl.textContent = inst.log;
        if (atBottom) logEl.scrollTop = logEl.scrollHeight;
    }

    const pyInput = $('#s2m-python');
    if (document.activeElement !== pyInput) pyInput.value = (status.settings && status.settings.python) || '';
    pyInput.disabled = !!status.env_override;
    $('#s2m-python-current').textContent = status.python + (status.env_override ? '  (set by STEMS2MIDI_PYTHON)' : '');
    const cfg = status.config || {};
    const d = cfg.defaults || {};
    const cfgEl = $('#s2m-config-info');
    if (cfgEl) {
        cfgEl.textContent = (cfg.files && cfg.files.length)
            ? `Defaults from ${cfg.files.join(', ')}` + (Object.keys(d).length ? ' — ' + Object.entries(d).map(([k, v]) => `${k === 'shift' ? 'lead silence' : k} ${v}`).join(', ') : '')
            : `No stems2midi.ini (optional: ${cfg.user_file || '%APPDATA%\\stems2midi\\stems2midi.ini'})`;
    }
    updateGo();
}

// ── form ───────────────────────────────────────────────────────────────
function chipValues(sel) {
    return Array.from(root.querySelectorAll(sel + ' input:checked')).map((i) => i.value);
}
function updateModelChips() {
    const six = ['htdemucs_6s', 'roformer'].includes($('#s2m-model').value);
    root.querySelectorAll('#s2m-stems input, #s2m-midi input').forEach((inp) => {
        if (inp.value === 'guitar' || inp.value === 'piano') {
            inp.disabled = !six;
            inp.closest('.s2m-chip').title = six ? '' : 'Only the 6-stem model separates guitar and piano';
        }
    });
}
function tuningValue() {
    const sel = $('#s2m-tuning').value;
    return sel === 'custom' ? ($('#s2m-tuning-custom').value.trim() || '4') : sel;
}
function updateTuningBox() {
    $('#s2m-tuning-custom-box').hidden = $('#s2m-tuning').value !== 'custom';
}
function updateGo() {
    if (!root) return;
    const c = (status && status.check) || {};
    const skip = $('#s2m-skip').checked;
    const engineOk = skip ? c.can_transcribe : (c.can_separate && c.can_transcribe);
    const go = $('#s2m-go');
    go.disabled = !file || !engineOk;
    go.textContent = skip ? 'Transcribe' : 'Separate & transcribe';
    const msg = $('#s2m-form-msg');
    if (!file) msg.textContent = '';
    else if (!engineOk) msg.textContent = 'Install the engine first (see Engine above).';
    else msg.textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
}
function setFile(f) {
    file = f || null;
    const drop = $('#s2m-drop');
    drop.classList.toggle('s2m-has', !!file);
    $('#s2m-drop-text').innerHTML = file
        ? `<strong>${esc(file.name)}</strong> — click to choose another`
        : '<strong>Drop an audio file</strong> or click to choose — MP3, WAV, FLAC, OGG, M4A';
    updateGo();
}
function saveForm() {
    const v = {
        model: $('#s2m-model').value, device: $('#s2m-device').value, quantize: $('#s2m-quantize').value,
        tuning: $('#s2m-tuning').value, tuningCustom: $('#s2m-tuning-custom').value,
        language: $('#s2m-language').value,
        engine: $('#s2m-engine').value, format: $('#s2m-format').value,
        backing: $('#s2m-backing').checked, stems: chipValues('#s2m-stems'), midi: chipValues('#s2m-midi'),
    };
    store('form', JSON.stringify(v));
}
function restoreForm() {
    let v = null;
    try { v = JSON.parse(store('form') || 'null'); } catch (_) { v = null; }
    if (!v) return;
    ['model', 'device', 'quantize', 'engine', 'format', 'tuning', 'language'].forEach((k) => { if (v[k]) $('#s2m-' + k).value = v[k]; });
    if (v.tuningCustom) $('#s2m-tuning-custom').value = v.tuningCustom;
    $('#s2m-backing').checked = v.backing !== false;
    if (Array.isArray(v.stems)) root.querySelectorAll('#s2m-stems input').forEach((i) => { i.checked = v.stems.includes(i.value); });
    if (Array.isArray(v.midi)) root.querySelectorAll('#s2m-midi input').forEach((i) => { i.checked = v.midi.includes(i.value); });
}

async function submit() {
    if (!file) return;
    const go = $('#s2m-go');
    go.disabled = true;
    const six = ['htdemucs_6s', 'roformer'].includes($('#s2m-model').value);
    const keep = (arr) => arr.filter((s) => six || (s !== 'guitar' && s !== 'piano'));
    const fd = new FormData();
    fd.append('file', file, file.name);
    fd.append('model', $('#s2m-model').value);
    fd.append('device', $('#s2m-device').value);
    fd.append('quantize', $('#s2m-quantize').value);
    fd.append('bpm', $('#s2m-bpm').value || '');
    fd.append('engine', $('#s2m-engine').value);
    fd.append('format', $('#s2m-format').value);
    fd.append('tuning', tuningValue());
    fd.append('slap', $('#s2m-slap').value.trim());
    fd.append('parts', $('#s2m-parts').value.trim());
    fd.append('only', $('#s2m-only').value.trim());
    fd.append('sustained', $('#s2m-sustained').checked ? 'true' : 'false');
    const nf = $('#s2m-notes-file').files[0];
    if (nf) {
        fd.append('notes_file', nf, nf.name);
        const use = $('#s2m-notes-use').value;
        fd.append('notes_use', use === 'bars' ? ($('#s2m-notes-bars').value.trim() || 'all') : use);
        fd.append('notes_track', $('#s2m-notes-track').value.trim());
    }
    fd.append('lyrics', $('#s2m-lyrics').value);
    fd.append('fetch_meta', $('#s2m-fetch').checked ? 'true' : 'false');
    fd.append('charter', $('#s2m-charter').value.trim());
    fd.append('tone', $('#s2m-tone').value.trim());
    fd.append('language', $('#s2m-language').value);
    fd.append('backing', $('#s2m-backing').checked ? 'true' : 'false');
    fd.append('skip_separation', $('#s2m-skip').checked ? 'true' : 'false');
    fd.append('stems', keep(chipValues('#s2m-stems')).join(',') || 'bass');
    fd.append('midi', keep(chipValues('#s2m-midi')).join(',') || 'none');
    $('#s2m-form-msg').textContent = 'Uploading…';
    try {
        await api('/jobs', { method: 'POST', body: fd });
        setFile(null);
        $('#s2m-file').value = '';
        $('#s2m-lyrics').value = '';          // lyrics and notes belong to that song
        $('#s2m-notes-file').value = '';
        $('#s2m-only').value = '';
        await loadJobs();
    } catch (e) {
        updateGo();
        $('#s2m-form-msg').textContent = 'Error: ' + e.message;
        return;
    }
    updateGo();
}

// ── jobs ───────────────────────────────────────────────────────────────
let jobsCache = [];
async function loadJobs() {
    try {
        const data = await api('/jobs');
        jobsCache = data.jobs || [];
    } catch (_) { return; }
    renderJobs();
    schedulePoll();
}

function schedulePoll() {
    clearTimeout(pollTimer);
    if (!active) return;
    const busy = jobsCache.some((j) => j.status === 'queued' || j.status === 'running');
    const installing = status && status.install && status.install.running;
    if (!busy && !installing) return;
    pollTimer = setTimeout(async () => {
        if (installing) {
            await loadStatus(false);
            // install just finished → re-probe the environment
            if (status && status.install && !status.install.running) await loadStatus(true);
        }
        if (busy) await loadJobs(); else schedulePoll();
    }, 1200);
}

function jobSignature(j) {
    return [j.status, j.status === 'running' ? Math.round((j.pct || 0) * 200) : '', j.msg || '', j.error || ''].join('|');
}

function renderJobs() {
    const host = $('#s2m-jobs');
    if (!host) return;
    if (!jobsCache.length) {
        host.innerHTML = '<p class="s2m-dim s2m-small">No jobs yet.</p>';
        rendered.clear();
        return;
    }
    if (host.querySelector('p.s2m-dim')) host.innerHTML = '';
    const ids = new Set(jobsCache.map((j) => j.id));
    host.querySelectorAll('.s2m-job').forEach((el) => {
        if (!ids.has(el.dataset.id)) { el.remove(); rendered.delete(el.dataset.id); stopRoll(el.dataset.id); }
    });
    let prev = null;
    jobsCache.forEach((j) => {
        let el = host.querySelector(`.s2m-job[data-id="${j.id}"]`);
        if (!el) {
            el = document.createElement('article');
            el.className = 's2m-card s2m-job';
            el.dataset.id = j.id;
        }
        const sig = jobSignature(j);
        if (rendered.get(j.id) !== sig) {
            if (j.status === 'running' && rendered.has(j.id) && el.querySelector('.s2m-bar')) {
                updateProgress(el, j);   // cheap in-place update while running
            } else {
                stopRoll(j.id);
                el.innerHTML = jobHtml(j);
                if (j.status === 'done') hydrateDone(el, j);
            }
            rendered.set(j.id, sig);
        }
        // keep newest-first order without re-creating nodes
        if (prev ? prev.nextSibling !== el : host.firstChild !== el) {
            if (prev) prev.after(el); else host.prepend(el);
        }
        prev = el;
    });
}

function updateProgress(el, j) {
    const bar = el.querySelector('.s2m-bar > i');
    if (bar) bar.style.width = Math.round((j.pct || 0) * 100) + '%';
    const msg = el.querySelector('[data-role="msg"]');
    if (msg) msg.textContent = `${Math.round((j.pct || 0) * 100)}% · ${j.msg || ''}`;
    const meta = el.querySelector('[data-role="meta"]');
    if (meta) meta.textContent = metaText(j);
}

function metaText(j) {
    const o = j.options || {};
    const bits = [o.skip_separation ? 'transcribe only' : o.model];
    if (j.started) bits.push(fmtDur(((j.finished || Date.now() / 1000) - j.started)));
    if (j.result && j.result.device && j.result.device !== 'n/a') bits.push(String(j.result.device).toUpperCase());
    if (j.result && j.result.bpm) bits.push(j.result.bpm + ' BPM');
    return bits.filter(Boolean).join(' · ');
}

function jobHtml(j) {
    const head = `
      <div class="s2m-row s2m-between">
        <div style="min-width:0">
          <div class="s2m-job-title">${esc(j.filename)}</div>
          <div class="s2m-meta" data-role="meta">${esc(metaText(j))}</div>
        </div>
        <div class="s2m-row">
          <span class="s2m-badge s2m-${esc(j.status)}">${esc(j.status)}</span>
          ${j.status === 'done' ? `<a class="s2m-btn s2m-sm" href="${API}/jobs/${j.id}/zip" download>Download all (.zip)</a>` : ''}
          ${j.status === 'running' || j.status === 'queued'
            ? `<button class="s2m-btn s2m-sm s2m-danger" data-act="cancel" data-id="${j.id}" type="button">Cancel</button>`
            : `<button class="s2m-btn s2m-sm s2m-danger" data-act="delete" data-id="${j.id}" type="button" title="Delete this result">Delete</button>`}
        </div>
      </div>`;
    if (j.status === 'running' || j.status === 'queued') {
        const pct = Math.round((j.pct || 0) * 100);
        return head + `<div class="s2m-bar"><i style="width:${pct}%"></i></div>
          <div class="s2m-small s2m-dim" data-role="msg">${j.status === 'queued' ? 'Waiting in queue…' : `${pct}% · ${esc(j.msg || '')}`}</div>`;
    }
    if (j.status === 'error') {
        return head + `<div class="s2m-err">${esc(j.error || 'Failed')}</div>` +
            (j.log_tail ? `<details><summary class="s2m-small s2m-dim">Log</summary><pre class="s2m-log">${esc(j.log_tail)}</pre></details>` : '');
    }
    if (j.status === 'cancelled') return head;

    const r = j.result || {};
    const ORDER = ['bass', 'guitar', 'drums', 'vocals', 'piano', 'other'];
    const rank = (n) => (ORDER.indexOf(n) < 0 ? 99 : ORDER.indexOf(n));
    const stems = Object.entries(r.stems || {}).sort((a, b) => rank(a[0]) - rank(b[0]));
    let html = head;
    const midiEntries = Object.entries(r.midi || {});
    midiEntries.forEach(([stem, mid]) => {
        html += `<div class="s2m-section-label">${esc(STEM_LABELS[stem] || stem)} MIDI</div>
          <div class="s2m-midi" data-stem="${esc(stem)}">
            <div class="s2m-row s2m-between s2m-midi-bar">
              <span class="s2m-small s2m-dim" data-role="midi-info">Loading notes…</span>
              <div class="s2m-row">
                <button class="s2m-btn s2m-sm" data-act="preview" type="button">▶ Hear MIDI${(r.stems || {}).backing_no_bass ? ' + backing' : ''}</button>
                ${(r.tabs || {})[stem] ? `<button class="s2m-btn s2m-sm" data-act="tab" data-file="${esc(r.tabs[stem])}" type="button">Tab</button>` : ''}
                <a class="s2m-btn s2m-sm s2m-primary" href="${fileUrl(j.id, mid, true)}" download>Download ${esc(mid)}</a>
                ${(r.rsxml || {})[stem] ? `<a class="s2m-btn s2m-sm s2m-primary" href="${fileUrl(j.id, r.rsxml[stem], true)}" download title="Rocksmith 2014 bass arrangement — import in EOF (File → Import → Rocksmith) or DLC Builder">Rocksmith XML</a>` : ''}
              </div>
            </div>
            <canvas class="s2m-roll" title="Click to play from here"></canvas>
            <div class="s2m-tab" hidden>
              <div class="s2m-row s2m-between s2m-tab-bar">
                <span class="s2m-small s2m-dim" data-role="tab-info"></span>
                ${(r.tabs || {})[stem] ? `<a class="s2m-btn s2m-sm" href="${fileUrl(j.id, r.tabs[stem], true)}" download>Download ${esc(r.tabs[stem])}</a>` : ''}
              </div>
              <pre class="s2m-log s2m-tab-pre"></pre>
            </div>
            ${stem === 'bass' && (r.tabs || {})[stem] ? rsForm(j) : ''}
          </div>`;
    });
    if ((r.stems || {}).vocals || r.lyrics) html += lyricsBlock(j);
    if (stems.length) {
        html += '<div class="s2m-section-label">Stems</div><div class="s2m-stems">';
        stems.forEach(([name, fname]) => {
            html += `<div class="s2m-stem">
                <span class="s2m-stem-name" title="${esc(fname)}">${esc(STEM_LABELS[name] || name)}</span>
                <audio controls preload="none" data-stem="${esc(name)}" src="${fileUrl(j.id, fname)}"></audio>
                <a class="s2m-dl" href="${fileUrl(j.id, fname, true)}" download title="Download ${esc(fname)}">${DL_ICON}</a>
              </div>`;
        });
        html += '</div>';
    }
    (r.warnings || []).forEach((w) => { html += `<div class="s2m-warn">⚠ ${esc(w)}</div>`; });
    return html;
}

// Rocksmith 2014 XML export (built by the server with FeedBack's own writer)
function rsForm(j) {
    const meta = (j.options && j.options.meta) || {};
    const field = (name, label, value, size) =>
        `<label class="s2m-small s2m-dim">${label} <input class="s2m-input s2m-rs-in" name="${name}" value="${esc(value || '')}" size="${size}" maxlength="120"></label>`;
    return `<details class="s2m-rs">
        <summary class="s2m-small">Rocksmith XML: other artist / title, or line it up with EOF's audio…</summary>
        <form class="s2m-row s2m-rs-form" id="s2m-rs-${esc(j.id)}" data-job="${esc(j.id)}">
          ${field('artist', 'Artist', meta.artist, 18)}
          ${field('title', 'Title', meta.title, 22)}
          ${field('album', 'Album', meta.album, 18)}
          ${field('year', 'Year', meta.year, 5)}
        </form>
        <div class="s2m-row s2m-rs-form">
          <label class="s2m-small s2m-dim" title="The audio the chart will play with — EOF's guitar.ogg has extra silence at the start; the notes are moved to match it">Line up with
            <input class="s2m-rs-in" type="file" name="align" accept="audio/*,.ogg" form="s2m-rs-${esc(j.id)}"></label>
          <label class="s2m-small s2m-dim" title="Or move everything by this many seconds (EOF lead-in)">or shift (s)
            <input class="s2m-input s2m-rs-in" name="shift" type="number" step="0.001" min="-60" max="60" placeholder="default" style="width:6em" form="s2m-rs-${esc(j.id)}"></label>
          <label class="s2m-small s2m-dim" title="Rocksmith phrases are what Riff Repeater loops: shorter phrases = shorter bits to practise. Sections (verse, chorus…) stay as they are; the same notes get the same phrase name.">phrases
            <select class="s2m-input s2m-inline s2m-rs-in" name="phrase_bars" form="s2m-rs-${esc(j.id)}" style="margin-left:0">
              <option value="">default</option><option value="2">every 2 bars</option><option value="4">every 4 bars</option>
              <option value="8">every 8 bars</option><option value="0">one per section</option></select></label>
          <button class="s2m-btn s2m-sm s2m-primary" type="submit" form="s2m-rs-${esc(j.id)}">Download XML</button>
          <span class="s2m-small s2m-dim" data-role="rs-msg"></span>
        </div>
        <div class="s2m-small s2m-dim">EOF adds silence at the start of the guitar.ogg it makes: choose that file here and the notes
          and beats are moved to match (measured from the audio). In EOF select the PART REAL_BASS track first, then
          File → Import → Rocksmith; the notes land in difficulty 0 (first difficulty tab), not Expert.</div>
      </details>`;
}

// Lyrics → UltraStar .txt + Rocksmith vocals XML
function lyricsLinks(j, ly) {
    if (!ly || !ly.ultrastar) return '';
    const how = ly.aligner === 'mms' ? 'MMS aligner' : 'simple aligner — check the words inside the lines';
    return `<span class="s2m-small s2m-dim">${ly.lines || '?'} lines, ${ly.syllables || '?'} syllables (${esc(how)})</span>
        <a class="s2m-btn s2m-sm s2m-primary" href="${fileUrl(j.id, ly.ultrastar, true)}" download title="Open in UltraStar Creator to fine-tune, or import in EOF: File → Import → Lyrics">UltraStar .txt</a>
        <a class="s2m-btn s2m-sm s2m-primary" href="${fileUrl(j.id, ly.xml, true)}" download title="Rocksmith 2014 vocals arrangement (like EOF's PART VOCALS_RS2.xml) — for DLC Builder">Vocals XML</a>`;
}

function lyricsBlock(j) {
    const r = j.result || {};
    const ly = r.lyrics || null;
    const canRedo = !!(r.stems || {}).vocals;
    return `<div class="s2m-section-label">Lyrics</div>
      <div class="s2m-lyr" data-job="${esc(j.id)}">
        <div class="s2m-row" data-role="lyr-links">${lyricsLinks(j, ly)}</div>
        ${canRedo ? `<details class="s2m-rs" ${ly ? '' : 'open'}>
          <summary class="s2m-small">${ly ? 'Line up other / corrected lyrics…' : 'Paste the lyrics to line them up with the vocals…'}</summary>
          <form class="s2m-lyr-form" data-job="${esc(j.id)}">
            <textarea class="s2m-input s2m-lyrics" name="lyrics" rows="5" spellcheck="false" placeholder="One sung line per line, every repeat written out. Syllables: automatic, or mark them: un-faith-ful"></textarea>
            <div class="s2m-row">
              <select class="s2m-input s2m-inline" name="language" style="margin-left:0"><option value="">language: auto</option><option value="pt">Português</option><option value="en">English</option><option value="es">Español</option><option value="fr">Français</option><option value="de">Deutsch</option><option value="it">Italiano</option></select>
              <label class="s2m-small s2m-dim" title="EOF lead-in (seconds added to every time); empty = lead_silence from stems2midi.ini">shift (s)
                <input class="s2m-input" name="shift" type="number" step="0.001" min="-60" max="60" placeholder="default" style="width:6em"></label>
              <button class="s2m-btn s2m-sm s2m-primary" type="submit">Line up lyrics</button>
              <span class="s2m-small s2m-dim" data-role="lyr-msg"></span>
            </div>
          </form>
          <div class="s2m-small s2m-dim">The first time, the aligner model (~1.2 GB) is downloaded — that run takes a few minutes.
            UltraStar file: #MP3 is guitar.ogg when there is a lead-in (open it with the EOF project's audio).</div>
        </details>` : ''}
      </div>`;
}

async function submitLyricsForm(form) {
    const box = form.closest('.s2m-lyr');
    const msg = form.querySelector('[data-role="lyr-msg"]');
    const btn = form.querySelector('button[type="submit"]');
    const fd = new FormData(form);
    if (!String(fd.get('lyrics') || '').trim()) { msg.textContent = 'Paste the lyrics first.'; return; }
    btn.disabled = true;
    msg.textContent = 'Lining up with the vocals… (first time: downloading the model)';
    try {
        const res = await fetch(`${API}/jobs/${form.dataset.job}/lyrics`, { method: 'POST', body: fd });
        let body = null;
        try { body = await res.json(); } catch (_) { body = null; }
        if (!res.ok) throw new Error((body && (body.detail || body.error)) || 'HTTP ' + res.status);
        const j = jobsCache.find((x) => x.id === form.dataset.job);
        if (j) { j.result = j.result || {}; j.result.lyrics = body.lyrics; }
        box.querySelector('[data-role="lyr-links"]').innerHTML = lyricsLinks(j || { id: form.dataset.job }, body.lyrics);
        msg.textContent = 'Done.';
    } catch (e) {
        msg.textContent = '⚠ ' + e.message;
    } finally {
        btn.disabled = false;
    }
}

async function submitRsForm(form) {
    const box = form.closest('.s2m-rs');
    const msg = box.querySelector('[data-role="rs-msg"]');
    const btn = box.querySelector('button[type="submit"]');
    const fd = new FormData(form);
    const file = box.querySelector('input[name="align"]').files[0];
    if (file) fd.append('align', file);
    fd.set('shift', box.querySelector('input[name="shift"]').value || '');
    btn.disabled = true;
    msg.textContent = file ? 'Lining up with ' + file.name + '…' : 'Building…';
    try {
        const res = await fetch(`${API}/jobs/${form.dataset.job}/rocksmith.xml`, { method: 'POST', body: fd });
        if (!res.ok) {
            let err = 'HTTP ' + res.status;
            try { const b = await res.json(); err = b.detail || b.error || err; } catch (_) { /* not JSON */ }
            throw new Error(err);
        }
        const blob = await res.blob();
        const cd = res.headers.get('content-disposition') || '';
        const m = /filename\*=UTF-8''([^;]+)/i.exec(cd);
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = m ? decodeURIComponent(m[1]) : 'bass_RS2.xml';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 5000);
        const sh = res.headers.get('x-stems2midi-shift');
        msg.textContent = sh && Number(sh) ? `Done — notes moved ${Number(sh) > 0 ? '+' : ''}${Number(sh).toFixed(3)} s.` : 'Done.';
    } catch (e) {
        msg.textContent = '⚠ ' + e.message;
    } finally {
        btn.disabled = false;
    }
}

// ── piano roll + MIDI preview ──────────────────────────────────────────
function hydrateDone(el, j) {
    const r = j.result || {};
    Object.entries(r.notes || {}).forEach(async ([stem, nfile]) => {
        const box = el.querySelector(`.s2m-midi[data-stem="${CSS.escape(stem)}"]`);
        if (!box) return;
        try {
            const res = await fetch(fileUrl(j.id, nfile));
            const data = await res.json();
            const ctl = makeRoll(j, box, data, el);
            rolls.set(j.id + ':' + stem, ctl);
        } catch (e) {
            const info = box.querySelector('[data-role="midi-info"]');
            if (info) info.textContent = 'Could not load notes: ' + e.message;
        }
    });
}

function stopRoll(jid) {
    for (const [key, ctl] of rolls) {
        if (key.startsWith(jid + ':')) { ctl.destroy(); rolls.delete(key); }
    }
}
function stopAllRolls() { for (const ctl of rolls.values()) ctl.stopPreview(); }

function makeRoll(job, box, data, jobEl) {
    const notes = data.notes || [];
    const canvas = box.querySelector('canvas');
    const info = box.querySelector('[data-role="midi-info"]');
    const previewBtn = box.querySelector('[data-act="preview"]');
    const stem = data.stem;
    const end = notes.reduce((m, n) => Math.max(m, n.t + n.d), 0) || 1;
    let lo = Infinity, hi = -Infinity;
    notes.forEach((n) => { lo = Math.min(lo, n.m); hi = Math.max(hi, n.m); });
    if (!isFinite(lo)) { lo = 28; hi = 55; }
    lo -= 2; hi += 2;
    info.textContent = notes.length
        ? `${notes.length} notes · ${noteName(lo + 2)}–${noteName(hi - 2)} · ${data.bpm} BPM · engine ${data.engine}${data.quantize ? ' · quantized 1/' + data.quantize : ''}`
        : 'No notes detected';
    if (data.tuning && notes.length) {
        const names = data.tuning.strings.map((m) => NOTE_NAMES[m % 12]);
        const st = data.tab_stats || {};
        info.innerHTML = esc(info.textContent) + ' · tab ' + names.map((nm, i) =>
            `<b style="color:rgb(${STRING_RGB[i % STRING_RGB.length]})">${esc(nm)}</b>`).join(' ') +
            esc(` · ${st.shifts || 0} position shifts · max fret ${st.max_fret || 0}`) +
            (st.slap != null ? esc(` · ${st.slap} slap ▲ · ${st.pop} pop ●`) : '') +
            (data.slap_auto ? esc(` · guessed slap sections: ${data.slap_sections}`) : '');
        if (Array.isArray(data.parts_auto) && data.parts_auto.length) {   // guessed song parts
            const line = document.createElement('div');
            line.className = 's2m-small s2m-parts-line';
            line.textContent = 'Parts (guessed — same letter = same music): ' + data.parts_auto.map((p) =>
                `${p.name} ${p.number} (${p.letter}) ${fmtClock(p.start)}`).join(' · ') + ' ';
            const ed = document.createElement('button');
            ed.type = 'button';
            ed.className = 's2m-btn s2m-btn-small';
            ed.textContent = 'Edit these parts';
            ed.title = 'Copy the parts into the Song parts field: fix the names, then run again';
            ed.addEventListener('click', () => {
                const f = $('#s2m-parts');
                f.value = data.parts || '';
                f.focus();
                f.scrollIntoView({ block: 'center' });
            });
            line.append(ed);
            box.querySelectorAll('.s2m-parts-line').forEach((x) => x.remove());
            box.querySelector('.s2m-midi-bar').after(line);
        }
        if (data.slap_auto) {   // let the user correct the guess and re-run
            const use = document.createElement('button');
            use.type = 'button';
            use.className = 's2m-btn s2m-btn-small';
            use.textContent = 'Edit these sections';
            use.title = 'Copy the guessed sections into the Slap sections field to correct them';
            use.addEventListener('click', () => {
                const f = $('#s2m-slap');
                f.value = data.slap_sections;
                f.focus();
                f.scrollIntoView({ block: 'center' });
            });
            info.append(' ', use);
        }
    }

    // tab text view
    const tabBtn = box.querySelector('[data-act="tab"]');
    const tabBox = box.querySelector('.s2m-tab');
    let tabLoaded = false;
    if (tabBtn && tabBox) {
        tabBtn.addEventListener('click', async () => {
            tabBox.hidden = !tabBox.hidden;
            tabBtn.classList.toggle('s2m-on', !tabBox.hidden);
            if (tabBox.hidden || tabLoaded) return;
            const pre = tabBox.querySelector('pre');
            try {
                const res = await fetch(fileUrl(job.id, tabBtn.dataset.file));
                pre.textContent = res.ok ? await res.text() : 'Could not load the tab.';
                tabLoaded = res.ok;
            } catch (e) { pre.textContent = 'Could not load the tab: ' + e.message; }
            const t = data.tuning || {};
            tabBox.querySelector('[data-role="tab-info"]').textContent =
                `Tuning ${t.label || ''} (${(t.strings || []).map(noteName).join(' ')}) · offsets ${(t.offsets || []).join(',')}`;
        });
    }

    // list of bars to review, under the roll
    const rv = (data.review || []).filter((b) => b.level !== 'ok').sort((a, b) => b.p - a.p);
    box.querySelectorAll('.s2m-review').forEach((x) => x.remove());
    if (Array.isArray(data.review) && data.review.length) {
        const nb = rv.filter((b) => b.level === 'bad').length;
        const det = document.createElement('details');
        det.className = 's2m-review';
        det.innerHTML = `<summary class="s2m-small">Bars to review: <b class="s2m-rv-bad">${nb} likely wrong</b> · ` +
            `<b class="s2m-rv-check">${rv.length - nb} worth a check</b> · ${data.review.length - rv.length} look fine ` +
            `<span class="s2m-dim">(red / amber marks on the roll; click a bar to hear it` +
            `${data.review.some((b) => b.relative) ? ' — the stem is hard to read: these are the worst bars of this song' : ''})</span></summary>` +
            `<ol class="s2m-rv-list">${rv.map((b) => `<li data-t="${b.t}" class="s2m-rvi-${b.level}">` +
            `bar ${b.bar} · ${fmtClock(b.t)} · ${Math.round(b.p * 100)}% · ${esc((b.why || []).join(', ') || '—')}</li>`).join('')}</ol>`;
        canvas.after(det);
        det.addEventListener('click', (ev) => {
            const li = ev.target.closest('li[data-t]');
            if (!li) return;
            const t = Math.max(0, Number(li.dataset.t) - 0.5);
            const a = stemAudio();
            if (a) { a.currentTime = t; a.play().catch(() => {}); } else { playhead = t; draw(); }
        });
    }

    const audios = () => Array.from(jobEl.querySelectorAll('audio'));
    const stemAudio = () => jobEl.querySelector(`audio[data-stem="${CSS.escape(stem)}"]`);
    const backingAudio = () => jobEl.querySelector('audio[data-stem="backing_no_bass"]');
    let playhead = null, raf = 0, preview = null;

    function draw() {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth, h = canvas.clientHeight;
        if (!w) return;
        if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
        const g = canvas.getContext('2d');
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, w, h);
        const rows = hi - lo + 1, rowH = h / rows;
        // octave guides (C lines) + E lines (bass open strings)
        for (let m = lo; m <= hi; m++) {
            const y = h - (m - lo + 1) * rowH;
            if (m % 12 === 0) { g.fillStyle = 'rgba(148,163,184,.10)'; g.fillRect(0, y + rowH - 1, w, 1); }
            if (m % 12 === 0 && rowH * 12 > 28) {
                g.fillStyle = 'rgba(148,163,184,.55)'; g.font = '10px system-ui, sans-serif';
                g.fillText(noteName(m), 4, y + rowH - 3);
            }
        }
        // beat grid
        const beat = 60 / (data.bpm || 120);
        if (end / beat < w / 3) {
            for (let t = 0, k = 0; t < end; t += beat, k++) {
                const x = t / end * w;
                g.fillStyle = k % 4 === 0 ? 'rgba(148,163,184,.14)' : 'rgba(148,163,184,.05)';
                g.fillRect(x, 0, 1, h);
            }
        }
        // review: bars most likely to need fixing (amber = check, red = likely wrong)
        (data.review || []).forEach((b) => {
            if (b.level === 'ok') return;
            const x = b.t / end * w, bw = Math.max(1, (b.end - b.t) / end * w);
            g.fillStyle = b.level === 'bad' ? 'rgba(239,68,68,.10)' : 'rgba(234,179,8,.07)';
            g.fillRect(x, 0, bw, h);
            g.fillStyle = b.level === 'bad' ? 'rgba(239,68,68,.85)' : 'rgba(234,179,8,.75)';
            g.fillRect(x, 0, bw, 4);
        });
        notes.forEach((n) => {
            const x = n.t / end * w, nw = Math.max(1.5, n.d / end * w);
            const y = h - (n.m - lo + 1) * rowH;
            const a = 0.45 + 0.55 * (n.v || 90) / 127;
            const on = playhead != null && playhead >= n.t && playhead < n.t + n.d;
            const rgb = n.s != null ? STRING_RGB[n.s % STRING_RGB.length] : '14,165,233';
            g.fillStyle = on ? '#e8c040' : `rgba(${rgb},${a.toFixed(2)})`;
            g.fillRect(x, y + 0.5, nw, Math.max(1.5, rowH - 1));
            if (n.tech) {   // ▲ slap (white) / ● pop (gold) above the note
                g.fillStyle = n.tech === 'pop' ? '#e8c040' : '#f8fafc';
                const cx = x + Math.min(nw, 6) / 2, cy = y - 3;
                g.beginPath();
                if (n.tech === 'pop') g.arc(cx, cy, 2.2, 0, Math.PI * 2);
                else { g.moveTo(cx - 2.5, cy + 1.5); g.lineTo(cx + 2.5, cy + 1.5); g.lineTo(cx, cy - 2.5); }
                g.fill();
            }
            if (n.f != null && nw >= 11 && rowH >= 7) {   // fret number on the note
                g.fillStyle = '#04131d';
                g.font = `bold ${Math.min(10, Math.floor(rowH + 1))}px system-ui, sans-serif`;
                g.fillText(String(n.f), x + 2, y + rowH - 1.5);
            }
        });
        if (playhead != null) {
            g.fillStyle = '#f8fafc';
            g.fillRect(playhead / end * w, 0, 1.5, h);
        }
    }

    function tick() {
        const playing = (preview && preview.ctx) ? preview : null;
        const a = audios().find((x) => !x.paused);
        if (playing) playhead = playing.offset + (playing.ctx.currentTime - playing.t0);
        else if (a) playhead = a.currentTime;
        draw();
        if (playing || a) raf = requestAnimationFrame(tick);
        else raf = 0;
    }
    function kick() { if (!raf) raf = requestAnimationFrame(tick); }

    const onPlay = (ev) => {
        // only one source at a time inside the page
        document.querySelectorAll('.s2m-root audio').forEach((x) => { if (x !== ev.target) x.pause(); });
        if (preview && ev.target !== backingAudio()) stopPreview();
        kick();
    };
    const onSeek = (ev) => { if (!preview) { playhead = ev.target.currentTime; draw(); } };
    audios().forEach((a) => { a.addEventListener('play', onPlay); a.addEventListener('seeked', onSeek); });

    function startPreview(from) {
        stopPreview();
        stopAllRolls();
        document.querySelectorAll('.s2m-root audio').forEach((x) => x.pause());
        const AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return;
        const ctx = new AC();
        const t0 = ctx.currentTime + 0.15;
        const master = ctx.createGain();
        master.gain.value = 0.5;
        const lp = ctx.createBiquadFilter();
        lp.type = 'lowpass'; lp.frequency.value = 1100; lp.Q.value = 0.8;
        lp.connect(master); master.connect(ctx.destination);
        // schedule in a sliding window so long songs don't create 1000s of nodes at once
        let idx = notes.findIndex((n) => n.t + n.d > from);
        if (idx < 0) idx = notes.length;
        preview = { ctx, t0, offset: from, idx, timer: 0 };
        const schedule = () => {
            if (!preview || preview.ctx !== ctx) return;
            const horizon = ctx.currentTime - t0 + from + 2.0;
            while (preview.idx < notes.length && notes[preview.idx].t < horizon) {
                const n = notes[preview.idx++];
                const st = t0 + Math.max(0, n.t - from), du = Math.max(0.05, n.d - Math.max(0, from - n.t));
                const f = 440 * Math.pow(2, (n.m - 69) / 12);
                const o = ctx.createOscillator(), o2 = ctx.createOscillator(), env = ctx.createGain();
                o.type = 'sawtooth'; o.frequency.value = f;
                o2.type = 'sine'; o2.frequency.value = f; // adds fundamental body
                const peak = 0.25 * (0.5 + 0.5 * (n.v || 90) / 127);
                env.gain.setValueAtTime(0, st);
                env.gain.linearRampToValueAtTime(peak, st + 0.006);
                env.gain.exponentialRampToValueAtTime(peak * 0.55, st + Math.min(du, 0.25));
                env.gain.setValueAtTime(peak * 0.55, st + du);
                env.gain.exponentialRampToValueAtTime(0.0001, st + du + 0.06);
                o.connect(env); o2.connect(env); env.connect(lp);
                o.start(st); o2.start(st); o.stop(st + du + 0.08); o2.stop(st + du + 0.08);
            }
            if (preview.idx >= notes.length && ctx.currentTime - t0 + from > end + 0.5) { stopPreview(); return; }
            preview.timer = setTimeout(schedule, 500);
        };
        schedule();
        const back = backingAudio();
        if (back) {
            back.currentTime = from;
            setTimeout(() => { if (preview && preview.ctx === ctx) back.play().catch(() => {}); }, 150);
        }
        previewBtn.textContent = '■ Stop';
        kick();
    }
    function stopPreview() {
        if (!preview) return;
        clearTimeout(preview.timer);
        try { preview.ctx.close(); } catch (_) { /* already closed */ }
        preview = null;
        const back = backingAudio();
        if (back && !back.paused) back.pause();
        previewBtn.textContent = '▶ Hear MIDI' + (backingAudio() ? ' + backing' : '');
    }

    previewBtn.addEventListener('click', () => {
        if (preview) stopPreview(); else startPreview(playhead != null && playhead < end ? playhead : 0);
    });
    canvas.addEventListener('click', (ev) => {
        const rect = canvas.getBoundingClientRect();
        const t = Math.max(0, (ev.clientX - rect.left) / rect.width * end);
        playhead = t;
        if (preview) { startPreview(t); return; }
        const a = audios().find((x) => !x.paused) || stemAudio();
        if (a) { a.currentTime = t; if (a.paused) a.play().catch(() => {}); }
        draw();
    });
    const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(() => draw()) : null;
    if (ro) ro.observe(canvas);
    draw();

    return {
        stopPreview,
        destroy() {
            stopPreview();
            if (raf) cancelAnimationFrame(raf);
            if (ro) ro.disconnect();
            audios().forEach((a) => { a.removeEventListener('play', onPlay); a.removeEventListener('seeked', onSeek); });
        },
    };
}

// ── wiring ─────────────────────────────────────────────────────────────
function bind() {
    root = document.querySelector('#' + SCREEN_ID + ' .s2m-root');
    if (!root || root.dataset.bound) return !!root;
    root.dataset.bound = '1';

    const drop = $('#s2m-drop'), input = $('#s2m-file');
    input.addEventListener('change', () => setFile(input.files && input.files[0]));
    drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
    ['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('s2m-over'); }));
    ['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('s2m-over'); }));
    drop.addEventListener('drop', (e) => {
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) setFile(f);
    });

    restoreForm();
    updateModelChips();
    updateTuningBox();
    root.addEventListener('change', (e) => {
        if (e.target.id === 's2m-model') updateModelChips();
        if (e.target.id === 's2m-tuning') updateTuningBox();
        if (e.target.closest('.s2m-card') && e.target.id !== 's2m-file' && e.target.id !== 's2m-python') saveForm();
        if (e.target.id === 's2m-skip') updateGo();
    });
    $('#s2m-go').addEventListener('click', submit);
    $('#s2m-refresh').addEventListener('click', () => loadStatus(true));
    $('#s2m-toggle-settings').addEventListener('click', (e) => {
        const box = $('#s2m-settings');
        box.hidden = !box.hidden;
        e.currentTarget.setAttribute('aria-expanded', String(!box.hidden));
    });
    $('#s2m-save-python').addEventListener('click', async () => {
        try {
            await api('/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ python: $('#s2m-python').value.trim() }) });
            await loadStatus(true);
        } catch (e) {
            $('#s2m-python-current').textContent = 'Error: ' + e.message;
        }
    });
    $('#s2m-install-btn').addEventListener('click', async () => {
        try {
            await api('/install', { method: 'POST' });
        } catch (e) { /* surfaced by the status refresh */ }
        await loadStatus(false);
        schedulePoll();
    });
    $('#s2m-notes-use').addEventListener('change', () => {
        $('#s2m-notes-bars').hidden = $('#s2m-notes-use').value !== 'bars';
    });
    $('#s2m-jobs').addEventListener('submit', (e) => {
        const lf = e.target.closest('form.s2m-lyr-form');
        if (lf) { e.preventDefault(); submitLyricsForm(lf); return; }
        const form = e.target.closest('form.s2m-rs-form');
        if (!form) return;
        e.preventDefault();
        submitRsForm(form);
    });
    $('#s2m-jobs').addEventListener('click', async (e) => {
        const btn = e.target.closest('button[data-act]');
        if (!btn || !btn.dataset.id) return;
        const id = btn.dataset.id;
        btn.disabled = true;
        try {
            if (btn.dataset.act === 'cancel') await api(`/jobs/${id}/cancel`, { method: 'POST' });
            if (btn.dataset.act === 'delete') {
                if (!window.confirm('Delete this result and its files?')) { btn.disabled = false; return; }
                stopRoll(id);
                await api(`/jobs/${id}`, { method: 'DELETE' });
            }
        } catch (_) { /* refreshed below */ }
        await loadJobs();
    });
    return true;
}

function enter() {
    if (!bind()) return;
    active = true;
    loadStatus(false);
    loadJobs();
}
function leave() {
    active = false;
    clearTimeout(pollTimer);
    stopAllRolls();
    if (root) root.querySelectorAll('audio').forEach((a) => a.pause());
}

function onScreenChanged(ev) {
    const id = (ev && ev.detail && ev.detail.id) || (ev && ev.id);
    if (id === SCREEN_ID) enter(); else if (active) leave();
}

const bus = window.feedBack || window.slopsmith;
if (bus && typeof bus.on === 'function') bus.on('screen:changed', onScreenChanged);
// The screen may already be visible when this script loads (deep link / reload).
const scr = document.getElementById(SCREEN_ID);
if (scr && (scr.classList.contains('active') || scr.offsetParent !== null)) enter();
})();
