# Stems → MIDI

Split a song (MP3, WAV, FLAC, OGG, M4A…) into **bass, guitar, drums, vocals,
piano and other** stems with [Demucs](https://github.com/adefossez/demucs), then
transcribe the **bass stem to a MIDI file**. Comes as a FeedBack plugin (screen
in the sidebar) and as a standalone command-line tool.

What you get per song:

| File | What it is |
|---|---|
| `bass.wav`, `guitar.wav`, … | the stems you selected (24-bit WAV or FLAC) |
| `backing_no_bass.wav` | the whole song minus the bass, for play-along |
| `bass.mid` | the bass line as MIDI (GM program 33, Electric Bass finger), tempo = detected BPM |
| `bass.notes.json` | the same notes as JSON (`t`, `d`, `m`, `v`) plus string `s` / fret `f`, the tuning and hand-position anchors |
| `bass.tab.txt` | plain-text bass tab (string & fret per note) for a quick check |
| `stems2midi.json` | run summary (model, device, BPM, files, warnings) |

## Install the engine

Separation needs PyTorch + Demucs (~2 GB). They're listed in
`requirements-ml.txt`, which is **deliberately not** `requirements.txt`, so the
plugin loader doesn't auto-download 2 GB on every fresh install.

**Easy way:** open *Stems → MIDI* in FeedBack and click **Install dependencies**.
This uses FeedBack's own Python and installs into a private folder
(`<config>/stems2midi/pylib`) that only the separation worker sees, so it can't
clash with the server's packages. On Windows that gives you the CPU build of
PyTorch.

**NVIDIA GPU (much faster — ~30 s instead of ~5 min per song):** make a
separate venv with the CUDA build of PyTorch and point the plugin at it:

```powershell
py -3.12 -m venv C:\stems-venv
C:\stems-venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
C:\stems-venv\Scripts\python -m pip install -r plugins\stems2midi\requirements-ml.txt
```

Use the CUDA index that matches your driver (see
https://pytorch.org/get-started/locally). Then in the plugin click **Python…**,
enter `C:\stems-venv\Scripts\python.exe` and **Save** (or set the
`STEMS2MIDI_PYTHON` environment variable). **Re-check** should now show a green
`GPU: …` pill.

The first separation downloads the model weights (~80 MB per model) into the
PyTorch cache. After that it runs offline.

## Windows installer (another PC)

`installer/` holds a one-click installer, packaged as `stems2midi-install.zip`
(top level: `INSTALAR.bat`, `install.ps1`, `stems2midi.bat`, `LEIA-ME.txt`,
`stems2midi/` = this plugin). `INSTALAR.bat` detects the NVIDIA driver's CUDA
version, finds/installs Python 3.10–3.13, creates `.venv`, installs the newest
CUDA build of PyTorch the driver supports (falling back build by build, then to
CPU), installs the rest, verifies the GPU and runs a 10 s end-to-end test. It
can also create a desktop shortcut (drop audio files on it), set
`STEMS2MIDI_PYTHON` for the FeedBack plugin and copy the plugin into a FeedBack
folder. Options: `-Cuda cu126`, `-Force`, `-SkipTest`, `-FeedBackDir <path>`, `-Yes`.

**Updating:** extract the new zip anywhere and run `ATUALIZAR.bat`
(`install.ps1 -Update`): it finds the earlier install (this folder, `install_dir`
in `stems2midi.ini`, or `STEMS2MIDI_PYTHON`), copies the new tool into it,
pip-installs only the small packages that are missing (no PyTorch / models),
runs `--check` and copies the plugin to FeedBack again — seconds.
`ATUALIZAR.bat -Url "<share link>"` downloads the zip first (Google Drive /
Dropbox share links are turned into direct downloads) and remembers the link
(`update_url` in `stems2midi.ini`): after that a plain `ATUALIZAR.bat` fetches
whatever the link holds. The updater also installs a `torchaudio` that works
with the installed PyTorch (same version, else the newest of the same CUDA
build, `--no-deps`) for the lyrics' MMS aligner — the alignment itself is
done in numpy, so torchaudio's compiled part need not match PyTorch. `INSTALAR.bat`
offers the same when it finds an earlier install elsewhere.

## Command line

```bash
python tools/stems2midi.py song.mp3                      # bass + guitar stems, backing, bass.mid
python tools/stems2midi.py song.mp3 --stems all --quantize 16
python tools/stems2midi.py song.mp3 --model htdemucs_ft --stems bass
python tools/stems2midi.py bass_only.wav --skip-separation --bpm 98   # transcribe only
python tools/stems2midi.py --check                       # what's installed, GPU or not
```

Main options (`--help` lists all):

| Option | Default | Meaning |
|---|---|---|
| `-o DIR` | `<song>_stems/` | output folder |
| `--model` | `htdemucs_6s` | `htdemucs_6s` (6 stems incl. guitar/piano), `htdemucs` (4 stems, fastest), `htdemucs_ft` (4 stems, cleanest bass, ~4× slower) |
| `--device` | `auto` | `cpu`, `cuda`, `mps` |
| `--stems` | `bass,guitar` | stems to save, or `all` |
| `--midi` | `bass` | stems to transcribe (`none` to skip) |
| `--engine` | `auto` | `pyin` (monophonic, default for bass) or `basic-pitch` (polyphonic, optional) |
| `--quantize` | `0` | snap to 1/8, 1/16 or 1/32 notes on the detected beat grid |
| `--bpm` | auto | force the MIDI tempo |
| `--tuning` | `auto` | bass tuning for the tab: `auto` (4-string E standard, or 5-string B standard when the song goes below E — see below), `4`, `5`, `5c`, `drop-d`, `eb`, `d-standard`, `drop-c`, `5-drop-a`, note names `"D1 A1 D2 G2"`, MIDI `26,33,38,43`, offsets `--tuning=-2,0,0,0`, or `none` |
| `--max-fret` | `24` | highest fret on the neck |
| `--fmin/--fmax` | `B0`/`G4` for bass | pitch range |
| `--shifts N` | `1` | Demucs test-time augmentation: slightly better, N× slower |
| `--no-backing` | | don't write the mix-without-bass |
| `--no-review` | | don't check the transcription against the stem (see *Review against the stem*) |
| `--notes-from FILE` | | take the bass notes from AnthemScore / Guitar Pro / MuseScore (MusicXML or MIDI) — see *Notes from another transcriber* |
| `--phrase-bars` | `4` | Rocksmith XML: a phrase (Riff Repeater unit) every N bars inside each section; `0` = one per section |
| `--lyrics FILE` | | line the lyrics up with the vocals → `vocals.ultrastar.txt` + `vocals_RS2.xml` (see *Lyrics*) |
| `--language` | guessed | lyrics language: `pt`, `en`, `es`, `fr`, `de`, `it` |
| `--lyrics-aligner` | `auto` | `mms` (speech model), `simple` (no model), `auto` = mms when it loads |
| `--repo DIR` | | load Demucs weights from a local folder (offline) |

## How the bass transcription works

1. The bass stem is downmixed and resampled to 11 kHz.
2. [pYIN](https://librosa.org/doc/main/generated/librosa.pyin.html) tracks the
   fundamental frame by frame (93 ms window, 11.6 ms hop), down to B0 (or the
   lowest open string, if lower); the global tuning offset of the recording (often 10–30 cents off
   A440) is measured and removed before rounding to semitones.
3. **Attacks first, one note per attack.** An onset curve (46 ms STFT) is
   peak-picked in two tiers: strong attacks relative to the loudest one
   nearby, and weaker re-plucks relative to the *typical* attack height,
   accepted only if the short-frame level really rises (≥ 3 dB). This finds
   fast repeated 8ths/16ths without turning the flux wiggles of a ringing note
   into extra notes.
4. Each note's pitch is the median over its first ~110 ms (the fretted note —
   slides and the flat sag of a dying string are ignored); it ends at the next
   attack, an 18 dB drop or when the string stops sounding. A new *stable*
   pitch held ≥ 90 ms without a new attack becomes its own note (hammer-on,
   pull-off, slide target). Attacks without pitch (dead notes) are skipped.
5. Velocity comes from the attack level. Tempo comes from the full mix's beat
   tracker, folded into 60–170 BPM (use `--bpm` for faster songs).
6. Optional quantisation snaps to a beat-synchronous grid whose phase is fitted
   to the notes themselves.

### Measured on real charts

`benchmark.py` scores the transcription against charts you already finished
(see below). On five songs (htdemucs_6s, CPU):

| Song | Note-sequence F1 | Precision | Recall | Your chart's own sync error |
|---|---|---|---|---|
| Delfins – Canção de Engate | 0.81 | 0.76 | 0.89 | ±20 ms |
| Manic Street Preachers – La Tristesse Durera | 0.80 | 0.74 | 0.85 | −110…+125 ms |
| Phil Collins – Do You Remember | 0.50 | 0.51 | 0.49 | ±135 ms |
| Xutos & Pontapés – Circo de Feras | 0.91 | 0.89 | 0.93 | ±135 ms |
| Xutos & Pontapés – À Minha Maneira | 0.81 | 0.81 | 0.81 | −70…+65 ms |

(The previous pitch-run segmentation scored 0.73 / 0.70 on the first two.)
Weak spots: sustained synth-like bass with high double stops (Phil Collins),
and techniques — slap/pop, slides, dead notes are not detected yet.
`htdemucs_ft` separates the bass slightly better (+0.015 F1 on a test clip)
at ~4× the time.

**Basic Pitch** (`pip install basic-pitch`; may not install on the newest
Python versions — use a 3.10/3.11 venv if it fails) adds a
polyphonic engine used automatically for guitar/piano MIDI; for bass it's
reduced to a single line (lowest note wins).

## Which model / settings (GPU)

`htdemucs_6s` (default) is the only model with a separate guitar stem.
For the **bass transcription** `htdemucs_ft` is a little better and `--shifts 2`
doesn't help (note F1 vs the charts):

| | `htdemucs_6s` | `htdemucs_6s --shifts 2` | `htdemucs_ft` |
|---|---|---|---|
| Delfins | 0.822 | 0.821 | 0.827 |
| Katy Perry (slap) | 0.579 | 0.584 | 0.606 |

`htdemucs_ft` runs four models, so separation takes ~4× longer: on a 2-core
CPU ~15 min per song instead of ~3–4; on an 8 GB RTX 3050 roughly 1–2 min
instead of under 30 s (estimate) — fine when you only need the bass. Demucs
works on ~8 s chunks, so VRAM use stays well under 8 GB; on an out-of-memory
error the job falls back to the CPU by itself.

## Held notes vs. re-plucks

A same-pitch note that starts right where the previous one ends is kept only
if its attack reaches 25 % of the song's strong attacks (97th percentile of
the onset curve); weaker "attacks" are the wobble of a held or fading note and
are merged into it. Before, a note fading out at the end of a song came out as
a row of short identical notes. Note F1 on 13 charted songs: Delfins
0.822 → 0.858 (18 → 2 stray notes in the fade), Katy Perry 0.579 → 0.659,
Relax 0.618 → 0.648, the others within ±0.005 (0.30 merged more but cost up to
0.013 elsewhere). `STEMS2MIDI_REPEAT_MIN_ATTACK=0` turns it off.

## Tuning: 4-string E or 5-string B

The bass is tracked down to B0 whatever tuning you pick. When **at least 10
notes and 8 % of all notes** fall below the lowest open string, the song most
likely needs lower strings: with `auto` (the default) the tab is made for a
5-string in B standard, otherwise a warning suggests it (and the low notes are
moved up an octave). Measured on 15 charted songs: Korn *Word Up* (D standard)
30 % of its notes below E1, Primus 17 %; the E-standard songs 0–6 % (slides,
octave slips) — tracking down to B0 changed the E-standard songs' scores by
< 0.01.

## String & fret choice (`fretting.py`)

A pitch can be played in several places (E2 = D-string 2, A-string 7, E-string
12). `fretting.assign_frets()` picks one place per note for the whole line at
once, with these priorities:

1. **fewest / smallest hand jumps** — the hand is modelled by its anchor (index
   finger fret, reaching 4 frets); moving costs `3 + 1 per fret`, ×1.5 inside
   fast runs (< 150 ms between notes), ×0.5 when there's a rest ≥ 0.5 s;
2. **at or below the 12th fret** — +0.5 per fret above 12;
3. tie-breakers: lower positions, no string skipping, no 4-finger stretch below
   the 5th fret, open strings free in the first positions but discouraged when
   the hand sits at the 5th fret or higher.

The optimum over the whole line is found by dynamic programming (Viterbi over
fingering × hand position), so it can accept a slightly higher position now to
avoid a jump later. Double stops/chords are assigned to distinct strings within
one hand span. Notes outside the instrument's range move by octaves (or are
dropped with `out_of_range="drop"`), with a warning. On random 300-note lines it
needs ~3× fewer position shifts than a "lowest fret" greedy choice. All weights
live in `fretting.Weights`. For Guitar Pro input the tab already has strings
and frets, so this is only needed to re-fret for another tuning/string count.

## Slap & pop

Slap bass follows a simple playing rule: **inside the parts you slap, notes
with their own attack on the lower strings (E/A, plus B on a 5-string) are
slaps and on the two highest (D/G) pops** (hammer-ons/pull-offs/slides stay
plain) — that's where the charters put them too (in the 11 charts, 79 % of
slaps on E/A — Relax and Primus also slap the D string — and 98 % of pops
on D/G). The transcription has no strings, so
the boundary is a pitch: up to E2 (A string, 7th fret) slap, F2 and up pop
(tested 0–5 semitones above open D: E2/F2 is best), and the notes are then
fretted with slaps on E/A and pops on D/G. So the work is to
know *where* the slap parts are; the per-note marking is then automatic.
Tested on 11 slap/pop CDLCs (Delfins, Dirty Loops, Primus, Korn, Frankie Goes
to Hollywood, Lenny Kravitz, Seinfeld, Katy Perry, Jamiroquai, 2× RHCP) and
4 fingerstyle ones.

* **Transcription:** `--slap all`, `--slap "0:32-1:05, 2:10-2:40"` or
  `--slap auto` (UI: *Slap sections*). Notes get `"tech": "slap"|"pop"` in
  `bass.notes.json`, ▲/● markers in the piano roll and an `S`/`P` row under
  the tab. With the sections taken from the charts, end-to-end slap F1 is
  0.87–1.00 on Delfins, Korn, Relax, Katy Perry, Can't Stop and Dark
  Necessities (pop 0.74–1.00 where the chart has pops); it is low where the
  transcription itself is (Primus, Seinfeld, Dirty Loops) and zero on Lenny
  Kravitz's *Fly Away*, whose slapped verses are almost inaudible in the mix
  (both Demucs models return a silent bass stem there).
* **`--slap auto` (experimental)** guesses the sections from the attack
  sound (`slapdetect.py`, a small model in `slap_model.json`; leave-one-song-out
  on the 15 songs: slap F1 0.86–0.97 on Delfins, Katy Perry and Dark
  Necessities, 0.4–0.7 on Korn, Relax, Primus, Seinfeld, near zero on Can't
  Stop and Fly Away; on fingerstyle songs it wrongly marks 5–17 % of the
  song). It prints the sections it chose — treat them as a first draft, fix
  them (UI: *Edit these sections*) and re-run.
* **Octave pops** (pop an octave above the slap just played) are fretted with
  the octave shape — two strings up, two frets higher (slap E-3 → pop D-5,
  A-5 → G-7), as charters do for 99 % of them (395/398 in the 11 charts); the
  plain fretting matched 84 %, now 96 %.
* **Pops an octave low:** when the slapped root keeps ringing under the pop
  (octave patterns), the transcription often keeps the root's pitch — 68 % of
  the pops in *California Gurls*, ~15 % in *Can't Stop*. A spectral fix was
  tried and rejected (it broke as many notes as it fixed), so check octave
  pops when correcting the notes.
* **Pop vs slap from the sound:** the high octaves (C6–C8, the click you see
  in AnthemScore) were measured on the 11 songs: pops *and* slaps both have
  them (that is what `--slap auto` uses to find the slapped parts), but they
  don't tell a pop from a slap — leave-one-song-out AUC 0.37–0.79, versus
  0.76–1.00 for plain pitch, which is the rule used.
* **Finished charts (`chart_tools.py`)** — the most useful today: after you've
  corrected the notes, mark slap/pop in EOF's XML in one go instead of note
  by note:

  ```bash
  python plugins/stems2midi/chart_tools.py slap "PART REAL_BASS_RS2.xml" --sections "0:21-0:42, 1:26-1:38, 2:46-3:25"
  ```

  writes `PART REAL_BASS_RS2.slap.xml` (all difficulty levels, existing marks
  kept, `--clear` to redo, `--include-legato` if your charts slap hammer-ons
  too). Against the charters' own marks: Delfins slap F1 0.95 / pop 0.91,
  Dirty Loops 0.85 / 0.77, Primus 0.71 / 0.78 (Primus also slaps the D
  string). Import it back in EOF (File → Import → Rocksmith) or take it to
  DLC Builder.

## Benchmark against your own charts (`benchmark.py`)

Put each finished song in its own folder with the Rocksmith bass XML from EOF
(`PART REAL_BASS_RS2.xml`) and the audio it is synced to (`guitar.ogg`), then:

```bash
python plugins/stems2midi/benchmark.py "C:\...\myCDLCs\Work"
python plugins/stems2midi/benchmark.py WORK --only Delfins --model htdemucs_ft
python plugins/stems2midi/benchmark.py WORK --reuse-stems      # re-transcribe, don't separate again
python plugins/stems2midi/benchmark.py WORK --slap-from-chart  # also score slap/pop (sections from each chart)
```

It writes `report.md` / `report.json` (plus each song's `bass.mid`,
`bass.tab.txt`, `bass.notes.json`) to `WORK/_stems2midi_benchmark/`.
Fret-hand mutes are left out (no pitch). **Note-sequence F1** matches same
pitch in the same order within ±0.3 s, so it doesn't depend on how well the
chart itself is synced; **strict F1** needs ±50 ms. The report also gives the
chart's own sync error vs. the detected attacks, string/fret agreement of
`fretting.py` with your tab, beat-tracking accuracy and the techniques your
chart uses.

## Defaults (`stems2midi.ini`)

Any command-line option can get a default in `stems2midi.ini` — the Windows
installer writes one to `%APPDATA%\stems2midi\stems2midi.ini` (Linux/macOS:
`~/.config/stems2midi/stems2midi.ini`; also read next to the plugin folder and
from `$STEMS2MIDI_CONFIG`). The command line always wins.

```ini
[defaults]
; seconds of silence EOF adds at the start of guitar.ogg — the Rocksmith XML
; is moved by this much (same as --shift); also used by the FeedBack plugin
lead_silence = 5
;tuning = auto
;model = htdemucs_6s
;language = pt            ; lyrics language (default: guessed from the text)
;phrase_bars = 4          ; Rocksmith phrase (Riff Repeater unit) length; 0 = one per section
;charter = ALSRC          ; author of the DLC Builder project (and its DLC key)
;fetch_meta = yes         ; look up album art + lyrics on the web (iTunes, LRCLIB)
;dlc_project = yes        ; write Artist_Title.rs2dlc + song.ogg next to the XML

[feedback]
; where the installer copies the plugin
plugins_dir = C:\Program Files\feedback\current\resources\slopsmith\plugins
```

The plugin screen shows which file it read (Engine panel).

## Song parts from the start (`--parts`)

```bash
python stems2midi.py "Band - Song.mp3" --parts "intro=0:00-0:21, verse=0:21-0:42, chorus=0:42-1:05" --slap verse,chorus
```

**`--parts auto`** finds the parts in the audio (`structure.py`: bar-by-bar
harmony and timbre, recurrence + Laplacian spectral clustering — McFee & Ellis
2014) and numbers them: `intro 1 (A) 0:01 · verse 1 (B) 0:37 · chorus 1 (C)
1:06 · verse 2 (B) 1:37 …` — the same letter means the same music. Names are a
guess (loudest repeated part → chorus, the other main repeated part → verse, a
part that always comes right before the chorus → prechorus, one-offs before /
after → intro / outro, in the middle → bridge). The full list is printed (UI:
*Edit these parts* copies it into the field) so you can fix the names and run
again. Against 7 charted songs: boundaries within ±2.5 s F 0.62, same-part
agreement 0.67, names right on 52 % of the bars (Can't Stop 96 %, Fly Away
76 %, Word Up 72 %, California Gurls 60 %; songs whose parts differ mostly in
the vocals score lower). `--slap verse,chorus` works with `auto` too.

`--parts` (UI: *Song parts*) names the first occurrence of each part (times
in the audio file you give); the repeats are found by their notes and the
XML gets the Rocksmith sections/phrases, as `chart_tools.py structure` does.
`--slap` then also takes part names: `--slap verse,chorus` slaps every verse
and chorus. A result folder can be redone the same way:
`stems2midi.py "Band - Song_stems" --parts "…" --slap verse`.

## Songs where the bass stem holds more than the bass (`--only`, `--sustained`)

Demucs puts *every* low instrument in the bass stem: orchestral cellos and
double basses, piano left hand, synth bass. In Queen's *Who Wants to Live
Forever* the bass guitar only comes in at 2:50 — before that the "bass" is the
orchestra (447 of 738 notes). Two options:

* `--only "2:50-5:20"` (times in the song, or bar numbers as in EOF:
  `--only "40-90"`) — notes elsewhere are dropped. On a result folder it works
  on the existing transcription (the full one is kept in
  `bass.notes.full.json`). UI: *Bass plays only in*.
* `--sustained` — ballads, long held notes over strings / pads: a repeated
  note counts only when the level clearly rises again (≥ 4 dB), and double
  triggers (a very short note and the same pitch right after) are merged. On
  that song: 291 → 87 notes after 2:50. Not the default: on the benchmark songs
  it costs a little on fast repeated notes. UI: *long held notes (ballad)*.

Slow songs may also get a doubled tempo (143.6 instead of ~72 BPM): the bar
lines come out half as long — `--bpm 72`.

## Second pass: options worth a re-run (`suggest.py`)

After the first pass the tool looks at what it found and prints the options
worth trying, with the reason, in the warnings and at the end of
`bass.review.txt`, plus one command that re-runs on the result folder (its
bass stem — no new separation; the beats, slap sections, parts, `--only`,
`--sustained` and a set `--bpm` of the earlier run are kept):

| Hint | When | Fired on |
|---|---|---|
| `--sustained` | ≥ 25 % of the notes start right after the same pitch with no dip in level (a held note cut in pieces) | Queen 44 %; others ≤ 17 % |
| `--only "1:40-4:32"` | ≥ 10 % of the notes (≥ 20, ≥ 20 s) come from stretches where the stem is 12+ dB under its loud part | Queen 42 %; others ≤ 1 % |
| `--bpm` half | tempo ≥ 135 with the held-notes hint | Queen (143.6 → 71.8) |
| `--bpm` double | tempo ≤ 85 with a note every ≤ 0.3 beat | Xutos *À minha maneira* (80.7 → 159) |
| `--notes-from` | the review finds most bars doubtful (+ how weak the fundamental is) | Bandeira (fundamental 9 dB under the harmonics) |
| `--model htdemucs_ft` | the bass stem's low end (40–250 Hz) is 12+ dB under the song's in its best quarter of 4 s windows — the separation put the bass elsewhere (a thin / high bass taken for guitar by the 6-stem model). Already on `htdemucs_ft`: `--skip-separation` with a stem from another separator. Runs the song again (new separation) | Delfins *Lugar ao Sol* (29.7 dB; a web separator's stem: 4.2); Bandeira 4.7 |

None fired on the other benchmark songs. They are hints, not automatic
changes: each option also hurts some songs (e.g. a 3 dB re-pluck rule gains
on Delfins and loses 3 points on Manic). Choosing between two transcriptions
bar by bar with the review score was tried and gained nothing (≤ 0.001 F1 on
7 songs): the review spots doubtful bars but can't tell which of two
plausible readings is right. `--no-suggest` turns the hints off; a 5-string
bass is already suggested by the tuning check.

## Notes from another transcriber (`--notes-from`, `notesimport.py`)

Some bass sounds are hard for the built-in tracker. When the bass has almost
no fundamental — thin mixes: in Delfins' *A Bandeira* the low E is 20–35 dB
below its 3rd harmonic (B2) in the stem — it locks on the harmonic (B2/C3
instead of E1) or hears no note at all, and notes go missing. A neural
transcriber such as **AnthemScore** copes better, but its MusicXML / MIDI sits
on a fixed tempo grid that drifts away from the band. `--notes-from` takes those
notes (pitches as they are) and puts them on the song's time:

1. global fit — tempo ratio and offset with the most notes on an attack of the
   bass stem; bass lines repeat, so fits one or two beats off score almost as
   well: the candidates are told apart by how well the stem sounds those
   pitches (or their harmonics) at those times. Files written at half / double
   (or 3:2) the song's tempo — common in hand-made tabs — are tried when the
   file's length says so, and when fewer than half the notes land on an
   attack every ratio from 0.4 to 2.6 is (a tab typed in at 100 BPM for a
   57.5 BPM song: Plutónio *Meu Deus*, ×1.74);
   when even the best ratio leaves the stem not sounding the file's pitches
   (the band's tempo changes where the file's doesn't — Iron Maiden *Fear of
   the Dark*, tab at a fixed 75 BPM), the notes are lined up one by one with a
   quick transcription of the stem (pitch classes, semi-global alignment),
   and the result kept if it fits better;
2. segment re-fit — every 16 notes may move up to ±1 s (a cost for moving
   keeps repetitive lines from jumping a beat): follows the places where the
   band holds a beat longer than the file;
3. local warp — the median distance to the nearest attack, smoothed over
   ±8 s, follows the tempo drift;
4. every start snapped to the attack it sits on (±60 ms); overlaps cut;
5. every end trimmed to where the stem goes quiet (10 dB under the note's
   peak): notation programs write bass lines legato, filling the rests, and
   the chart would otherwise hold notes through silences.

On a result folder, the slap sections and song parts of the earlier run are
kept unless `--slap` / `--parts` are given.

**Only some bars from the file** — `--use`:

* `all` (default) — every note from the file;
* `doubtful` — the built-in transcription, and the bars the review flags
  (*likely wrong* / *worth a check*) taken from the file;
* `"12-30, 45-60"` (bar numbers as in EOF) or `"1:02-1:30"` (times as in EOF,
  i.e. with the lead-in) — those bars from the file.

The result says which bars came from the file. On stems the review can hardly
read (no fundamental — *A Bandeira*) both versions score alike, so `all` is
usually better there; `doubtful` helps where the built-in transcription is
mostly right. UI: the select next to the file.

Then the usual: beat map, strings/frets, slap/pop, song parts, phrases,
Rocksmith XML, review.

```bash
python stems2midi.py "Delfins - Bandeira.mp3" --notes-from "Bandeira (AnthemScore).musicxml"
python stems2midi.py "Delfins - Bandeira_stems" --notes-from bandeira.mid     # on an earlier result
```

UI: *Bass notes from another transcriber* (file) in the form. The result says
how many notes fell on an attack and the tempo difference (*A Bandeira*:
749 notes, 80 % on an attack, the file 1.5 % slower than the song,
+2.63 s), and the stretches where the file's notes mostly miss the attacks (a
part missing, repeated or played differently in the file). Accepted:
`.musicxml`, `.xml`, `.mxl`, `.mid`, and Guitar Pro (below).

### Guitar Pro tabs (`.gp3` `.gp4` `.gp5`, `.gpx` `.gp`)

A tab of the song (Ultimate Guitar, Songsterr exports, your own) given to
`--notes-from` is read with FeedBack's Guitar Pro reader (`lib/gp2rs.py`, the
GP import's; `.gpx`/`.gp` through `lib/gp2rs_gpx.py`; the installer ships both
in `vendor/`; needs PyGuitarPro) and lined up the same way. On top of the
notes it keeps what the tab says:

* **the bass track** — the first track marked as bass (MIDI program 32–39 or
  a bass tuning, but not one named strings / synth / vocals… — MIDI-made
  files put those on low tunings; a name with bass/baixo breaks ties); a file
  with no bass track says so and lists the tracks; `--track 2` or
  `--track "Bass 2"` for another (UI: the box under the file);
* **as performed** — repeats, voltas, D.S./D.C./Coda/Fine expanded, tempo
  changes followed (`.gpx`/`.gp`: FeedBack's GP6–8 reader plays repeats
  once — the tool warns; save as `.gp5` then);
* **the tuning** (with *Bass tuning: auto*: 5-string, drop D, E♭… as in the
  tab) and **each note's string and fret** — the tab's fingering instead of
  the tool's;
* **slap / pop** and **hammer-on / pull-off / slide** marks;
* dead notes help the fit and are then left out (no pitch);
* a bass tabbed on a guitar-tuned track (E2 A2 D3 G3 B3 E4) is moved an
  octave down and fretted by the tool.

Measured on the five songs with a tab and a hand-made chart (bass stem from
the MP3; aligned F1 vs. the chart, strings/frets in agreement):

| Song | Tab | Lined up | F1 | Strings/frets |
|---|---|---|---|---|
| Delfins — Canção de Engate | 110 BPM | +5.83 s | 0.998 | 100 % |
| Manic — La Tristesse Durera | 98 BPM, 142 dead notes | +3.27 s | 0.958 | 100 % |
| Phil Collins — Do You Remember | 93 BPM | +3.29 s | 0.972 | 100 % |
| Xutos — À minha maneira | 158 BPM | +13.35 s | 0.924 | 100 % |
| Xutos — Circo de feras | 187 BPM for a 96 BPM song | ×1.94, +5.07 s | 0.982 | 100 % |

(The built-in transcription on the same stems: 0.79–0.86.) Without the
segment re-fit Delfins was 0.853 (the band drifts a beat's worth after 2:50);
without the double-time search Circo was 0.17. The AnthemScore MusicXML of
Circo de feras gains the same way (0.17 → 0.98).

```bash
python stems2midi.py "Xutos & Pontapés - Circo de feras.mp3" --notes-from "Circo de feras.gp5"
python stems2midi.py "Circo de feras_stems" --notes-from "Circo de feras.gp5" --track 2
```

When most bars of a song look doubtful to the review (a stem like this), it
ranks the bars within the song instead (the worst 15 % = *likely wrong*, the
next 15 % = *worth a check*) and says so.

## Review against the stem (`review.py`)

After transcribing, every bar is checked against the bass stem and the bars
most likely to have mistakes are listed, so the review in EOF can start there
instead of listening to the whole song. Per bar it measures:

* **attacks with no note** — clear attacks in the stem with no note starting
  there (often pops / ghost notes the pitch tracker heard as unpitched, or fast
  runs);
* **notes without a clear attack** — possible ghosts of a held note;
* **pitch not clearly in the stem** / **a different note sounds stronger** /
  **octave below/above sounds stronger** — wrong note or wrong octave;
* **bass sounds where there is no note** — missing or too-short notes;
* **bass very quiet** — quiet bars are guesswork (fades, breaks).

A small model (`review_model.json`, trained on 15 hand-made charts) turns these
into the chance the bar has mistakes: **bad** (red) and **check** (amber).
Tested leave-one-song-out: the bars flagged as most suspect (top quarter) had
mistakes 83 % of the time, against 52 % for the least suspect quarter
(AUC 0.71; 0.6–0.86 on most songs, weaker on very busy or synth bass).

Where to see it:

* `bass.review.txt` in the result folder — bars sorted most suspect first,
  numbered like the bars in EOF, with the time in the song and in EOF
  (`[…]`, when `lead_silence` / `--shift` is set);
* the piano roll in FeedBack — red/amber bands over the bars, and a
  *Bars to review* list under it (click a bar to hear the stem there);
* `notes.json` → `review`.

`--no-review` skips it (it takes a few seconds).

## Lyrics → UltraStar Creator / EOF / Rocksmith (`lyrics.py`)

Give the lyrics as plain text — **one sung line per line, every repeat written
out** (no `[Chorus] x2`; lines like `[Chorus]` or `(x2)` are ignored) — and the
tool lines every syllable up with the vocals stem:

```bash
python stems2midi.py "Band - Song.mp3" --lyrics letra.txt          # with the transcription
python stems2midi.py "Band - Song_stems" --lyrics letra.txt         # later, on a result folder
python lyrics.py "Band - Song_stems" --lyrics letra.txt --language pt  # same, lyrics only
```

In FeedBack: paste them in *Lyrics* before starting, or later under the result
(*Lyrics → Paste the lyrics…*; needs the vocals stem, which a run with lyrics
always saves).

You get, in the result folder:

* `vocals.ultrastar.txt` — UltraStar song (`#BPM:400`, `#GAP` = first
  syllable, line breaks from your lines, the sung pitch per syllable). Open it
  in **UltraStar Creator** to fine-tune, or import it in **EOF** (PART VOCALS
  track → File › Import › Lyrics). With a lead-in (`lead_silence` / `--shift`)
  the times follow EOF's `guitar.ogg` and `#MP3:guitar.ogg`, like the files you
  make by hand; without one, the song's own audio.
* `vocals_RS2.xml` — the Rocksmith 2014 vocals arrangement (`-` = the word goes
  on, `+` = end of line, like EOF's `PART VOCALS_RS2.xml`) for DLC Builder.
* both files are written **without accents** (ã → a, ç → c, é → e; curly
  quotes and long dashes made plain): Rocksmith's lyric font, EOF and some
  UltraStar tools only take plain letters. `--keep-accents` (or
  `keep_accents = yes` in `stems2midi.ini`) keeps them;
* `vocals.lyrics.json` — the timed syllables (as given, with accents); `lyrics.py <folder> --shift 5`
  (no `--lyrics`) writes the two files again with another shift / names.

An existing **UltraStar .txt** can be given as the lyrics too: its syllables,
words and line breaks are kept and only the times are redone (useful to
re-time a file you split by hand, or one made for another version of the song).

Syllables are split with the `pyphen` hyphenation dictionaries (pt, en, es,
fr, de, it…; installed by the installer) or, without it, simple rules. To
control the split, write it yourself: `un-faith-ful`, `can-ção`.

Two aligners:

* **mms** (default when it loads) — torchaudio's MMS forced aligner, a speech
  model that knows the letters of 1100+ languages. Every syllable is placed;
  a "anything" token between lines absorbs ad-libs, backing vocals and solos.
  Downloads ~1.2 GB once (to `%USERPROFILE%\.cache\torch\hub`); on the GPU it
  takes seconds, on CPU a minute or two.
* **simple** — no model: lyric lines are matched to the sung phrases (pauses
  in the vocals stem), then syllables to the attacks inside each line. Line
  starts are usually right (on a 4-minute test song most lines within 0.3 s),
  words inside lines are rough, and songs whose lines repeat (same chorus line
  four times) can slip by a line — check in UltraStar Creator.

To measure it against a song you already did by hand:
`lyrics.py "<folder>" --lyrics letra.txt --compare "PART VOCALS_RS2.xml"`
prints how far line and word starts are from yours.

## Workflow with EOF (Editor on Fire)

1. Transcribe the song (with a bass tuning; give *Song parts* and *Slap
   sections* — times or part names — if you know them). You get `bass_RS2.xml`
   and `bass.review.txt` (bars to check first).
2. If the EOF project's `guitar.ogg` has EOF's lead-in silence, set
   `lead_silence` in `stems2midi.ini` once, or line the XML up with it: UI *Rocksmith XML … line it up with EOF's audio* (choose
   `guitar.ogg`), or `--align-to guitar.ogg` / `--shift <seconds>`.
3. In EOF, **select the PART REAL_BASS track first**, then File › Import ›
   Rocksmith. Import replaces the *active* track; on a track that isn't a pro
   bass/guitar track it silently does nothing.
4. The notes land in **difficulty 0** (the first difficulty tab), not Expert:
   EOF numbers the populated difficulties from 0 when it exports and reads them
   back the same way. Leave them there — the export works the same and DLC
   Builder makes the other levels.
5. The bar lines don't need fixing bar by bar: notes are placed in seconds,
   the beat map only drives the bar lines, phrase starts and EOF's grid. Fix it
   when the tempo was misread (e.g. half/double — re-run with `--bpm`) or bar 1
   is on the wrong beat. If you move beats, check that EOF didn't drag the
   notes along.
6. Sections and slap/pop: see below (`chart_tools.py structure`), or by hand in
   EOF:
   * slap in bulk: select the notes (drag, or Ctrl+A) → Note › Rocksmith ›
     Toggle › Slap (Ctrl+Shift+S);
   * then pops: Edit › Selection › Conditional deselect (Alt+D) to keep only
     the notes on D/G (the "lanes" are the strings) → Note › Rocksmith ›
     Remove › Slap, then Toggle › Pop (Ctrl+Shift+P);
   * sections: Beat › Rocksmith › Place RS Section (Shift+S) with "Also add as
     RS phrase".
7. Lyrics: import `vocals.ultrastar.txt` on the PART VOCALS track (File ›
   Import › Lyrics), or fix it in UltraStar Creator first; or give DLC Builder
   `vocals_RS2.xml` directly.
8. Export from EOF and build the psarc in DLC Builder as usual.

## Song parts and bulk slap/pop (`chart_tools.py structure`)

Name each part once — its first occurrence — and the repeats are found by
their notes (same pitch at the same beat position for ≥ 60 % of the notes,
bars from the XML's beat map):

```bash
python plugins/stems2midi/chart_tools.py structure "PART REAL_BASS_RS2.xml" \
    --define "intro=0:05-0:21, verse=0:21-0:42, chorus=0:42-1:05" --copy-tech
```

* writes Rocksmith **sections + phrases** (verse 1/2/3…, phrases verse1,
  verse2…; stretches in between become `riff` or `noguitar`) — names must be
  Rocksmith ones (intro, verse, prechorus, chorus, bridge, solo, outro, riff…);
* `--copy-tech`: slap/pop and hammer-on/pull-off of each **first** occurrence
  are copied to the matching notes of its repeats — mark one verse by hand
  in EOF, export, run this, import again;
* `--slap verse,chorus`: mark slap (E/A) / pop (D/G) in every occurrence of
  those parts at once, then fix the few exceptions in EOF;
* prints what it found, with how alike each repeat is (`verse (67%) 1:35`).

Works on EOF's export or on `bass_RS2.xml`; the result is
`<name>.structure.xml` (import with File → Import → Rocksmith, PART REAL_BASS
selected). Plain Python, no extra packages; installer: `chart_tools.bat`.

In EOF itself: select notes (drag, or Ctrl+A) → Note › Rocksmith › Toggle ›
Slap (Ctrl+Shift+S) / Pop (Ctrl+Shift+P); Edit › Selection › Conditional
deselect (Alt+D) keeps only notes on chosen strings; Beat › Rocksmith › Place
RS Section (Shift+S) places one section at a time.

## DLC Builder project, album art and lyrics (`dlcproject.py`, `metafetch.py`)

Next to `bass_RS2.xml` the tool writes a **DLC Builder project**,
`Artist_Title.rs2dlc`, and the song audio for it (`song.ogg`, or
`song_lead5s.ogg` with 5 s of silence added when the XML is moved for EOF's
lead-in): open the project in DLC Builder, add the tones, build. It holds:

* **Author** = your charter name (`--charter ALSRC`, or once in
  `stems2midi.ini`: `charter = ALSRC`; without it DLC Builder uses the one in
  its settings) and a **DLC key** made of it + artist + title
  (`ALSRCDelfinsLugarAoSol`).
* artist / title / album / year (sort fields in plain letters, a leading
  "The" moved off, and the charter name at the end of the title sort —
  DLC Builder names the .psarc from them: `Delfins_Lugar-ao-Sol-ALSRC_v1_p.psarc`;
  `chart_tools.py dlcfix` does the same to an XML saved by EOF), `cover.jpg` when
  there is one, and the arrangements in the folder: bass, the 5-string Rhythm
  XML, `vocals_RS2.xml`. Tuning and A440/A220 come from each XML.
* Paths relative to the project, so the folder can be moved. Running again on
  the folder (`--shift`, `--parts`, `--lyrics`…) **updates** the project and
  keeps what you set in DLC Builder (tones, preview, volumes, IDs, other
  arrangements). `--no-dlc-project` (ini `dlc_project = no`) turns it off.
* With `--skip-separation` the input is a stem, not the song: the audio is
  left for you to set in DLC Builder (or taken from an earlier `song.ogg`, or
  rebuilt from `bass` + `backing_no_bass` in the folder).

**`--fetch-meta`** (ini `fetch_meta = yes`, a checkbox on the plugin screen)
looks the song up by artist and title (from "Artist - Title.mp3", or
`--artist/--title`; track numbers, accents and "(Remastered)" don't matter):

* **album art** from iTunes Search (1000×1000), else MusicBrainz + Cover Art
  Archive → `cover.jpg` (DLC Builder converts it);
* **lyrics** from LRCLIB (lrclib.net, open lyrics database) → `lyrics.txt`
  (section headers out, one sung line per line) and `lyrics.lrc` when it has
  line timings. Without `--lyrics` they are lined up with the vocals as if
  given — check them; edit `lyrics.txt` and run on the folder with
  `--lyrics lyrics.txt` to redo;
* a missing album / year from the iTunes match.

Files already in the folder are kept. No network or no match is just a note.
Genius has no lyrics API and its terms don't allow fetching its pages, so it
isn't used.

## Rocksmith XML (`rsexport.py`)

Every transcription with a bass tab also writes **`bass_RS2.xml`**, a Rocksmith
2014 bass arrangement, next to `bass.mid` — no extra step. It is written with
FeedBack's own writer (`lib/gp2rs.py`, the Guitar Pro import's), found next to
the plugin (`plugins/stems2midi → ../../lib`), via `FEEDBACK_DIR`, or in the
copy the Windows installer ships (`vendor/gp2rs.py`); it runs in any Python —
no numpy, and PyGuitarPro isn't needed for writing.

* Artist / title come from the file name **"Artist - Title.mp3"**; set them
  (and album / year) with `--artist --title --album --year`. They are written
  **without accents** (Canção → Cancao): EOF garbles them ("CanAAo") and
  Rocksmith's font has no accented letters. `--keep-accents` keeps them;
  `chart_tools.py structure` does the same to EOF's XML.
* **DLC Builder names the .psarc** from artist / title (their sort fields). If
  the project went through EOF with accents in its song properties, EOF writes
  them garbled ("Pontapés" → "PontapAs", "Canção" → "CanAAo") and that ends up
  in the package name. Fix them in EOF (*Song > Properties*) or in DLC Builder,
  or run `chart_tools.py dlcfix "PART REAL_BASS_RS2.xml" --from-folder` (artist
  and title from the folder name "Artist - Title", without accents).
* **No hand-position change on a hammer-on / pull-off** (DLC Builder:
  *position shift into pull-off*): for each run of legato notes with an anchor
  change inside it, one anchor at the picked note covers the run (width up to
  6 frets) and the hand position after the run comes back at the next note; a
  run too wide for one hand position loses the hammer-on/pull-off where the
  hand moves. XMLs edited in EOF: `chart_tools.py dlcfix`.
* UI: **Rocksmith XML** button next to *Download bass.mid*; under the roll,
  *Rocksmith XML with other artist / title…* re-exports with other metadata.
* To redo only the XML of an earlier result (other metadata, or a result made
  before this existed), give the result folder instead of a song:

  ```bash
  python plugins/stems2midi/stems2midi.py "song_stems" --artist Delfins --title "Canção de Engate" --year 1996
  ```

  (Windows installer: drag the `<song>_stems` folder onto `stems2midi.bat`.)
  `--no-rs-xml` skips the file.
* **EOF's lead-in.** When EOF makes a project from an MP3 it adds silence at
  the start of `guitar.ogg` (3–10 s), so a chart made from the MP3 plays early.
  `--align-to guitar.ogg` (with a song or a result folder) measures the offset
  from the audio itself (low-band onset cross-correlation, ±20 ms on 5 test
  songs) and moves notes, beats and phrases; `--shift 5.0` does it by hand.
  UI: *Rocksmith XML: … line it up with EOF's audio* → choose `guitar.ogg`.
* **Sections and phrases.** Sections are the song parts (`--parts`, or a
  `riff` every 8 bars); inside them a **phrase every 4 bars** — in Rocksmith
  the phrase is what Riff Repeater loops (and what difficulty levels are built
  on), so a long verse is not one 30-second block. Stretches with the same
  notes get the same phrase name (`verse1`, `verse2`, `verse1`…), like
  hand-made charts. `--phrase-bars 2|8` or `0` (one phrase per section),
  `phrase_bars = …` in `stems2midi.ini`, or *phrases* in the UI's Rocksmith XML
  form; `chart_tools.py structure` does the same on EOF's XML.
* **5-string tabs.** Rocksmith 2014 only has 4-string bass, so a 5-string tab
  is exported the way custom-song authors chart it (DLC Builder otherwise says
  *Invalid strings used*): the 4 lowest strings, B E A D written like the
  5-string CDLC do (Miley Cyrus' *Midnight Sky*): offsets +7 +7 +7 +7 with
  `centOffset` −1200 (an octave down); notes played on the top string move to
  the string below, 5 frets up. The tab (`bass.tab.txt`, FeedBack) stays
  5-string. An XML made earlier: `chart_tools.py fourstring bass_RS2.xml`.
* **5-string part as Rhythm.** When the top string carries many notes (≥ 10
  and ≥ 5 %), `bass5_rhythm_RS2.xml` is written too: a *Rhythm* (guitar path)
  bonus arrangement an octave down (A220, `centOffset` −1200) tuned B E A D G
  (the unused 6th string also on G: offsets −5 −5 −5 −5 −4 −9) with
  `tonebase` bass — as in 5-string CDLC (Tears for Fears' *Raoul and the Kings
  of Spain*) — so a 5-string bass plays it string for string. Add it next to
  the bass in DLC Builder.
  `--rs-5string rhythm` always writes it, `bass` never (ini `rs_5string`);
  for an XML made earlier: `chart_tools.py rhythm5 bass_RS2.xml`.
* All times are written with 3 decimals, as Rocksmith/EOF expect — EOF reads
  the digits after the point as milliseconds (`0.314906` → 314.9 s), which
  made it drop every note of the first version of this export.

What it writes: beat map from the beats tracked on the mix (smoothed, re-phased
to the bass attacks, 4/4 with bar 1 where the bass hits hardest/lowest),
COUNT / `riff` phrase every 8 bars (`noguitar` on rests) / END, notes with
string/fret, sustain (≥ 0.25 s), hammer-on/pull-off on legato notes on the same
string, `slap="1"` / `pluck="1"`, anchors from the fretting, tuning,
arrangement properties. Note times are moved 15 ms earlier (charters place
notes a median 16 ms before our detected attacks). One difficulty level — DLC
Builder generates the rest.

Checked on 12 charted songs: the notes match the transcription exactly
(FeedBack reads the file back), and the beat map sits within ±50 ms of the
charter's beats on 94–99 % of beats for Delfins, Katy Perry, Korn and RHCP;
where it doesn't, the tempo was misread (Fly Away: 107 vs 80 BPM — use
`--bpm`) or the chart itself is offset (Manic, Phil Collins ≈ 0.1 s). The
workflow: **import the XML in EOF** (File → Import → Rocksmith), fix notes and
beat map, export, then DLC Builder → psarc.

## In FeedBack

Sidebar → **Stems → MIDI**. Drop a file, pick stems and options, and press
**Separate & transcribe**. Jobs run one at a time in a separate process (the
server stays responsive; a crash or out-of-memory can't take it down) and keep
running if you leave the screen. Each result shows:

* a **piano roll** of the MIDI (click to play from that point),
* **▶ Hear MIDI + backing** — a quick synth render of the MIDI over the
  bass-less backing track, the fastest way to judge the transcription by ear,
* players and download links for every stem, `bass.mid`, and a zip of everything,
* with a bass tuning: the tab, slap/pop markers and the **Rocksmith XML** export.

Jobs and outputs live in `<config>/stems2midi/jobs/` (last 30 kept).

## Tests

```bash
pytest plugins/stems2midi/tests -v
```

Logic tests need only numpy; transcription tests need librosa; route tests use
a fake CLI, so nothing downloads model weights.
