"""Lyrics → syllables → UltraStar / Rocksmith vocals."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s2m_lyrics", HERE / "lyrics.py")
lyrics = importlib.util.module_from_spec(spec)
sys.modules["s2m_lyrics"] = lyrics          # dataclasses look the module up
spec.loader.exec_module(lyrics)


def test_parse_lines_skips_tags_and_keeps_hyphens():
    lyr = lyrics.parse_lyrics("[Chorus]\nLife has been un-faith-ful\n\n(x2)\nAnd it all\n", "en")
    assert len(lyr.lines) == 2
    assert lyr.lines[0][3] == ["un", "faith", "ful"]
    syl = lyr.syllables()
    assert [s.text for s in syl][:6] == ["Life", "has", "been", "un", "faith", "ful"]
    assert syl[3].first and not syl[3].last and syl[5].last
    assert syl[6].line == 1 and syl[6].first


def test_language_guess():
    assert lyrics.guess_language("Não sei o que fazer com a minha canção") == "pt"
    assert lyrics.guess_language("I am just a petrified cry and the rain") == "en"


def test_portuguese_rules_without_pyphen():
    split = lambda w: lyrics._split_rule(w, "pt")
    assert split("canção") == ["can", "ção"]
    assert split("bandeira") == ["ban", "dei", "ra"]
    assert split("guitarra") == ["gui", "tar", "ra"]
    assert split("minha") == ["mi", "nha"]
    assert split("quero") == ["que", "ro"]
    assert "".join(split("pássaro")) == "pássaro"


def test_english_rules_without_pyphen():
    split = lambda w: lyrics._split_rule(w, "en")
    assert split("Life") == ["Life"] and split("time") == ["time"] and split("smile") == ["smile"]
    assert split("unfaithful") == ["un", "faith", "ful"] and split("table") == ["ta", "ble"]
    assert split("loved") == ["loved"] and split("wanted") == ["wan", "ted"]
    assert split("sigh,") == ["sigh,"] and "".join(split("relic.")) == "relic."


def test_punctuation_stays_on_the_word():
    syl = lyrics.syllabify("sigh,", "en")
    assert "".join(syl) == "sigh,"
    assert lyrics.normalize("Canção,") == "cancao"


def _timed(text="Life has been un-faith-ful\nAnd it all pro-mised"):
    syl = lyrics.parse_lyrics(text, "en").syllables()
    t = 10.0
    for i, s in enumerate(syl):
        if i and s.line != syl[i - 1].line:
            t += 2.0
        s.t, s.end, t = t, t + 0.3, t + 0.4
    return syl


def test_ultrastar_file():
    txt = lyrics.ultrastar(_timed(), {"artist": "A", "title": "T", "language": "en"}, shift=5)
    assert txt.endswith("E\r\n") and "\r\n" in txt
    lines = txt.split("\r\n")
    assert "#MP3:guitar.ogg" in lines and "#BPM:400" in lines and "#GAP:15000" in lines
    notes = [ln for ln in lines if ln.startswith(":")]
    assert notes[0] == ": 0 8 0 Life "                      # 0.3 s = 8 beats of 37.5 ms
    assert notes[3].endswith(" un") and notes[5].endswith(" ful ")   # word end = trailing space
    brk = [ln for ln in lines if ln.startswith("- ")]
    assert len(brk) == 1
    b = int(brk[0].split()[1])
    assert int(notes[5].split()[1]) < b < int(notes[6].split()[1])
    starts = [int(n.split()[1]) for n in notes]
    assert starts == sorted(starts)


def test_vocals_xml_marks_words_and_lines():
    xml = lyrics.vocals_xml(_timed(), shift=5)
    assert 'count="11"' in xml
    assert 'time="15.000"' in xml and 'lyric="un-"' in xml and 'lyric="ful+"' in xml
    assert 'lyric="mised+"' in xml


def test_outputs_written_again_with_another_shift(tmp_path):
    syl = _timed()
    lyr = lyrics.parse_lyrics("Life has been un-faith-ful\nAnd it all pro-mised", "en")
    lyrics.save(tmp_path, lyr, syl, "simple", "x")
    info = lyrics.write_outputs(tmp_path, {"artist": "A", "title": "T"}, shift=0)
    assert info["syllables"] == 11 and info["lines"] == 2
    assert "#GAP:10000" in (tmp_path / info["ultrastar"]).read_text(encoding="utf-8")
    lyrics.write_outputs(tmp_path, {}, shift=5)
    assert "#GAP:15000" in (tmp_path / lyrics.US_NAME).read_text(encoding="utf-8")
    ref = lyrics.read_vocals_xml(tmp_path / lyrics.XML_FILE)
    assert ref[0] == (15.0, "Life")
    rep = lyrics.compare(syl, ref, shift=5)
    assert "line starts (2): median error 0.00 s" in rep


def _sung(lines_syl, sr=16000):
    """Synthetic singing: a vowel-like tone per syllable, a pause between lines."""
    t, y, truth = 1.0, np.zeros(int(sr * 40), np.float32), []
    tt = np.arange(int(sr * 0.32)) / sr
    for li, n in enumerate(lines_syl):
        truth.append(t)
        for k in range(n):
            f = 196 * 2 ** ((k % 5) / 12)
            tone = sum(np.sin(2 * np.pi * f * h * tt) / h for h in (1, 2, 3, 4))
            env = np.minimum(1, tt / 0.02) * np.minimum(1, (tt[-1] - tt) / 0.04)
            s = int(t * sr)
            y[s:s + len(tt)] += 0.3 * (tone * env).astype(np.float32)
            t += 0.38
        t += 1.6 + 0.4 * (li % 2)
    return y, truth


def test_simple_aligner_finds_line_starts():
    pytest.importorskip("librosa")
    text = "one two three four\nfive six seven\neight nine ten eleven twelve\nthir teen four"
    y, truth = _sung([4, 3, 5, 3])
    lyr, syl, used = lyrics.run(y, 16000, text, "en", aligner="simple", pitch=False)
    assert used == "simple"
    starts = [s.t for i, s in enumerate(syl) if i == 0 or s.line != syl[i - 1].line]
    assert np.max(np.abs(np.array(starts) - truth)) < 0.12
    assert all(b.t > a.t for a, b in zip(syl, syl[1:]))


def test_mms_code_path_with_a_tiny_untrained_model():
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    try:
        from torchaudio.pipelines._wav2vec2 import utils as U
    except ImportError:
        pytest.skip("torchaudio internals changed")
    if not lyrics.mms_available():
        pytest.skip("no MMS_FA in this torchaudio")
    b = torchaudio.pipelines.MMS_FA
    prm = dict(b._params, encoder_num_layers=1, encoder_embed_dim=32, encoder_num_heads=2,
               encoder_ff_interm_features=64, encoder_pos_conv_kernel=8, encoder_pos_conv_groups=2)
    torch.manual_seed(0)
    m = U._extend_model(U._get_model(b._model_type, prm), normalize_waveform=True,
                        apply_log_softmax=True, append_star=True).eval()
    syl = lyrics.parse_lyrics("Can-ção de en-ga-te\nmais uma", "pt").syllables()
    y = np.random.default_rng(0).normal(0, 0.1, 16000 * 25).astype(np.float32)
    lyrics.align_mms(y, syl, model=m)
    assert all(b.t >= a.t for a, b in zip(syl, syl[1:]))
    assert all(s.end > s.t for s in syl)


def test_ultrastar_file_as_lyrics():
    us = ("#TITLE:T\r\n#BPM:398\r\n#GAP:13080\r\n: 0 2 0 Life \r\n: 38 2 0 has \r\n: 56 1 0 un\r\n"
          ": 68 1 0 faith\r\n: 70 1 0 ~\r\n: 82 1 0 ful\r\n- 105\r\n: 128 1 0 And \r\n* 137 1 0 it \r\nE\r\n")
    assert lyrics.is_ultrastar(us) and not lyrics.is_ultrastar("Life has been\nunfaithful\n")
    lyr = lyrics.parse_lyrics(us)
    assert lyr.lang == "en"
    assert lyr.lines == [[["Life"], ["has"], ["un", "faith", "ful"]], [["And"], ["it"]]]


def test_line_alignment_never_gives_up():
    ph = [(1.0 + 3 * i, 2.5 + 3 * i) for i in range(4)]          # 4 sung phrases
    blocks = lyrics.align_lines(ph, [3, 3, 3, 3, 3, 3], rate=0.4, max_lines=3)
    assert [b[0] for b in blocks] == sorted(b[0] for b in blocks)
    with pytest.raises(ValueError):
        lyrics.align_lines(ph[:1], [3] * 9, rate=0.4, max_lines=3)
    spans = lyrics._spread_lines(ph[:1], [3] * 9)
    assert len(spans) == 9 and spans[0][0] == 1.0 and all(a < b for a, b in spans)


def test_compare_reports_a_constant_offset():
    syl = _timed()
    ref = [(s.t + 2.0, s.text + ("" if s.last else "-") + ("+" if i in (5, 10) else ""))
           for i, s in enumerate(syl)]
    rep = lyrics.compare(syl, ref)
    assert "2.00 s earlier than the reference" in rep and "without that offset: median 0.00 s" in rep


def test_mms_problem_is_explained():
    why = lyrics.mms_problem()
    assert isinstance(why, str) and (why == "") == lyrics.mms_available()


def test_numpy_ctc_alignment_matches_torchaudio():
    torch = pytest.importorskip("torch")
    AF = pytest.importorskip("torchaudio.functional")
    if not hasattr(AF, "forced_align"):
        pytest.skip("no torchaudio forced_align to compare with")
    rng = np.random.default_rng(3)
    T, C = 400, 29
    logits = rng.normal(0, 1, (T, C)).astype(np.float32)
    tokens = [28, 5, 7, 7, 3, 28, 9, 2, 28]          # repeated letter and "*" tokens
    for k, tok in enumerate(tokens):                    # make each token likely somewhere in order
        logits[20 + 40 * k: 30 + 40 * k, tok] += 6
    logp = torch.log_softmax(torch.from_numpy(logits), -1)
    ali, sc = AF.forced_align(logp.unsqueeze(0), torch.tensor([tokens], dtype=torch.int32), blank=0)
    ref = [(s.start, s.end) for s in AF.merge_tokens(ali[0], sc[0].exp())]
    got = [(a, b) for a, b, _p in lyrics.ctc_align(logp.numpy(), tokens)]
    assert got == ref


def test_silence_mask_keeps_lines_out_of_a_solo():
    T, C = 300, 6
    star, letters = 5, [1, 2, 3]
    logp = np.full((T, C), -6.0, np.float32)
    logp[:, 0] = -0.1                       # blank everywhere
    logp[:, star] = 0.0                     # "anything" is free
    for a in (40, 200):                     # the line fits equally well twice …
        for k, tok in enumerate(letters):
            logp[a + 5 * k: a + 5 * k + 4, tok] = -0.2
    logp[40:60, letters] -= 0.05            # … the first place a bit worse
    tokens = [star] + letters + [star]
    free = lyrics.ctc_align(logp, tokens)
    assert free[1][0] >= 200                # unconstrained: the later place
    voiced = np.zeros(T, bool)
    voiced[30:80] = True                    # but only the first place has voice
    masked = lyrics.ctc_align(lyrics.mask_silence(logp, voiced, letters), tokens)
    assert 40 <= masked[1][0] < 60


def test_lyrics_files_without_accents(tmp_path):
    assert lyrics.to_ascii("Canção, não… «olá» — ÁGUA ß") == 'Cancao, nao... "ola" - AGUA ss'
    lyr = lyrics.parse_lyrics("Can-ção de en-ga-te\nnão há", "pt")
    syl = lyr.syllables()
    for i, s in enumerate(syl):
        s.t, s.end = 1 + i * 0.4, 1.3 + i * 0.4
    lyrics.save(tmp_path, lyr, syl, "simple", "x")
    lyrics.write_outputs(tmp_path, {"title": "Canção de Engate", "artist": "Delfins"})
    us = (tmp_path / lyrics.US_NAME).read_text(encoding="utf-8")
    xml = (tmp_path / lyrics.XML_FILE).read_text(encoding="utf-8")
    assert "#TITLE:Cancao de Engate" in us and " cao " in us and "nao" in us and "ç" not in us + xml
    assert 'lyric="cao"' in xml or 'lyric="cao-"' in xml
    assert all(ord(c) < 128 for c in us + xml)
    lyrics.write_outputs(tmp_path, {"title": "Canção"}, keep_accents=True)
    assert "ção" in (tmp_path / lyrics.US_NAME).read_text(encoding="utf-8")
