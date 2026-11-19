@echo off
rem lyrics - alinhar a letra com o stem da voz -> UltraStar .txt + XML de voz do Rocksmith.
rem   lyrics.bat "Banda - Musica_stems" --lyrics letra.txt
rem   lyrics.bat "Banda - Musica_stems" --lyrics letra.txt --compare "PART VOCALS_RS2.xml"
rem   lyrics.bat "Banda - Musica_stems" --shift 5      (so escrever de novo com outro desvio)
setlocal
set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"
rem pasta sem .venv (zip novo extraido noutro sitio): a instalacao indicada no stems2midi.ini ou STEMS2MIDI_PYTHON
if not exist "%PY%" for /f "usebackq tokens=1,* delims==" %%a in (`findstr /b /i /c:"install_dir" "%APPDATA%\stems2midi\stems2midi.ini" 2^>nul`) do for /f "tokens=* delims= " %%c in ("%%b") do if exist "%%c\.venv\Scripts\python.exe" set "PY=%%c\.venv\Scripts\python.exe"
if not exist "%PY%" if defined STEMS2MIDI_PYTHON if exist "%STEMS2MIDI_PYTHON%" set "PY=%STEMS2MIDI_PYTHON%"
if not exist "%PY%" set "PY=python"
rem a linha de comandos tal como foi escrita (caminhos com espacos / barra final)
set S2M_RAW_ARGS=%*
"%PY%" "%HERE%stems2midi\lyrics.py" %*
exit /b %errorlevel%
