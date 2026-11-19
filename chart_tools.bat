@echo off
rem chart_tools - editar um XML do Rocksmith (exportado pelo EOF ou pela ferramenta).
rem   chart_tools.bat structure "PART REAL_BASS_RS2.xml" --define "verse=0:21-0:42, chorus=0:42-1:05" --copy-tech
rem   chart_tools.bat slap "PART REAL_BASS_RS2.xml" --sections "0:21-0:42"
setlocal
set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"
rem pasta sem .venv (zip novo extraido noutro sitio): a instalacao indicada no stems2midi.ini ou STEMS2MIDI_PYTHON
if not exist "%PY%" for /f "usebackq tokens=1,* delims==" %%a in (`findstr /b /i /c:"install_dir" "%APPDATA%\stems2midi\stems2midi.ini" 2^>nul`) do for /f "tokens=* delims= " %%c in ("%%b") do if exist "%%c\.venv\Scripts\python.exe" set "PY=%%c\.venv\Scripts\python.exe"
if not exist "%PY%" if defined STEMS2MIDI_PYTHON if exist "%STEMS2MIDI_PYTHON%" set "PY=%STEMS2MIDI_PYTHON%"
if not exist "%PY%" set "PY=python"
rem a linha de comandos tal como foi escrita (caminhos com espacos / barra final)
set S2M_RAW_ARGS=%*
"%PY%" "%HERE%stems2midi\chart_tools.py" %*
exit /b %errorlevel%
