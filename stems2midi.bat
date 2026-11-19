@echo off
rem stems2midi - separa stems e gera o MIDI do baixo.
rem   * Arraste um ou mais ficheiros de audio para cima deste ficheiro, ou
rem   * arraste uma pasta "<musica>_stems" para (re)fazer so o XML do Rocksmith, ou
rem   * stems2midi.bat musica.mp3 [opcoes]     (stems2midi.bat --help)
rem Opcoes por omissao no modo "arrastar": variavel STEMS2MIDI_ARGS,
rem   ex.:  set STEMS2MIDI_ARGS=--stems all --quantize 16
setlocal
set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"
rem pasta sem .venv (zip novo extraido noutro sitio): a instalacao indicada no stems2midi.ini ou STEMS2MIDI_PYTHON
if not exist "%PY%" for /f "usebackq tokens=1,* delims==" %%a in (`findstr /b /i /c:"install_dir" "%APPDATA%\stems2midi\stems2midi.ini" 2^>nul`) do for /f "tokens=* delims= " %%c in ("%%b") do if exist "%%c\.venv\Scripts\python.exe" set "PY=%%c\.venv\Scripts\python.exe"
if not exist "%PY%" if defined STEMS2MIDI_PYTHON if exist "%STEMS2MIDI_PYTHON%" set "PY=%STEMS2MIDI_PYTHON%"
set "CORE=%HERE%stems2midi\stems2midi.py"
if not exist "%PY%" (
  echo O ambiente ainda nao esta instalado. Corra primeiro INSTALAR.bat
  pause
  exit /b 1
)
set "FIRST=%~1"
if not defined FIRST goto usage
if not exist "%FIRST%" goto passthru
set "SECOND=%~2"
if defined SECOND if "%SECOND:~0,1%"=="-" goto passthru

:dragloop
if "%~1"=="" goto done
echo.
set "ARG=%~1"
if "%ARG:~-1%"=="\" set "ARG=%ARG:~0,-1%"
echo === "%~nx1"
"%PY%" "%CORE%" "%ARG%" %STEMS2MIDI_ARGS%
shift
goto dragloop

:done
echo.
echo Concluido. Os resultados ficam em pastas "<musica>_stems" ao lado de cada ficheiro
echo (bass_RS2.xml = arranjo Rocksmith 2014 para importar no EOF / DLC Builder).
pause
exit /b 0

:passthru
rem a linha de comandos tal como foi escrita (caminhos com espacos / barra final)
set S2M_RAW_ARGS=%*
"%PY%" "%CORE%" %*
exit /b %errorlevel%

:usage
echo Arraste um ficheiro de audio (MP3, WAV, FLAC...) para cima de stems2midi.bat
echo ou use:  stems2midi.bat musica.mp3 [opcoes]
echo.
"%PY%" "%CORE%" --help
echo.
pause
exit /b 0
