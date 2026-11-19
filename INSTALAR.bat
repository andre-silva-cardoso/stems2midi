@echo off
rem Instala o stems2midi (stems + MIDI do baixo) com suporte para placas NVIDIA.
rem Opcoes extra sao passadas ao install.ps1, ex.:  INSTALAR.bat -Cuda cu126
title stems2midi - instalacao
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
