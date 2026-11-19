@echo off
rem Atualiza so a ferramenta stems2midi numa instalacao existente (sem voltar a
rem descarregar o PyTorch e os modelos) e volta a copiar o plugin para o FeedBack.
rem Extraia o .zip novo (para qualquer pasta) e corra este ficheiro.
title stems2midi - atualizacao
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Update %*
echo.
pause
