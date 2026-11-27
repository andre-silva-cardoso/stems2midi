<#
.SYNOPSIS
  Instala o stems2midi (separação de stems + baixo -> MIDI) num ambiente Python
  próprio, com PyTorch para a placa NVIDIA quando existir.

.DESCRIPTION
  1. Deteta a placa NVIDIA e a versão de CUDA suportada pelo driver (nvidia-smi).
  2. Encontra (ou instala via winget) um Python 3.10–3.13, de preferência 3.12.
  3. Cria o ambiente .venv nesta pasta.
  4. Instala o PyTorch com CUDA (tenta cu130 → cu128 → cu126 → … até um servir),
     depois Demucs, librosa, soundfile e mido.
  5. Confirma que a GPU é usada e faz um teste completo com áudio sintético
     (descarrega também o modelo, ~80 MB).
  6. Opcional: liga o plugin do FeedBack a este Python e cria um atalho.

  Pode ser corrido várias vezes: reaproveita o que já está instalado.

.PARAMETER Cuda
  Força uma build do PyTorch, ex.: cu128. Use "cpu" para forçar CPU.
.PARAMETER Force
  Apaga e recria o ambiente .venv.
.PARAMETER SkipTest
  Não faz o teste final.
.PARAMETER FeedBackDir
  Pasta de plugins do FeedBack (ou a pasta do FeedBack que a contém) para onde
  copiar o plugin. Por omissão: a do stems2midi.ini ou
  C:\Program Files\feedback\current\resources\slopsmith\plugins.
.PARAMETER Yes
  Não faz perguntas (aceita as opções por omissão).
.PARAMETER Update
  Só atualiza a ferramenta numa instalação existente (ATUALIZAR.bat): copia os
  ficheiros novos para a pasta já instalada, instala apenas as dependências que
  faltarem (sem PyTorch / modelos) e volta a copiar o plugin para o FeedBack.
  Demora segundos.

.PARAMETER Url
  Com -Update: descarrega o .zip da versão nova deste link (ex.: um link de
  partilha do Google Drive, "qualquer pessoa com o link") e atualiza a partir
  dele. O link fica guardado: da próxima vez basta ATUALIZAR.bat.

.PARAMETER Roformer
  Instala também o audio-separator, para o separador BS-RoFormer
  (--model roformer: melhor stem do baixo). O modelo (~0,7 GB) é descarregado
  na primeira utilização. Funciona com -Update (ATUALIZAR.bat -Roformer).

.EXAMPLE
  .\install.ps1
  .\install.ps1 -Update
  .\install.ps1 -Update -Url "https://drive.google.com/file/d/…/view?usp=sharing"
  .\install.ps1 -Update -Url "https://github.com/andre-silva-cardoso/stems2midi"   (o .zip do GitHub)
  .\install.ps1 -Cuda cu126
  .\install.ps1 -FeedBackDir "C:\Software\FeedBack\feedBack" -Yes
#>
[CmdletBinding()]
param(
    [string]$Cuda = "",
    [switch]$Force,
    [switch]$SkipTest,
    [string]$FeedBackDir = "",
    [switch]$Yes,
    [switch]$Update,
    [string]$Url = "",
    [switch]$Local,
    [string]$InstallDir = "",
    [switch]$Roformer
)

# Native tools (pip, nvidia-smi) write to stderr; with "Stop" Windows
# PowerShell 5.1 would turn that into terminating errors. Exit codes are
# checked explicitly instead.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

$Root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$Plugin  = Join-Path $Root "stems2midi"
$Core    = Join-Path $Plugin "stems2midi.py"
$Req     = Join-Path $Plugin "requirements-ml.txt"
$Venv    = Join-Path $Root ".venv"
$VPy     = Join-Path $Venv "Scripts\python.exe"
$Log     = Join-Path $Root "install.log"
$Launch  = Join-Path $Root "stems2midi.bat"
$TorchIx = "https://download.pytorch.org/whl"

try { Start-Transcript -Path $Log -Force | Out-Null } catch { }

function Step([string]$n, [string]$t) { Write-Host ""; Write-Host "[$n] $t" -ForegroundColor Cyan }
function Ok([string]$t)   { Write-Host "    OK  $t" -ForegroundColor Green }
function Info([string]$t) { Write-Host "        $t" }
function Warn([string]$t) { Write-Host "    !!  $t" -ForegroundColor Yellow }
function Fail([string]$t) {
    Write-Host ""
    Write-Host "ERRO: $t" -ForegroundColor Red
    Write-Host "Registo completo em: $Log"
    try { Stop-Transcript | Out-Null } catch { }
    exit 1
}
function Ask([string]$q, [bool]$def = $true) {
    if ($Yes) { return $def }
    $hint = "s/N"
    if ($def) { $hint = "S/n" }
    $a = Read-Host "    $q [$hint]"
    if ([string]::IsNullOrWhiteSpace($a)) { return $def }
    $a = $a.Trim().ToLower()
    return ($a.StartsWith("s") -or $a.StartsWith("y"))
}
# preferências (stems2midi.ini por utilizador; lido pela ferramenta e pelo plugin)
$CfgDir  = Join-Path $env:APPDATA "stems2midi"
$CfgFile = Join-Path $CfgDir "stems2midi.ini"
function Read-Ini([string]$file, [string]$key) {
    if (-not (Test-Path $file)) { return "" }
    foreach ($line in (Get-Content -Path $file -Encoding UTF8)) {
        if ($line -match ("^\s*" + [regex]::Escape($key) + "\s*=\s*(.*?)\s*$")) { return $Matches[1].Trim('"') }
    }
    return ""
}
# pasta de plugins do FeedBack: -FeedBackDir, a do stems2midi.ini, ou a da instalação normal
function Guess-PluginsDir {
    $cands = @((Read-Ini $CfgFile "plugins_dir"),
               (Join-Path $env:ProgramFiles "feedback\current\resources\slopsmith\plugins"),
               (Join-Path $env:LOCALAPPDATA "Programs\feedback\current\resources\slopsmith\plugins"))
    if (${env:ProgramFiles(x86)}) { $cands += (Join-Path ${env:ProgramFiles(x86)} "feedback\current\resources\slopsmith\plugins") }
    return ($cands | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1)
}
# copia a pasta do plugin para <plugins>\stems2midi (pede administrador se for preciso)
function Copy-Plugin([string]$src, [string]$pluginsDir) {
    $dest = Join-Path $pluginsDir "stems2midi"
    $copied = $false
    try {
        New-Item -ItemType Directory -Force -Path $dest -ErrorAction Stop | Out-Null
        Get-ChildItem -Path $src -Force | Where-Object { $_.Name -ne "__pycache__" } |
            Copy-Item -Destination $dest -Recurse -Force -ErrorAction Stop
        $copied = $true
    } catch {
        Info "Sem permissão para escrever em $pluginsDir — a pedir autorização de administrador…"
        $q = { param($x) "'" + ($x -replace "'", "''") + "'" }
        $cmd = "New-Item -ItemType Directory -Force -Path " + (& $q $dest) + " | Out-Null; " +
               "Copy-Item -Path " + (& $q (Join-Path $src "*")) + " -Destination " + (& $q $dest) + " -Recurse -Force"
        try {
            Start-Process -FilePath "powershell.exe" -Verb RunAs -Wait -WindowStyle Hidden `
                -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $cmd)
            $copied = Test-Path -LiteralPath (Join-Path $dest "stems2midi.py")
        } catch { Warn "Cópia como administrador cancelada: $_" }
    }
    if ($copied) { Ok "Plugin copiado para $dest" }
    else { Warn "Não consegui copiar o plugin — copie a pasta stems2midi à mão para $pluginsDir" }
    return $copied
}
# pasta de uma instalação anterior (com .venv): esta, a do ini, ou a de STEMS2MIDI_PYTHON
function Find-Install {
    $c = @($InstallDir, $Root, (Read-Ini $CfgFile "install_dir"))
    foreach ($py in @($env:STEMS2MIDI_PYTHON, [Environment]::GetEnvironmentVariable("STEMS2MIDI_PYTHON", "User"))) {
        if ($py) { try { $c += (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $py))) } catch { } }
    }
    foreach ($d in $c) {
        if ($d -and (Test-Path -LiteralPath (Join-Path $d ".venv\Scripts\python.exe"))) { return $d }
    }
    return $null
}
function Save-IniKey([string]$key, [string]$value) {
    try {
        New-Item -ItemType Directory -Force -Path $CfgDir | Out-Null
        $text = ""
        if (Test-Path $CfgFile) { $text = [IO.File]::ReadAllText($CfgFile) }
        $line = "$key = " + $value.Replace('$', '$$')
        if ($text -match ("(?m)^\s*" + $key + "\s*=")) {
            $text = [regex]::Replace($text, ("(?m)^\s*" + $key + "\s*=.*$"), $line)
        } elseif ($text -match "(?m)^\[feedback\]\s*$") {
            $text = [regex]::Replace($text, "(?m)^\[feedback\]\s*$", ("[feedback]`r`n" + $line))
        } else {
            $text = $text.TrimEnd() + "`r`n`r`n[feedback]`r`n$key = $value`r`n"
        }
        [IO.File]::WriteAllText($CfgFile, $text, (New-Object Text.UTF8Encoding $false))
    } catch { }
}
function Save-InstallDir([string]$dir) { Save-IniKey "install_dir" $dir }

# versão da ferramenta numa pasta (plugin.json)
function Tool-Version([string]$dir) {
    try { return [string]((Get-Content -Raw -Path (Join-Path $dir "plugin.json") | ConvertFrom-Json).version) }
    catch { return "?" }
}

# torchaudio com a mesma versão/build do torch (o alinhador da letra precisa dele)
# saída de uma linha de Python (texto vazio se falhar — nunca $null)
function Py-Out([string]$py, [string]$code) {
    $o = & $py -c $code 2>$null
    if ($null -eq $o) { return "" }
    return ("" + ($o | Select-Object -Last 1)).Trim()
}
function Install-Roformer([string]$py) {
    Info "A instalar o audio-separator (BS-RoFormer)…"
    & $py -m pip install "audio-separator[cpu]" audioread --disable-pip-version-check --quiet
    if ($LASTEXITCODE -ne 0) {
        Warn "Não consegui instalar o audio-separator — o resto funciona; --model roformer não estará disponível."
    } else {
        Ok "audio-separator instalado: use --model roformer (ou 'BS-RoFormer' no FeedBack). O modelo (~0,7 GB) é descarregado na 1.ª vez."
    }
}

function Ensure-Torchaudio([string]$py) {
    $tv = Py-Out $py "import torch;print(torch.__version__)"
    if (-not $tv) { Warn "Não consegui ver a versão do torch em $py — torchaudio não verificado."; return }
    $mms = "import torchaudio;print(torchaudio.__version__ if hasattr(torchaudio.pipelines,'MMS_FA') else '')"
    $ta = Py-Out $py $mms
    if ($ta) { Ok "torchaudio $ta (torch $tv)"; return }
    $base = ($tv -split "\+")[0]
    $build = $(if ($tv -match "\+(cu\d+|cpu)") { $Matches[1] } else { "cpu" })
    Info "A instalar o torchaudio para o torch $tv — para alinhar a letra (modelo MMS)…"
    # 1) a mesma versão; 2) a mais recente desta build (o torchaudio deixou de acompanhar
    #    todas as versões do torch; o alinhamento só usa a parte em Python); 3) PyPI.
    # --no-deps: nunca mexer no torch instalado.
    $tries = @(
        @("torchaudio==$base", "--index-url", "$TorchIx/$build"),
        @("torchaudio", "--index-url", "$TorchIx/$build"),
        @("torchaudio")
    )
    foreach ($t in $tries) {
        & $py -m pip install @t --no-deps --upgrade --disable-pip-version-check --quiet 2>$null
        if ($LASTEXITCODE -eq 0) {
            $ta = Py-Out $py $mms
            if ($ta) { Ok "torchaudio $ta instalado (torch $tv)"; return }
        }
    }
    Warn "Não consegui instalar um torchaudio que funcione com o torch $tv — a letra usa o alinhamento simples."
}

# "3.12|C:\...\python.exe" or $null
function Probe-Python([string]$exe, [string[]]$pre) {
    try {
        $code = "import sys;print('%d.%d|%s' % (sys.version_info[0], sys.version_info[1], sys.executable))"
        $out = & $exe @pre -c $code 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return ([string]($out | Select-Object -Last 1)).Trim() }
    } catch { }
    return $null
}
function Torch-Info {
    # -> @{ok; version; cuda; gpu}
    $code = "import json,torch;print(json.dumps({'version':torch.__version__,'cuda':torch.version.cuda or '','avail':torch.cuda.is_available(),'gpu':(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')}))"
    $out = & $VPy -c $code 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    try { return (([string]($out | Select-Object -Last 1)) | ConvertFrom-Json) } catch { return $null }
}

Write-Host ""
$NewVersion = Tool-Version $Plugin
Write-Host "  stems2midi $NewVersion — instalação" -ForegroundColor White
Write-Host "  Pasta: $Root"

# ── atualização a partir de um link (Google Drive…) ────────────────────────
function Direct-Url([string]$u) {
    # GitHub: the repository page (or a branch) → its .zip
    if ($u -match "^https?://github\.com/([^/]+)/([^/?#]+?)(?:\.git)?/?(?:tree/([^/?#]+))?/?$") {
        $branch = $(if ($Matches[3]) { $Matches[3] } else { "main" })
        return "https://github.com/$($Matches[1])/$($Matches[2])/archive/refs/heads/$branch.zip"
    }
    if ($u -match "drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:.*&)?id=)([A-Za-z0-9_-]+)") {
        return "https://drive.google.com/uc?export=download&id=" + $Matches[1]
    }
    if ($u -match "dropbox\.com/") { return ($u -replace "([?&])dl=0", '$1dl=1') }
    return $u
}
if ($Update -and -not $Local) {
    $src = $Url
    if (-not $src) { $src = Read-Ini $CfgFile "update_url" }
    if ($src) {
        Write-Host ""
        Write-Host "  stems2midi — atualização a partir de $src" -ForegroundColor White
        $tmpZip = Join-Path $env:TEMP "stems2midi-update.zip"
        $tmpDir = Join-Path $env:TEMP "stems2midi-update"
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri (Direct-Url $src) -OutFile $tmpZip -UseBasicParsing -ErrorAction Stop
        } catch { Fail "Não consegui descarregar o .zip: $_" }
        $head = [IO.File]::ReadAllBytes($tmpZip)[0..1]
        if (-not ($head[0] -eq 0x50 -and $head[1] -eq 0x4B)) {
            Fail "O link não deu o .zip (deu uma página). No Google Drive: Partilhar → 'Qualquer pessoa com o link'."
        }
        Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
        Expand-Archive -Path $tmpZip -DestinationPath $tmpDir -Force
        $ps = Get-ChildItem -Path $tmpDir -Recurse -Filter "install.ps1" | Select-Object -First 1
        if (-not $ps) { Fail "O .zip não tem install.ps1 — é o instalador do stems2midi?" }
        Get-ChildItem -Path $ps.Directory.FullName -Recurse -File | Unblock-File -ErrorAction SilentlyContinue
        if ($Url) { Save-IniKey "update_url" $Url; Info "Link guardado: da próxima vez basta ATUALIZAR.bat" }
        $inst = Find-Install
        if (-not $inst) { Fail "Não encontrei a instalação (pasta com .venv). Corra o ATUALIZAR.bat que está na pasta da instalação." }
        $args2 = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ps.FullName, "-Update", "-Local", "-InstallDir", $inst)
        if ($FeedBackDir) { $args2 += @("-FeedBackDir", $FeedBackDir) }
        if ($Yes) { $args2 += "-Yes" }
        try { Stop-Transcript | Out-Null } catch { }
        & powershell.exe @args2
        exit $LASTEXITCODE
    }
}

# ── 0. ficheiros ───────────────────────────────────────────────────────────
if (-not (Test-Path $Core) -or -not (Test-Path $Req)) {
    Fail "Não encontro stems2midi\stems2midi.py ao lado deste script. Extraia o .zip completo e corra de novo."
}
try {
    # files extracted from a downloaded zip carry the "from the internet" mark
    @(Get-ChildItem -Path $Root -File) + @(Get-ChildItem -Path $Plugin -Recurse -File) |
        Unblock-File -ErrorAction SilentlyContinue
} catch { }

# ── atualização rápida ─────────────────────────────────────────────────────
$Existing = Find-Install
if (-not $Update -and -not $Force -and $Existing -and $Existing -ne $Root) {
    Info "Já existe uma instalação em $Existing"
    if (Ask "Atualizar só a ferramenta lá (segundos — sem voltar a descarregar o PyTorch e os modelos)?") { $Update = $true }
}
if ($Update) {
    Step "1/4" "Instalação existente"
    if (-not $Existing) {
        if ($Local) { Fail "Não encontrei a instalação (pasta com .venv)." }
        Warn "Não encontrei uma instalação anterior (.venv) — vou fazer a instalação completa."
        $Update = $false
    }
}
if ($Update) {
    $OldVersion = Tool-Version (Join-Path $Existing "stems2midi")
    Ok "$Existing  (versão instalada: $OldVersion → nova: $NewVersion)"
    if ($Existing -ne $Root) {
        try {
            $dst = Join-Path $Existing "stems2midi"
            New-Item -ItemType Directory -Force -Path $dst | Out-Null
            Get-ChildItem -Path $Plugin -Force | Where-Object { $_.Name -ne "__pycache__" } |
                Copy-Item -Destination $dst -Recurse -Force -ErrorAction Stop
            Get-ChildItem -Path $Root -File | Where-Object { $_.Extension -in ".bat", ".ps1", ".txt" -and $_.Name -ne "install.log" } |
                Copy-Item -Destination $Existing -Force -ErrorAction Stop
            Ok "Ferramenta copiada para $Existing"
        } catch { Fail "Não consegui copiar os ficheiros para $Existing : $_" }
    }
    $Root = $Existing
    $Plugin = Join-Path $Root "stems2midi"
    $Core = Join-Path $Plugin "stems2midi.py"
    $VPy = Join-Path $Root ".venv\Scripts\python.exe"
    Get-ChildItem -Path $Plugin -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    Step "2/4" "Dependências novas"
    $code = "import importlib.util as u;print(' '.join(p for m,p in [('torch','torch'),('demucs','demucs'),('librosa','librosa'),('soundfile','soundfile'),('mido','mido'),('pyphen','pyphen'),('guitarpro','pyguitarpro')] if u.find_spec(m) is None))"
    $missing = Py-Out $VPy $code
    if ($missing -match "torch|demucs") {
        Fail "Falta o PyTorch/Demucs nesta instalação — corra INSTALAR.bat (instalação completa)."
    }
    if ($missing) {
        Info "A instalar: $missing"
        & $VPy -m pip install @($missing -split " ") --disable-pip-version-check --quiet
        if ($LASTEXITCODE -ne 0) { Warn "Não consegui instalar $missing (a ferramenta funciona na mesma; corra INSTALAR.bat mais tarde)." }
        else { Ok "Instalado: $missing" }
    } else {
        Ok "Nada a instalar"
    }
    Ensure-Torchaudio $VPy
    if ($Roformer) { Install-Roformer $VPy }

    Step "3/4" "Verificação"
    $out = & $VPy $Core --check --json-progress 2>$null
    $check = $null
    try { $check = ([string]($out | Select-Object -Last 1)) | ConvertFrom-Json } catch { }
    if (-not $check) { Fail "O stems2midi não arrancou: & `"$VPy`" `"$Core`" --check" }
    Ok ("stems2midi {0} · torch {1} · torchaudio {2} · {3}" -f $check.stems2midi, $check.torch,
        $(if ($check.torchaudio) { $check.torchaudio } else { "em falta" }),
        $(if ($check.cuda) { "GPU: " + $check.gpu } else { "CPU" }))
    Info "Python: $VPy"

    Step "4/4" "Plugin do FeedBack"
    $PluginsDir = $(if ($FeedBackDir) { $FeedBackDir.Trim().Trim('"') } else { Guess-PluginsDir })
    if ($PluginsDir -and (Split-Path -Leaf $PluginsDir) -ne "plugins" -and (Test-Path -LiteralPath (Join-Path $PluginsDir "plugins"))) {
        $PluginsDir = Join-Path $PluginsDir "plugins"
    }
    if ($PluginsDir -and (Test-Path -LiteralPath $PluginsDir)) {
        if (Copy-Plugin $Plugin $PluginsDir) { Info "Reinicie o FeedBack para usar a versão nova." }
    } else {
        Warn 'Não encontrei a pasta de plugins do FeedBack (ATUALIZAR.bat -FeedBackDir "C:\...\plugins").'
    }
    Save-InstallDir $Root
    Write-Host ""
    Write-Host "  Atualizado para a versão $NewVersion (antes: $OldVersion)." -ForegroundColor Green
    try { Stop-Transcript | Out-Null } catch { }
    exit 0
}

# ── 1. GPU ─────────────────────────────────────────────────────────────────
Step "1/7" "Placa gráfica"
$cudaVer = $null
$smi = $null
$cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($cmd) { $smi = $cmd.Source }
elseif (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe") { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }

if ($smi) {
    $gpuLine = (& $smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    $smiOut  = (& $smi 2>$null | Out-String)
    if ($smiOut -match "CUDA Version:\s*([0-9]+)\.([0-9]+)") {
        $cudaVer = [version]("{0}.{1}" -f $Matches[1], $Matches[2])
    }
    if ($gpuLine) { Ok "NVIDIA: $gpuLine" }
    if ($cudaVer) { Ok "O driver suporta CUDA até $cudaVer" }
    else { Warn "nvidia-smi não indicou a versão de CUDA; vou tentar as builds mais recentes." }
} else {
    Warn "Não encontrei placa NVIDIA (nvidia-smi). Será instalada a versão para CPU (mais lenta)."
    Info "Se tem placa NVIDIA, instale/atualize o driver em https://www.nvidia.com/Download e corra de novo."
}

$builds = @(
    @{ tag = "cu130"; v = [version]"13.0" },
    @{ tag = "cu129"; v = [version]"12.9" },
    @{ tag = "cu128"; v = [version]"12.8" },
    @{ tag = "cu126"; v = [version]"12.6" },
    @{ tag = "cu124"; v = [version]"12.4" },
    @{ tag = "cu121"; v = [version]"12.1" },
    @{ tag = "cu118"; v = [version]"11.8" }
)
$candidates = @()
if ($Cuda) {
    $candidates = @($Cuda.ToLower())
} elseif ($smi) {
    foreach ($b in $builds) {
        if (-not $cudaVer -or $b.v -le $cudaVer) { $candidates += $b.tag }
    }
} else {
    $candidates = @("cpu")
}
$wantGpu = ($candidates.Count -gt 0 -and $candidates[0] -ne "cpu")

# ── 2. Python ──────────────────────────────────────────────────────────────
Step "2/7" "Python"
$basePy = $null
$venvOk = $false
if ((Test-Path $VPy) -and -not $Force) {
    $p = Probe-Python $VPy @()
    if ($p) { $venvOk = $true; Ok "Ambiente existente: Python $($p.Split('|')[0]) em .venv" }
}
if (-not $venvOk) {
    $tries = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11", "3.13", "3.10")) { $tries += , @("py", @("-$v")) }
    }
    $tries += , @("python", @())
    foreach ($v in @("312", "311", "313", "310")) {
        $tries += , @("$env:LOCALAPPDATA\Programs\Python\Python$v\python.exe", @())
        $tries += , @("$env:ProgramFiles\Python$v\python.exe", @())
    }
    foreach ($t in $tries) {
        $exe = $t[0]; $pre = [string[]]$t[1]
        if ($exe -like "*\*" -and -not (Test-Path $exe)) { continue }
        $p = Probe-Python $exe $pre
        if (-not $p) { continue }
        $ver = [version]$p.Split("|")[0]
        if ($ver -ge [version]"3.10" -and $ver -le [version]"3.13") { $basePy = $p.Split("|", 2)[1]; break }
    }
    if (-not $basePy) {
        Warn "Não encontrei Python 3.10–3.13 (o PyTorch ainda pode não suportar versões mais recentes)."
        if ((Get-Command winget -ErrorAction SilentlyContinue) -and (Ask "Instalar o Python 3.12 agora com o winget?")) {
            & winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
            $cand = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
            if (Test-Path $cand) { $p = Probe-Python $cand @(); if ($p) { $basePy = $p.Split("|", 2)[1] } }
        }
        if (-not $basePy) {
            Fail "Instale o Python 3.12 (https://www.python.org/downloads/, marque 'Add python.exe to PATH') e corra de novo."
        }
    }
    Ok "A usar $basePy"
}

# ── 3. venv ────────────────────────────────────────────────────────────────
Step "3/7" "Ambiente Python (.venv)"
if ($Force -and (Test-Path $Venv)) {
    Info "A apagar o ambiente anterior…"
    Remove-Item -Recurse -Force $Venv -ErrorAction SilentlyContinue
}
if (-not (Test-Path $VPy)) {
    & $basePy -m venv $Venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VPy)) { Fail "Não consegui criar o ambiente em $Venv" }
    Ok "Criado"
} else {
    Ok "Já existe"
}
& $VPy -m pip install --upgrade pip --disable-pip-version-check --quiet
if ($LASTEXITCODE -ne 0) { Warn "Não consegui atualizar o pip (continua mesmo assim)." }

# ── 4. PyTorch ─────────────────────────────────────────────────────────────
Step "4/7" "PyTorch"
$tag = $null
$ti = Torch-Info
if ($ti -and ($ti.cuda -or -not $wantGpu)) {
    $tag = "cpu"
    if ($ti.cuda) { $tag = "cu" + ($ti.cuda -replace "\.", "") }
    Ok "Já instalado: torch $($ti.version)"
} else {
    if ($ti -and $wantGpu) {
        Info "Existe um PyTorch só para CPU ($($ti.version)); vou substituí-lo pela versão CUDA."
        & $VPy -m pip uninstall -y torch torchvision torchaudio --disable-pip-version-check
    }
    if ($wantGpu) {
        Info "Download grande (~2,5 GB) — pode demorar vários minutos."
        foreach ($c in $candidates) {
            Info "A tentar a build $c …"
            & $VPy -m pip install torch --index-url "$TorchIx/$c" --disable-pip-version-check
            if ($LASTEXITCODE -eq 0) { $tag = $c; break }
            Warn "Build $c indisponível para este Python/driver; a tentar a seguinte."
        }
        if (-not $tag) { Warn "Nenhuma build CUDA instalou. Vou instalar a versão para CPU." }
    }
    if (-not $tag) {
        & $VPy -m pip install torch --disable-pip-version-check
        if ($LASTEXITCODE -ne 0) { Fail "Não consegui instalar o PyTorch. Veja a mensagem do pip acima." }
        $tag = "cpu"
    }
    Ok "PyTorch instalado ($tag)"
}

# ── 5. Restantes dependências ─────────────────────────────────────────────
Step "5/7" "Demucs, librosa, soundfile, mido, pyphen, PyGuitarPro"
$extra = @()
if ($tag -and $tag -ne "cpu") { $extra = @("--extra-index-url", "$TorchIx/$tag") }
& $VPy -m pip install -r $Req @extra --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Fail "Falhou a instalação das dependências (ver acima)." }
Ok "Dependências instaladas"
Ensure-Torchaudio $VPy
if ($Roformer) { Install-Roformer $VPy }

# ── 6. Verificação + teste ─────────────────────────────────────────────────
Step "6/7" "Verificação"
$check = $null
$out = & $VPy $Core --check --json-progress 2>$null
try { $check = ([string]($out | Select-Object -Last 1)) | ConvertFrom-Json } catch { }
if (-not $check) { Fail "O stems2midi não arrancou: & `"$VPy`" `"$Core`" --check" }
Info ("torch {0} · demucs {1} · librosa {2}" -f $check.torch, $check.demucs, $check.librosa)
if ($check.cuda) {
    Ok "GPU ativa: $($check.gpu)"
} elseif ($wantGpu) {
    Warn "A placa NVIDIA existe mas o PyTorch não a está a usar."
    Info "Tente: .\install.ps1 -Force   (ou indique a build: .\install.ps1 -Force -Cuda cu126)"
} else {
    Warn "Vai correr em CPU (uma música de 4 min demora ~3–8 min)."
}
if (-not $check.can_separate -or -not $check.can_transcribe) { Fail "Faltam dependências (ver o relatório acima)." }

if (-not $SkipTest) {
    Info "Teste completo com 10 s de áudio sintético (a primeira vez descarrega o modelo, ~80 MB)…"
    $tmp = Join-Path $env:TEMP "stems2midi_test"
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $gen = Join-Path $tmp "gen.py"
    @'
import sys
import numpy as np, soundfile as sf
sr = 44100
beat = 60 / 110
line = [40, 40, 43, 45, 40, 47, 45, 33, 35, 36, 28, 31]
y = np.zeros(int(10 * sr))
rng = np.random.default_rng(0)
for i, m in enumerate(line):
    t0 = 0.5 + i * beat * 0.5
    f = 440 * 2 ** ((m - 69) / 12)
    tt = np.arange(int(beat * 0.45 * sr)) / sr
    env = np.exp(-tt * 4) * np.minimum(1, tt / 0.004)
    i0 = int(t0 * sr)
    y[i0:i0 + tt.size] += 0.35 * env * sum(np.sin(2 * np.pi * f * k * tt) / k for k in range(1, 8))
for k in range(int(9 / beat)):  # kick + hats
    i0 = int((0.5 + k * beat) * sr)
    tt = np.arange(int(0.12 * sr)) / sr
    y[i0:i0 + tt.size] += 0.5 * np.sin(2 * np.pi * (50 + 60 * np.exp(-tt * 25)) * tt) * np.exp(-tt * 20)
    j = int((0.5 + (k + 0.5) * beat) * sr)
    n = int(0.04 * sr)
    y[j:j + n] += 0.08 * rng.standard_normal(n) * np.exp(-np.arange(n) / sr * 90)
sf.write(sys.argv[1], np.stack([y, y]).T, sr)
print(len(line))
'@ | Set-Content -Path $gen -Encoding ASCII
    $expected = & $VPy $gen (Join-Path $tmp "teste.wav")
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $VPy $Core (Join-Path $tmp "teste.wav") -o (Join-Path $tmp "out") --stems bass --bpm 110
    $sw.Stop()
    $summary = Join-Path $tmp "out\stems2midi.json"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $summary)) {
        Fail "O teste falhou (ver mensagens acima). Se foi no download do modelo, verifique a ligação à internet."
    }
    $res = Get-Content $summary -Raw | ConvertFrom-Json
    $notes = (Get-Content (Join-Path $tmp "out\bass.notes.json") -Raw | ConvertFrom-Json).notes.Count
    Ok ("Teste concluído em {0:N0} s em {1}: {2} notas de baixo detetadas (a linha de teste tem {3})" -f `
        $sw.Elapsed.TotalSeconds, ([string]$res.device).ToUpper(), $notes, ([string]$expected).Trim())
    Info "(o tempo inclui o download do modelo; as próximas músicas são mais rápidas)"
    Info "Resultado do teste: $tmp\out"
}

# ── 7. Integração ──────────────────────────────────────────────────────────
Step "7/7" "Atalhos, FeedBack e preferências"

$OldPlugins = Read-Ini $CfgFile "plugins_dir"
$OldLead    = Read-Ini $CfgFile "lead_silence"

if (Test-Path $Launch) {
    if (Ask "Criar atalho 'Stems para MIDI' no Ambiente de Trabalho (arraste músicas para cima dele)?") {
        try {
            $desk = [Environment]::GetFolderPath("Desktop")     # funciona com OneDrive / "Área de Trabalho"
            if (-not $desk -or -not (Test-Path -LiteralPath $desk)) { $desk = Join-Path $env:USERPROFILE "Desktop" }
            $lnkPath = Join-Path $desk "Stems para MIDI.lnk"     # (sem '>' — não é válido em nomes de ficheiro)
            $ws = New-Object -ComObject WScript.Shell
            $lnk = $ws.CreateShortcut($lnkPath)
            $lnk.TargetPath = $Launch
            $lnk.WorkingDirectory = $Root
            $lnk.Description = "Arraste um MP3 para separar stems e gerar o MIDI do baixo"
            $lnk.Save()
            Ok "Atalho criado: $lnkPath"
        } catch { Warn "Não consegui criar o atalho: $_" }
    }
}

if (Ask "Configurar o plugin do FeedBack para usar este Python (variável STEMS2MIDI_PYTHON)?") {
    [Environment]::SetEnvironmentVariable("STEMS2MIDI_PYTHON", $VPy, "User")
    Ok "STEMS2MIDI_PYTHON = $VPy  (reinicie o FeedBack para aplicar)"
}

# pasta de plugins do FeedBack: -FeedBackDir, a do stems2midi.ini, ou a da instalação normal
$PluginsDir = ""
if ($FeedBackDir) { $PluginsDir = $FeedBackDir.Trim().Trim('"') }
if (-not $PluginsDir) {
    $guess = Guess-PluginsDir
    if ($Yes) {
        $PluginsDir = $guess
    } else {
        if ($guess) {
            $a = Read-Host "    Pasta de plugins do FeedBack [$guess]  (Enter = esta, '-' = não copiar)"
            if ([string]::IsNullOrWhiteSpace($a)) { $PluginsDir = $guess } elseif ($a.Trim() -ne "-") { $PluginsDir = $a.Trim().Trim('"') }
        } else {
            $PluginsDir = (Read-Host "    Pasta de plugins do FeedBack (Enter para saltar)").Trim().Trim('"')
        }
    }
}
if ($PluginsDir) {
    # aceita a pasta do FeedBack (com plugins\ lá dentro) ou a própria pasta de plugins
    if ((Split-Path -Leaf $PluginsDir) -ne "plugins" -and (Test-Path -LiteralPath (Join-Path $PluginsDir "plugins"))) {
        $PluginsDir = Join-Path $PluginsDir "plugins"
    }
    if (-not (Test-Path -LiteralPath $PluginsDir)) {
        Warn "A pasta $PluginsDir não existe — copie a pasta stems2midi à mão para a pasta de plugins do FeedBack."
        $PluginsDir = ""
    }
}
if ($PluginsDir) {
    if (Copy-Plugin $Plugin $PluginsDir) {
        Info "(depois de atualizar o FeedBack, corra ATUALIZAR.bat para voltar a copiar o plugin)"
    }
}

# silêncio inicial do EOF → o XML do Rocksmith é deslocado por este valor
$Lead = $OldLead
if (-not $Yes) {
    $def = $(if ($OldLead) { $OldLead } else { "0" })
    $a = Read-Host "    Silêncio que o EOF acrescenta no início do guitar.ogg, em segundos [$def]"
    if (-not [string]::IsNullOrWhiteSpace($a)) { $Lead = $a.Trim().Replace(",", ".") } else { $Lead = $def }
}
if (-not $Lead) { $Lead = "0" }
$leadNum = 0.0
if (-not [double]::TryParse($Lead, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$leadNum)) {
    Warn "Valor inválido '$Lead' — fica 0"; $Lead = "0"
}
try {
    New-Item -ItemType Directory -Force -Path $CfgDir | Out-Null
    if (Test-Path $CfgFile) {
        $text = [IO.File]::ReadAllText($CfgFile)
        $text = [regex]::Replace($text, "(?m)^\s*lead_silence\s*=.*$", "lead_silence = $Lead")
        if ($text -notmatch "(?m)^\s*lead_silence\s*=") { $text = $text -replace "(?m)^\[defaults\]\s*$", "[defaults]`r`nlead_silence = $Lead" }
        if ($PluginsDir) {
            $text = [regex]::Replace($text, "(?m)^\s*plugins_dir\s*=.*$", "plugins_dir = $PluginsDir")
            if ($text -notmatch "(?m)^\s*plugins_dir\s*=") { $text = $text.TrimEnd() + "`r`n`r`n[feedback]`r`nplugins_dir = $PluginsDir`r`n" }
        }
    } else {
        $text = @"
; stems2midi — preferências (lidas pela ferramenta stems2midi.bat e pelo plugin do FeedBack)
; Qualquer opção da linha de comandos pode ter aqui um valor por omissão.

[defaults]
; segundos de silêncio que o EOF acrescenta no início do guitar.ogg:
; o XML do Rocksmith é deslocado por este valor (como --shift)
lead_silence = $Lead
; afinação do baixo: auto (4 cordas E ou 5 cordas B), 4, 5, drop-d, …
;tuning = auto
; modelo Demucs: htdemucs_6s (com guitarra), htdemucs_ft (melhor baixo, 4x mais lento)
;model = htdemucs_6s

[feedback]
plugins_dir = $PluginsDir
"@
    }
    [IO.File]::WriteAllText($CfgFile, $text, (New-Object Text.UTF8Encoding $false))
    Ok "Preferências em $CfgFile (lead_silence = $Lead)"
} catch { Warn "Não consegui escrever $CfgFile : $_" }

Save-InstallDir $Root

Write-Host ""
Write-Host "  Pronto!" -ForegroundColor Green
Write-Host "  • Arraste um MP3 para cima de stems2midi.bat (ou do atalho) — o resultado fica numa pasta <música>_stems."
Write-Host "  • Linha de comandos:  stems2midi.bat musica.mp3 --stems all --quantize 16"
Write-Host "  • No FeedBack: reinicie-o e abra 'Stems → MIDI'. O indicador deve mostrar 'GPU: …'."
Write-Host "    (Python a usar, se não definiu a variável: $VPy)"
try { Stop-Transcript | Out-Null } catch { }
exit 0
