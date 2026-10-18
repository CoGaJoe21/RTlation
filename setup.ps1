<#
.SYNOPSIS
  Richtet RTlation komplett im Projektordner ein: uv, Python, Pakete,
  CUDA-Bibliotheken und Modelle. Kann beliebig oft erneut gestartet werden,
  bereits vorhandene Teile werden uebersprungen.

.PARAMETER LimitRate
  Bandbreitenlimit fuer grosse Downloads im curl-Format, z.B. 7M fuer 7 MB/s.

.PARAMETER Stt
  Modelle fuer die Spracherkennung: large-v3-turbo (schnell, Standard) und
  large-v3 (groesser, 3 GB). Mehrere durch Komma getrennt; in der Oberflaeche
  laesst sich zwischen den eingerichteten umschalten.

.PARAMETER Tts
  Sprachausgabe-Modelle, die eingerichtet werden: chatterbox, qwen3 oder
  beide durch Komma getrennt. "none" laesst die Sprachausgabe weg.

.PARAMETER Quiet
  Keine Fortschrittsanzeige der Downloads (fuer Log-Dateien).

.PARAMETER SkipOllama
  Das Uebersetzungsmodell nicht ueber Ollama laden.

.EXAMPLE
  .\setup.ps1 -LimitRate 7M -Tts chatterbox,qwen3
#>
param(
    [string]$LimitRate = "",
    [string[]]$Stt = @("large-v3-turbo"),
    [string[]]$Tts = @("chatterbox"),
    [switch]$Quiet,
    [switch]$SkipOllama
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root   = $PSScriptRoot
$Tools  = Join-Path $Root ".tools"
$Models = Join-Path $Root "models"

# Festgelegte Versionen, damit das Setup auf jedem Rechner dasselbe ergibt.
$UvVersion    = "0.13.0"
$CudaWheels   = @(
    @{ Name = "nvidia-cublas-cu12"; Version = "12.9.2.10" },
    @{ Name = "nvidia-cudnn-cu12";  Version = "9.27.0.42" }
)
$WhisperFiles  = @("config.json", "preprocessor_config.json", "tokenizer.json", "vocabulary.json", "model.bin")
$WhisperModels = @{
    "large-v3-turbo" = @{ Repo = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"; Rev = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf" }
    "large-v3"       = @{ Repo = "Systran/faster-whisper-large-v3";              Rev = "edaa852ec7e145841d8ffdb056a99866b5f0a478" }
}
$VadUrl       = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx"
$OllamaModel  = "translategemma:4b"

# Sprachausgabe: Jedes Modell bekommt eine eigene Python-Umgebung, weil sich
# ihre Abhaengigkeiten gegenseitig ausschliessen. PyTorch teilen sie sich.
$TorchBase    = "https://download.pytorch.org/whl/cu124"
$TorchWheels  = @(
    @{ File = "torch-2.6.0+cu124-cp311-cp311-win_amd64.whl";      Sha256 = "6a1fb2714e9323f11edb6e8abf7aad5f79e45ad25c081cde87681a18d99c29eb" },
    @{ File = "torchaudio-2.6.0+cu124-cp311-cp311-win_amd64.whl"; Sha256 = "a25e146ce66ea9a6aed39008cc2001891bdf75253af479a4c32096678b2073b3" }
)
$TtsEngines   = @{
    chatterbox = @{
        Repo  = "ResembleAI/chatterbox"
        Rev   = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
        Dir   = "chatterbox"
        Files = @("ve.pt", "t3_mtl23ls_v3.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json")
    }
    qwen3 = @{
        Repo  = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        Rev   = "5d83992436eae1d760afd27aff78a71d676296fc"
        Dir   = "qwen3-tts-0.6b-base"
        Files = $null   # ganzes Repository
    }
}
# Mit "powershell -File" kommt "a,b" als ein einzelner Wert an.
$Stt = @($Stt | ForEach-Object { $_.Split(",") } | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
foreach ($name in $Stt) {
    if (-not $WhisperModels.ContainsKey($name)) { throw "Unbekanntes Whisper-Modell: $name (moeglich: $($WhisperModels.Keys -join ', '))" }
}
$Tts = @($Tts | ForEach-Object { $_.Split(",") } | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ -and $_ -ne "none" })
foreach ($name in $Tts) {
    if (-not $TtsEngines.ContainsKey($name)) { throw "Unbekanntes Sprachausgabe-Modell: $name (moeglich: $($TtsEngines.Keys -join ', '))" }
}

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "== $Text" -ForegroundColor Yellow
}

function Invoke-Curl([string[]]$CurlArgs) {
    & curl.exe @CurlArgs
    return $LASTEXITCODE
}

# Laedt eine Datei mit curl (Wiederaufnahme, optionales Bandbreitenlimit).
function Get-File([string]$Url, [string]$Dest) {
    if (Test-Path $Dest) { return }
    New-Item -ItemType Directory -Force (Split-Path $Dest) | Out-Null
    $part = "$Dest.part"
    # --speed-limit/--speed-time: haengt die Verbindung 60 s lang, gilt das als Abbruch.
    # Kein --retry von curl: Damit meldet curl 8.x einen Fehler teils als Erfolg
    # (gemessen); die Wiederholung uebernimmt die Schleife unten.
    $curlArgs = @("--location", "--fail", "--speed-limit", "1000", "--speed-time", "60", "--output", $part)
    if ($LimitRate) { $curlArgs += @("--limit-rate", $LimitRate) }
    if ($Quiet) { $curlArgs += @("--silent", "--show-error") }
    Write-Host "  -> $(Split-Path $Dest -Leaf)"
    # Bricht die Verbindung mittendrin ab (z.B. curl 56), geht es beim naechsten
    # Versuch an derselben Stelle weiter.
    $code = 0
    for ($attempt = 1; $attempt -le 8; $attempt++) {
        $code = Invoke-Curl ($curlArgs + @("--continue-at", "-", $Url))
        if ($code -eq 0 -and (Test-Path $part)) { break }
        if ($code -eq 0) { $code = -1 }   # Erfolg gemeldet, aber nichts angekommen
        if ($code -eq 33) {
            # Server kann nicht fortsetzen: von vorne beginnen.
            Remove-Item -Force $part -ErrorAction SilentlyContinue
        }
        Write-Host "     Verbindung abgebrochen (curl $code), neuer Versuch $attempt von 8 ..." -ForegroundColor DarkYellow
        Start-Sleep -Seconds ([Math]::Min(30, 3 * $attempt))
    }
    if ($code -ne 0) {
        throw "Download fehlgeschlagen (curl $code): $Url - setup.bat einfach erneut starten, es geht an derselben Stelle weiter."
    }
    Move-Item -Force $part $Dest
}

function Get-Json([string]$Url, [string]$Accept = "application/json") {
    $text = (& curl.exe --silent --show-error --location --fail --header "Accept: $Accept" $Url) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Abfrage fehlgeschlagen: $Url" }
    return $text | ConvertFrom-Json
}

function Assert-Sha256([string]$Path, [string]$Expected) {
    $actual = (Get-FileHash $Path -Algorithm SHA256).Hash
    if ($actual -ne $Expected) {
        Remove-Item -Force $Path
        throw "Pruefsumme stimmt nicht: $(Split-Path $Path -Leaf)"
    }
}

# Laedt Dateien eines Hugging-Face-Repositorys in einem festen Stand.
# Ohne $Files wird das ganze Repository geholt.
function Get-HfFiles([string]$Repo, [string]$Rev, [string]$Dest, [string[]]$Files) {
    if (-not $Files) {
        $meta  = Get-Json "https://huggingface.co/api/models/$Repo/revision/$Rev"
        $Files = $meta.siblings | ForEach-Object { $_.rfilename } | Where-Object { $_ -notin @(".gitattributes", "README.md") }
    }
    foreach ($name in $Files) {
        Get-File "https://huggingface.co/$Repo/resolve/$Rev/$name" (Join-Path $Dest ($name -replace "/", "\"))
    }
}

# ---------------------------------------------------------------- uv + Python
Write-Step "uv und Python-Umgebung"
$UvDir = Join-Path $Tools "uv"
$Uv    = Join-Path $UvDir "uv.exe"
if (-not (Test-Path $Uv)) {
    $zip = Join-Path $Tools "uv.zip"
    Get-File "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" $zip
    Expand-Archive -Path $zip -DestinationPath $UvDir -Force
    Remove-Item -Force $zip
    if (-not (Test-Path $Uv)) { throw "uv.exe wurde nach dem Entpacken nicht gefunden." }
}
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools "python"
& $Uv sync --project $Root
if ($LASTEXITCODE -ne 0) { throw "uv sync fehlgeschlagen." }

# ------------------------------------------------------------ CUDA-Bibliotheken
# faster-whisper braucht cuBLAS und cuDNN. Die DLLs kommen aus den offiziellen
# NVIDIA-Wheels und liegen danach in .tools\cuda\bin.
Write-Step "CUDA-Bibliotheken (cuBLAS, cuDNN)"
$CudaDir    = Join-Path $Tools "cuda"
$CudaBin    = Join-Path $CudaDir "bin"
$CudaMarker = Join-Path $CudaDir "versions.txt"
$CudaWanted = ($CudaWheels | ForEach-Object { "$($_.Name)==$($_.Version)" }) -join "`n"
$CudaHave   = if (Test-Path $CudaMarker) { (Get-Content $CudaMarker) -join "`n" } else { "" }
if ($CudaHave -ne $CudaWanted) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    Remove-Item -Recurse -Force $CudaBin -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $CudaBin | Out-Null
    foreach ($wheel in $CudaWheels) {
        $meta = Get-Json "https://pypi.org/pypi/$($wheel.Name)/$($wheel.Version)/json"
        $file = $meta.urls | Where-Object { $_.filename -like "*win_amd64.whl" } | Select-Object -First 1
        if (-not $file) { throw "Kein Windows-Paket fuer $($wheel.Name) $($wheel.Version) gefunden." }
        $whl = Join-Path $Tools $file.filename
        Get-File $file.url $whl
        Assert-Sha256 $whl $file.digests.sha256
        $archive = [System.IO.Compression.ZipFile]::OpenRead($whl)
        try {
            foreach ($entry in $archive.Entries) {
                if ($entry.Name -like "*.dll") {
                    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $CudaBin $entry.Name), $true)
                }
            }
        } finally {
            $archive.Dispose()
        }
        Remove-Item -Force $whl
    }
    Set-Content -Path $CudaMarker -Value $CudaWanted -Encoding ascii
} else {
    Write-Host "  bereits vorhanden"
}

# --------------------------------------------------------------------- Modelle
foreach ($name in $Stt) {
    Write-Step "Whisper-Modell $name"
    Get-HfFiles $WhisperModels[$name].Repo $WhisperModels[$name].Rev (Join-Path $Models "whisper\$name") $WhisperFiles
}

Write-Step "Modell fuer die Pausenerkennung"
Get-File $VadUrl (Join-Path $Models "vad\silero_vad.onnx")

# --------------------------------------------------------------- Sprachausgabe
$WheelDir = Join-Path $Tools "wheels"
foreach ($name in $Tts) {
    $engine = $TtsEngines[$name]
    Write-Step "Sprachausgabe: $name"

    $venv      = Join-Path $Tools "venvs\$name"
    $venvPy    = Join-Path $venv "Scripts\python.exe"
    $marker    = Join-Path $venv ".installed"
    $reqFile   = Join-Path $Root "workers\$name\requirements.txt"
    $ovrFile   = Join-Path $Root "workers\$name\overrides.txt"
    $reqHash   = (Get-FileHash $reqFile -Algorithm SHA256).Hash
    if (Test-Path $ovrFile) { $reqHash += (Get-FileHash $ovrFile -Algorithm SHA256).Hash }
    $installed = if (Test-Path $marker) { (Get-Content $marker) -join "" } else { "" }
    if ($installed -ne $reqHash) {
        $wheelPaths = @()
        foreach ($wheel in $TorchWheels) {
            $path = Join-Path $WheelDir $wheel.File
            Get-File "$TorchBase/$($wheel.File -replace '\+', '%2B')" $path
            Assert-Sha256 $path $wheel.Sha256
            $wheelPaths += $path
        }
        & $Uv venv $venv --python 3.11 --allow-existing
        if ($LASTEXITCODE -ne 0) { throw "Python-Umgebung fuer $name konnte nicht angelegt werden." }
        & $Uv pip install --python $venvPy @wheelPaths
        if ($LASTEXITCODE -ne 0) { throw "PyTorch konnte fuer $name nicht installiert werden." }
        $extra = if (Test-Path $ovrFile) { @("--override", $ovrFile) } else { @() }
        & $Uv pip install --python $venvPy -r $reqFile @extra
        if ($LASTEXITCODE -ne 0) { throw "Pakete fuer $name konnten nicht installiert werden." }
        Set-Content -Path $marker -Value $reqHash -Encoding ascii
    } else {
        Write-Host "  Umgebung bereits vorhanden"
    }

    Get-HfFiles $engine.Repo $engine.Rev (Join-Path $Models "tts\$($engine.Dir)") $engine.Files
}
# Die PyTorch-Pakete sind jetzt in den Umgebungen; die Downloads braucht es nicht mehr.
Remove-Item -Recurse -Force $WheelDir -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------- Ollama
# Ollama selbst kann Downloads nicht drosseln. Mit -LimitRate werden die
# Modelldateien deshalb mit curl geholt, geprueft und in den Ollama-Speicher
# gelegt; "ollama pull" findet sie dann vor und laedt nur noch das Manifest.
function Import-OllamaBlobs([string]$Model) {
    $name, $tag = $Model.Split(":")
    $base = "https://registry.ollama.ai/v2/library/$name"
    $manifest = Get-Json "$base/manifests/$tag" "application/vnd.docker.distribution.manifest.v2+json"
    $store = if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS } else { Join-Path $env:USERPROFILE ".ollama\models" }
    $blobDir = Join-Path $store "blobs"
    New-Item -ItemType Directory -Force $blobDir | Out-Null
    foreach ($layer in @($manifest.config) + @($manifest.layers)) {
        $hex  = $layer.digest.Split(":")[1]
        $dest = Join-Path $blobDir "sha256-$hex"
        if (Test-Path $dest) { continue }
        $tmp = Join-Path $Tools "ollama-$hex"
        Get-File "$base/blobs/$($layer.digest)" $tmp
        Assert-Sha256 $tmp $hex
        Move-Item -Force $tmp $dest
    }
}

if (-not $SkipOllama) {
    Write-Step "Uebersetzungsmodell ($OllamaModel) ueber Ollama"
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Warning "Ollama ist nicht installiert. Bitte von https://ollama.com installieren und setup.ps1 erneut starten."
    } else {
        $listed = & ollama list
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Ollama laeuft nicht. Bitte Ollama starten und setup.ps1 erneut ausfuehren."
        } elseif ($listed -match [regex]::Escape($OllamaModel)) {
            Write-Host "  bereits vorhanden"
        } else {
            if ($LimitRate) {
                try {
                    Import-OllamaBlobs $OllamaModel
                } catch {
                    Write-Warning "Gedrosselter Download nicht moeglich ($($_.Exception.Message)). Ollama laedt ungedrosselt."
                }
            }
            & ollama pull $OllamaModel
            if ($LASTEXITCODE -ne 0) { throw "ollama pull fehlgeschlagen." }
        }
    }
}

# ------------------------------------------------- virtuelles Mikrofon
Write-Step "Virtuelles Mikrofon fuer Discord, TeamSpeak usw."
$cable = Get-CimInstance Win32_SoundDevice -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*VB-Audio*" }
if ($cable) {
    Write-Host "  VB-Cable ist installiert. In RTlation 'CABLE Input' als Ausgabegeraet waehlen,"
    Write-Host "  in Discord oder TeamSpeak 'CABLE Output' als Mikrofon."
} else {
    Write-Host "  Nicht gefunden. Wer die Uebersetzung in Discord, TeamSpeak usw. senden will,"
    Write-Host "  installiert einmal VB-Cable (kostenlos): https://vb-audio.com/Cable/"
    Write-Host "  Danach waehlt RTlation es automatisch als Ausgabegeraet."
}

Write-Step "Fertig"
Write-Host "Start mit:  run.bat"
