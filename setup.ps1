<#
One-shot setup for Windows. Run via setup.bat, or:
    powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1 [-Reinstall]
  - finds a real Python 3.10+ (skips the Microsoft Store stub; installs via winget if missing)
  - creates .venv and installs this package into it, retrying flaky downloads
  - installs ffmpeg (winget, or a direct download) and copies it into .\bin
  - runs `audio-tools doctor` at the end
Re-running is safe and fast; -Reinstall forces the pip install to run again.
#>
param([switch]$Reinstall)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"   # PS 5.1 progress bars make downloads/extracts 10x slower
Set-Location -LiteralPath $PSScriptRoot
$Root = $PSScriptRoot

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072  # add TLS 1.2
    [Net.WebRequest]::DefaultWebProxy.Credentials = [Net.CredentialCache]::DefaultNetworkCredentials  # corporate proxy auth
} catch { }

$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PIP_DEFAULT_TIMEOUT = "60"
$env:PYTHONUTF8 = "1"

$Venv = Join-Path $Root ".venv"
$PyInVenv = Join-Path $Venv "Scripts\python.exe"
$BinDir = Join-Path $Root "bin"
$PipRetries = 3
$script:TrustPyPI = $false

function Say($m)  { Write-Host "`n==> $m" -ForegroundColor White }
function Ok($m)   { Write-Host "    $m" -ForegroundColor Green }
function Warn($m) { Write-Host "    $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

# Run a native program. Returns its exit code, or -1 if it could not start
# (e.g. blocked by AppLocker / WDAC, which does NOT update $LASTEXITCODE).
function Run-Native {
    param([string]$Exe, [string[]]$Arguments = @(), [switch]$Quiet)
    $global:LASTEXITCODE = -1
    try {
        if ($Quiet) { & $Exe @Arguments 2>&1 | Out-Null } else { & $Exe @Arguments 2>&1 | ForEach-Object { "$_" } | Out-Host }
    } catch {
        Warn "Could not start '$Exe': $($_.Exception.Message)"
        return -1
    }
    if ($null -eq $LASTEXITCODE) { return -1 }
    return $LASTEXITCODE
}

if ($ExecutionContext.SessionState.LanguageMode -ne "FullLanguage") {
    Warn "PowerShell is running in $($ExecutionContext.SessionState.LanguageMode). Parts of this script may be blocked by policy."
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user;$env:LOCALAPPDATA\Microsoft\WinGet\Links"
}

# ---------------------------------------------------------------- python
function Test-Python($cmd, $cmdArgs) {
    $code = Run-Native $cmd ($cmdArgs + @("-c", "import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)")) -Quiet
    return ($code -eq 0)
}

function Find-Python {
    $candidates = @(
        @{ cmd = "py";       args = @("-3.12") },
        @{ cmd = "py";       args = @("-3.13") },
        @{ cmd = "py";       args = @("-3.11") },
        @{ cmd = "py";       args = @("-3") },
        @{ cmd = "python";   args = @() },
        @{ cmd = "python3";  args = @() }
    )
    foreach ($c in $candidates) {
        if (Get-Command $c.cmd -ErrorAction SilentlyContinue) {
            if (Test-Python $c.cmd $c.args) { return $c }
        }
    }
    $globs = @("$env:LOCALAPPDATA\Programs\Python\Python31*\python.exe", "C:\Python31*\python.exe", "C:\Program Files\Python31*\python.exe")
    foreach ($g in $globs) {
        foreach ($exe in (Get-ChildItem $g -ErrorAction SilentlyContinue | Sort-Object FullName -Descending)) {
            if (Test-Python $exe.FullName @()) { return @{ cmd = $exe.FullName; args = @() } }
        }
    }
    return $null
}

Say "Looking for Python 3.10+"
$py = Find-Python
if (-not $py) {
    Warn "No usable Python found (the Microsoft Store placeholder does not count)."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Warn "Installing Python 3.12 with winget..."
        Run-Native "winget" @("install", "-e", "--id", "Python.Python.3.12", "--source", "winget", "--scope", "user",
                              "--accept-package-agreements", "--accept-source-agreements", "--silent") | Out-Null
        Refresh-Path
        $py = Find-Python
    }
    if (-not $py) {
        Warn "Opening the Python download page. Install it, tick 'Add python.exe to PATH', then re-run setup.bat."
        Start-Process "https://www.python.org/downloads/windows/"
        Die "Python is required."
    }
}
$pyVersion = (& $py.cmd @($py.args) --version 2>&1 | Out-String).Trim()
Ok "Using $($py.cmd) $($py.args -join ' ') ($pyVersion)"

# ------------------------------------------------------------------ venv
Say "Creating virtual environment in .venv"
$venvUsable = $false
if (Test-Path -LiteralPath $PyInVenv) {
    $venvUsable = (Run-Native $PyInVenv @("-c", "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)") -Quiet) -eq 0
    if ($venvUsable) { $venvUsable = (Run-Native $PyInVenv @("-m", "pip", "--version") -Quiet) -eq 0 }
}
if ($venvUsable) {
    Ok "Existing venv is usable."
} else {
    if (Test-Path -LiteralPath $Venv) {
        Warn "Existing venv is broken; moving it aside."
        Move-Item -LiteralPath $Venv ("$Venv.broken." + (Get-Date -Format "yyyyMMdd-HHmmss"))
    }
    $code = Run-Native $py.cmd ($py.args + @("-m", "venv", $Venv))
    if ($code -ne 0 -or -not (Test-Path -LiteralPath $PyInVenv)) { Die "Could not create the venv." }
    $probe = Run-Native $PyInVenv @("-c", "import sys") -Quiet
    if ($probe -ne 0) {
        Die "The new venv's python.exe cannot run. On managed PCs this usually means application control (AppLocker / WDAC) blocks programs under your user folder. Ask IT to allow '$PyInVenv', or move this folder somewhere allowed."
    }
    Ok "Created."
}

# ------------------------------------------------------------------- pip
function Pip-Install {
    param([string[]]$PipArgs)
    $base = @("-m", "pip", "install", "--prefer-binary", "--retries", "2")
    $trust = @("--trusted-host", "pypi.org", "--trusted-host", "files.pythonhosted.org")
    $log = Join-Path $env:TEMP ("audio-tools-pip-" + [guid]::NewGuid().ToString("N") + ".log")
    for ($i = 1; $i -le $PipRetries; $i++) {
        $extra = @(); if ($script:TrustPyPI) { $extra = $trust }
        $global:LASTEXITCODE = -1
        try {
            & $PyInVenv @base @extra @PipArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log | Out-Host
        } catch { Warn "Could not start pip: $($_.Exception.Message)"; return $false }
        if ($LASTEXITCODE -eq 0) { Remove-Item -LiteralPath $log -ErrorAction SilentlyContinue; return $true }
        Warn "pip attempt $i/$PipRetries failed."
        $text = ""; if (Test-Path -LiteralPath $log) { $text = Get-Content -LiteralPath $log -Raw }
        if (-not $script:TrustPyPI -and $text -match "SSL|CERTIFICATE_VERIFY_FAILED") {
            Warn "SSL errors talking to PyPI (common behind corporate proxies)."
            Warn "Retrying while trusting pypi.org / files.pythonhosted.org without certificate checks."
            $script:TrustPyPI = $true
        } else {
            Start-Sleep -Seconds 2
        }
    }
    Remove-Item -LiteralPath $log -ErrorAction SilentlyContinue
    return $false
}

$alreadyInstalled = (Run-Native $PyInVenv @("-c", "import audio_tools, numpy") -Quiet) -eq 0
if ($alreadyInstalled -and -not $Reinstall) {
    Say "audio-tools is already installed in .venv (use -Reinstall to force)"
} else {
    Say "Upgrading pip"
    if (-not (Pip-Install @("--upgrade", "pip"))) { Warn "Could not upgrade pip; continuing with the bundled one." }

    Say "Installing audio-tools (+ scipy for faster filtering, if available)"
    if (-not (Pip-Install @("-e", ".[fast]"))) {
        Warn "Install with scipy failed; installing without it (slower, but works)."
        if (-not (Pip-Install @("-e", "."))) { Die "Installing the package failed. Check your internet connection / proxy and re-run." }
    }
    Ok "Installed."
}

# ---------------------------------------------------------------- ffmpeg
function Test-Tool([string]$exe) {
    if (-not (Test-Path -LiteralPath $exe)) { return $false }
    return (Run-Native $exe @("-version") -Quiet) -eq 0
}

function Have-LocalFFmpeg { return (Test-Tool (Join-Path $BinDir "ffmpeg.exe")) -and (Test-Tool (Join-Path $BinDir "ffprobe.exe")) }

function Copy-FromPath {
    # Copy a PATH-visible ffmpeg into .\bin so the app finds it even in this (stale-PATH) cmd window.
    $src = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if (-not $src) { return $false }
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    foreach ($exe in @("ffmpeg.exe", "ffprobe.exe")) {
        $from = Join-Path (Split-Path $src.Source) $exe
        if (Test-Path -LiteralPath $from) { Copy-Item -LiteralPath $from (Join-Path $BinDir $exe) -Force }
    }
    return (Have-LocalFFmpeg)
}

function Download-FFmpeg {
    $url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    $tmp = Join-Path $env:TEMP ("audio-tools-ffmpeg-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tmp, $BinDir | Out-Null
    $zip = Join-Path $tmp "ffmpeg.zip"
    Warn "Downloading ffmpeg from $url (about 100 MB)..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    } catch {
        Warn "Download failed: $($_.Exception.Message)"
        return $false
    }
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::ExtractToDirectory($zip, $tmp)
    } catch {
        Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
    }
    $found = Get-ChildItem -LiteralPath $tmp -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    if (-not $found) { Warn "ffmpeg.exe was not inside the archive."; return $false }
    foreach ($exe in @("ffmpeg.exe", "ffprobe.exe")) {
        $src = Join-Path $found.DirectoryName $exe
        if (Test-Path -LiteralPath $src) {
            Copy-Item -LiteralPath $src (Join-Path $BinDir $exe) -Force
            Unblock-File -LiteralPath (Join-Path $BinDir $exe) -ErrorAction SilentlyContinue
        }
    }
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    return (Have-LocalFFmpeg)
}

Say "Checking ffmpeg"
if (Have-LocalFFmpeg) {
    Ok "ffmpeg found in .\bin"
} elseif ((Get-Command ffmpeg -ErrorAction SilentlyContinue) -and (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Ok "ffmpeg found on PATH."
} else {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Warn "Installing ffmpeg with winget..."
        Run-Native "winget" @("install", "-e", "--id", "Gyan.FFmpeg", "--source", "winget",
                              "--accept-package-agreements", "--accept-source-agreements", "--silent") | Out-Null
        Refresh-Path
        if (Copy-FromPath) { Ok "ffmpeg installed and copied into .\bin" }
    }
    if (-not (Have-LocalFFmpeg)) {
        if (Download-FFmpeg) { Ok "ffmpeg placed in .\bin" }
        else { Warn "Could not install ffmpeg automatically. Download it from https://www.gyan.dev/ffmpeg/builds/ and copy ffmpeg.exe + ffprobe.exe into the 'bin' folder here." }
    }
}

# ---------------------------------------------------------------- doctor
Say "Checking the install"
$status = Run-Native $PyInVenv @("-m", "audio_tools", "doctor")
Write-Host ""
if ($status -eq 0) { Ok "Setup complete. Start the menu by double-clicking run.bat" }
else { Warn "Setup finished with problems (see above). Fix them and re-run setup.bat" }
exit $status
