$ErrorActionPreference = "Stop"

$Webapp = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Webapp
$Frontend = Join-Path $Webapp "frontend"
$Venv = Join-Path $Webapp ".build-venv"

Set-Location $Frontend
npm ci
npm run build

Set-Location $Root
$Dot = Get-Command dot.exe -ErrorAction SilentlyContinue
if (-not $Dot) {
  choco install graphviz --yes --no-progress
  $GraphvizBin = Join-Path $env:ProgramFiles "Graphviz\bin"
  if (-not (Test-Path (Join-Path $GraphvizBin "dot.exe"))) {
    throw "Graphviz installation completed but dot.exe was not found."
  }
  $env:Path = "$GraphvizBin;$env:Path"
  $Dot = Get-Command dot.exe -ErrorAction Stop
}
$env:FLOWPILOT_GRAPHVIZ_ROOT = Split-Path -Parent (Split-Path -Parent $Dot.Source)

if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
py -3.11 -m venv $Venv
& "$Venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Venv\Scripts\python.exe" -m pip install -r "$Webapp\backend\requirements.txt" pyinstaller
& "$Venv\Scripts\python.exe" -m PyInstaller `
  --noconfirm `
  --clean `
  --distpath "$Webapp\dist" `
  --workpath "$Webapp\build" `
  "$Webapp\FlowPilot.spec"

$Release = Join-Path $Webapp "release"
New-Item -ItemType Directory -Force -Path $Release | Out-Null
$Zip = Join-Path $Release "FlowPilot-Windows-x64.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Compress-Archive -Path "$Webapp\dist\FlowPilot\*" -DestinationPath $Zip

Write-Host "Built: $Webapp\dist\FlowPilot\FlowPilot.exe"
Write-Host "Portable package: $Zip"
