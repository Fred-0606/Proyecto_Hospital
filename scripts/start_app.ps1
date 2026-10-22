[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$serverPath = Join-Path $projectRoot "app\server.py"
$applicationUrl = "http://127.0.0.1:8000"
$healthUrl = "$applicationUrl/api/health"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "No se encontro el entorno Python en .venv. Debe prepararse antes de abrir la aplicacion."
}

if (-not (Test-Path -LiteralPath $serverPath -PathType Leaf)) {
    throw "No se encontro app/server.py. Verifica que el proyecto este completo."
}

function Test-ApplicationReady {
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        return $health.status -eq "ok" -and $health.real_inference -eq $true
    }
    catch {
        return $false
    }
}

if (-not (Test-ApplicationReady)) {
    Start-Process -FilePath $pythonPath `
        -ArgumentList @("app/server.py") `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden

    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline -and -not (Test-ApplicationReady)) {
        Start-Sleep -Milliseconds 500
    }

    if (-not (Test-ApplicationReady)) {
        throw "El servidor no estuvo listo despues de 90 segundos. Ejecuta app/server.py manualmente para ver el error."
    }
}

Start-Process $applicationUrl
