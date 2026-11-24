# Start API + Vite UI on Windows (PowerShell).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (Test-Path (Join-Path $Root ".venv\Scripts\Activate.ps1")) {
    . (Join-Path $Root ".venv\Scripts\Activate.ps1")
}
$env:PYTHONPATH = Join-Path $Root "backend"
$ApiPort = if ($env:YOLOFLOW_API_PORT) { $env:YOLOFLOW_API_PORT } else { "8000" }
$UiPort = if ($env:YOLOFLOW_UI_PORT) { $env:YOLOFLOW_UI_PORT } else { "5173" }

$api = Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","app.main:app","--app-dir","backend","--reload","--host","127.0.0.1","--port",$ApiPort -WorkingDirectory $Root -PassThru
try {
    Set-Location (Join-Path $Root "frontend")
    npm run dev -- --host 127.0.0.1 --port $UiPort
}
finally {
    if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
}
