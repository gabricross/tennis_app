param(
    [switch]$SoloJugadores
)

$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
$appHome = Join-Path $env:LOCALAPPDATA 'TennisLab'
if ($env:TENNISLAB_HOME) {
    $appHome = [IO.Path]::GetFullPath($env:TENNISLAB_HOME)
}
$environment = Join-Path $appHome '.venv'
$python = Join-Path $environment 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $launcher) {
        throw 'Instala Python 3.14 para Windows y vuelve a ejecutar este archivo.'
    }
    & $launcher.Source -3.14 -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw 'No se ha podido crear el entorno de Tennis Lab.' }
}
& $python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 'Se requiere Python 3.14. Usa un entorno TennisLab nuevo con TENNISLAB_HOME.')"
if ($LASTEXITCODE -ne 0) { throw 'El entorno existente no utiliza Python 3.14.' }
$requirements = 'requirements-lock.txt'
if ($SoloJugadores) { $requirements = 'requirements-runtime-lock.txt' }
& $python -m pip install -r (Join-Path $project $requirements)
if ($LASTEXITCODE -ne 0) { throw 'No se han podido instalar las dependencias.' }
$downloadArguments = @('--download-only')
if ($SoloJugadores) { $downloadArguments += '--players-only' }
& $python (Join-Path $project 'setup_models.py') @downloadArguments
if ($LASTEXITCODE -ne 0) { throw 'No se han podido verificar las descargas.' }
$models = @('court', 'ball')
if ($SoloJugadores) { $models = @() }
foreach ($model in $models) {
    if (-not (Test-Path -LiteralPath (Join-Path $appHome ('models\' + $model + '.onnx')))) {
        & $python (Join-Path $project 'setup_models.py') --convert $model
        if ($LASTEXITCODE -ne 0) { throw ('No se ha podido convertir el modelo: ' + $model) }
    }
}
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path $project 'Abrir Tennis Lab.lnk'))
$shortcut.TargetPath = $python
$shortcut.Arguments = '"' + (Join-Path $project 'run_server.py') + '" --open'
$shortcut.WorkingDirectory = $project
$shortcut.Description = 'Tennis Lab: análisis privado de clips en este ordenador'
$shortcut.Save()
[void][Runtime.InteropServices.Marshal]::ReleaseComObject($shell)
Write-Output 'Preparado. Abre el acceso directo Abrir Tennis Lab.'
