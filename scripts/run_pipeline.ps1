# Ejecuta cada etapa en una sesión independiente de Quarto y se detiene ante
# el primer error. Las rutas se resuelven desde la raíz del repositorio.

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "No existe .venv\Scripts\python.exe. Cree primero el entorno virtual."
}

$env:QUARTO_PYTHON = $pythonPath
$stages = @(
    "pipeline/00_contexto_y_objetivos.qmd",
    "pipeline/01_etl_extraccion_anonimizacion.qmd",
    "pipeline/02_diseno_analitico_y_seleccion.qmd",
    "pipeline/03_etl_transformacion_calidad.qmd",
    "pipeline/04_conjuntos_datos_y_particion_temporal.qmd",
    "pipeline/05_eda_desarrollo.qmd",
    "pipeline/06_feature_engineering_y_preprocesamiento.qmd",
    "pipeline/07_modelo_baseline.qmd",
    "pipeline/08_modelos_comparativos.qmd",
    "pipeline/09_evaluacion_final.qmd",
    "pipeline/10_interpretabilidad.qmd",
    "pipeline/11_conclusiones.qmd"
)

Push-Location $projectRoot
try {
    foreach ($stage in $stages) {
        Write-Host "Renderizando $stage"
        & quarto render $stage
        if ($LASTEXITCODE -ne 0) {
            throw "Falló la etapa $stage con código $LASTEXITCODE."
        }
    }
}
finally {
    Pop-Location
}
