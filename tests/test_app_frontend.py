"""Pruebas estructurales básicas de la interfaz web estática."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"


def test_frontend_assets_exist() -> None:
    """Comprueba que los tres activos principales de la aplicación existen."""
    for filename in ("index.html", "styles.css", "app.js"):
        path = APP_DIR / filename
        assert path.is_file(), f"Falta el activo web requerido: {path}"
        assert path.stat().st_size > 0


def test_frontend_exposes_complete_workflow() -> None:
    """Verifica los puntos de interacción esenciales del flujo solicitado."""
    html = (APP_DIR / "index.html").read_text(encoding="utf-8")
    required_ids = (
        'id="fileInput"',
        'id="modelos"',
        'id="metricas"',
        'id="metricChart"',
        'id="simulador"',
        'id="simForm"',
        'id="pipelineRunner"',
        'id="shutdownButton"',
    )
    for identifier in required_ids:
        assert identifier in html


def test_frontend_documents_real_model_and_clinical_limitation() -> None:
    """Presenta la inferencia real sin reemplazar el criterio profesional."""
    html = (APP_DIR / "index.html").read_text(encoding="utf-8").lower()
    assert "pipeline entrenado" in html
    assert "no sustituir" in html
    assert "predicción real" in html


def test_frontend_accepts_excel_and_uses_project_name() -> None:
    """Comprueba el contrato Excel y el título solicitado para la aplicación."""
    html = (APP_DIR / "index.html").read_text(encoding="utf-8")
    assert ".xlsx,.xls" in html
    assert "PREDICCIÓN DE ESTANCIAS" in html
    assert "PROLONGADAS EN URGENCIAS" in html
    assert "CSV o JSON" not in html


def test_excel_ingestion_requires_censo_and_second_header_row() -> None:
    """Alinea la lectura web con read_censo_sheet de src.data_io."""
    javascript = (APP_DIR / "app.js").read_text(encoding="utf-8")
    assert "SheetNames.includes('CENSO')" in javascript
    assert "workbook.Sheets.CENSO" in javascript
    assert "range:1" in javascript
    assert "workbook.Sheets[workbook.SheetNames[0]]" not in javascript


def test_frontend_requests_controlled_local_shutdown() -> None:
    """Comprueba que el botón usa exclusivamente la ruta local de cierre."""
    javascript = (APP_DIR / "app.js").read_text(encoding="utf-8")
    assert "shutdownApplication" in javascript
    assert "`${API_BASE}/api/shutdown`" in javascript
    assert "method:'POST'" in javascript
