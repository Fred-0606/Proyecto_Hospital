"""Servidor HTTP local para la interfaz y la inferencia de los modelos."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    load_serialized_models,
    predict_new_patient,
    prepare_censo_context,
    summarize_censo_context,
)


class PredictionHandler(SimpleHTTPRequestHandler):
    """Sirve activos estáticos y una API JSON de predicción."""

    models: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    censo_context: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Autoriza solicitudes locales desde una vista previa en otro origen."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "models": sorted(self.models),
                    "real_inference": True,
                    "censo_loaded": self.censo_context is not None,
                }
            )
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/shutdown":
            self._shutdown_application()
            return
        if self.path == "/api/context":
            self._load_context()
            return
        if self.path != "/api/predict":
            self._send_json({"error": "Ruta no encontrada."}, HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 100_000:
                raise ValueError("El cuerpo de la solicitud es inválido.")
            payload = json.loads(self.rfile.read(content_length))
            if self.censo_context is None:
                raise ValueError("Cargue primero un Excel válido con la hoja CENSO.")
            result = predict_new_patient(
                str(payload.get("model", "")),
                payload.get("patient", {}),
                self.censo_context,
                self.models,
                self.metadata,
            )
            self._send_json(result)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def _load_context(self) -> None:
        """Carga el Excel en memoria y prepara el contexto sin persistirlo."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 75_000_000:
                raise ValueError("El archivo Excel está vacío o supera 75 MB.")
            content = self.rfile.read(content_length)
            context = prepare_censo_context(content)
            PredictionHandler.censo_context = context
            self._send_json(summarize_censo_context(context))
        except (ValueError, TypeError, KeyError, OSError) as error:
            PredictionHandler.censo_context = None
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def _shutdown_application(self) -> None:
        """Apaga el servidor cuando la solicitud proviene de la interfaz local."""
        origin = self.headers.get("Origin", "")
        allowed_origins = {
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        }
        if origin not in allowed_origins:
            self._send_json(
                {"error": "El cierre solo está disponible desde la aplicación local."},
                HTTPStatus.FORBIDDEN,
            )
            return

        self._send_json({"status": "closing"})
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Carga una vez los modelos verificados e inicia el servidor local."""
    models, metadata = load_serialized_models()
    PredictionHandler.models = models
    PredictionHandler.metadata = metadata
    server = ThreadingHTTPServer((host, port), PredictionHandler)
    print(f"Aplicación disponible en http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    run_server(arguments.host, arguments.port)
