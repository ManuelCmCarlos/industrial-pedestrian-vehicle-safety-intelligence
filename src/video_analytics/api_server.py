"""REST API local de leitura para os eventos persistidos do Project 4.

Executar a partir da raiz do repositório com PYTHONPATH=src:
    .venv/bin/python -m video_analytics.api_server --database CAMINHO/events_integrated.db

Disponibiliza GET /events?limit=3 e dashboard /dashboard em http://127.0.0.1:8000.
Não inicia processamento de vídeo nem grava eventos.
"""

import argparse
import json
import re
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from video_analytics.storage.database import Database
from video_analytics.storage.event_repository import EventRepository


def read_events(database_path: Path, limit: int) -> list[dict]:
    """Recupera registos reais e prepara os campos da resposta JSON."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Base de dados não encontrada: {database_path}")

    repository = EventRepository(Database(database_path))
    records = repository.get_recent_records(limit=limit)

    return [
        {
            "id": record.row_id,
            "timestamp": record.event.timestamp.isoformat(),
            "event_type": record.event.event_type,
            "track_id": record.event.track_id,
            "object_class": record.event.object_class,
            "confidence": record.event.confidence,
            "source_id": record.event.source_id,
            "rule_id": record.event.rule_id,
            "direction": record.event.direction,
            "metadata": record.event.metadata,
        }
        for record in records
    ]


def make_handler(database_path: Path):
    """Cria o gestor de pedidos HTTP associado à base escolhida."""

    dashboard_path = Path(__file__).resolve().parent / "web" / "dashboard.html"

    class EventsHandler(BaseHTTPRequestHandler):
        def send_dashboard(self) -> None:
            """Entrega a página do painel pelo mesmo servidor da API."""
            try:
                body = dashboard_path.read_bytes()
            except OSError as error:
                self.log_error("Falha ao ler o dashboard: %s", error)
                self.send_json(500, {"detail": "Dashboard indisponível."})
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        def send_json(self, status: int, payload) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            request = urlsplit(self.path)
            if request.path == "/dashboard":
                self.send_dashboard()
                return
            if request.path != "/events":
                self.send_json(404, {"detail": "Recurso não encontrado."})
                return

            params = parse_qs(request.query, keep_blank_values=True)
            values = params.get("limit", ["20"])
            if (
                set(params) - {"limit"}
                or len(values) != 1
                or re.fullmatch(r"[0-9]+", values[0]) is None
                or not 1 <= int(values[0]) <= 100
            ):
                self.send_json(
                    400,
                    {"detail": "Use limit como número inteiro entre 1 e 100."},
                )
                return

            try:
                events = read_events(database_path, limit=int(values[0]))
            except (sqlite3.Error, OSError, ValueError, TypeError) as error:
                self.log_error("Falha na leitura dos eventos: %s", error)
                self.send_json(500, {"detail": "Falha ao consultar a base de dados."})
                return

            self.send_json(200, events)

    return EventsHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="API HTTP local de eventos do Project 4")
    parser.add_argument("--database", required=True, type=Path, help="SQLite de eventos existente")
    parser.add_argument("--port", type=int, default=8000, help="Porta HTTP local (padrão: 8000)")
    parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Endereço de escuta do servidor (padrão: 127.0.0.1)",
    )
    args = parser.parse_args()

    database_path = args.database.resolve()
    if not database_path.is_file():
        parser.error(f"A base de dados não existe: {database_path}")

    with ThreadingHTTPServer((args.host, args.port), make_handler(database_path)) as server:
        print(f"API local: http://127.0.0.1:{server.server_port}/events?limit=3", flush=True)
        print(f"Dashboard: http://127.0.0.1:{server.server_port}/dashboard", flush=True)
        print(f"SQLite: {database_path}", flush=True)
        print("Para terminar: Control+C", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
