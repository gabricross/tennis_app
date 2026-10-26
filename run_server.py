from __future__ import annotations

import argparse
import http.client
import json
import logging
import threading
import time
import webbrowser

import uvicorn

from config import HOST, PORT

LOGGER = logging.getLogger("tennis.launcher")


def running() -> bool:
    connection = http.client.HTTPConnection(HOST, PORT, timeout=1)
    try:
        connection.request("GET", "/api/health")
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"El puerto {PORT} está ocupado por otro servicio.")
        body = json.loads(response.read())
        if body.get("app") != "TennisLab":
            raise RuntimeError(f"El puerto {PORT} no pertenece a Tennis Lab.")
        return True
    except (ConnectionRefusedError, TimeoutError):
        return False
    finally:
        connection.close()


def open_when_ready() -> None:
    for _ in range(30):
        if running():
            webbrowser.open(f"http://{HOST}:{PORT}/")
            return
        time.sleep(0.4)
    LOGGER.error("No se ha podido abrir el navegador: el servidor no respondió a tiempo.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Iniciar Tennis Lab, sin servicios de pago.")
    parser.add_argument("--open", action="store_true", help="Abrir la interfaz en el navegador.")
    arguments = parser.parse_args()
    if running():
        print(f"Tennis Lab ya está abierto en http://{HOST}:{PORT}/")
        if arguments.open:
            webbrowser.open(f"http://{HOST}:{PORT}/")
        return
    if arguments.open:
        threading.Thread(target=open_when_ready, daemon=True).start()
    print(f"\nTennis Lab: http://{HOST}:{PORT}/\nCierra esta ventana o pulsa Ctrl+C para detenerlo.\n")
    uvicorn.run("app:app", host=HOST, port=PORT, workers=1, access_log=False, log_level="info")


if __name__ == "__main__":
    main()
