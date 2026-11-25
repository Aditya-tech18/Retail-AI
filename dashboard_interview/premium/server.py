from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os, urllib.request, urllib.error

ROOT = Path(__file__).resolve().parent
BACKEND = os.environ.get("RETAILAI_BACKEND", "http://127.0.0.1:8501")

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/"):
            try:
                with urllib.request.urlopen(
                    BACKEND + self.path, timeout=20
                ) as response:
                    data = response.read()
                    self.send_response(response.status)
                    self.send_header(
                        "Content-Type",
                        response.headers.get("Content-Type", "application/json")
                    )
                    self.end_headers()
                    self.wfile.write(data)
            except urllib.error.HTTPError as e:
                self.send_error(e.code, "Backend endpoint unavailable")
            except Exception:
                self.send_error(502, "Backend not reachable")
            return
        return super().do_GET()

if __name__ == "__main__":
    print("RetailAI Premium: http://127.0.0.1:8502")
    ThreadingHTTPServer(("127.0.0.1", 8502), Handler).serve_forever()
