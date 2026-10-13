"""Local HTTP/SSE transport for read-only rendering snapshots."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from render.render3D.launch import open_cot_window

def handler_for(demo):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args):
            pass

        def respond(self, payload, status=200):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                # Closing a browser can interrupt an in-flight frame response.
                pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/api/frame':
                query = parse_qs(url.query)
                if query.get('token', [''])[0] != demo.view_token:
                    return self.respond({'error': 'stale CoT window'}, 403)
                try:
                    after = int(query.get('after', ['-1'])[0])
                except ValueError:
                    return self.respond({'error': 'invalid frame version'}, 400)
                # Safari can leave an SSE connection waiting. A normal JSON
                # request can bootstrap the first pose and recover later poses.
                with demo.condition:
                    demo.condition.wait_for(lambda: (demo.latest is not None
                        and demo.version > after) or getattr(demo, 'closed', False)
                        or getattr(demo, 'failure', None) is not None, timeout=1)
                    if getattr(demo, 'failure', None) is not None:
                        return self.respond({'error': demo.failure}, 503)
                    if demo.latest is None:
                        return self.respond({'ready': False}, 202)
                    packet = dict(demo.latest)
                    packet['jackal_png'] = 'data:image/png;base64,' + demo.png if demo.png else ''
                return self.respond(packet)
            if url.path == "/api/state":
                return self.respond({**(demo.latest or {}), "acks": demo.acks, "validation": demo.validation})
            if url.path == "/api/manifest":
                return self.respond(demo.info)
            if url.path != "/api/events":
                return self.respond({"error": "not found"}, 404)
            view = parse_qs(url.query).get("view", [""])[0]
            if view != "3d":
                return self.respond({"error": "Jackal 2D is embedded in the CoT view"}, 400)
            token = parse_qs(url.query).get("token", [""])[0]
            if token != demo.view_token:
                return self.respond({"error": "stale CoT window"}, 403)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            version = -1
            try:
                self.wfile.write(b'retry: 1000\n: connected\n\n')
                self.wfile.flush()
                while True:
                    with demo.condition:
                        demo.condition.wait_for(
                            lambda: demo.version != version or token != demo.view_token
                                or getattr(demo, 'closed', False) or getattr(demo, 'failure', None) is not None,
                            timeout=10)
                        if token != demo.view_token:
                            return
                        if getattr(demo, 'closed', False) or getattr(demo, 'failure', None) is not None:
                            body = {'error': demo.failure} if getattr(demo, 'failure', None) is not None else {}
                            self.wfile.write(('event:closed\ndata:' + json.dumps(body) + '\n\n').encode())
                            self.wfile.flush()
                            return
                        if demo.version == version:
                            data = b": keepalive\n\n"
                        else:
                            version = demo.version
                            if demo.latest is None:
                                continue
                            packet = dict(demo.latest)
                            packet["jackal_png"] = "data:image/png;base64," + demo.png if demo.png else ''
                            data = ("data:" + json.dumps(packet, separators=(",", ":")) + "\n\n").encode()
                    self.wfile.write(data)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                # Close arcade framing raises PNG detail; the acknowledgement
                # carries the initial/contact/final screenshot as base64.
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("invalid request length")
                # Vite proxies the CoT browser's render acknowledgements.
                origin = self.headers.get("Origin", "")
                if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
                    return self.respond({"error": "local origins only"}, 403)
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('JSON payload must be an object')
                if self.path == "/api/window":
                    if payload.get('view_token') != demo.view_token:
                        return self.respond({'error': 'stale CoT window'}, 403)
                    action, client = payload.get('action'), payload.get('client_id')
                    if action not in ('open', 'close') or not isinstance(client, str) or not 0 < len(client) <= 128:
                        raise ValueError('invalid window event')
                    if hasattr(demo, 'window_event'):
                        demo.window_event(action, client)
                elif self.path == "/api/ack":
                    if payload.get("view") != "3d":
                        raise ValueError("offscreen 2D receipts are internal")
                    demo.acknowledge(payload)
                elif self.path == "/api/error":
                    if payload.get('view_token') != demo.view_token:
                        return self.respond({'error': 'stale CoT window'}, 403)
                    message = payload.get('message')
                    if not isinstance(message, str) or not 0 < len(message) <= 4000:
                        raise ValueError('invalid renderer error')
                    if not hasattr(demo, 'fail'):
                        return self.respond({'error': 'error reporting is unsupported'}, 409)
                    demo.fail('3D 渲染失败：' + message)
                elif self.path == "/api/restart":
                    if not getattr(demo, 'supports_restart', False):
                        return self.respond({'error': 'this viewer does not support restarting'}, 409)
                    with demo.condition:
                        previous_run = demo.run_id
                        demo.restart_requested = True
                        demo.condition.notify_all()
                        if not demo.condition.wait_for(
                            lambda: demo.run_id > previous_run
                            and demo.latest is not None and demo.latest["run_id"] == demo.run_id, timeout=15):
                            raise ValueError("restart timed out")
                        url = f"http://127.0.0.1:5174/?view=3d&token={demo.view_token}"
                    threading.Thread(target=open_cot_window, args=(url,), daemon=True).start()
                    return self.respond({"ok": True, "run_id": demo.run_id, "url": url})
                else:
                    return self.respond({"error": "not found"}, 404)
                self.respond({"ok": True})
            except (ValueError, TypeError, KeyError) as exc:
                self.respond({"error": str(exc)}, 400)
    return Handler


class LocalDemoServer(ThreadingHTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        # SSE and per-step acknowledgements must not wait for TCP's delayed
        # ACK/Nagle cycle on the localhost keep-alive connection.
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection, address
