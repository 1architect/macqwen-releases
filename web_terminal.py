#!/usr/bin/env python3
"""web_terminal.py - reach the chat from your phone, on the same WiFi.

Not a reimplementation of the terminal: it runs the real one inside a
pseudo-terminal and pipes it to a browser. The model loads once and lives in
one process, so the phone and the Mac share a session rather than each holding
13 GB. Everything the terminal does works, including approval prompts, because
it IS the terminal.

    python3 web_terminal.py            starts the chat and prints a URL
    python3 web_terminal.py --model v5 passes the flag through

A token is required. Anything on the network that reaches this port can run
shell commands on the Mac, so an open port is not an option.
"""
import argparse, html, json, os, pty, re, secrets, select, signal, socket, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from macqwen.commands import web_shortcuts

ROOT = os.path.dirname(os.path.abspath(__file__))
# A line buffer, not a byte stream. The chat redraws progress bars with a
# carriage return, which overwrites the line in a real terminal. Appending
# every redraw turned one progress bar into hundreds of identical lines on the
# phone, so carriage returns are applied here instead.
LINES = [""]
BUF_LOCK = threading.Lock()
MAX_LINES = 600
VERSION = [0]
CLIENTS = []
def _token():
    """Stable across restarts so the URL can be bookmarked on the phone."""
    f = os.path.expanduser("~/.frankenstein/web_token")
    os.makedirs(os.path.dirname(f), exist_ok=True)
    if os.path.exists(f):
        t = open(f).read().strip()
        if t:
            return t
    t = secrets.token_urlsafe(12)
    with open(f, "w") as fh:
        fh.write(t)
    os.chmod(f, 0o600)
    return t


TOKEN = _token()
master_fd = None
child_pid = None


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def spawn(args):
    """Run chat.sh under a PTY so it behaves exactly as it does in Terminal."""
    global master_fd, child_pid
    child_pid, master_fd = pty.fork()
    if child_pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.chdir(ROOT)
        os.execv("/bin/bash", ["/bin/bash", os.path.join(ROOT, "chat.sh")] + args)
    threading.Thread(target=pump, daemon=True).start()


def pump():
    while True:
        try:
            r, _, _ = select.select([master_fd], [], [], 0.3)
            if not r:
                continue
            data = os.read(master_fd, 65536)
        except OSError:
            break
        if not data:
            break
        feed(data)
    # The child exited. Say so in the buffer: a blank page with no explanation
    # is the worst possible failure mode for something you look at on a phone.
    code = None
    try:
        _, status = os.waitpid(child_pid, os.WNOHANG)
        code = os.waitstatus_to_exitcode(status)
    except (ChildProcessError, OSError):
        pass
    feed(f"\r\n\x1b[31m[chat process exited{'' if code is None else f' (code {code})'}. "
         f"Most likely out of memory: this model needs about 13 GB. "
         f"Restart with a smaller one: web_terminal.py -- --v4 flat]\x1b[0m\r\n"
         .encode())


_pending_cr = [False]


def feed(data: bytes):
    """Apply the byte stream to the line buffer with terminal semantics.

    A PTY translates every newline into CR LF, so "\r" here means one of two
    completely different things: end of line, or rewrite this line. Treating
    them alike erased the whole banner, one line at a time, as each line ended.
    Only a CR that is NOT followed by LF is a carriage return.
    """
    text = data.decode("utf-8", "replace")
    if _pending_cr[0]:
        text = "\r" + text
        _pending_cr[0] = False
    if text.endswith("\r"):          # might be the CR of a CR LF split across reads
        _pending_cr[0] = True
        text = text[:-1]
    text = text.replace("\r\n", "\n")
    with BUF_LOCK:
        for ch in text:
            if ch == "\n":
                LINES.append("")
            elif ch == "\r":
                LINES[-1] = ""          # carriage return: rewrite this line
            elif ch == "\b":
                LINES[-1] = LINES[-1][:-1]
            else:
                LINES[-1] += ch
        if len(LINES) > MAX_LINES:
            del LINES[: len(LINES) - MAX_LINES]
        VERSION[0] += 1


def render() -> str:
    with BUF_LOCK:
        snapshot = "\n".join(LINES)
    return to_html(snapshot.encode("utf-8", "replace"))


ANSI = re.compile(rb"\x1b\[([0-9;]*)m")
OTHER_ESC = re.compile(rb"\x1b\][^\x07]*\x07|\x1b\[[0-9;?]*[A-Za-z]|\r")
COLOURS = {"30": "#555", "31": "#f66", "32": "#6c6", "33": "#dc6", "34": "#69f",
           "35": "#c9f", "36": "#5cc", "37": "#ddd", "90": "#888", "2": "#999",
           "1": "font-weight:bold"}


def to_html(data: bytes) -> str:
    """ANSI colours to spans. The terminal uses colour to mean things."""
    out, pos, open_spans = [], 0, 0
    for m in ANSI.finditer(data):
        chunk = OTHER_ESC.sub(b"", data[pos:m.start()])
        out.append(html.escape(chunk.decode("utf-8", "replace")))
        codes = m.group(1).decode() or "0"
        for c in codes.split(";"):
            if c in ("0", ""):
                out.append("</span>" * open_spans)
                open_spans = 0
            elif c in COLOURS:
                v = COLOURS[c]
                style = v if ":" in v else f"color:{v}"
                out.append(f'<span style="{style}">')
                open_spans += 1
        pos = m.end()
    chunk = OTHER_ESC.sub(b"", data[pos:])
    out.append(html.escape(chunk.decode("utf-8", "replace")))
    out.append("</span>" * open_spans)
    return "".join(out)


def _shortcut_buttons() -> str:
    """Keep phone actions aligned with the shared chat command metadata."""
    buttons = [
        '<button onclick="send(\'y\')">y</button>',
        '<button onclick="send(\'n\')">n</button>',
        '<button onclick="raw(String.fromCharCode(3))">ctrl-c</button>',
    ]
    for label, command in web_shortcuts():
        buttons.append(
            f'<button onclick="send({html.escape(json.dumps(command), quote=True)})">'
            f'{html.escape(label)}</button>'
        )
    return "\n  ".join(buttons)


PAGE = """<!doctype html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,viewport-fit=cover">
<title>MACQWEN</title>
<style>
  :root{color-scheme:dark}
  *{box-sizing:border-box}
  body{margin:0;background:#12141a;color:#d6dae3;
       font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
       height:100dvh;display:flex;flex-direction:column;overscroll-behavior:none}
  #out{flex:1;overflow-y:auto;white-space:pre-wrap;word-break:break-word;
       padding:12px 12px 4px;-webkit-overflow-scrolling:touch}
  #bar{display:flex;gap:8px;padding:8px 10px;
       padding-bottom:calc(8px + env(safe-area-inset-bottom));
       background:#1a1d25;border-top:1px solid #2b303b}
  #in{flex:1;background:#0e1015;color:#e6e9ef;border:1px solid #333947;
      border-radius:9px;padding:11px 12px;font:inherit;min-width:0}
  #in:focus{outline:none;border-color:#4a5568}
  button{background:#2f3646;color:#e6e9ef;border:0;border-radius:9px;
         padding:11px 15px;font:inherit;font-weight:600}
  button:active{background:#3d4557}
  #keys{display:flex;gap:6px;padding:0 10px 6px;overflow-x:auto}
  #keys button{padding:7px 12px;font-size:12px;background:#232833;font-weight:400}
</style></head><body>
<div id="out"></div>
<div id="keys">
  {{SHORTCUTS}}
</div>
<div id="bar">
  <input id="in" autocomplete="off" autocapitalize="off" autocorrect="off"
         spellcheck="false" placeholder="message, or / command">
  <button onclick="go()">send</button>
</div>
<script>
const out=document.getElementById('out'), inp=document.getElementById('in');
const T=new URLSearchParams(location.search).get('t')||'';
let stick=true;
out.addEventListener('scroll',()=>{stick=out.scrollHeight-out.scrollTop-out.clientHeight<60});
function render(h){out.innerHTML=h;
  if(stick)out.scrollTop=out.scrollHeight;}
const es=new EventSource('/stream?t='+T);
es.onmessage=e=>render(JSON.parse(e.data));
async function raw(s){await fetch('/input?t='+T,{method:'POST',body:JSON.stringify({raw:s})});}
async function send(s){await fetch('/input?t='+T,{method:'POST',body:JSON.stringify({line:s})});}
function go(){const v=inp.value;inp.value='';send(v);stick=true;out.scrollTop=out.scrollHeight;}
inp.addEventListener('keydown',e=>{if(e.key==='Enter')go()});
</script></body></html>"""
PAGE = PAGE.replace("{{SHORTCUTS}}", _shortcut_buttons())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def ok(self):
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        return q.get("t", [""])[0] == TOKEN

    def do_GET(self):
        if not self.ok():
            self.send_response(403); self.end_headers()
            self.wfile.write(b"forbidden"); return
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            import json
            seen = -1
            try:
                while True:
                    v = VERSION[0]
                    if v != seen:
                        seen = v
                        self.wfile.write(b"data: "
                                         + json.dumps(render()).encode() + b"\n\n")
                    else:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    time.sleep(0.3)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.ok():
            self.send_response(403); self.end_headers(); return
        import json
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            payload = {}
        if "raw" in payload:
            os.write(master_fd, payload["raw"].encode())
        else:
            os.write(master_fd, (payload.get("line", "") + "\n").encode())
        self.send_response(204); self.end_headers()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("rest", nargs=argparse.REMAINDER,
                   help="passed straight to chat.sh, e.g. -- --v4")
    a = p.parse_args()
    args = [x for x in a.rest if x != "--"] or ["--v4"]
    spawn(args)
    url = f"http://{lan_ip()}:{a.port}/?t={TOKEN}"
    print("\n" + "=" * len(url))
    print(url)
    print("=" * len(url))
    print("\nSame session as the Mac terminal. Ctrl-C here stops both.\n")
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        if child_pid:
            os.kill(child_pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
