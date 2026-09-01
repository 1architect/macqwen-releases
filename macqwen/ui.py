"""Terminal presentation shared by every model and profile."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time

from macqwen.api_keys import sanitized_environment
from macqwen.text import CompletedTextBuffer

C = {
    "dim": "\033[2m",
    "gray": "\033[90m",
    "b": "\033[1m",
    "g": "\033[32m",
    "y": "\033[33m",
    "r": "\033[31m",
    "c": "\033[36m",
    "clear": "\033[2K\r",
    "0": "\033[0m",
}


def human_size(size: int) -> str:
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.2f} GB"
    return f"{size / 1_000_000:.0f} MB"


def token_limit_text(limit: int) -> str:
    return "off" if limit < 0 else str(limit)


def rss_gb() -> float:
    """Resident set size in GB.

    This does not count MLX Metal buffers. A run holding 3.4 GB of model
    weights and several GB of allocator cache can report under 1.5 GB here,
    which hid a large-prompt leak for a long time. Report
    `mx.get_active_memory()` and `mx.get_cache_memory()` alongside it.
    """
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=True,
        env=sanitized_environment(),
    )
    return int(result.stdout.strip()) / 1048576


def filter_thinking(piece: str, inside: bool, show: bool) -> tuple[str, bool]:
    """Remove thinking tags and optionally their contents from one token."""
    visible = ""
    if inside:
        piece = piece.lstrip("\r\n")
    while piece:
        if inside:
            end = piece.find("</think>")
            if end < 0:
                return visible + (piece if show else ""), True
            if show:
                visible += piece[:end].rstrip("\r\n") + "\n\n"
            piece = piece[end + len("</think>") :].lstrip("\r\n")
            inside = False
        else:
            start = piece.find("<think>")
            if start < 0:
                return visible + piece, False
            visible += piece[:start]
            piece = piece[start + len("<think>") :].lstrip("\r\n")
            inside = True
    return visible, inside


class IngestGlow:
    """Animate prefill with a moving foreground glow.

    The bar starts pending. Backends then report measured layer or chunk
    completion, rate, and remaining time through `update`.
    """

    def __init__(self, output=None):
        self.output = output or sys.stdout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._active = False
        self._done = 0
        self._total = 0
        self._chunked = False
        self._started = 0.0
        self._label = "Loading context"
        self._last_done = 0
        self._last_update = 0.0
        self._rate = 0.0

    @staticmethod
    def colorize(text: str, center: float) -> str:
        base = (92, 101, 122)
        peak = (220, 248, 255)
        radius = 8.0
        out = []
        previous = None
        for index, char in enumerate(text):
            strength = max(0.0, 1.0 - abs(index - center) / radius)
            strength *= strength
            strength = round(strength * 4) / 4
            rgb = tuple(
                round(start + (end - start) * strength)
                for start, end in zip(base, peak)
            )
            if rgb != previous:
                out.append(f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m")
                previous = rgb
            out.append(char)
        return "".join(out) + C["0"]

    def start(self, total: int = 0, label: str = "Loading context") -> None:
        self.finish()
        if not self.output.isatty():
            return
        with self._lock:
            self._done = 0
            self._total = total
            self._chunked = False
            self._started = time.perf_counter()
            self._last_update = self._started
            self._last_done = 0
            self._rate = 0.0
            self._label = label
            self._active = True
            self._stop.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._render()
        self._thread.start()

    def update(self, done: int, total: int) -> bool:
        """Report progress. Switches the display to a bar."""
        with self._lock:
            if not self._active:
                return False
            now = time.perf_counter()
            delta = done - self._last_done
            seconds = now - self._last_update
            if delta > 0 and seconds > 0:
                current = delta / seconds
                self._rate = (
                    current if not self._rate
                    else self._rate * 0.65 + current * 0.35
                )
            self._last_done = done
            self._last_update = now
            self._done = done
            self._total = total
            self._chunked = self._chunked or done > 0
        if total and done >= total:
            self.finish()
        return True

    def set_label(self, label: str) -> None:
        """Change the active progress label without restarting its animation."""
        with self._lock:
            if not self._active:
                return
            self._label = label
        self._render()

    def finish(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._stop.set()
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=0.25)
        self.output.write(C["clear"])
        self.output.flush()

    def _line(self, done: int, total: int, elapsed: float, chunked: bool,
              current_rate: float = 0.0, width: int = 28) -> str:
        if not total:
            return f"{self._label}  •"
        rate = current_rate or done / max(elapsed, 1e-6)
        eta = (total - done) / rate if rate > 0 and total else 0
        filled = min(width, int(width * done / total)) if total else 0
        bar = f"{'█' * filled}{'░' * (width - filled)}"
        percent = min(100, round(done * 100 / total))
        if not chunked:
            return f"{self._label}  {bar}  0/{total:,} tok  -- tok/s  estimating"
        return (
            f"{self._label}  {bar}  {done:,}/{total:,} tok  {percent:>3}%  "
            f"{rate:5.1f} tok/s  {eta:,.0f}s left"
        )

    def _render(self) -> None:
        with self._lock:
            if not self._active:
                return
            done, total = self._done, self._total
            chunked = self._chunked
            rate = self._rate
            elapsed = time.perf_counter() - self._started
        columns = shutil.get_terminal_size(fallback=(100, 24)).columns
        width = max(8, min(28, columns - 74))
        line = self._line(done, total, elapsed, chunked, rate, width)
        travel = len(line) + 16
        center = ((elapsed / 1.35) * travel) % travel - 8
        self.output.write("\r" + self.colorize(line, center) + "\033[K")
        self.output.flush()

    def _animate(self) -> None:
        while not self._stop.wait(0.10):
            self._render()


TOOL_ACTIONS = {
    "api_docs": ("Reading API documentation", "Read API documentation"),
    "web_search": ("Searching the web", "Searched the web"),
    "find_files": ("Finding files", "Found files"),
    "list_dir": ("Listing files", "Listed files"),
    "read_file": ("Reading file", "Read file"),
    "search": ("Searching files", "Searched files"),
    "write_file": ("Creating file", "Created file"),
    "replace_text": ("Editing file", "Edited file"),
    "run_command": ("Running command", "Ran command"),
}


def tool_action(name: str, args: dict, complete: bool = False) -> str:
    """Return a compact user-facing description for one tool call."""
    labels = TOOL_ACTIONS.get(name, (f"Using {name}", f"Used {name}"))
    label = labels[1 if complete else 0]
    subject = (
        args.get("path") or args.get("pattern") or args.get("query")
        or args.get("library") or args.get("command")
    )
    if subject:
        subject = str(subject).replace("\n", " ")[:72]
        return f"{label}: {subject}"
    return label


class AgentUI:
    """Present agent prefill and tools without exposing model protocol text."""

    def __init__(self, output=None):
        self.output = output or sys.stdout
        self.glow = IngestGlow(self.output)
        self._tool = None
        self._tool_started = None
        self._tool_pending = False

    MIN_TOOL_SECONDS = 0.20

    def start_turn(self, _turn: int, total: int) -> None:
        """Start model prefill activity for one agent generation segment."""
        self.glow.start(total)

    def prefill_progress(self, done: int, total: int) -> None:
        """Forward measured backend prefill progress to the active display."""
        self.glow.update(done, total)

    def prefilled(self) -> None:
        """Remove prefill activity before answer decoding starts."""
        self.glow.finish()

    def tool_started(self, name: str, args: dict) -> None:
        """Attach parsed tool details to an existing pending tool display."""
        self._tool = (name, args)
        self._tool_started = None
        label = tool_action(name, args)
        if self._tool_pending:
            self.glow.set_label(label)
        else:
            self.output.write("\n")
            self.output.flush()
            self._tool_pending = True
            self.glow.start(label=label)

    def tool_pending(self, name: str | None = None) -> None:
        """Show activity as soon as hidden tool protocol enters the stream."""
        label = tool_action(name, {}) if name else "Preparing tool"
        if self._tool_pending:
            self.glow.set_label(label)
            return
        self._tool_pending = True
        self.glow.start(label=label)

    def tool_executing(self) -> None:
        """Start the execution-only duration clock immediately before dispatch."""
        self._tool_started = time.perf_counter()

    def tool_finished(self, error: bool = False) -> None:
        """Stop execution timing, then satisfy the visual display interval."""
        finished = time.perf_counter()
        elapsed = (
            finished - self._tool_started
            if self._tool_started is not None else 0.0
        )
        remaining = self.MIN_TOOL_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self.glow.finish()
        if self._tool is None:
            return
        name, args = self._tool
        label = tool_action(name, args, complete=True)
        prefix = f"{C['r']}Failed{C['0']}  " if error else ""
        self.output.write(f"{prefix}{C['dim']}{label}  {elapsed:.1f}s{C['0']}\n")
        self.output.flush()
        self._tool = None
        self._tool_started = None
        self._tool_pending = False

    def finish(self) -> None:
        """Clear any live progress line during interruption or final shutdown."""
        self.glow.finish()


class WordAnimator:
    """Show only complete words, fading each one in.

    A word is written several times in a rising grey before it lands in its
    final colour. Backspaces return the cursor, so every step covers the same
    cells and the finished screen holds exactly the reply text.

    The fade is self-limiting. It never spends more than half the time the
    model spent producing the word, so a slow model gets the whole animation
    and a fast one prints at full speed. The measurement excludes the time
    this class spends animating, so the two cannot feed each other.
    """

    FADE = (
        "\033[38;5;237m",
        "\033[38;5;241m",
        "\033[38;5;245m",
        "\033[38;5;249m",
    )
    THINK_FADE = (
        "\033[38;5;234m",
        "\033[38;5;236m",
        "\033[38;5;238m",
        C["gray"],
    )
    # Below this a fade reads as a flicker, so the word is written once.
    FLOOR = 0.006
    # The live chat runs this fade on its output worker. The model callback
    # only queues text, so a visible fade does not stop token generation.
    # MACQWEN_FADE_MS sets the budget in milliseconds and 0 turns it off.
    DEFAULT_DELAY = max(0.0, float(os.environ.get("MACQWEN_FADE_MS", "96"))) / 1000

    def __init__(self, output=None, delay: float | None = None, is_tty=None,
                 fade=None, clock=None, sleep=None):
        self.output = output or sys.stdout
        self.delay = self.DEFAULT_DELAY if delay is None else delay
        self.is_tty = self.output.isatty() if is_tty is None else is_tty
        self.buffer = CompletedTextBuffer()
        self.fade = self.FADE if fade is None else tuple(fade)
        self._clock = clock or time.perf_counter
        self._sleep = sleep or time.sleep
        self._ready = None

    def _budget(self) -> float:
        """How long this word may spend fading."""
        if self._ready is None:
            return self.delay
        idle = self._clock() - self._ready
        return max(0.0, min(self.delay, idle * 0.5))

    def _show(self, text: str, style: str = "") -> None:
        body = text.rstrip(" \t\r\n")
        tail = text[len(body):]
        budget = self._budget()
        shades = self.THINK_FADE if style == C["gray"] else self.fade
        if (
            self.is_tty
            and body
            and body.isascii()
            and "\n" not in body
            and shades
            and budget >= self.FLOOR
        ):
            pause = budget / len(shades)
            back = "\b" * len(body)
            for shade in shades:
                self.output.write(f"{shade}{body}{C['0']}")
                self.output.flush()
                self._sleep(pause)
                self.output.write(back)
        self.output.write(f"{style}{body}{C['0'] if style else ''}{tail}")
        self.output.flush()
        self._ready = self._clock()

    def feed(self, piece: str, style: str = "") -> None:
        for complete in self.buffer.feed(piece):
            self._show(complete, style)

    def finish(self, style: str = "") -> None:
        for complete in self.buffer.finish():
            self._show(complete, style)


class AsyncWordAnimator:
    """Animate terminal output without blocking token generation."""

    def __init__(self, animator=None, enabled: bool = True):
        self.animator = animator or WordAnimator()
        if not enabled:
            self.animator.delay = 0.0
        self._items = queue.Queue()
        self._cancelled = threading.Event()
        self._closed = False
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._items.get()
            if item is None:
                return
            method, text, style = item
            if self._cancelled.is_set():
                continue
            try:
                if method == "feed":
                    self.animator.feed(text, style)
                else:
                    self.animator.finish(style)
            except BaseException as exc:
                self._error = exc
                self._cancelled.set()

    def feed(self, piece: str, style: str = "") -> None:
        if not self._closed:
            self._items.put(("feed", piece, style))

    def finish(self, style: str = "") -> None:
        if self._closed:
            return
        self._closed = True
        self._items.put(("finish", "", style))
        self._items.put(None)
        self._thread.join()
        if self._error is not None:
            raise self._error

    def cancel(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancelled.set()
        self._items.put(None)
        self._thread.join(timeout=0.25)
