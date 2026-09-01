"""Terminal input with safe multiline bracketed paste support."""
from __future__ import annotations

import os
import sys
import termios


PASTE_START = b"\x1b[200~"
PASTE_END = b"\x1b[201~"
ENABLE_PASTE = b"\x1b[?2004h"
DISABLE_PASTE = b"\x1b[?2004l"


def _suffix_prefix_length(data: bytearray, marker: bytes) -> int:
    maximum = min(len(data), len(marker) - 1)
    for size in range(maximum, 0, -1):
        if data[-size:] == marker[:size]:
            return size
    return 0


def _safe_echo(data: bytes) -> bytes:
    """Prevent pasted control bytes from changing terminal state."""
    out = bytearray()
    for value in data:
        if value in (9, 10) or value >= 32:
            out.append(value)
        elif value == 27:
            out.extend(b"^[")
    return bytes(out)


def _append_paste(content: bytearray, data: bytes, output_fd: int) -> None:
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    content.extend(normalized)
    os.write(output_fd, _safe_echo(normalized))


def _erase_last(content: bytearray, output_fd: int) -> None:
    if not content:
        return
    if content[-1] == 10:
        content.pop()
        os.write(output_fd, b"\x1b[A\x1b[999C")
        return
    start = len(content) - 1
    while start > 0 and content[start] & 0xC0 == 0x80:
        start -= 1
    del content[start:]
    os.write(output_fd, b"\b \b")


def read_prompt(prompt: str) -> str:
    """Read one turn. Pasted newlines do not submit until a later Enter."""
    if not sys.stdin.isatty():
        return input(prompt)

    input_fd = sys.stdin.fileno()
    output_fd = sys.stdout.fileno()
    original = termios.tcgetattr(input_fd)
    raw = termios.tcgetattr(input_fd)
    raw[3] &= ~(termios.ICANON | termios.ECHO)
    raw[6][termios.VMIN] = 1
    raw[6][termios.VTIME] = 0

    content = bytearray()
    pending = bytearray()
    in_paste = False
    submitted = False

    termios.tcsetattr(input_fd, termios.TCSADRAIN, raw)
    try:
        os.write(output_fd, ENABLE_PASTE + prompt.encode("utf-8"))
        while not submitted:
            chunk = os.read(input_fd, 4096)
            if not chunk:
                raise EOFError
            pending.extend(chunk)

            while pending and not submitted:
                if in_paste:
                    end = pending.find(PASTE_END)
                    if end >= 0:
                        _append_paste(content, bytes(pending[:end]), output_fd)
                        del pending[: end + len(PASTE_END)]
                        in_paste = False
                        continue
                    keep = _suffix_prefix_length(pending, PASTE_END)
                    consume = len(pending) - keep
                    if consume:
                        _append_paste(
                            content, bytes(pending[:consume]), output_fd
                        )
                        del pending[:consume]
                    break

                if pending.startswith(PASTE_START):
                    del pending[: len(PASTE_START)]
                    in_paste = True
                    continue
                if PASTE_START.startswith(pending):
                    break

                value = pending[0]
                if value == 27:
                    if len(pending) == 1:
                        break
                    if pending[1] == ord("["):
                        final = next(
                            (
                                index
                                for index, item in enumerate(pending[2:], 2)
                                if 0x40 <= item <= 0x7E
                            ),
                            None,
                        )
                        if final is None:
                            break
                        del pending[: final + 1]
                    else:
                        del pending[:2]
                    continue

                del pending[0]
                if value in (10, 13):
                    os.write(output_fd, b"\r\n")
                    submitted = True
                elif value == 3:
                    raise KeyboardInterrupt
                elif value == 4:
                    if not content:
                        raise EOFError
                elif value in (8, 127):
                    _erase_last(content, output_fd)
                elif value == 21:
                    content.clear()
                    os.write(output_fd, b"^U")
                elif value == 9 or value >= 32:
                    content.append(value)
                    os.write(output_fd, bytes((value,)))
    finally:
        termios.tcsetattr(input_fd, termios.TCSADRAIN, original)
        os.write(output_fd, DISABLE_PASTE)

    return content.decode("utf-8", errors="replace")
