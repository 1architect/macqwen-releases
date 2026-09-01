from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
import time
import unittest


class TerminalInputTests(unittest.TestCase):
    def test_multiline_paste_waits_for_manual_enter(self):
        master, slave = pty.openpty()
        script = (
            "from macqwen.terminal import read_prompt; "
            "print(repr(read_prompt('you> ')), flush=True)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
        )
        os.close(slave)
        try:
            self._read_until(master, b"you> ")
            os.write(master, b"\x1b[200~first\nsecond\n\x1b[201~")
            time.sleep(0.1)
            self.assertIsNone(process.poll())
            os.write(master, b"\r")
            output = self._read_until(master, b"'first\\nsecond\\n'")
            self.assertIn(b"'first\\nsecond\\n'", output)
            self.assertEqual(process.wait(timeout=2), 0)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2)
            os.close(master)

    @staticmethod
    def _read_until(fd: int, marker: bytes) -> bytes:
        output = bytearray()
        deadline = time.time() + 3
        while marker not in output and time.time() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            try:
                output.extend(os.read(fd, 4096))
            except OSError:
                break
        if marker not in output:
            raise AssertionError(f"missing {marker!r} in {bytes(output)!r}")
        return bytes(output)


if __name__ == "__main__":
    unittest.main()
