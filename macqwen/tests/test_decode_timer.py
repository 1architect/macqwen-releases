"""The reported rate must count model time only.

The streaming callback runs inside the decode loop. The terminal fade costs a
different amount for every word, so counting it made the rate wander for
reasons that had nothing to do with the model.
"""
from __future__ import annotations

import unittest

from macqwen.backends.base import DecodeTimer


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def read(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class DecodeTimerTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.timer = DecodeTimer(clock=self.clock.read)

    def test_elapsed_counts_model_time(self):
        self.clock.advance(2.0)
        self.assertAlmostEqual(self.timer.elapsed(), 2.0)

    def test_elapsed_excludes_the_streaming_callback(self):
        self.clock.advance(1.0)
        with self.timer.emitting():
            self.clock.advance(0.5)
        self.clock.advance(1.0)
        self.assertAlmostEqual(self.timer.elapsed(), 2.0)
        self.assertAlmostEqual(self.timer.emitted, 0.5)

    def test_a_slower_terminal_does_not_change_the_rate(self):
        rates = []
        for fade in (0.0, 0.05, 0.2):
            clock = FakeClock()
            timer = DecodeTimer(clock=clock.read)
            for _token in range(10):
                clock.advance(0.35)
                with timer.emitting():
                    clock.advance(fade)
            rates.append(10 / timer.elapsed())
        self.assertAlmostEqual(min(rates), max(rates), places=6)
        self.assertAlmostEqual(rates[0], 1 / 0.35, places=6)

    def test_a_callback_that_raises_still_counts(self):
        with self.assertRaises(RuntimeError):
            with self.timer.emitting():
                self.clock.advance(0.4)
                raise RuntimeError("terminal failed")
        self.clock.advance(0.6)
        self.assertAlmostEqual(self.timer.emitted, 0.4)
        self.assertAlmostEqual(self.timer.elapsed(), 0.6)

    def test_a_mark_measures_a_later_span_the_same_way(self):
        self.clock.advance(1.0)
        with self.timer.emitting():
            self.clock.advance(0.3)
        mark = self.timer.mark()
        self.clock.advance(2.0)
        with self.timer.emitting():
            self.clock.advance(0.9)
        self.assertAlmostEqual(self.timer.since(mark), 2.0)
        self.assertAlmostEqual(self.timer.elapsed(), 3.0)

    def test_spans_never_go_negative(self):
        mark = self.timer.mark()
        with self.timer.emitting():
            self.clock.advance(1.0)
        self.assertGreaterEqual(self.timer.since(mark), 0.0)
        self.assertGreaterEqual(self.timer.elapsed(), 0.0)


if __name__ == "__main__":
    unittest.main()
