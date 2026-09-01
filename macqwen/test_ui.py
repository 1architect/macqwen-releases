from __future__ import annotations

import re
import threading
import unittest
import unittest.mock
from unittest.mock import patch
from io import StringIO

from macqwen.ui import (
    AgentUI,
    AsyncWordAnimator,
    C,
    IngestGlow,
    WordAnimator,
    filter_thinking,
    human_size,
    token_limit_text,
)


class UITests(unittest.TestCase):
    def test_human_size(self):
        self.assertEqual(human_size(2_500_000_000), "2.50 GB")
        self.assertEqual(human_size(4_000_000), "4 MB")

    def test_token_limit_text(self):
        self.assertEqual(token_limit_text(-1), "off")
        self.assertEqual(token_limit_text(512), "512")

    def test_thinking_hidden_by_default(self):
        piece, inside = filter_thinking("<think>reasoning</think>answer", False, False)
        self.assertEqual(piece, "answer")
        self.assertFalse(inside)

    def test_thinking_shown_when_asked(self):
        piece, inside = filter_thinking("<think>reasoning</think>answer", False, True)
        self.assertEqual(piece, "reasoning\n\nanswer")
        self.assertFalse(inside)

    def test_thinking_boundary_has_one_blank_line(self):
        piece, inside = filter_thinking(
            "\nreasoning\n\n\n</think>\n\n\nanswer", True, True
        )
        self.assertEqual(piece, "reasoning\n\nanswer")
        self.assertFalse(inside)

    def test_thinking_state_carries_across_tokens(self):
        piece, inside = filter_thinking("<think>half", False, False)
        self.assertEqual(piece, "")
        self.assertTrue(inside)
        piece, inside = filter_thinking(" rest</think>done", inside, False)
        self.assertEqual(piece, "done")
        self.assertFalse(inside)


class GlowLineTests(unittest.TestCase):
    """One prefill shows a count; a chunked one shows a bar."""

    def setUp(self):
        self.glow = IngestGlow()

    def test_whole_prefill_shows_the_token_count(self):
        line = self.glow._line(0, 4096, 1.0, chunked=False)
        self.assertIn("Loading context", line)
        self.assertIn("0/4,096 tok", line)
        self.assertIn("estimating", line)

    def test_chunked_prefill_shows_progress(self):
        line = self.glow._line(512, 4096, 2.0, chunked=True)
        self.assertIn("512/4,096 tok", line)
        self.assertIn("█", line)
        self.assertIn("░", line)
        self.assertIn("12%", line)
        self.assertIn("tok/s", line)
        self.assertIn("left", line)

    def test_start_renders_before_the_animation_thread_ticks(self):
        class TTYOutput(StringIO):
            def isatty(self):
                return True

        output = TTYOutput()
        glow = IngestGlow(output)
        glow.start(100)
        try:
            self.assertIn("Loading context", output.getvalue())
            self.assertIn("tok/s", output.getvalue())
        finally:
            glow.finish()

    def test_update_before_start_is_refused(self):
        self.assertFalse(self.glow.update(1, 10))

    def test_redirected_output_does_not_start_animation(self):
        with patch("sys.stdout.isatty", return_value=False):
            self.glow.start(100)
        self.assertFalse(self.glow._active)


class AgentUITests(unittest.TestCase):
    def test_pending_tool_renders_immediately(self):
        class TTYOutput(StringIO):
            def isatty(self):
                return True

        output = TTYOutput()
        ui = AgentUI(output)
        ui.tool_pending("list_dir")
        try:
            self.assertIn("Listing files", output.getvalue())
        finally:
            ui.finish()

    def test_completed_tool_has_no_success_badge(self):
        class TTYOutput(StringIO):
            def isatty(self):
                return True

        output = TTYOutput()
        ui = AgentUI(output)
        ui.MIN_TOOL_SECONDS = 0
        ui.tool_started("read_file", {"path": "notes.txt"})
        ui.tool_executing()
        ui.tool_finished()
        written = output.getvalue()
        self.assertIn("Read file: notes.txt", written)
        self.assertNotIn("Done", written)
        self.assertNotIn(C["g"], written)
        self.assertNotIn("✓", written)
        self.assertNotIn("×", written)

    def test_visibility_delay_does_not_change_reported_tool_time(self):
        output = StringIO()
        ui = AgentUI(output)
        ui.tool_started("list_dir", {})
        with patch("macqwen.ui.time.perf_counter", side_effect=(10.0, 10.06)), \
                patch("macqwen.ui.time.sleep") as sleep:
            ui.tool_executing()
            ui.tool_finished()
        self.assertIn("Listed files  0.1s", output.getvalue())
        sleep.assert_called_once()


class FakeClock:
    """A clock the test moves by hand, so no test waits on real time."""

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def read(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def idle(self, seconds):
        self.now += seconds


def visible(text: str) -> str:
    """What the terminal still shows once the fade has overwritten itself."""
    body = re.sub(r"\033\[[0-9;]*m", "", text)
    screen = []
    for character in body:
        if character == "\b":
            if screen:
                screen.pop()
        else:
            screen.append(character)
    return "".join(screen)


class WordAnimatorTests(unittest.TestCase):
    def animator(self, clock, **options):
        return WordAnimator(
            output=StringIO(), is_tty=True, clock=clock.read,
            sleep=clock.sleep, **options,
        )

    def test_redirected_chat_never_prints_partial_words_or_ansi(self):
        output = StringIO()
        animator = WordAnimator(output=output, is_tty=False)
        animator.feed("par")
        self.assertEqual(output.getvalue(), "")
        animator.feed("tial word ")
        self.assertEqual(output.getvalue(), "partial word ")
        animator.finish()
        self.assertNotIn("\033", output.getvalue())

    def test_a_word_fades_through_every_shade(self):
        clock = FakeClock()
        animator = self.animator(clock)
        animator.feed("hello ")
        written = animator.output.getvalue()
        for shade in WordAnimator.FADE:
            self.assertIn(shade, written)

    def test_thinking_fade_lands_on_the_final_gray(self):
        clock = FakeClock()
        animator = self.animator(clock)
        animator.feed("thought ", C["gray"])
        written = animator.output.getvalue()
        for shade in WordAnimator.THINK_FADE:
            self.assertIn(shade, written)
        landing = (
            f"{C['gray']}thought{C['0']}" + "\b" * len("thought")
            + f"{C['gray']}thought{C['0']} "
        )
        self.assertIn(landing, written)
        self.assertNotIn(WordAnimator.FADE[-1], written)

    def test_the_fade_leaves_only_the_reply_on_screen(self):
        clock = FakeClock()
        animator = self.animator(clock)
        animator.feed("one two three ")
        animator.finish()
        self.assertEqual(visible(animator.output.getvalue()), "one two three ")

    def test_the_fade_never_costs_more_than_its_delay(self):
        clock = FakeClock()
        animator = self.animator(clock, delay=0.04)
        animator.feed("word ")
        self.assertAlmostEqual(sum(clock.slept), 0.04, places=6)

    def test_a_fast_model_is_not_slowed_down(self):
        clock = FakeClock()
        animator = self.animator(clock)
        animator.feed("first ")
        clock.slept.clear()
        clock.idle(0.002)
        animator.feed("second ")
        self.assertEqual(clock.slept, [])
        self.assertEqual(visible(animator.output.getvalue()), "first second ")

    def test_a_slow_model_gets_the_whole_fade(self):
        clock = FakeClock()
        animator = self.animator(clock, delay=0.04)
        animator.feed("first ")
        clock.slept.clear()
        clock.idle(0.5)
        animator.feed("second ")
        self.assertAlmostEqual(sum(clock.slept), 0.04, places=6)

    def test_the_fade_spends_at_most_half_the_model_time(self):
        clock = FakeClock()
        animator = self.animator(clock, delay=1.0)
        animator.feed("first ")
        clock.slept.clear()
        clock.idle(0.05)
        animator.feed("second ")
        self.assertAlmostEqual(sum(clock.slept), 0.025, places=6)

    def test_a_non_ascii_word_skips_the_fade(self):
        clock = FakeClock()
        animator = self.animator(clock)
        animator.feed("fotossintese\u00e9 ")
        self.assertEqual(clock.slept, [])
        self.assertNotIn("\b", animator.output.getvalue())

    def test_a_newline_lands_after_the_fade_not_inside_it(self):
        clock = FakeClock()
        animator = self.animator(clock)
        animator.feed("linha\nnova ")
        written = animator.output.getvalue()
        self.assertEqual(visible(written), "linha\nnova ")
        # the cursor never travels back over a line break, so no faded step
        # may carry one: the boundary rule leaves it in the tail
        faded = re.findall(r"\033\[38;5;\d+m(.*?)\033\[0m", written, re.DOTALL)
        self.assertTrue(faded)
        for step in faded:
            self.assertNotIn("\n", step)


class AsyncWordAnimatorTests(unittest.TestCase):
    def test_worker_leaves_the_expected_text(self):
        clock = FakeClock()
        output = StringIO()
        worker = AsyncWordAnimator(WordAnimator(
            output=output,
            is_tty=True,
            clock=clock.read,
            sleep=clock.sleep,
        ))
        worker.feed("one two ")
        worker.feed("three")
        worker.finish()
        self.assertEqual(visible(output.getvalue()), "one two three")

    def test_feed_does_not_wait_for_the_animation(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingAnimator:
            def feed(self, _text, _style):
                entered.set()
                release.wait(1.0)

            def finish(self, _style):
                pass

        worker = AsyncWordAnimator(BlockingAnimator())
        worker.feed("word ")
        self.assertTrue(entered.wait(0.2))
        release.set()
        worker.finish()

    def test_disabled_animation_keeps_words_without_fade_codes(self):
        output = StringIO()
        worker = AsyncWordAnimator(
            WordAnimator(output=output, is_tty=True), enabled=False
        )
        worker.feed("plain words ")
        worker.finish()
        self.assertEqual(output.getvalue(), "plain words ")


if __name__ == "__main__":
    unittest.main()


class FadeBudgetTests(unittest.TestCase):
    def test_the_fade_can_be_turned_off(self):
        clock = FakeClock()
        animator = WordAnimator(
            output=StringIO(), is_tty=True, delay=0.0,
            clock=clock.read, sleep=clock.sleep,
        )
        animator.feed("word ")
        self.assertEqual(clock.slept, [])
        self.assertEqual(animator.output.getvalue(), "word ")

    def test_the_environment_sets_the_default_budget(self):
        import importlib

        import macqwen.ui as ui

        with unittest.mock.patch.dict("os.environ", {"MACQWEN_FADE_MS": "50"}):
            reloaded = importlib.reload(ui)
            self.assertAlmostEqual(reloaded.WordAnimator.DEFAULT_DELAY, 0.05)
        importlib.reload(ui)
