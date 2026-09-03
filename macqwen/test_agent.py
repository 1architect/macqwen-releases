"""The agent loop, driven by a scripted engine so no model is needed."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from macqwen.agent import Limits, run_agent
from macqwen.tools.repo import Repo


@dataclass
class Stats:
    finish: str = "stop"
    host_free_gb: float = 32.0
    swap_gb: float = 0.0


class ScriptedEngine:
    """Replays prepared model turns and records what the loop fed back."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.thinking_enabled = False
        self.requested_limits = []
        self.pending = []
        self.appended_text = []
        self.tool_results = []
        self.invariant = True
        self.opened = None

    def open_conversation(self, system, task, tools, reasoning_effort):
        self.opened = (system, task, tools, reasoning_effort)

    def generate(self, max_tokens, out, on_prefilled=None,
                 on_prefill_progress=None):
        self.requested_limits.append(max_tokens)
        if on_prefill_progress is not None:
            on_prefill_progress(len(self.pending), len(self.pending))
        if on_prefilled is not None:
            on_prefilled()
        if not self.turns:
            return "", Stats(finish="length")
        return self.turns.pop(0)

    def check_invariant(self):
        return self.invariant

    def append_text(self, text):
        self.appended_text.append(text)

    def append_tool_results(self, results):
        self.tool_results.append(results)


CALL = """<tool_call>
<function=read_file>
<parameter=path>
notes.txt
</parameter>
</function>
</tool_call>"""

WRITE = """<tool_call>
<function=write_file>
<parameter=path>
created.txt
</parameter>
<parameter=content>
new
</parameter>
</function>
</tool_call>"""


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        Path(self.dir.name, "notes.txt").write_text("hello\n")
        self.repo = Repo(self.dir.name)
        self.said = []

    def tearDown(self):
        self.dir.cleanup()

    def out(self, text=""):
        self.said.append(str(text))

    def test_final_answer_stops(self):
        engine = ScriptedEngine([("the answer", Stats(finish="stop"))])
        self.assertEqual(run_agent(engine, self.repo, self.out), "answer")

    def test_final_answer_clears_live_ui(self):
        engine = ScriptedEngine([("the answer", Stats(finish="stop"))])
        ui = MagicMock()
        self.assertEqual(run_agent(engine, self.repo, self.out, ui=ui), "answer")
        ui.finish.assert_called_once_with()

    def test_each_model_segment_reports_stats(self):
        expected = Stats(finish="stop")
        engine = ScriptedEngine([("the answer", expected)])
        seen = []
        run_agent(engine, self.repo, self.out, on_stats=seen.append)
        self.assertEqual(seen, [expected])

    def test_thinking_has_capacity_outside_the_answer_limit(self):
        engine = ScriptedEngine([("the answer", Stats(finish="stop"))])
        engine.thinking_enabled = True
        run_agent(
            engine,
            self.repo,
            self.out,
            Limits(max_tokens=120, think_tokens=512),
        )
        self.assertEqual(engine.requested_limits, [632])

    def test_tool_call_runs_and_feeds_back(self):
        engine = ScriptedEngine([
            (CALL, Stats(finish="stop")),
            ("done", Stats(finish="stop")),
        ])
        self.assertEqual(run_agent(engine, self.repo, self.out), "answer")
        self.assertEqual(len(engine.tool_results), 1)
        self.assertIn("hello", engine.tool_results[0][0])

    def test_tool_error_is_reported_not_raised(self):
        bad = CALL.replace("notes.txt", "../escape.txt")
        engine = ScriptedEngine([
            (bad, Stats(finish="stop")),
            ("recovered", Stats(finish="stop")),
        ])
        self.assertEqual(run_agent(engine, self.repo, self.out), "answer")
        self.assertIn("error", engine.tool_results[0][0].lower())

    def test_denied_mutating_tool_does_not_run(self):
        engine = ScriptedEngine([
            (WRITE, Stats(finish="stop")),
            ("done", Stats(finish="stop")),
        ])
        reason = run_agent(
            engine, self.repo, self.out, approve=lambda _name, _args: False
        )
        self.assertEqual(reason, "answer")
        self.assertFalse(Path(self.dir.name, "created.txt").exists())
        self.assertIn("denied", engine.tool_results[0][0].lower())

    def test_approved_mutating_tool_runs(self):
        engine = ScriptedEngine([
            (WRITE, Stats(finish="stop")),
            ("done", Stats(finish="stop")),
        ])
        run_agent(engine, self.repo, self.out, approve=lambda _name, _args: True)
        self.assertEqual(Path(self.dir.name, "created.txt").read_text(), "new")

    def test_ui_stops_progress_before_approval_and_resumes_execution(self):
        engine = ScriptedEngine([
            (WRITE, Stats(finish="stop")),
            ("done", Stats(finish="stop")),
        ])
        events = []
        ui = MagicMock()
        ui.tool_started.side_effect = lambda *_args: events.append("started")
        ui.tool_approval.side_effect = lambda: events.append("approval")
        ui.tool_executing.side_effect = lambda: events.append("executing")
        ui.tool_finished.side_effect = lambda **_kwargs: events.append("finished")
        run_agent(
            engine, self.repo, self.out,
            approve=lambda *_args: (events.append("approved"), True)[1],
            ui=ui,
        )
        self.assertEqual(
            events[:5], ["started", "approval", "approved", "executing", "finished"]
        )

    def test_denial_stops_progress_without_execution(self):
        engine = ScriptedEngine([
            (WRITE, Stats(finish="stop")),
            ("done", Stats(finish="stop")),
        ])
        events = []
        ui = MagicMock()
        ui.tool_started.side_effect = lambda *_args: events.append("started")
        ui.tool_approval.side_effect = lambda: events.append("approval")
        ui.tool_executing.side_effect = lambda: events.append("executing")
        ui.tool_finished.side_effect = lambda **_kwargs: events.append("finished")
        run_agent(
            engine, self.repo, self.out,
            approve=lambda *_args: (events.append("denied"), False)[1],
            ui=ui,
        )
        self.assertEqual(events[:4], ["started", "approval", "denied", "finished"])

    def test_unclosed_think_is_closed_once_then_gives_up(self):
        turns = [("<think>thinking", Stats(finish="length"))] * 5
        engine = ScriptedEngine(turns)
        reason = run_agent(engine, self.repo, self.out,
                           Limits(max_forced_closes=2))
        self.assertEqual(reason, "truncated")
        self.assertEqual(engine.appended_text.count("\n</think>\n\n"), 2)

    def test_broken_invariant_stops_immediately(self):
        engine = ScriptedEngine([(CALL, Stats())])
        engine.invariant = False
        self.assertEqual(run_agent(engine, self.repo, self.out), "invariant")

    def test_low_memory_stops(self):
        engine = ScriptedEngine([(CALL, Stats(host_free_gb=0.5))])
        reason = run_agent(engine, self.repo, self.out, Limits(min_free_gb=2.0))
        self.assertEqual(reason, "memory")

    def test_swap_growth_stops(self):
        engine = ScriptedEngine([(CALL, Stats(swap_gb=9.0))])
        reason = run_agent(engine, self.repo, self.out,
                           Limits(max_swap_growth_gb=4.0),
                           host_memory=lambda: (32.0, 0.0))
        self.assertEqual(reason, "swap")

    def test_turn_budget_is_honoured(self):
        engine = ScriptedEngine([(CALL, Stats())] * 10)
        reason = run_agent(engine, self.repo, self.out, Limits(max_turns=3))
        self.assertEqual(reason, "max-turns")
        self.assertEqual(len(engine.tool_results), 3)

    def test_missing_memory_reporting_skips_those_guards(self):
        # a runtime that cannot report memory must not stop for it
        engine = ScriptedEngine([("answer", Stats(host_free_gb=None, swap_gb=None))])
        self.assertEqual(run_agent(engine, self.repo, self.out), "answer")


if __name__ == "__main__":
    unittest.main()
