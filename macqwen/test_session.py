from __future__ import annotations

import tempfile
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

from macqwen import preferences
from macqwen.session import (
    Session,
    ask_approval,
    run_benchmark,
    run_turn_plain,
    token_stats_text,
)
from macqwen.ui import IngestGlow


class FakeBackend:
    tape = []
    pending = []
    thinking_enabled = False

    def reset(self):
        pass


class BenchmarkBackend(FakeBackend):
    routing_profile = "exact-quality"

    def __init__(self):
        self.tape = []
        self.pending = []

    def open_conversation(self, *_args, **_kwargs):
        self.pending = [10, 11]

    def generate(self, max_tokens):
        self.tape.extend(self.pending)
        self.pending = []
        self.tape.extend([7, 8][:max_tokens])
        return "ok", SimpleNamespace(
            tokens=min(2, max_tokens), prompt_tokens=2,
            prompt_rate=4.0, rate=2.0,
        )


class ConfigurableBackend(FakeBackend):
    def __init__(self):
        self.threshold = 0.85

    def configure(self, argument):
        name, value = argument.split()
        self.threshold = float(value)
        return f"{name}: {value}"


class InterruptingBackend(FakeBackend):
    def __init__(self):
        self.tape = []
        self.pending = []
        self.reset_called = False

    def open_conversation(self, *_args, **_kwargs):
        self.pending = [1, 2]

    def generate(self, **_kwargs):
        raise KeyboardInterrupt

    def reset(self):
        self.reset_called = True
        self.tape = []
        self.pending = []


class FakeTools:
    def __init__(self, repo):
        self.repo = repo


class SessionTests(unittest.TestCase):
    def test_agent_token_stats_combine_model_segments(self):
        stats = [
            SimpleNamespace(
                prompt_tokens=2000, prefill_seconds=100.0,
                tokens=100, seconds=50.0,
                tail_tokens=92, tail_seconds=40.0,
            ),
            SimpleNamespace(
                prompt_tokens=300, prefill_seconds=10.0,
                tokens=20, seconds=5.0,
                tail_tokens=12, tail_seconds=5.0,
            ),
        ]
        text = token_stats_text(stats, context=2420, elapsed=166.0)
        self.assertIn("2,300 new tok @ 20.9 tok/s", text)
        self.assertIn("gen 120 @ 2.2 tok/s", text)
        self.assertIn("tail 104 @ 2.3 tok/s", text)
        self.assertIn("ctx 2,420", text)

    def test_agent_token_stats_omit_an_unavailable_tail(self):
        stats = [SimpleNamespace(
            prompt_tokens=10, prefill_seconds=2.0,
            tokens=4, seconds=2.0,
        )]
        text = token_stats_text(stats, context=14, elapsed=4.0)
        self.assertNotIn("tail", text)

    def test_approval_accepts_english_and_portuguese(self):
        for answer in ("y", "yes", "s", "sim"):
            with self.subTest(answer=answer):
                self.assertTrue(
                    ask_approval("write_file", {"path": "x"}, lambda _prompt: answer)
                )

    def test_approval_denies_unknown_or_empty_input(self):
        answers = iter(("", "maybe"))
        self.assertFalse(
            ask_approval("run_command", {"command": "make"}, lambda _prompt: next(answers))
        )

    def test_agent_status_reads_the_toolbox_repo(self):
        with tempfile.TemporaryDirectory() as root:
            prefs = dict(preferences.DEFAULTS, workspace=root)
            with patch("macqwen.session.Toolbox.build", side_effect=FakeTools), \
                    patch("macqwen.session.rss_gb", return_value=0.1):
                session = Session(
                    FakeBackend(), "agent", prefs, "unused.json", Path(root) / "keys.json"
                )
                self.assertIn(str(session.repo.root), session.status())

    def test_reset_rebuilds_tools_after_workspace_change(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            prefs = dict(preferences.DEFAULTS, workspace=first)
            with patch("macqwen.session.Toolbox.build", side_effect=FakeTools):
                session = Session(
                    FakeBackend(), "agent", prefs, "unused.json", Path(first) / "keys.json"
                )
                prefs["workspace"] = second
                session.reset()
                self.assertEqual(session.repo.root, session.repo.root.__class__(second).resolve())

    def test_plain_prompt_omits_unused_workspace_facts(self):
        with tempfile.TemporaryDirectory() as root:
            prefs = dict(preferences.DEFAULTS, workspace=root)
            session = Session(
                FakeBackend(), "plain", prefs, "unused.json", Path(root) / "keys.json"
            )
            self.assertNotIn(f"workspace root: {root}", session.current_system_prompt())

    def test_prompt_file_can_be_edited_outside_the_chat(self):
        with tempfile.TemporaryDirectory() as root:
            prefs_path = Path(root) / "preferences.json"
            session = Session(
                FakeBackend(), "plain", dict(preferences.DEFAULTS), prefs_path,
                Path(root) / "keys.json",
            )
            session.system_prompt_path().write_text("external prompt\n")
            self.assertEqual(session.current_system_prompt(), "external prompt")

    def test_benchmark_session_does_not_migrate_or_write_preferences(self):
        with tempfile.TemporaryDirectory() as root:
            prefs_path = Path(root) / "preferences.json"
            prefs = dict(preferences.DEFAULTS, system_prompt="legacy prompt")
            session = Session(
                FakeBackend(), "plain", prefs, prefs_path,
                Path(root) / "keys.json", migrate_system_prompt=False,
            )
            self.assertFalse(prefs_path.exists())
            self.assertFalse(session.system_prompt_path().exists())
            self.assertEqual(prefs["system_prompt"], "legacy prompt")

    def test_each_profile_has_a_different_prompt_file(self):
        with tempfile.TemporaryDirectory() as root:
            prefs_path = Path(root) / "preferences.json"
            session = Session(
                FakeBackend(), "plain", dict(preferences.DEFAULTS), prefs_path,
                Path(root) / "keys.json",
            )
            plain_path = session.system_prompt_path()
            plain_path.write_text("plain custom\n")
            with patch("macqwen.session.Toolbox.build", side_effect=FakeTools):
                session.set_profile("agent")
                agent_path = session.system_prompt_path()
                agent_path.write_text("agent custom\n")
                self.assertNotEqual(plain_path, agent_path)
                self.assertEqual(session.current_system_prompt(), "agent custom")
                session.set_profile("plain")
            self.assertEqual(session.current_system_prompt(), "plain custom")

    def test_profile_change_clears_agent_tools(self):
        with tempfile.TemporaryDirectory() as root:
            prefs = dict(preferences.DEFAULTS, workspace=root)
            with patch("macqwen.session.Toolbox.build", side_effect=FakeTools):
                session = Session(
                    FakeBackend(), "agent", prefs, Path(root) / "preferences.json",
                    Path(root) / "keys.json",
                )
                self.assertIsNotNone(session.tools)
                self.assertTrue(session.set_profile("plain"))
                self.assertIsNone(session.tools)

    def test_prefill_interrupt_resets_the_conversation(self):
        backend = InterruptingBackend()
        with tempfile.TemporaryDirectory() as root:
            session = Session(
                backend,
                "plain",
                dict(preferences.DEFAULTS),
                "unused.json",
                Path(root) / "keys.json",
            )
            output = StringIO()
            with redirect_stdout(output):
                run_turn_plain(session, "hello", IngestGlow())
            self.assertTrue(backend.reset_called)
            self.assertFalse(session.opened)
            self.assertIn("conversation reset", output.getvalue())

    def test_benchmark_reports_generated_token_ids(self):
        prefs = dict(preferences.DEFAULTS, max_tokens=2)
        with tempfile.TemporaryDirectory() as root:
            session = Session(
                BenchmarkBackend(), "plain", prefs, "unused.json", Path(root) / "keys.json"
            )
            result = run_benchmark(session, "hello", 1.0)
            self.assertEqual(result["token_ids"], [7, 8])
            self.assertEqual(result["generated_tokens"], 2)

    def test_api_key_input_is_hidden_and_saved(self):
        with tempfile.TemporaryDirectory() as root:
            key_path = Path(root) / "api_keys.json"
            session = Session(
                FakeBackend(),
                "plain",
                dict(preferences.DEFAULTS),
                "unused.json",
                key_path,
            )
            with patch("macqwen.session.getpass.getpass", return_value="private-value"):
                result = session.set_api_key("tavily")
            self.assertNotIn("private-value", result)
            self.assertIn('"tavily": "private-value"', key_path.read_text())

    def test_model_settings_apply_only_to_the_live_session(self):
        with tempfile.TemporaryDirectory() as root:
            preferences_path = Path(root) / "preferences.json"
            session = Session(
                ConfigurableBackend(),
                "plain",
                dict(preferences.DEFAULTS),
                preferences_path,
                Path(root) / "keys.json",
            )
            result = session.model_settings("threshold 1.0")
            self.assertEqual(result, "threshold: 1.0")
            self.assertEqual(session.backend.threshold, 1.0)
            self.assertFalse(preferences_path.exists())


if __name__ == "__main__":
    unittest.main()
