from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from macqwen import preferences
from macqwen.backends.base import GenerationCancelled
from macqwen.session import (
    Session,
    _effective_turn_budgets,
    ask_approval,
    main,
    run_benchmark,
    run_turn_plain,
    token_stats_text,
    _run_generation,
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
        self.requested_limits = []

    def open_conversation(self, *_args, **_kwargs):
        self.pending = [10, 11]

    def generate(self, max_tokens):
        self.requested_limits.append(max_tokens)
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
    def test_generation_runner_observes_ctrl_c_while_worker_is_busy(self):
        def work(should_cancel):
            while not should_cancel():
                time.sleep(0.01)
            raise GenerationCancelled

        timer = threading.Timer(
            0.05, lambda: os.kill(os.getpid(), signal.SIGINT)
        )
        timer.start()
        try:
            result, error, presses = _run_generation(work)
        finally:
            timer.cancel()

        self.assertIsNone(result)
        self.assertIsInstance(error, GenerationCancelled)
        self.assertEqual(presses, 1)

    def test_generation_runner_reuses_one_worker_thread(self):
        idents = []

        def work(should_cancel):
            del should_cancel
            idents.append(threading.get_ident())
            return 7

        first, first_error, _ = _run_generation(work)
        second, second_error, _ = _run_generation(work)
        self.assertEqual((first, second), (7, 7))
        self.assertIsNone(first_error)
        self.assertIsNone(second_error)
        self.assertEqual(len(idents), 2)
        self.assertEqual(idents[0], idents[1])

    def test_ctrl_c_stages_close_then_quit(self):
        class LoadedBackend(FakeBackend):
            def __init__(self):
                self.tape = []
                self.pending = []
                self.reset_called = 0

            def reset(self):
                self.reset_called += 1

        backend = LoadedBackend()
        with tempfile.TemporaryDirectory() as root, \
                patch.object(sys, "argv", [
                    "session.py", "--model", "qwen27b",
                    "--model-path", str(Path(root) / "model"),
                    "--preferences-file", str(Path(root) / "preferences.json"),
                    "--api-keys-file", str(Path(root) / "keys.json"),
                ]), \
                patch("macqwen.session.build_backend", return_value=backend), \
                patch("macqwen.session.read_prompt", side_effect=(
                    "hello", KeyboardInterrupt, KeyboardInterrupt, KeyboardInterrupt,
                )), \
                patch("macqwen.session.run_turn_plain", return_value=1), \
                redirect_stdout(StringIO()) as output:
            self.assertEqual(main(), 0)

        self.assertEqual(backend.reset_called, 1)
        self.assertIn("answer stopped", output.getvalue())
        self.assertIn("conversation closed", output.getvalue())

    def test_seed_is_applied_after_backend_load_before_generation(self):
        import mlx.core as mx

        events = []

        class LoadedBackend(FakeBackend):
            def __init__(self):
                self.tape = []
                self.pending = []

        def load_backend(*_args, **_kwargs):
            events.append("backend")
            return LoadedBackend()

        def generate(*_args, **_kwargs):
            events.append("generation")

        with tempfile.TemporaryDirectory() as root, \
                patch.object(sys, "argv", [
                    "session.py", "--model", "qwen27b",
                    "--model-path", str(Path(root) / "model"),
                    "--preferences-file", str(Path(root) / "preferences.json"),
                    "--api-keys-file", str(Path(root) / "keys.json"),
                    "--seed", "37",
                ]), \
                patch("macqwen.session.build_backend", side_effect=load_backend), \
                patch("macqwen.session.run_turn_plain", side_effect=generate), \
                patch("macqwen.session.read_prompt", side_effect=("hello", "/quit")), \
                patch.object(
                    mx.random, "seed",
                    side_effect=lambda value: events.append(("seed", value)),
                ), redirect_stdout(StringIO()):
            self.assertEqual(main(), 0)

        self.assertEqual(events, ["backend", ("seed", 37), "generation"])

    def test_reap_xhigh_shared_thinking_budget_stays_one_total_ceiling(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "reap"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(
                '{"model_type": "qwen4_exp", '
                '"reap_prune": {"kept_experts": 288}}'
            )
            backend = SimpleNamespace(model_path=checkpoint)
            prefs = dict(
                preferences.DEFAULTS,
                thinking_enabled=True,
                effort="xhigh",
                max_tokens=2048,
                # Zero exercises the legacy shared-total sentinel before
                # preference-file migration turns it into the default.
                think_budget=0,
            )

            answer, requested, think, total = _effective_turn_budgets(
                backend, prefs, "agent"
            )

            self.assertEqual((answer, requested, think, total),
                             (2048, None, 4096, 2048))

    def test_reap_xhigh_caps_a_large_shared_total_without_adding_reasoning(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "reap"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(
                '{"reap_prune": {"kept_experts": 288}}'
            )
            backend = SimpleNamespace(model_path=checkpoint)
            prefs = dict(
                preferences.DEFAULTS,
                thinking_enabled=True,
                effort="xhigh",
                max_tokens=8192,
                think_budget=0,
            )

            answer, requested, think, total = _effective_turn_budgets(
                backend, prefs, "plain"
            )

            self.assertEqual((answer, requested, think, total),
                             (8192, None, 4096, 8192))

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

    def test_load_session_marks_pending_only_conversation_opened(self):
        class PendingBackend(FakeBackend):
            def load_session(self, name):
                self.tape = []
                self.pending = [5]
                return "loaded work"

        with tempfile.TemporaryDirectory() as root:
            session = Session(
                PendingBackend(), "plain", dict(preferences.DEFAULTS),
                "unused.json", Path(root) / "keys.json",
            )
            session.load_session("work")
            self.assertTrue(session.opened)

    def test_benchmark_reports_generated_token_ids(self):
        prefs = dict(preferences.DEFAULTS, max_tokens=2)
        with tempfile.TemporaryDirectory() as root:
            session = Session(
                BenchmarkBackend(), "plain", prefs, "unused.json", Path(root) / "keys.json"
            )
            result = run_benchmark(session, "hello", 1.0)
            self.assertEqual(result["token_ids"], [7, 8])
            self.assertEqual(result["generated_tokens"], 2)

    def test_benchmark_max_tokens_bounds_saved_thinking_budget(self):
        prefs = dict(
            preferences.DEFAULTS,
            thinking_enabled=True,
            max_tokens=2,
            think_budget=4096,
        )
        with tempfile.TemporaryDirectory() as root:
            backend = BenchmarkBackend()
            session = Session(
                backend, "plain", prefs, "unused.json", Path(root) / "keys.json"
            )
            run_benchmark(session, "hello", 1.0)
            self.assertEqual(backend.requested_limits, [2])

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


class PreparedQmmSessionTests(unittest.TestCase):
    def _bonsai_args(self, **overrides):
        args = SimpleNamespace(
            model_path="/models/b2",
            prefill_step_size=512,
            session_dir=None,
            prepared_qmm="on",
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_default_on_and_explicit_off_construct_different_settings(self):
        from macqwen.session import build_backend

        captured = {}

        class FakeBonsai:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.prepared_qmm_metadata = kwargs["prepared_qmm_metadata"]
                self._setting_sources = {}

        prefs = dict(preferences.DEFAULTS)
        with patch("models.bonsai2.backend.BonsaiBackend", FakeBonsai):
            build_backend("bonsai2", self._bonsai_args(prepared_qmm="on"), prefs)
            on_value = captured["prepared_qmm_metadata"]
            build_backend("bonsai2", self._bonsai_args(prepared_qmm="off"), prefs)
            off_value = captured["prepared_qmm_metadata"]
        self.assertTrue(on_value)
        self.assertFalse(off_value)
        self.assertNotEqual(on_value, off_value)

    def test_missing_flag_defaults_to_on(self):
        from macqwen.session import build_backend

        captured = {}

        class FakeBonsai:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.prepared_qmm_metadata = kwargs["prepared_qmm_metadata"]
                self._setting_sources = {}

        args = SimpleNamespace(
            model_path="/models/b2", prefill_step_size=512, session_dir=None
        )
        prefs = dict(preferences.DEFAULTS)
        with patch("models.bonsai2.backend.BonsaiBackend", FakeBonsai):
            build_backend("bonsai2", args, prefs)
        self.assertTrue(captured["prepared_qmm_metadata"])

    def test_cli_parsing_default_on_explicit_off_and_rejects_bad(self):
        from macqwen.session import build_backend

        seen = {}

        def fake_build(name, args, prefs):
            seen["prepared_qmm"] = getattr(args, "prepared_qmm", None)
            return FakeBackend()

        with tempfile.TemporaryDirectory() as root, patch.object(
            sys, "argv", [
                "session.py", "--model", "qwen27b",
                "--model-path", str(Path(root) / "model"),
                "--preferences-file", str(Path(root) / "preferences.json"),
                "--api-keys-file", str(Path(root) / "keys.json"),
            ]), patch(
                "macqwen.session.build_backend", side_effect=fake_build
            ), patch(
                "macqwen.session.read_prompt", side_effect=("/quit",)
            ), redirect_stdout(StringIO()):
            self.assertEqual(main(), 0)
        self.assertEqual(seen["prepared_qmm"], "on")

        with tempfile.TemporaryDirectory() as root, patch.object(
            sys, "argv", [
                "session.py", "--model", "qwen27b",
                "--model-path", str(Path(root) / "model"),
                "--preferences-file", str(Path(root) / "preferences.json"),
                "--api-keys-file", str(Path(root) / "keys.json"),
                "--prepared-qmm", "off",
            ]), patch(
                "macqwen.session.build_backend", side_effect=fake_build
            ), patch(
                "macqwen.session.read_prompt", side_effect=("/quit",)
            ), redirect_stdout(StringIO()):
            self.assertEqual(main(), 0)
        self.assertEqual(seen["prepared_qmm"], "off")

        with tempfile.TemporaryDirectory() as root, patch.object(
            sys, "argv", [
                "session.py", "--model", "qwen27b",
                "--model-path", str(Path(root) / "model"),
                "--preferences-file", str(Path(root) / "preferences.json"),
                "--api-keys-file", str(Path(root) / "keys.json"),
                "--prepared-qmm", "maybe",
            ]), redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit):
                main()

    def test_source_reporting_cli_vs_default(self):
        from macqwen.session import build_backend

        class FakeBonsai:
            def __init__(self, **kwargs):
                self.prepared_qmm_metadata = kwargs["prepared_qmm_metadata"]

        prefs = dict(preferences.DEFAULTS)
        with patch("models.bonsai2.backend.BonsaiBackend", FakeBonsai):
            with patch.object(sys, "argv", ["session.py"]):
                backend = build_backend(
                    "bonsai2", self._bonsai_args(prepared_qmm="on"), prefs
                )
                self.assertEqual(
                    backend._setting_sources.get("prepared-qmm"), "default"
                )
            with patch.object(
                sys, "argv", ["session.py", "--prepared-qmm", "off"]
            ):
                backend = build_backend(
                    "bonsai2", self._bonsai_args(prepared_qmm="off"), prefs
                )
                self.assertEqual(backend._setting_sources.get("prepared-qmm"), "CLI")

    def test_write_refusal_surfaces_restart_through_model_settings(self):
        class RefusingBackend(FakeBackend):
            def configure(self, argument):
                if argument == "prepared-qmm":
                    return "prepared-qmm        on"
                if argument.startswith("prepared-qmm "):
                    raise ValueError(
                        "prepared-qmm applies at startup; restart the model"
                    )
                return "Bonsai-2 settings"

        with tempfile.TemporaryDirectory() as root:
            session = Session(
                RefusingBackend(),
                "plain",
                dict(preferences.DEFAULTS),
                "unused.json",
                Path(root) / "keys.json",
            )
            self.assertIn("prepared-qmm", session.model_settings("prepared-qmm"))
            refused = session.model_settings("prepared-qmm off")
            self.assertIn("could not change settings", refused)
            self.assertIn("restart", refused.lower())


if __name__ == "__main__":
    unittest.main()
