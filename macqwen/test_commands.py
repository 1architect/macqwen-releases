"""The command table, driven without a model."""
from __future__ import annotations

import unittest

from macqwen import commands, preferences


class FakeSession:
    def __init__(self, profile="agent"):
        self.profile = profile
        self.preferences = dict(preferences.DEFAULTS)
        self.saved = 0
        self.stopped = False
        self.was_reset = False
        self.opened = False
        self.prompt = "default prompt"
        self.keys = []
        self.server_requested = False
        self.prompt_path = "/tmp/system-prompt-agent.txt"

    def save_preferences(self):
        self.saved += 1

    def reset(self):
        self.was_reset = True

    def stop(self):
        self.stopped = True

    def start_server(self):
        self.server_requested = True
        self.stopped = True

    def status(self):
        return "status text"

    def current_system_prompt(self):
        return self.preferences["system_prompt"] or self.prompt

    def set_system_prompt(self, value):
        self.preferences["system_prompt"] = value
        self.save_preferences()

    def system_prompt_path(self):
        return self.prompt_path

    def set_profile(self, profile):
        changed = profile != self.profile
        self.profile = profile
        self.preferences["profile"] = profile
        self.save_preferences()
        if changed:
            self.reset()
        return changed

    def list_api_keys(self):
        return "key status"

    def set_api_key(self, service):
        self.keys.append(("set", service))
        return f"set {service}"

    def delete_api_key(self, service):
        self.keys.append(("delete", service))
        return f"deleted {service}"

    def save_session(self, name):
        return f"saved {name}"

    def load_session(self, name):
        return f"loaded {name}"

    def list_sessions(self):
        return "no saved sessions"

    def delete_session(self, name):
        return f"deleted {name}"

    def model_settings(self, argument):
        return f"settings {argument}".strip()


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()

    def test_plain_text_is_not_a_command(self):
        self.assertIsNone(commands.dispatch(self.session, "hello there"))

    def test_multiline_paste_is_never_a_command(self):
        # pasted code that starts with a slash must reach the model
        self.assertIsNone(commands.dispatch(self.session, "/usr/bin/env\nsecond line"))

    def test_unknown_command_is_reported(self):
        out = commands.dispatch(self.session, "/nope")
        self.assertIn("unknown command", out)

    def test_aliases_reach_the_same_handler(self):
        for name in ("/thinking", "/think"):
            self.assertIn("thinking:", commands.dispatch(self.session, name))

    def test_deprecated_portuguese_aliases_are_rejected(self):
        for name in (
            "/limite", "/salvar", "/carregar", "/sessoes",
            "/apagar", "/ajuda", "/sair",
        ):
            with self.subTest(name=name):
                self.assertIn("unknown command", commands.dispatch(self.session, name))

    def test_api_keys_never_accept_an_inline_secret(self):
        self.assertIn("usage:", commands.dispatch(self.session, "/keys set tavily secret"))
        self.assertEqual(self.session.keys, [])

    def test_api_key_commands_route_by_service(self):
        self.assertEqual(commands.dispatch(self.session, "/keys"), "key status")
        commands.dispatch(self.session, "/keys set tavily")
        commands.dispatch(self.session, "/keys delete context7")
        self.assertEqual(self.session.keys, [("set", "tavily"), ("delete", "context7")])

    def test_settings_reaches_the_model_configurator(self):
        self.assertEqual(
            commands.dispatch(self.session, "/settings threshold 1.0"),
            "settings threshold 1.0",
        )

    def test_server_command_leaves_chat_and_requests_server(self):
        self.assertIn("starting", commands.dispatch(self.session, "/server"))
        self.assertTrue(self.session.server_requested)
        self.assertTrue(self.session.stopped)


class SettingTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()

    def test_thinking_toggles_and_persists(self):
        commands.dispatch(self.session, "/thinking on")
        self.assertTrue(self.session.preferences["thinking_enabled"])
        self.assertEqual(self.session.saved, 1)
        commands.dispatch(self.session, "/thinking hide")
        self.assertFalse(self.session.preferences["show_thinking"])

    def test_thinking_rejects_nonsense_without_saving(self):
        out = commands.dispatch(self.session, "/thinking sideways")
        self.assertIn("usage:", out)
        self.assertEqual(self.session.saved, 0)

    def test_max_tokens_accepts_a_limit_and_off(self):
        commands.dispatch(self.session, "/max-tokens 512")
        self.assertEqual(self.session.preferences["max_tokens"], 512)
        output = commands.dispatch(self.session, "/max-tokens off")
        self.assertEqual(self.session.preferences["max_tokens"], -1)
        self.assertIn("default (2048)", output)

    def test_thinking_budget_accepts_a_limit_and_off(self):
        commands.dispatch(self.session, "/think-budget 384")
        self.assertEqual(self.session.preferences["think_budget"], 384)
        commands.dispatch(self.session, "/think-budget off")
        self.assertEqual(self.session.preferences["think_budget"], -1)

    def test_thinking_budget_refuses_zero_and_text(self):
        for bad in ("0", "-5", "many"):
            with self.subTest(value=bad):
                output = commands.dispatch(self.session, f"/think-budget {bad}")
                self.assertIn("usage:", output)

    def test_max_tokens_refuses_zero_and_text(self):
        for bad in ("0", "-5", "many"):
            with self.subTest(value=bad):
                self.assertIn("usage:", commands.dispatch(self.session, f"/max-tokens {bad}"))

    def test_effort_is_restricted_to_known_levels(self):
        self.assertIn("usage:", commands.dispatch(self.session, "/effort extreme"))
        commands.dispatch(self.session, "/effort xhigh")
        self.assertEqual(self.session.preferences["effort"], "xhigh")

    def test_animation_toggles_and_persists(self):
        self.assertEqual(commands.dispatch(self.session, "/animate off"), "animate: off")
        self.assertFalse(self.session.preferences["animate"])
        self.assertEqual(self.session.saved, 1)
        self.assertIn("usage:", commands.dispatch(self.session, "/animate maybe"))

    def test_prompt_can_be_set_and_restored(self):
        shown = commands.dispatch(self.session, "/prompt")
        self.assertIn("default prompt", shown)
        self.assertIn(self.session.prompt_path, shown)
        commands.dispatch(self.session, "/prompt custom")
        self.assertEqual(self.session.preferences["system_prompt"], "custom")
        commands.dispatch(self.session, "/prompt default")
        self.assertEqual(self.session.preferences["system_prompt"], "")

    def test_profile_changes_and_resets_the_conversation(self):
        out = commands.dispatch(self.session, "/profile plain")
        self.assertEqual(self.session.profile, "plain")
        self.assertEqual(self.session.preferences["profile"], "plain")
        self.assertTrue(self.session.was_reset)
        self.assertIn("conversation reset", out)

    def test_profile_rejects_unknown_values(self):
        out = commands.dispatch(self.session, "/profile expert")
        self.assertIn("usage:", out)
        self.assertEqual(self.session.profile, "agent")

    def test_every_setting_command_writes_a_valid_preference(self):
        for text in ("/thinking on", "/max-tokens 64", "/think-budget 32", "/effort low",
                     "/stream off", "/animate off", "/approval auto"):
            commands.dispatch(self.session, text)
        for key, value in self.session.preferences.items():
            _, valid = preferences.SCHEMA[key]
            self.assertTrue(valid(value), f"{key}={value!r} is not valid")


class ProfileTests(unittest.TestCase):
    def test_agent_only_commands_are_hidden_from_plain(self):
        plain = FakeSession(profile="plain")
        out = commands.dispatch(plain, "/approval auto")
        self.assertIn("agent profile", out)
        self.assertEqual(plain.preferences["approval"], "ask")

    def test_help_lists_fewer_commands_in_plain(self):
        self.assertLess(
            len(commands.available("plain")), len(commands.available("agent"))
        )

    def test_help_mentions_every_available_command(self):
        text = commands.render_help("agent")
        for command in commands.available("agent"):
            self.assertIn(command.name, text)

    def test_help_hides_compatibility_aliases(self):
        text = commands.render_help("agent")
        for suffix in ("(/think)", "(/api-keys)", "(/exit, /q)"):
            self.assertNotIn(suffix, text)

    def test_help_summary_starts_after_the_longest_usage(self):
        lines = commands.render_help("agent").splitlines()
        by_name = dict(zip(
            (command.name for command in commands.available("agent")), lines
        ))
        longest = max(len(command.usage) for command in commands.available("agent"))
        for command in commands.available("agent"):
            self.assertGreaterEqual(
                by_name[command.name].index(command.summary), longest + 4
            )

    def test_help_omits_deprecated_portuguese_aliases(self):
        text = commands.render_help("agent")
        for name in (
            "/limite", "/salvar", "/carregar", "/sessoes",
            "/apagar", "/ajuda", "/sair",
        ):
            self.assertNotIn(name, text)

    def test_quit_stops_the_session(self):
        session = FakeSession()
        commands.dispatch(session, "/quit")
        self.assertTrue(session.stopped)


if __name__ == "__main__":
    unittest.main()
