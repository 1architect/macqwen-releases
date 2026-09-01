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
    def setUp(self):
        self.session = FakeSession()

    def test_agent_only_commands_are_hidden_from_plain(self):
        plain = FakeSession(profile="plain")
        out = commands.dispatch(plain, "/approval auto")
        self.assertIn("agent profile", out)
        self.assertEqual(plain.preferences["approval"], "ask")

    def test_help_keeps_six_primary_commands_for_each_profile(self):
        self.assertEqual(len(commands.primary_commands("plain")), 6)
        self.assertEqual(len(commands.primary_commands("agent")), 6)

    def test_help_all_keeps_profile_specific_commands_filtered(self):
        self.assertIn("/approval", commands.render_help("agent", all_commands=True))
        self.assertNotIn("/approval", commands.render_help("plain", all_commands=True))

    def test_config_help_shows_only_relevant_tool_controls(self):
        plain = commands.dispatch(FakeSession("plain"), "/config")
        agent = commands.dispatch(FakeSession("agent"), "/config")
        for text in (plain, agent):
            self.assertIn("/config keys", text)
        self.assertNotIn("/config approval", plain)
        self.assertNotIn("/config workspace", plain)
        self.assertIn("/config approval", agent)
        self.assertIn("/config workspace", agent)

    def test_web_shortcuts_use_primary_command_metadata(self):
        self.assertEqual(
            commands.web_shortcuts(),
            (("help", "/help"), ("new", "/new"),
             ("config", "/config"), ("status", "/status")),
        )

    def test_help_mentions_every_available_command(self):
        text = commands.render_help("agent")
        for command in commands.primary_commands("agent"):
            self.assertIn(command.name, text)

    def test_help_hides_compatibility_aliases(self):
        text = commands.render_help("agent")
        for suffix in ("(/think)", "(/api-keys)", "(/exit, /q)"):
            self.assertNotIn(suffix, text)

    def test_help_summary_starts_after_the_longest_usage(self):
        lines = commands.render_help("agent").splitlines()
        by_name = dict(zip(
            (command.name for command in commands.primary_commands("agent")), lines
        ))
        longest = max(len(command.usage) for command in commands.primary_commands("agent"))
        for command in commands.primary_commands("agent"):
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

    def test_new_and_reset_share_the_same_behavior(self):
        new = commands.dispatch(self.session, "/new")
        self.session.was_reset = False
        reset = commands.dispatch(self.session, "/reset")
        self.assertEqual(new, reset)
        self.assertTrue(self.session.was_reset)

    def test_session_groups_state_commands(self):
        self.assertIn("saved demo", commands.dispatch(self.session, "/session save demo"))
        self.assertIn("loaded demo", commands.dispatch(self.session, "/session load demo"))
        self.assertIn("no saved sessions", commands.dispatch(self.session, "/session list"))
        self.assertIn("deleted demo", commands.dispatch(self.session, "/session delete demo"))

    def test_config_groups_setting_commands(self):
        self.assertIn("thinking: on", commands.dispatch(self.session, "/config thinking on"))
        self.assertIn("settings threshold 1.0", commands.dispatch(self.session, "/config model threshold 1.0"))
        self.assertIn("animate: off", commands.dispatch(self.session, "/config display animate off"))

    def test_help_all_shows_compatibility_reference(self):
        text = commands.dispatch(self.session, "/help all")
        self.assertIn("Compatibility commands:", text)
        for name in ("/server", "/settings", "/reset", "/save"):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()


class EffortLevelTests(unittest.TestCase):
    """`/effort` carried its own copy of the level list and rejected `high`
    after the schema gained it. The command and the schema now read the same
    tuple, and this test fails if anyone splits them again."""

    def test_the_command_accepts_every_schema_level(self):
        from macqwen.preferences import EFFORT_LEVELS, SCHEMA

        _, valid = SCHEMA["effort"]
        for level in EFFORT_LEVELS:
            self.assertTrue(valid(level), level)

    def test_the_command_body_holds_no_second_copy(self):
        import inspect

        from macqwen import commands

        body = inspect.getsource(commands._effort)
        self.assertIn("EFFORT_LEVELS", body)
        self.assertNotIn('"xhigh"', body)
