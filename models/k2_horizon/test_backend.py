from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from macqwen.conversation import EXTRA_REASONING

from models.k2_horizon.backend import K2HorizonBackend


class FakeTokenizer:
    eos_token_ids = [1]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids):
        return "".join(chr(value) for value in ids)

    def convert_tokens_to_ids(self, token):
        return 999 if token == "<|ifm|im_end|>" else None

    def apply_chat_template(self, messages, **options):
        self.messages = messages
        self.options = options
        return "rendered"


class BackendTests(unittest.TestCase):
    def backend(self):
        tokenizer = FakeTokenizer()
        patches = (
            patch(
                "models.k2_horizon.backend.resolve_k2_horizon",
                return_value=Path("/models/k2"),
            ),
            patch("mlx_lm.load", return_value=(object(), tokenizer)),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return K2HorizonBackend("k2"), tokenizer

    def test_template_maps_xhigh_and_keeps_k2_options_inside_the_model(self):
        backend, tokenizer = self.backend()
        backend.open_conversation(
            "system", "user", tools=[], enable_thinking=True,
            reasoning_effort="xhigh",
        )
        self.assertEqual(tokenizer.options["reasoning_effort"], "high")
        self.assertEqual(tokenizer.options["tool_call_format"], "json")

    def test_server_style_high_prompt_is_restored_to_native_k2_high(self):
        backend, tokenizer = self.backend()
        backend.tokenizer.apply_chat_template(
            [{
                "role": "system",
                "content": EXTRA_REASONING["high"] + "\n\nsystem",
            }],
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="medium",
        )
        self.assertEqual(tokenizer.messages[0]["content"], "system")
        self.assertEqual(tokenizer.options["reasoning_effort"], "high")

    def test_no_thinking_uses_the_matching_k2_tag(self):
        backend, _tokenizer = self.backend()
        backend.open_conversation(
            "system", "user", enable_thinking=False,
            reasoning_effort="medium",
        )
        rendered = "".join(chr(value) for value in backend.pending)
        self.assertTrue(rendered.endswith(
            "<|ifm|im_start|>assistant\n<ifm|think_fast>\n"
            "</ifm|think_fast>"
        ))

    def test_server_history_adds_the_thinking_field_k2_requires(self):
        backend, tokenizer = self.backend()
        backend.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "again"},
            ],
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="medium",
        )
        self.assertEqual(tokenizer.messages[1]["think_fast"], "")

    def test_stop_is_rewound_so_the_shared_tape_contract_stays_unchanged(self):
        backend, _tokenizer = self.backend()
        backend.pending = [10, 11]
        backend.cache = [SimpleNamespace(offset=0)]

        def generate_step(prompt, _model, **options):
            backend.cache[0].offset += len(prompt)
            options["prompt_progress_callback"](len(prompt), len(prompt))
            for value in (65, 66, 999):
                backend.cache[0].offset += 1
                yield value, None

        with patch("mlx_lm.generate.generate_step", generate_step):
            text, stats = backend.generate(3)

        self.assertEqual(text, "AB")
        self.assertEqual(stats.finish, "stop")
        self.assertEqual(stats.tokens, 2)
        self.assertEqual(backend.tape, [10, 11, 65, 66])
        self.assertTrue(backend.check_invariant())

    def test_session_replays_the_saved_tape(self):
        backend, _tokenizer = self.backend()
        with tempfile.TemporaryDirectory() as directory:
            backend.session_dir = Path(directory)
            backend.tape = [1, 2, 3]
            backend.turn_closed = False
            self.assertTrue(backend.save_session("work").startswith("saved work"))
            backend.reset()
            self.assertTrue(backend.load_session("work").startswith("loaded work"))
        self.assertEqual(backend.tape, [1, 2, 3])
        self.assertFalse(backend.turn_closed)
        self.assertTrue(backend._replay_needed)


if __name__ == "__main__":
    unittest.main()
