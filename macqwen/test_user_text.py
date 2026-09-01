from __future__ import annotations

import unittest

from macqwen.text import build_user_encoder


class _Added:
    def __init__(self, content: str):
        self.content = content


class _Tokenizer:
    """Minimal stand-in that turns marker strings into single control ids."""

    MARKERS = {"<think>": 900, "</think>": 901, "<|im_end|>": 902}

    def __init__(self):
        self.added_tokens_decoder = {
            value: _Added(key) for key, value in self.MARKERS.items()
        }

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = []
        position = 0
        while position < len(text):
            for marker, value in self.MARKERS.items():
                if text.startswith(marker, position):
                    ids.append(value)
                    position += len(marker)
                    break
            else:
                ids.append(ord(text[position]))
                position += 1
        return {"input_ids": ids}

    def decode(self, ids: list[int]) -> str:
        reverse = {value: key for key, value in self.MARKERS.items()}
        return "".join(reverse.get(value, chr(value)) for value in ids)


class UserTextTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _Tokenizer()
        self.pattern, self.encode = build_user_encoder(self.tokenizer)

    def test_pasted_markers_stay_plain_text(self):
        paste = "continue:\n<think>\nold notes\n</think>\ndone<|im_end|>tail"
        ids = self.encode(paste)
        self.assertEqual(len(self.pattern.findall(paste)), 3)
        self.assertFalse([value for value in ids if value >= 900])
        self.assertEqual(self.tokenizer.decode(ids), paste)

    def test_plain_text_is_untouched(self):
        message = "faca um resumo curto"
        self.assertFalse(self.pattern.findall(message))
        self.assertEqual(
            self.encode(message), self.tokenizer(message)["input_ids"]
        )

    def test_marker_at_both_edges(self):
        for paste in ("<think>x", "x</think>", "<think></think>"):
            ids = self.encode(paste)
            self.assertFalse([value for value in ids if value >= 900])
            self.assertEqual(self.tokenizer.decode(ids), paste)


if __name__ == "__main__":
    unittest.main()
