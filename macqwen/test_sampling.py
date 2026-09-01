"""The sampler decides what the chat says, so its boundaries matter.

Greedy has to stay reachable and exact, because every benchmark in this
project proves a change left the trajectory alone by comparing token IDs.
"""
from __future__ import annotations

import unittest

import mlx.core as mx

from macqwen.sampling import INSTRUCT, THINKING, Sampler, Sampling


class SettingsTests(unittest.TestCase):
    def test_the_default_matches_qwens_thinking_mode(self):
        s = Sampling()
        self.assertEqual(s.temperature, THINKING["temperature"])
        self.assertEqual(s.top_p, THINKING["top_p"])
        self.assertEqual(s.top_k, THINKING["top_k"])
        self.assertEqual(s.presence_penalty, THINKING["presence_penalty"])
        self.assertFalse(s.greedy)

    def test_instruct_mode_carries_the_repetition_penalty(self):
        self.assertEqual(INSTRUCT["presence_penalty"], 1.5)

    def test_temperature_zero_is_greedy(self):
        self.assertTrue(Sampling.greedy_settings().greedy)
        self.assertTrue(Sampling(temperature=0.0).greedy)

    def test_preferences_round_trip(self):
        s = Sampling.from_preferences(
            {"temperature": 0.7, "top_p": 0.8, "top_k": 5,
             "min_p": 0.1, "presence_penalty": 1.5}
        )
        self.assertEqual((s.temperature, s.top_p, s.top_k), (0.7, 0.8, 5))
        self.assertEqual((s.min_p, s.presence_penalty), (0.1, 1.5))

    def test_describe_names_greedy(self):
        self.assertEqual(Sampling.greedy_settings().describe(), "greedy")
        self.assertIn("temp", Sampling().describe())


class SamplerTests(unittest.TestCase):
    def logits(self, values):
        return mx.array([values], dtype=mx.float32)

    def test_every_path_returns_shape_one(self):
        # The decode loop feeds `token[None]` to the model, so a scalar here
        # raises "not enough values to unpack" one layer down.
        row = self.logits([0.1, 5.0, 0.2, 0.3])
        for settings in (
            Sampling.greedy_settings(),
            Sampling(),
            Sampling(temperature=1.0, top_k=0, top_p=1.0),
            Sampling(temperature=1.0, min_p=0.5),
            Sampling(temperature=1.0, presence_penalty=1.0),
        ):
            token = Sampler(settings)(row)
            self.assertEqual(token.shape, (1,), settings.describe())
            self.assertEqual(token[None].shape, (1, 1))

    def test_greedy_returns_the_argmax(self):
        sampler = Sampler(Sampling.greedy_settings())
        token = sampler(self.logits([0.1, 5.0, 0.2, 0.3]))
        self.assertEqual(int(token.item()), 1)

    def test_greedy_is_deterministic_across_calls(self):
        sampler = Sampler(Sampling.greedy_settings())
        row = self.logits([1.0, 2.0, 3.0, 2.5])
        picks = {int(sampler(row).item()) for _ in range(20)}
        self.assertEqual(picks, {2})

    def test_top_k_one_collapses_to_the_argmax(self):
        sampler = Sampler(Sampling(temperature=1.0, top_k=1, top_p=1.0))
        row = self.logits([0.0, 9.0, 0.0, 0.0])
        picks = {int(sampler(row).item()) for _ in range(20)}
        self.assertEqual(picks, {1})

    def test_sampling_can_pick_something_other_than_the_argmax(self):
        sampler = Sampler(Sampling(temperature=2.0, top_k=0, top_p=1.0))
        row = self.logits([1.0, 1.05, 1.0, 1.0])
        picks = {int(sampler(row).item()) for _ in range(60)}
        self.assertGreater(len(picks), 1)

    def test_a_dominant_token_survives_top_p(self):
        # carried - ordered < top_p keeps the first token even when it alone
        # exceeds the nucleus, so the candidate set is never empty.
        sampler = Sampler(Sampling(temperature=1.0, top_p=0.5, top_k=0))
        row = self.logits([0.0, 20.0, 0.0, 0.0])
        picks = {int(sampler(row).item()) for _ in range(20)}
        self.assertEqual(picks, {1})

    def test_presence_penalty_pushes_a_seen_token_down(self):
        sampler = Sampler(Sampling(temperature=0.01, top_k=0, top_p=1.0,
                                   presence_penalty=2.0))
        row = self.logits([0.0, 1.0, 0.9, 0.0])
        self.assertEqual(int(sampler(row).item()), 1)
        sampler.observe(1)
        picks = {int(sampler(row).item()) for _ in range(20)}
        self.assertNotIn(1, picks)

    def test_reset_forgets_what_was_seen(self):
        sampler = Sampler(Sampling(presence_penalty=1.0))
        sampler.observe(7)
        self.assertTrue(sampler._seen)
        sampler.reset()
        self.assertFalse(sampler._seen)

    def test_greedy_records_nothing_so_it_stays_cheap(self):
        sampler = Sampler(Sampling.greedy_settings())
        sampler.observe(7)
        self.assertFalse(sampler._seen)


if __name__ == "__main__":
    unittest.main()
