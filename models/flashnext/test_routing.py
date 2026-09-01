from __future__ import annotations

import unittest
import os
import unittest.mock
from types import SimpleNamespace

from models.flashnext.routing import RoutingProfile


class FakeStore:
    _read_mode = "unset"
    _track_residency = False

    def __init__(self):
        self.pins = []

    def unpin_all(self):
        pass

    def pin_rows(self, name, experts):
        self.pins.append((name, tuple(experts)))
        return len(experts)

    def pin_size(self, _name, experts):
        return len(experts)


def fake_language():
    layers = []
    for layer in range(48):
        cache = SimpleNamespace(prefix=f"layers.{layer}.gate_proj.weight")
        gate = SimpleNamespace(cache=cache)
        switch = SimpleNamespace(gate_proj=gate)
        layers.append(SimpleNamespace(mlp=SimpleNamespace(switch_mlp=switch)))
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


class RoutingTests(unittest.TestCase):
    def make(self, mode):
        return RoutingProfile(mode, FakeStore(), object())

    def test_only_known_profiles_are_accepted(self):
        with self.assertRaises(ValueError):
            self.make("turbo")

    def test_quality_profiles_collect_a_warmup(self):
        self.assertFalse(self.make("standard").quality)
        self.assertFalse(self.make("fast").quality)
        self.assertTrue(self.make("fast-quality").quality)
        self.assertTrue(self.make("exact-quality").quality)
        self.assertTrue(self.make("cache-aware").quality)
        self.assertTrue(self.make("fused-quality").quality)

    def test_exact_session_profile_matches_the_legacy_shape(self):
        profile = self.make("exact-quality").session_profile({2, 1})
        self.assertEqual(profile["mode"], "exact-quality")
        self.assertEqual(profile["stop_ids"], [1, 2])
        self.assertEqual(profile["resident_experts"], 32)
        self.assertEqual(profile["pin_budget_gb"], 6.0)
        self.assertEqual(profile["renorm"], 1.0)
        self.assertFalse(profile["speculative_fast"])

    def test_fused_profile_reports_its_pinned_experts(self):
        profile = self.make("fused-quality").session_profile({1})
        self.assertEqual(profile["resident_experts"], 32)

    def test_cache_aware_profile_reports_its_epsilon(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("FLASHNEXT_SWAP_RESIDENT", None)
            item = self.make("cache-aware")
        profile = item.session_profile({1})
        self.assertTrue(item.cache_aware)
        self.assertEqual(profile["resident_experts"], 32)
        self.assertEqual(profile["swap_epsilon"], 0.02)

    def test_fast_profile_forces_the_measured_threshold(self):
        profile = self.make("fast").session_profile({1})
        self.assertEqual(profile["threshold"], 0.20)
        self.assertEqual(profile["renorm"], 0.0)

    def test_fast_quality_pins_omitted_experts_after_warmup(self):
        store = FakeStore()
        profile = RoutingProfile(
            "fast-quality", store, fake_language(), tail_experts=1, warmup=1
        )
        profile.begin_decode()
        profile._observe(0, [[1, 2, 3]], [[0.7, 0.2, 0.1]], [1])
        profile.after_token(2, 3)
        profile.finish_decode()
        self.assertEqual(store._read_mode, "shared_mmap")
        self.assertEqual(profile.pinned_bytes, 9)
        self.assertEqual({experts for _name, experts in store.pins}, {(2,)})

    def test_pin_waits_for_the_full_route_warmup(self):
        store = FakeStore()
        profile = RoutingProfile(
            "fast-quality", store, fake_language(), tail_experts=1, warmup=2
        )
        profile._observe(0, [[1, 2]], [[0.7, 0.3]], [1])
        self.assertFalse(profile.after_token(2, 4))
        self.assertEqual(store.pins, [])
        profile._observe(0, [[1, 2]], [[0.6, 0.4]], [1])
        self.assertTrue(profile.after_token(3, 4))
        self.assertTrue(store.pins)
        self.assertEqual(profile.observed[0], 2)

    def test_pin_budget_is_a_hard_limit(self):
        store = FakeStore()
        profile = RoutingProfile(
            "fast-quality", store, fake_language(), tail_experts=2, warmup=1,
            pin_budget_gb=9e-9,
        )
        profile._observe(0, [[1, 2, 3]], [[0.6, 0.3, 0.1]], [1])
        profile.after_token(2, 3)
        self.assertEqual(profile.pinned_bytes, 9)
        self.assertLessEqual(profile.pinned_bytes, profile.pin_budget)
        self.assertEqual(profile.pinned, {0: {2}})


if __name__ == "__main__":
    unittest.main()


class ReadModePerProfileTests(unittest.TestCase):
    """Today's read-path work must reach every profile that uses it."""

    def profile(self, mode):
        import models.flashnext.adaptive_topk as topk

        item = RoutingProfile(mode, FakeStore(), fake_language())
        item.reset()
        return item

    def test_every_profile_selects_a_known_read_mode(self):
        from models.flashnext.routing import DEFAULT_READ_MODE, PROFILES

        forced = {"fast": "shared_mmap"}
        for mode in PROFILES:
            with self.subTest(mode=mode):
                item = self.profile(mode)
                self.assertEqual(
                    item.store._read_mode, forced.get(mode, DEFAULT_READ_MODE)
                )

    def test_the_read_mode_is_not_hardcoded_past_the_environment(self):
        import importlib

        import models.flashnext.routing as routing

        with unittest.mock.patch.dict("os.environ", {"FLASHNEXT_READ": "resident"}):
            reloaded = importlib.reload(routing)
            self.assertEqual(reloaded.DEFAULT_READ_MODE, "resident")
            item = reloaded.RoutingProfile("exact-quality", FakeStore(), fake_language())
            item.reset()
            self.assertEqual(item.store._read_mode, "resident")
        importlib.reload(routing)


class PinPartsTests(unittest.TestCase):
    """The pin budget can buy few whole experts or many scale sets."""

    def test_the_default_pins_the_whole_expert(self):
        from models.flashnext.routing import pin_parts

        with unittest.mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("FLASHNEXT_PIN_PARTS", None)
            self.assertEqual(pin_parts(), ("weight", "scales", "biases"))

    def test_scales_mode_drops_the_weight(self):
        from models.flashnext.routing import pin_parts

        with unittest.mock.patch.dict(
            "os.environ", {"FLASHNEXT_PIN_PARTS": "scales"}
        ):
            self.assertEqual(pin_parts(), ("scales", "biases"))

    def test_an_unknown_value_falls_back_to_the_whole_expert(self):
        from models.flashnext.routing import pin_parts

        with unittest.mock.patch.dict(
            "os.environ", {"FLASHNEXT_PIN_PARTS": "nonsense"}
        ):
            self.assertEqual(pin_parts(), ("weight", "scales", "biases"))

    def test_scales_mode_pins_fewer_names_per_expert(self):
        from models.flashnext.routing import PIN_PARTS

        self.assertLess(len(PIN_PARTS["scales"]), len(PIN_PARTS["all"]))
