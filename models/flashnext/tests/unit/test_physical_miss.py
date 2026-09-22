import unittest

from models.flashnext.physical_miss import (
    add_observation,
    allocate_physical_miss_slots,
    allocation_summary,
    empty_profile,
)


class TestPhysicalMissAllocation(unittest.TestCase):
    def test_requires_measured_physical_bytes(self):
        profile = empty_profile("unit")
        add_observation(profile, 0, 1, physical_miss_bytes=0, samples=20)
        self.assertEqual(
            allocate_physical_miss_slots(profile, 4, min_slots=2, max_slots=2),
            {},
        )

    def test_selects_highest_long_run_physical_misses(self):
        profile = empty_profile("unit")
        for expert, value in ((1, 900), (2, 800), (3, 700), (4, 10)):
            add_observation(
                profile, 0, expert, physical_miss_bytes=value, samples=4
            )
        for expert, value in ((1, 600), (2, 500), (3, 400)):
            add_observation(
                profile, 1, expert, physical_miss_bytes=value, samples=4
            )
        allocation = allocate_physical_miss_slots(
            profile, 4, min_slots=2, max_slots=2, num_layers=2
        )
        self.assertEqual(allocation, {0: [1, 2], 1: [1, 2]})
        summary = allocation_summary(profile, allocation)
        self.assertEqual(summary["selected_slots"], 4)
        self.assertGreater(summary["selected_fraction"], 0.7)

    def test_minimum_samples_filters_short_observations(self):
        profile = empty_profile("unit")
        add_observation(profile, 0, 1, physical_miss_bytes=900, samples=1)
        add_observation(profile, 0, 2, physical_miss_bytes=100, samples=4)
        allocation = allocate_physical_miss_slots(
            profile, 1, min_slots=1, max_slots=1, min_samples=2
        )
        self.assertEqual(allocation, {0: [2]})


if __name__ == "__main__":
    unittest.main()
