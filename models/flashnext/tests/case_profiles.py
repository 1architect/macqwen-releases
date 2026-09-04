from .api import script_case

TEST = script_case(
    test_id="profiles", title="Routing profile comparison", category="performance",
    explanation="Compares retained routing profiles with fixed prompts and reversed rounds.",
    why="It measures the throughput cost of each routing policy without changing chat settings.",
    filename="bench_profiles.py", arguments=("--tokens", "{tokens}", "--rounds", "2"), promotion=True,
)
