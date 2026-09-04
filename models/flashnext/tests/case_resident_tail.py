from .api import script_case

TEST = script_case(
    test_id="resident-tail", title="Pinned-tail residency", category="performance",
    explanation="Measures steady decode after selecting and pinning routed experts.",
    why="It tests expert pin depth, warmup length, and the resident read path.",
    filename="bench_resident_tail.py", arguments=("--tokens", "{tokens}", "--exact"), promotion=True,
)
