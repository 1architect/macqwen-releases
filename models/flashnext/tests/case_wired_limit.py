from .api import script_case

TEST = script_case(
    test_id="wired-limit", title="Pre-load wired limit", category="performance",
    explanation="Compares zero and two GB MLX wired limits in reversed fresh instances.",
    why="It tests whether static Metal residency stabilizes streaming buffers.",
    filename="bench_wired_limit.py", arguments=("--tokens", "{tokens}", "--pairs", "{pairs}"), promotion=True,
)
