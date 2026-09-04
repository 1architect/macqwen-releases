from .api import script_case

TEST = script_case(
    test_id="host-window", title="Exclusive host windows", category="diagnostic",
    explanation="Measures host intervals while both SSD and GPU work are idle.",
    why="It tests whether dependency-safe bookkeeping can move outside the critical path.",
    filename="bench_host_window.py", arguments=("--tokens", "{tokens}", "--passes", "{pairs}", "{model_args}"),
)
