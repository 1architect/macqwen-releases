from .api import script_case

TEST = script_case(
    test_id="score-sync", title="Score synchronization attribution", category="diagnostic",
    explanation="Reports per-token score-sync wall time, physical bytes, and pool state.",
    why="It tests which deferred graph work is paid when routing scores reach the host.",
    filename="bench_score_sync.py", arguments=("--tokens", "{tokens}", "--passes", "3", "{model_args}"),
)
