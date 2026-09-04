from .api import script_case

TEST = script_case(
    test_id="contribution-oracle", title="Expert contribution oracle", category="diagnostic",
    explanation="Compares router rank with the measured contribution of routed experts.",
    why="It tests whether a better subset selector can reduce expert reads without losing the selected output.",
    filename="bench_contribution_oracle.py", arguments=("--tokens", "{tokens}", "{model_args}"),
)
