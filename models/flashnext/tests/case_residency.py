from .api import script_case

TEST = script_case(
    test_id="residency", title="Residency gate accuracy", category="diagnostic",
    explanation="Compares the process residency tracker with mincore observations.",
    why="It tests whether mapped reads can safely replace positioned copies.",
    filename="bench_residency.py", arguments=("--tokens", "{tokens}", "{model_args}"),
)
