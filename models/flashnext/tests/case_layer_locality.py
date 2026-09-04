from .api import script_case

TEST = script_case(
    test_id="layer-locality", title="Cross-layer locality", category="microbenchmark",
    explanation="Compares repeated execution of one layer with execution across distinct layers.",
    why="It tests whether dense-weight locality explains the production GPU stretch.",
    filename="bench_layer_locality.py", arguments=("--arms", "{pairs}", "{model_args}"),
)
