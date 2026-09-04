from .api import script_case

TEST = script_case(
    test_id="layer-split", title="Dependency-correct layer split", category="microbenchmark",
    explanation="Measures chained layer components and the complete decoder layer.",
    why="It corrects invalid component sums caused by independent graph overlap.",
    filename="bench_layer_split.py", arguments=("--arms", "{pairs}", "{model_args}"),
)
