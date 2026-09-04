from .api import script_case

TEST = script_case(
    test_id="runtime-layer", title="Custom Metal layer runtime", category="microbenchmark",
    explanation="Compares stock MLX and the custom executor on fixed real routes.",
    why="It isolates custom kernel performance from changing token trajectories.",
    filename="bench_runtime_layer.py", arguments=("--arms", "{pairs}", "--require-custom", "{model_args}"),
)
