from .api import script_case

TEST = script_case(
    test_id="glue", title="MoE glue operations", category="microbenchmark",
    explanation="Measures routing masks, normalization, and layer glue around expert projections.",
    why="It tests whether small elementwise chains explain score synchronization time.",
    filename="bench_glue.py", arguments=("--arms", "{pairs}", "{model_args}"),
)
