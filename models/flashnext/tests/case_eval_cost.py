from .api import script_case

TEST = script_case(
    test_id="eval-cost", title="MLX evaluation cost", category="diagnostic",
    explanation="Measures eval counts, completion blocks, cache limits, and optional injected work.",
    why="It distinguishes synchronization overhead from deferred GPU graph work.",
    filename="bench_eval_cost.py", arguments=("--tokens", "{tokens}", "{model_args}"),
)
