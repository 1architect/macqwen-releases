from .api import script_case

TEST = script_case(
    test_id="parallel", title="Parallel instance throughput", category="performance",
    explanation="Runs multiple independent FlashNext instances and checks their token IDs.",
    why="It tests whether process-level overlap improves aggregate throughput.",
    filename="bench_parallel.py", arguments=("--tokens", "{tokens}"), promotion=True,
)
