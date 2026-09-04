from .api import script_case

TEST = script_case(
    test_id="read-ceiling-disk", title="All-cold read floor", category="microbenchmark",
    explanation="Routes fresh expert sets to force storage traffic.",
    why="It establishes the cold-storage floor and prices physical misses.",
    filename="bench_read_ceiling.py", arguments=("--mode", "disk", "--tokens", "{tokens}"),
)
