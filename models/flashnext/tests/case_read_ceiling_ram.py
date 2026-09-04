from .api import script_case

TEST = script_case(
    test_id="read-ceiling-ram", title="All-resident read ceiling", category="microbenchmark",
    explanation="Runs fixed routed experts with zero intended storage misses.",
    why="It establishes the compute ceiling when routed rows remain resident.",
    filename="bench_read_ceiling.py", arguments=("--mode", "ram", "--tokens", "{tokens}"),
)
