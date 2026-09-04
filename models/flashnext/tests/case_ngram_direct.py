from .api import script_case

TEST = script_case(
    test_id="ngram-direct", title="Direct n-gram shard dispatch", category="performance",
    explanation="Compares legacy all-shard n-gram lookup with owning-shard dispatch.",
    why="It tests whether unnecessary positioned reads limit decode.",
    filename="bench_ngram_direct.py", arguments=("--tokens", "{tokens}"), promotion=True,
)
