from .api import script_case

TEST = script_case(
    test_id="slab-sweep", title="Resident slab efficiency sweep", category="performance",
    explanation="Compares resident capacity and layer distributions with bytes and active memory.",
    why="It tests physical MB saved per resident MB added.",
    filename="bench_slab_sweep.py", arguments=("--tokens", "{tokens}"), promotion=True,
)
