from .api import script_case

TEST = script_case(
    test_id="prefill-scaling", title="Prefill amortization curve", category="performance",
    explanation="Measures prompt lengths in round-robin order with bytes and expert counts.",
    why="It explains why long prefill is fast and whether that mechanism transfers to decode.",
    filename="bench_prefill_scaling.py", arguments=("--rounds", "{pairs}", "{model_args}"), promotion=True,
)
