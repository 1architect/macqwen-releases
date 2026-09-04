from .api import script_case

TEST = script_case(
    test_id="gather-qmm", title="Routed Q4 gather throughput", category="microbenchmark",
    explanation="Measures real gate, up, and down gather_qmm projections on resident arrays.",
    why="It tests whether routed expert matmul contains the missing GPU block.",
    filename="bench_gather_qmm.py", arguments=("--arms", "{pairs}", "{model_args}"),
)
