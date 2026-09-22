from macqwen.testsuite.api import script_case

TEST = script_case(
    runtime="flashnext",
    test_id="native-scheduler", title="Native Metal scheduler", category="microbenchmark",
    explanation="Compares serial encoders, buffer barriers, and encoder fences.",
    why="It isolates Metal dependency scheduling outside MLX.",
    filename="bench_native_scheduler.py", arguments=("--arms", "{pairs}"),
)
