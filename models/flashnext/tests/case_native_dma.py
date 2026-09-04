from .api import script_case

TEST = script_case(
    test_id="native-dma", title="Native Metal under physical DMA", category="microbenchmark",
    explanation="Compares serial, barrier, and fence schedulers during verified F_NOCACHE reads.",
    why="It tests whether Metal barriers amplify latency during real NVMe DMA.",
    filename="bench_native_dma_contention.py", arguments=("--arms", "{pairs}"),
)
