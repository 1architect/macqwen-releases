from .api import script_case

TEST = script_case(
    test_id="decode-split", title="Decode time and physical-read split", category="diagnostic",
    explanation="Attributes token time to expert reads, n-gram reads, synchronization, and remaining work.",
    why="Aggregate throughput could not identify the current limiting stage.",
    filename="bench_decode_split.py", arguments=("--tokens", "{tokens}", "--passes", "{pairs}", "{model_args}"),
)
