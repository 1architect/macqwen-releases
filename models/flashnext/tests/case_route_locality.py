from .api import script_case

TEST = script_case(
    test_id="route-locality", title="Prompt route locality", category="diagnostic",
    explanation="Measures expert reuse, distinct sets, and top-rank coverage.",
    why="It explains prompt-dependent throughput and tests residency predictions.",
    filename="bench_route_locality.py", arguments=("--tokens", "{tokens}", "{model_args}"),
)
