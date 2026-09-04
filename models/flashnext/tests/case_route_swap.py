from .api import script_case

TEST = script_case(
    test_id="route-swap", title="Cache-aware routing opportunity", category="diagnostic",
    explanation="Counts cold selected experts with near-equal resident alternatives.",
    why="It estimates the byte opportunity before changing model computation.",
    filename="bench_route_swap.py", arguments=("--tokens", "{tokens}"),
)
