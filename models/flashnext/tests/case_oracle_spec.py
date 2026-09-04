from .api import script_case

TEST = script_case(
    test_id="oracle-spec", title="Perfect-draft speculation oracle", category="diagnostic",
    explanation="Verifies perfect future tokens at several block sizes.",
    why="It establishes the maximum possible speculative rate before any draft cost.",
    filename="bench_oracle_spec.py", arguments=("--tokens", "{tokens}", "--exact-verifier"),
)
