from .api import script_case

TEST = script_case(
    test_id="context-decay", title="Context-length decode curve", category="performance",
    explanation="Measures decode windows at several context lengths with physical reads.",
    why="It separates true context cost from the short-prompt warm working-set transient.",
    filename="bench_context_decay.py", arguments=("--tokens", "{tokens}", "--rounds", "{pairs}", "{model_args}"),
    promotion=True,
)
