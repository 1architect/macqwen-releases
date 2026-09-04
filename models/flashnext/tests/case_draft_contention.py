from .api import script_case

TEST = script_case(
    test_id="draft-contention", title="Off-process draft contention", category="diagnostic",
    explanation="Runs a target beside a duty-cycled draft process and measures target retention.",
    why="It tests whether independent draft computation can fit inside target idle windows.",
    filename="bench_draft_contention.py", arguments=("--tokens", "{tokens}", "--arms", "{pairs}"),
)
