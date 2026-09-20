from .api import script_case

TEST = script_case(
    test_id="session-roundtrip-real",
    title="Real-model session cache round trip",
    category="verification",
    explanation="Checks one-token parity after saving and restoring a real FlashNext cache.",
    why="It protects the checkpoint-dependent session path without importing MLX during unit discovery.",
    filename="session_roundtrip_real.py",
    arguments=("{model_args}",),
    promotion=False,
)
