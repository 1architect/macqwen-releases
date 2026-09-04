from .api import TestSpec

TEST = TestSpec(
    id="quality-chat-xhigh", title="Manual chat quality evaluation", category="quality",
    explanation="The user evaluates a final performance candidate through chat.sh with sampling and xhigh effort.",
    why="Qwen documents xhigh for the intended reasoning behavior. Automated benchmark text is not the quality authority.",
    metrics=("user judgment",), controls={"launcher": "chat.sh", "sampling": "normal", "effort": "xhigh"},
    source="chat.sh", status="manual",
)
