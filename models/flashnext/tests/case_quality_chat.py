from .api import TestSpec

QUALITY_PROMPT = (
    "crie uma extensão para sketchup que extrude várias faces ao mesmo tempo até "
    "uma altura definida pelo usuário. produza o código para eu salvar em um arquivo .rb"
)

TEST = TestSpec(
    id="quality-chat-xhigh", title="Manual chat quality evaluation", category="quality",
    explanation="The user evaluates a final performance candidate through chat.sh with sampling and xhigh effort.",
    why="Qwen documents xhigh for the intended reasoning behavior. Automated benchmark text is not the quality authority.",
    metrics=("user judgment",), controls={
        "launcher": "chat.sh", "sampling": "normal", "effort": "xhigh",
        "seed": "same explicit --seed in both conditions",
        "prompt": QUALITY_PROMPT,
    },
    source="chat.sh", status="manual",
)
