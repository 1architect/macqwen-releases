"""Launch and runtime defaults owned by the Bonsai-2 model package."""

MODEL_NAME = "bonsai2"
PYTHON_ENV = "MACQWEN_BONSAI2_PYTHON"
CHECKPOINT_ENV = "MACQWEN_BONSAI2_MODEL"
REQUIRED_MODULES = ("mlx", "mlx_lm", "transformers")
SESSION_DIR = "~/.cache/bonsai2/sessions"
CHECKPOINT_DIR = "Ternary-Bonsai-2-27B-mlx-2bit"
ALIASES = {"b2": CHECKPOINT_DIR, "bonsai2": CHECKPOINT_DIR}
