"""Launch and runtime defaults owned by the K2-Horizon model package."""

MODEL_NAME = "k2-horizon"
PYTHON_ENV = "MACQWEN_K2_HORIZON_PYTHON"
CHECKPOINT_ENV = "MACQWEN_K2_HORIZON_MODEL"
REQUIRED_MODULES = ("mlx", "mlx_lm", "transformers")
SESSION_DIR = "~/.cache/k2-horizon/sessions"
CHECKPOINT_DIR = "K2-Horizon-7B-MLX-8bit"
ALIASES = {"k2": CHECKPOINT_DIR, "k2-horizon": CHECKPOINT_DIR}
