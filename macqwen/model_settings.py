"""Model presets that must stay identical across the CLI and live settings."""

FLASHNEXT_DEFAULTS = {
    "routing": "exact-quality",
    "swap_epsilon": 0.02,
    "threshold": 0.85,
    "resident_experts": 32,
    "pin_budget_gb": 6.0,
    "tail_experts": 6,
    "tail_warmup": 8,
    "fusion_block": 23,
    "fusion_min_margin": 1.0,
    "fusion_min_block": 20,
    "fusion_margin_tokens": 8,
    "fusion_max_prompt": 512,
    "fusion_model": "",
}
