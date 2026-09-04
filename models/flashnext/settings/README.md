# Backend setting providers

Add related FlashNext settings to a `setting_*.py` provider. Export `SETTING`,
`SETTINGS`, or `get_settings()`. Each `Setting` must declare its lifecycle,
visibility, effective reader, and source. Keep providers pure Python. Do not
import MLX during discovery.

Use `live` only when the backend owns a safe setter. Use `startup` when the
runtime captures the value during model construction. Mark diagnostics
`research-only` so `/config model` hides them and `/config model all` shows them.
