# Qwen27B setting providers

Add related startup settings to a `setting_*.py` provider. Export `SETTING`,
`SETTINGS`, or `get_settings()`. Providers use the shared pure-Python
`macqwen.backend_settings.Setting` type and do not load the model.
