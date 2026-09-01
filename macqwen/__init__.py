"""The chat, shared by every model.

Pure Python on purpose. The 27B and Flash-Next runtimes need different
Python environments, so this package must import in both. Anything that
touches MLX belongs in a backend, not here. See docs/RESTRUCTURE.md.
"""
