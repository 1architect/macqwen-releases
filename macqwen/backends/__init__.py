"""Model runtimes. Each one satisfies macqwen.backends.base.Backend.

These are the only modules that import MLX. The rest of macqwen stays pure
Python so it can be imported by every environment, which matters here
because the two models need different ones.
"""
