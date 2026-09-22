from macqwen.testsuite.api import production_case

TEST = production_case(
    "metal-runtime",
    "Tests the custom Metal MoE executor against the MLX FlashNext reference on Q4/G32.",
)
