from macqwen.testsuite.api import production_case

TEST = production_case(
    "g64-kernel",
    "Compares opt-in Q4/G64 Metal execution with reference streaming on the same checkpoint.",
)
