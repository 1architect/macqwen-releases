#!/usr/bin/env python3
"""Does cache-aware routing change what the model says?

Every other change measured here preserves the output, so token identity does
the checking. This one substitutes a resident expert for a cold one when the
scores are close, so it can change the reply. Exact routing is the reference:
the same prompts run with the swap off and on, and the divergence between them
is the quality cost.

Some prompts have answers that can be checked without the model, so an outright
break shows up as a wrong answer rather than as a diff.

`fused-quality` was rejected on a probability question that exact routing
answered correctly. That prompt was not recorded, only its answer, so this
carries its own checkable set.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HEAD = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# `answer` is a string the reply must contain. None means judge by divergence.
# A checkable prompt has to reach its answer inside the token budget, or both
# conditions fail for the same irrelevant reason and the check proves nothing.
# Each one below demands the answer first.
PROMPTS = [
    ("What is 17 times 23? Reply with the number and nothing else.", "391"),
    ("A bag holds 3 red balls and 2 blue balls. You draw two without "
     "replacement. State the probability that both are red as a fraction in "
     "lowest terms. Give the fraction first, then one sentence.", "3/10"),
    ("How many days are in three consecutive non-leap years? Give the number "
     "first, then one sentence.", "1095"),
    ("A fair coin is flipped three times. State the probability of exactly "
     "two heads as a fraction. Give the fraction first.", "3/8"),
    ("Name the capital of Australia. Answer with the city name first.",
     "Canberra"),
    ("Explique a fotossintese em duas frases.", None),
    ("Write a Python function that returns the nth Fibonacci number.", None),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.02)
    args = parser.parse_args()

    os.environ["FLASHNEXT_TRACK_RESIDENT"] = "1"
    os.environ.setdefault("FLASHNEXT_TOPK_THRESHOLD", "0.85")
    os.environ["FLASHNEXT_SWAP_RESIDENT"] = "0"
    os.environ["FLASHNEXT_SWAP_EPSILON"] = str(args.epsilon)

    from macqwen.backends.flashnext import FlashNextBackend

    backend = FlashNextBackend()

    def reply(prompt: str) -> tuple[str, tuple]:
        backend.reset()
        backend.append_text(HEAD.format(prompt))
        text, stats = backend.generate(max_tokens=args.tokens)
        produced = tuple(backend.tape[-stats.tokens:]) if stats.tokens else ()
        return text, produced

    # Warm the residency tracker, or the gate has nothing to swap toward.
    reply(PROMPTS[0][0])

    results = []
    for prompt, expected in PROMPTS:
        os.environ["FLASHNEXT_SWAP_RESIDENT"] = "0"
        exact_text, exact_ids = reply(prompt)
        os.environ["FLASHNEXT_SWAP_RESIDENT"] = "1"
        swap_text, swap_ids = reply(prompt)
        os.environ["FLASHNEXT_SWAP_RESIDENT"] = "0"

        diverged = None
        for index, (a, b) in enumerate(zip(exact_ids, swap_ids)):
            if a != b:
                diverged = index
                break
        if diverged is None and len(exact_ids) != len(swap_ids):
            diverged = min(len(exact_ids), len(swap_ids))
        results.append((prompt, expected, exact_text, swap_text, diverged,
                        len(exact_ids)))

    print()
    print(f"  {'prompt':<44}{'exact':>7}{'swap':>7}{'diverged':>10}")
    broke = 0
    for prompt, expected, exact_text, swap_text, diverged, length in results:
        if expected is None:
            exact_ok = swap_ok = "-"
        else:
            exact_ok = "yes" if expected.lower() in exact_text.lower() else "NO"
            swap_ok = "yes" if expected.lower() in swap_text.lower() else "NO"
            if exact_ok == "yes" and swap_ok == "NO":
                broke += 1
        where = "identical" if diverged is None else f"token {diverged}/{length}"
        print(f"  {prompt[:42]:<44}{exact_ok:>7}{swap_ok:>7}{where:>10}")

    identical = sum(1 for r in results if r[4] is None)
    print()
    print(f"  identical replies       {identical} of {len(results)}")
    print(f"  checkable answers lost  {broke}")
    print()
    if broke:
        print("  Cache-aware routing broke an answer that exact routing got")
        print("  right. Do not enable it.")
    elif identical == len(results):
        print("  No reply changed. The swap did not fire, or it never changed")
        print("  a token. Confirm it fired before reading this as a pass.")
    else:
        print("  Replies changed without losing a checkable answer. Read the")
        print("  diverged replies below before calling that acceptable.")
        for prompt, _e, exact_text, swap_text, diverged, _l in results:
            if diverged is None:
                continue
            print()
            print(f"  --- {prompt[:60]}")
            print(f"  exact: {exact_text.strip()[:200]}")
            print(f"  swap : {swap_text.strip()[:200]}")


if __name__ == "__main__":
    main()
