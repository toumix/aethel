"""
Round-trip every proof of the æthel dump through the discopy bridge:
``decode(encode(term)) == term`` with equal types, both directions
type-checking on construction. Reports coverage and buckets any failures
by exception, so the residue can be filed as issues.

Usage: ``python experiments/roundtrip.py PATH_TO_DUMP [--limit N]``
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from aethel import ProofBank

import bridge


def check(sample) -> None:
    """Assert the round-trip identity on one sample."""
    term = sample.proof.term
    encoded = bridge.encode(term)
    if encoded.cod != bridge.encode_type(term.type):
        raise AssertionError("encoded type mismatch")
    decoded = bridge.decode(encoded)
    if decoded != term:
        raise AssertionError("decoded term mismatch")
    if decoded.type != term.type:
        raise AssertionError("decoded type mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sys.setrecursionlimit(1000000)
    bank = ProofBank.load_data(args.dump)
    samples = bank.samples[:args.limit] if args.limit else bank.samples

    failures, by_subset = Counter(), Counter()
    for i, sample in enumerate(samples):
        try:
            check(sample)
        except Exception as error:
            failures[f"{type(error).__name__}: {error}"] += 1
            by_subset[sample.subset] += 1
            print(f"FAIL {sample.name}: {type(error).__name__}: {error}")
        if (i + 1) % 10000 == 0:
            print(f"{i + 1}/{len(samples)} checked, {sum(failures.values())} failures")

    total, failed = len(samples), sum(failures.values())
    print(f"\n{total - failed}/{total} proofs round-trip exactly "
          f"({100 * (total - failed) / total:.4f}%)")
    for message, count in failures.most_common():
        print(f"{count:6d}  {message}")
    for subset, count in sorted(by_subset.items()):
        print(f"failures in {subset}: {count}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
