"""
The supertagging view of the æthel dump: one line per sample with its
lexical phrases and their types in prefix notation, split by the official
train/dev/test segmentation.

Usage: ``python experiments/data.py PATH_TO_DUMP OUT_DIR``

Writes ``{train,dev,test}.jsonl.gz``, a ``symbols.json`` vocabulary of
type symbols for the constructive decoder, and prints word-level token
statistics by train frequency — frequent (≥ 100), uncommon (10–99),
rare (1–9) and unseen (0). A type is assigned per lexical phrase but
counted once per word, matching the published protocol: the test total
reproduces ``BASELINES.md`` exactly (95,331 words) while the bins land
within 0.2% of the thesis table (91,503 / 2,639 / 826 / 363), whose
exact thresholds are not stated in any primary source.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import sys
from collections import Counter

from aethel import ProofBank

BINS = (("frequent", 100, sys.maxsize), ("uncommon", 10, 99),
        ("rare", 1, 9), ("unseen", 0, 0))


def rows(bank: ProofBank) -> dict[str, list[dict]]:
    """The samples of each subset as json-ready supertagging rows."""
    result = {"train": [], "dev": [], "test": []}
    for sample in bank.samples:
        result[sample.subset].append({
            "name": sample.name,
            "words": [phrase.string for phrase in sample.lexical_phrases],
            "sizes": [len(phrase) for phrase in sample.lexical_phrases],
            "types": [
                phrase.type.prefix() for phrase in sample.lexical_phrases]})
    return result


def symbols(subsets: dict[str, list[dict]]) -> list[str]:
    """The vocabulary of type symbols over the whole dump, sorted."""
    return sorted({
        symbol for subset in subsets.values() for row in subset
        for prefix in row["types"] for symbol in prefix.split()})


def statistics(subsets: dict[str, list[dict]]) -> dict[str, Counter]:
    """Token counts of each subset binned by train-set type frequency."""
    train_counts = Counter(
        prefix for row in subsets["train"] for prefix in row["types"])
    result = {}
    for subset, data in subsets.items():
        bins = Counter()
        for row in data:
            for prefix, size in zip(row["types"], row["sizes"]):
                count = train_counts[prefix]
                for name, low, high in BINS:
                    if low <= count <= high:
                        bins[name] += size
        result[subset] = bins
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subsets = rows(ProofBank.load_data(args.dump))

    for subset, data in subsets.items():
        with gzip.open(out_dir / f"{subset}.jsonl.gz", "wt") as file:
            for row in data:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "symbols.json").write_text(
        json.dumps(symbols(subsets), ensure_ascii=False, indent=0))

    for subset, bins in statistics(subsets).items():
        print(subset, sum(bins.values()), "words:", dict(bins))


if __name__ == "__main__":
    main()
