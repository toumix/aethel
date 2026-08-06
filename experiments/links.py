"""
Gold axiom links for the differentiable parser, extracted with æthel's own
``mill.nets.proof_to_links`` and serialised next to the supertagging view.

Leaves are numbered by the same convention the tagger decodes in: phrase
types in phrase order, atoms of each type in prefix order, then the atoms
of the goal (conclusion) type, which the tagger learns to predict as an
extra ``[GOAL]`` position. ``links`` maps each negative leaf to its
positive partner; the sample is skipped (and counted) if æthel's own
round-trip ``links_to_proof`` does not reproduce the proof term.

Usage: ``python experiments/links.py PATH_TO_DUMP OUT_DIR``
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import sys
from collections import Counter

from aethel import ProofBank
from aethel.mill.nets import links_to_proof, proof_to_links


def atoms(type_) -> int:
    """The number of atoms of a type."""
    return sum(
        not symbol.startswith(("◇", "□", "⟶"))
        for symbol in type_.prefix().split())


def extract(sample) -> dict | None:
    """
    The sample's gold links, or ``None`` when æthel cannot round-trip.

    ``proof_to_links`` numbers leaves over the constants used by the proof;
    the pairs are renumbered to the tagger's convention — every phrase in
    order, atoms in prefix order, goal type last — so the atoms of unused
    phrases (e.g. punctuation) keep their positions and simply stay
    unlinked.
    """
    links, lex_trees, conclusion = proof_to_links(sample.proof)
    if links_to_proof(links, lex_trees, conclusion).term != sample.proof.term:
        return None
    starts, total = [], 0
    for phrase in sample.lexical_phrases:
        starts.append(total)
        total += atoms(phrase.type)
    remap, cursor = {}, 0
    for index in sorted(lex_trees):
        count = atoms(sample.lexical_phrases[index].type)
        for k in range(count):
            remap[cursor + k] = starts[index] + k
        cursor += count
    for k in range(atoms(sample.proof.term.type)):
        remap[cursor + k] = total + k
    pairs = sorted(
        (remap[negative.index], remap[positive.index])
        for negative, positive in links.items())
    return {
        "name": sample.name,
        "goal": sample.proof.term.type.prefix(),
        "unused": len(sample.lexical_phrases) - len(lex_trees),
        "links": pairs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    sys.setrecursionlimit(1000000)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = ProofBank.load_data(args.dump)

    counts, files = Counter(), {}
    try:
        for subset in ("train", "dev", "test"):
            files[subset] = gzip.open(out_dir / f"links-{subset}.jsonl.gz", "wt")
        for i, sample in enumerate(bank.samples):
            try:
                row = extract(sample)
            except Exception as error:
                row = None
                counts[f"error: {type(error).__name__}"] += 1
            if row is None:
                counts[f"skipped {sample.subset}"] += 1
                row = {"name": sample.name, "goal": None, "links": None}
            else:
                counts[f"ok {sample.subset}"] += 1
                counts["unused phrases"] += row["unused"]
            files[sample.subset].write(
                json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 10000 == 0:
                print(i + 1, dict(counts))
    finally:
        for file in files.values():
            file.close()
    print(dict(counts))


if __name__ == "__main__":
    main()
