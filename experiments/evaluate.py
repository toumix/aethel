"""
End-to-end parsing evaluation: greedy-decode the frame, score the axiom
links with the joint model, snap them to a discrete matching with the
Hungarian algorithm (exact assignment per sort block), rebuild the proof
with ``mill.nets.links_to_proof`` — æthel's constructors type-check it —
and compare the term with the gold proof's.

Reports parsability (a proof was built) and strict term accuracy against
SPINDLE's 55.63, see ``BASELINES.md``.

Usage::

    python experiments/evaluate.py --data DATA_DIR --dump PICKLE \\
        --checkpoint best.pt [--subset dev] [--encoder ...] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModel, AutoTokenizer

from aethel import ProofBank
from aethel.mill.nets import LeafFT, links_to_proof, type_to_tree
from aethel.mill.types import Type

from parser import Parser, polarized_leaves
from tagger import Vocabulary
from train import batches, load, tensorise, with_goals


@torch.no_grad()
def parse(model: Parser, prefixes: list[str], states, gold) -> bool:
    """
    Whether the model's predicted frame and links rebuild the gold term.

    Parameters:
        model : The joint parser.
        prefixes : The predicted types of the phrases, goal last.
        states : The decoder states of the sample, ``(P, T, D)``.
        gold : The gold proof.
    """
    table, sorts = [], {}
    for j, prefix in enumerate(prefixes):
        for step, sort, pol in polarized_leaves(
                prefix.split(), polarity=j != len(prefixes) - 1):
            leaf = len(table)
            table.append((j, step, sort, pol))
            sorts.setdefault(sort, ([], []))[pol].append(leaf)
    links = {}
    for sort, (negatives, positives) in sorts.items():
        if not negatives or not positives:
            continue
        negative = model.negative(states[
            [table[n][0] for n in negatives],
            [table[n][1] for n in negatives]])
        positive = model.positive(states[
            [table[p][0] for p in positives],
            [table[p][1] for p in positives]])
        scores = (negative @ positive.T / model.link_dim ** .5).cpu()
        for row, column in zip(*linear_sum_assignment(-scores.numpy())):
            links[LeafFT(sort, negatives[row], False)]\
                = LeafFT(sort, positives[column], True)
    running, lex_trees, first_leaf = 0, {}, {}
    for j, prefix in enumerate(prefixes[:-1]):
        first_leaf[j] = running
        lex_trees[j], running = type_to_tree(
            Type.parse_prefix(prefix), True, running)
    conclusion, _ = type_to_tree(
        Type.parse_prefix(prefixes[-1]), False, running)
    linked = {leaf.index for pair in links.items() for leaf in pair}
    used = {
        j: tree for j, tree in lex_trees.items()
        if any(index in linked for index, entry in enumerate(table)
               if entry[0] == j)}
    proof = links_to_proof(links, used, conclusion)
    return proof.term == gold.term


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", required=True)
    argparser.add_argument("--dump", required=True)
    argparser.add_argument("--checkpoint", required=True)
    argparser.add_argument(
        "--encoder", default="DTAI-KULeuven/robbert-2023-dutch-large")
    argparser.add_argument("--subset", default="dev")
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--link-dim", type=int, default=128)
    argparser.add_argument("--limit", type=int, default=None)
    args = argparser.parse_args()

    sys.setrecursionlimit(1000000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = pathlib.Path(args.data)
    bank = ProofBank.load_data(args.dump)
    by_name = {sample.name: sample for sample in bank.samples}
    rows = with_goals(
        load(data_dir / f"{args.subset}.jsonl.gz"),
        load(data_dir / f"links-{args.subset}.jsonl.gz"))
    rows = [row for row in rows if row.get("links") is not None][:args.limit]

    vocab = Vocabulary(json.loads((data_dir / "symbols.json").read_text()))
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = Parser(
        AutoModel.from_pretrained(args.encoder), vocab,
        link_dim=args.link_dim).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    counts = Counter()
    for chunk in batches(rows, args.batch_size):
        input_ids, attention_mask, phrase_ids, targets, offsets = tensorise(
            chunk, tokenizer, vocab, device)
        decoded, states = model.greedy(
            input_ids, attention_mask, phrase_ids, targets.shape[0],
            max_length=max(64, targets.shape[1]), return_states=True)
        prefixes = [vocab.decode(row.tolist()) for row in decoded]
        for i, row in enumerate(chunk):
            start, stop = offsets[i], offsets[i] + len(row["words"])
            counts["total"] += 1
            try:
                match = parse(
                    model, prefixes[start:stop], states[start:stop],
                    by_name[row["name"]].proof)
            except Exception:
                counts["unparsable"] += 1
                continue
            counts["parsed"] += 1
            counts["match"] += match
        if counts["total"] % 1000 < args.batch_size:
            print(dict(counts))
    result = {
        "subset": args.subset,
        "parsability": counts["parsed"] / counts["total"],
        "term_accuracy": counts["match"] / counts["total"]}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
