"""
CPU smoke tests for the constructive tagger: the vocabulary round-trips
real prefixes, the loss backpropagates, and every arity-constrained greedy
decode parses back into a well-formed æthel type — with a random encoder,
so no pretrained download is needed.

Usage: ``python experiments/test_tagger.py DATA_DIR``
"""
from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import torch
from aethel.mill.types import Type

from tagger import Tagger, Vocabulary


class RandomEncoder(torch.nn.Module):
    "A stand-in encoder: an embedding layer with a config.hidden_size."

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(100, hidden_size)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(
            last_hidden_state=self.embedding(input_ids))


def main(data_dir: str) -> None:
    torch.manual_seed(0)
    vocab = Vocabulary(json.loads(
        (pathlib.Path(data_dir) / "symbols.json").read_text()))

    prefixes = ["NP", "⟶ ◇obj1 NP ⟶ ◇su NP SMAIN", "□det ⟶ N NP"]
    for prefix in prefixes:
        assert vocab.decode(vocab.encode(prefix)) == prefix
        Type.parse_prefix(prefix)

    model = Tagger(RandomEncoder(), vocab)
    batch, seq, phrases = 2, 7, 5
    input_ids = torch.randint(0, 100, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    phrase_ids = torch.tensor(
        [[-1, 0, 0, 1, 2, -1, -1], [-1, 3, 3, 3, 4, 4, -1]])
    targets = torch.full((phrases, 8), vocab.pad)
    for i in range(phrases):
        encoded = vocab.encode(prefixes[i % len(prefixes)])
        targets[i, :len(encoded)] = torch.tensor(encoded)

    loss = model(input_ids, attention_mask, phrase_ids, targets)
    loss.backward()
    assert loss.isfinite()

    decoded = model.greedy(
        input_ids, attention_mask, phrase_ids, phrases, max_length=32)
    for row in decoded:
        prefix = vocab.decode(row.tolist())
        Type.parse_prefix(prefix)

    from parser import GOAL, Parser, blocks, polarized_leaves

    goal_prefix = "⟶ ◇obj1 NP ⟶ ◇su NP SMAIN"
    assert [leaf[1:] for leaf in polarized_leaves(
        goal_prefix.split(), True)]\
        == [("NP", False), ("NP", False), ("SMAIN", True)]

    rows = [{
        "words": ["NP1", "NP2", "verb", GOAL],
        "types": ["NP", "NP", goal_prefix, "SMAIN"],
        "links": [[2, 0], [3, 1], [5, 4]]}]
    link_blocks = blocks(rows, offsets=[0])
    assert len(link_blocks) == 2

    joint = Parser(RandomEncoder(), vocab)
    seq = 9
    input_ids = torch.randint(0, 100, (1, seq))
    attention_mask = torch.ones(1, seq, dtype=torch.long)
    phrase_ids = torch.tensor([[-1, 0, 1, 2, 2, 3, -1, -1, -1]])
    targets = torch.full((4, 8), vocab.pad)
    for i, prefix in enumerate(rows[0]["types"]):
        encoded = vocab.encode(prefix)
        targets[i, :len(encoded)] = torch.tensor(encoded)
    tagging, linking = joint(
        input_ids, attention_mask, phrase_ids, targets,
        link_blocks=link_blocks)
    (tagging + linking).backward()
    assert tagging.isfinite() and linking.isfinite()

    print("test_tagger:", len(prefixes), "prefixes round-trip,",
          f"loss {loss.item():.3f} backpropagates,",
          phrases, "greedy decodes parse as types,",
          f"joint losses {tagging.item():.3f}/{linking.item():.3f}",
          "backpropagate")


if __name__ == "__main__":
    main(sys.argv[1])
