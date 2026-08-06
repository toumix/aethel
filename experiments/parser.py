"""
The end-to-end differentiable parser: the constructive tagger plus a
Sinkhorn-relaxed axiom linker, trained jointly through the shared encoder.

Every atom the decoder emits is one leaf of the proof frame, so the
decoder's hidden state at that emission step is the leaf's embedding —
tagger and linker literally share representations. Leaves of the same
atomic sort form one square block per sentence (proof nets balance
positive and negative occurrences); each block's score matrix is
normalised in the log domain by Sinkhorn iterations and the linking loss
is the negative log-likelihood of the gold matching. The goal type is
decoded like any phrase, from a ``[GOAL]`` position appended to the
sentence.
"""
from __future__ import annotations

import torch
from torch import nn

from tagger import ARROW, MODAL_PREFIXES, Tagger

GOAL = "[GOAL]"


def polarized_leaves(symbols: list[str],
                     polarity: bool) -> list[tuple[int, str, bool]]:
    """
    The atoms of a type in prefix notation, in emission order, with their
    polarity: ``(step, sort, polarity)`` for each atom, the argument of an
    arrow flipping polarity exactly as ``mill.nets.type_to_tree``.
    """
    result = []

    def go(i: int, pol: bool) -> int:
        symbol = symbols[i]
        if symbol == ARROW:
            return go(go(i + 1, not pol), pol)
        if symbol.startswith(MODAL_PREFIXES):
            return go(i + 1, pol)
        result.append((i, symbol, pol))
        return i + 1

    go(0, polarity)
    return result


def blocks(rows: list[dict], offsets: list[int]) -> list[tuple]:
    """
    The per-sort square blocks of a chunk: for each linked row, tuples
    ``(negatives, positives, gold)`` where negatives and positives are
    ``(flat phrase index, step)`` pairs and ``gold[i]`` is the index in
    ``positives`` matched to ``negatives[i]``.
    """
    result = []
    for row, offset in zip(rows, offsets):
        if row.get("links") is None:
            continue
        table, sorts = [], {}
        for j, prefix in enumerate(row["types"]):
            for step, sort, pol in polarized_leaves(
                    prefix.split(), polarity=row["words"][j] != GOAL):
                leaf = len(table)
                table.append((offset + j, step, pol))
                sorts.setdefault(sort, ([], []))[pol].append(leaf)
        matching = {neg: pos for neg, pos in row["links"]}
        linked = set(matching) | set(matching.values())
        for negatives, positives in sorts.values():
            negatives = [n for n in negatives if n in linked]
            positives = [p for p in positives if p in linked]
            if not negatives:
                continue
            position = {leaf: i for i, leaf in enumerate(positives)}
            result.append((
                torch.tensor([table[n][:2] for n in negatives]),
                torch.tensor([table[p][:2] for p in positives]),
                torch.tensor([position[matching[n]] for n in negatives])))
    return result


def sinkhorn(scores: torch.Tensor, iterations: int) -> torch.Tensor:
    """Log-domain Sinkhorn normalisation of a square score matrix."""
    for _ in range(iterations):
        scores = scores - scores.logsumexp(dim=1, keepdim=True)
        scores = scores - scores.logsumexp(dim=0, keepdim=True)
    return scores


class Parser(Tagger):
    """
    The joint tagger and linker.

    Parameters:
        encoder : As in :class:`Tagger`.
        vocab : As in :class:`Tagger`.
        embedding_dim : As in :class:`Tagger`.
        label_smoothing : As in :class:`Tagger`.
        link_dim : The dimension of the occurrence embeddings.
        iterations : The number of Sinkhorn iterations.
    """

    def __init__(self, encoder, vocab, embedding_dim: int = 256,
                 label_smoothing: float = 0., link_dim: int = 128,
                 iterations: int = 10):
        super().__init__(encoder, vocab, embedding_dim, label_smoothing)
        hidden = encoder.config.hidden_size
        self.link_dim, self.iterations = link_dim, iterations
        self.negative = nn.Linear(hidden, link_dim)
        self.positive = nn.Linear(hidden, link_dim)

    def forward(self, input_ids, attention_mask, phrase_ids, targets,
                link_blocks=None):
        """
        The pair of teacher-forced tagging loss and linking loss, the
        latter ``0`` when the chunk carries no link blocks.
        """
        pooled, keys, mask = self.phrase_states(
            input_ids, attention_mask, phrase_ids, targets.shape[0])
        bos = torch.full(
            (targets.shape[0], 1), self.vocab.bos, device=targets.device)
        inputs = torch.cat([bos, targets[:, :-1]], dim=1)
        outputs, _ = self.gru(self.embedding(inputs), pooled[None])
        context = self.attend(outputs, keys, mask)
        logits = self.output(torch.cat([outputs, context], dim=-1))
        tagging = nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten(),
            ignore_index=self.vocab.pad,
            label_smoothing=self.label_smoothing)
        linking, count = outputs.new_zeros(()), 0
        for negatives, positives, gold in link_blocks or ():
            negatives, positives, gold = (
                x.to(outputs.device) for x in (negatives, positives, gold))
            neg = self.negative(outputs[negatives[:, 0], negatives[:, 1]])
            pos = self.positive(outputs[positives[:, 0], positives[:, 1]])
            log_p = sinkhorn(
                neg @ pos.T / self.link_dim ** .5, self.iterations)
            linking = linking - log_p[
                torch.arange(len(gold), device=gold.device), gold].sum()
            count += len(gold)
        return tagging, linking / max(count, 1)
