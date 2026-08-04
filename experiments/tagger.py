"""
The constructive supertagger: a pretrained encoder contextualises a
sentence, each lexical phrase's subword states are mean-pooled, and a GRU
head decodes the phrase's type symbol by symbol in prefix notation — so
unseen types are first-class outputs rather than a missing softmax class.

Decoding is constrained by symbol arity (2 for ``⟶``, 1 for modalities,
0 for atoms): a prefix is a well-formed type exactly when its number of
open slots reaches zero, at which point the end symbol is forced, so every
prediction parses back through ``aethel.mill.types.parse_prefix``.
"""
from __future__ import annotations

import torch
from torch import nn

ARROW = "⟶"
MODAL_PREFIXES = ("□", "◇", "!")


class Vocabulary:
    """The type-symbol vocabulary with padding, start and end symbols."""

    def __init__(self, symbols: list[str]):
        self.symbols = ["<pad>", "<bos>", "<eos>"] + list(symbols)
        self.index = {symbol: i for i, symbol in enumerate(self.symbols)}
        self.pad, self.bos, self.eos = 0, 1, 2
        self.arities = torch.tensor([
            2 if symbol == ARROW
            else 1 if symbol.startswith(MODAL_PREFIXES)
            else 0 for symbol in self.symbols])

    def __len__(self) -> int:
        return len(self.symbols)

    def encode(self, prefix: str) -> list[int]:
        """The symbol indices of a type in prefix notation, then the end."""
        return [self.index[symbol] for symbol in prefix.split()] + [self.eos]

    def decode(self, indices: list[int]) -> str:
        """The prefix notation of a decoded index sequence."""
        return " ".join(
            self.symbols[i] for i in indices
            if i not in (self.pad, self.bos, self.eos))


class Tagger(nn.Module):
    """
    The encoder-decoder supertagger.

    Parameters:
        encoder : A HuggingFace-style module whose forward returns an
            object with a ``last_hidden_state`` of shape ``(B, S, D)``.
        vocab : The type-symbol vocabulary.
        embedding_dim : The dimension of the symbol embeddings.
    """

    def __init__(self, encoder, vocab: Vocabulary, embedding_dim: int = 256):
        super().__init__()
        self.encoder, self.vocab = encoder, vocab
        hidden = encoder.config.hidden_size
        self.embedding = nn.Embedding(
            len(vocab), embedding_dim, padding_idx=vocab.pad)
        self.gru = nn.GRU(embedding_dim, hidden, batch_first=True)
        self.output = nn.Linear(hidden, len(vocab))

    def phrase_states(self, input_ids, attention_mask, phrase_ids,
                      n_phrases: int):
        """
        Mean-pool the encoder's subword states into one state per phrase.

        Parameters:
            input_ids : Subword indices of shape ``(B, S)``.
            attention_mask : The attention mask of shape ``(B, S)``.
            phrase_ids : For each subword, the flat index of its phrase
                across the batch, or ``-1`` for special subwords.
            n_phrases : The total number of phrases in the batch.
        """
        states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask).last_hidden_state
        flat, ids = states.flatten(0, 1), phrase_ids.flatten()
        keep = ids >= 0
        flat, ids = flat[keep], ids[keep]
        total = torch.zeros(
            n_phrases, flat.shape[-1],
            device=flat.device, dtype=flat.dtype).index_add_(0, ids, flat)
        count = torch.zeros(
            n_phrases, device=flat.device, dtype=flat.dtype).index_add_(
            0, ids, torch.ones_like(ids, dtype=flat.dtype))
        return total / count.clamp(min=1).unsqueeze(-1)

    def forward(self, input_ids, attention_mask, phrase_ids, targets):
        """
        The teacher-forced cross-entropy loss on a batch of targets of
        shape ``(P, L)``, padded with ``vocab.pad``.
        """
        states = self.phrase_states(
            input_ids, attention_mask, phrase_ids, targets.shape[0])
        bos = torch.full(
            (targets.shape[0], 1), self.vocab.bos, device=targets.device)
        inputs = torch.cat([bos, targets[:, :-1]], dim=1)
        outputs, _ = self.gru(self.embedding(inputs), states[None])
        logits = self.output(outputs)
        return nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten(),
            ignore_index=self.vocab.pad)

    @torch.no_grad()
    def greedy(self, input_ids, attention_mask, phrase_ids, n_phrases: int,
               max_length: int = 64):
        """
        Greedy arity-constrained decoding: one well-formed type per phrase,
        as a ``(P, max_length)`` tensor padded with ``vocab.pad``.
        """
        states = self.phrase_states(
            input_ids, attention_mask, phrase_ids, n_phrases)[None]
        arities = self.vocab.arities.to(states.device)
        tokens = torch.full(
            (n_phrases, 1), self.vocab.bos, device=states.device)
        slots = torch.ones(n_phrases, dtype=torch.long, device=states.device)
        result = []
        for _ in range(max_length):
            outputs, states = self.gru(self.embedding(tokens), states)
            logits = self.output(outputs[:, -1])
            logits[:, self.vocab.pad] = logits[:, self.vocab.bos] = -torch.inf
            done = slots == 0
            logits[done] = -torch.inf
            logits[done, self.vocab.eos] = 0.
            logits[~done, self.vocab.eos] = -torch.inf
            tokens = logits.argmax(-1, keepdim=True)
            slots = slots + arities[tokens[:, 0]] - 1
            slots[done] = 0
            result.append(tokens[:, 0])
        result = torch.stack(result, dim=1)
        mask = (result == self.vocab.eos).cumsum(dim=1) > 0
        return result.masked_fill(mask, self.vocab.pad)
