"""
Train and evaluate the constructive supertagger on the æthel dump.

Usage::

    python experiments/train.py --data DATA_DIR --out OUT_DIR \\
        [--encoder DTAI-KULeuven/robbert-2023-dutch-base] [--epochs 5] \\
        [--batch-size 32] [--limit N] [--test]

Reports word-level accuracy overall and by train-frequency bin
(frequent / uncommon / rare / unseen, see ``data.py``) plus frame
accuracy, on dev after every epoch and on test with ``--test``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import random
from collections import Counter

import torch
from transformers import AutoModel, AutoTokenizer

from data import BINS
from parser import GOAL, Parser, blocks
from tagger import Tagger, Vocabulary


def load(path: pathlib.Path) -> list[dict]:
    with gzip.open(path, "rt") as file:
        return [json.loads(line) for line in file]


def batches(rows: list[dict], size: int, shuffle: bool = False):
    rows = sorted(rows, key=lambda row: len(row["words"]))
    chunks = [rows[i:i + size] for i in range(0, len(rows), size)]
    if shuffle:
        random.shuffle(chunks)
    return chunks


def with_goals(rows: list[dict], links: list[dict]) -> list[dict]:
    """Append the goal type as a zero-size ``[GOAL]`` phrase to each row."""
    by_name = {link["name"]: link for link in links}
    result = []
    for row in rows:
        link = by_name.get(row["name"])
        if link is None or link["goal"] is None:
            result.append(row)
            continue
        result.append(row | {
            "words": row["words"] + [GOAL],
            "sizes": row["sizes"] + [0],
            "types": row["types"] + [link["goal"]],
            "links": link["links"]})
    return result


def tensorise(chunk, tokenizer, vocab, device):
    """Tokenise a chunk of rows into the tensors the tagger consumes."""
    encoding = tokenizer(
        [row["words"] for row in chunk], is_split_into_words=True,
        padding=True, truncation=True, return_tensors="pt")
    offsets, total = [], 0
    for row in chunk:
        offsets.append(total)
        total += len(row["words"])
    phrase_ids = torch.full(encoding["input_ids"].shape, -1)
    for i, offset in enumerate(offsets):
        for j, word in enumerate(encoding.word_ids(i)):
            if word is not None:
                phrase_ids[i, j] = offset + word
    length = max(
        len(vocab.encode(prefix))
        for row in chunk for prefix in row["types"])
    targets = torch.full((total, length), vocab.pad)
    for i, row in enumerate(chunk):
        for j, prefix in enumerate(row["types"]):
            encoded = vocab.encode(prefix)
            targets[offsets[i] + j, :len(encoded)] = torch.tensor(encoded)
    return (encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
            phrase_ids.to(device), targets.to(device), offsets)


@torch.no_grad()
def evaluate(model, rows, tokenizer, vocab, train_counts, device,
             batch_size: int) -> dict:
    """Word-level accuracy overall, by frequency bin, and per frame."""
    model.eval()
    correct, bins, frames, goals = Counter(), Counter(), [0, 0], [0, 0]
    for chunk in batches(rows, batch_size):
        input_ids, attention_mask, phrase_ids, targets, _ = tensorise(
            chunk, tokenizer, vocab, device)
        decoded = model.greedy(
            input_ids, attention_mask, phrase_ids, targets.shape[0],
            max_length=max(64, targets.shape[1]))
        predictions = [vocab.decode(row.tolist()) for row in decoded]
        offset = 0
        for row in chunk:
            frame = True
            for word, prefix, size in zip(
                    row["words"], row["types"], row["sizes"]):
                hit = predictions[offset] == prefix
                offset += 1
                if word == GOAL:
                    goals[0], goals[1] = goals[0] + hit, goals[1] + 1
                    continue
                frame = frame and hit
                correct.update({"hit": size if hit else 0, "total": size})
                count = train_counts[prefix]
                for name, low, high in BINS:
                    if low <= count <= high:
                        bins.update({
                            f"{name}_hit": size if hit else 0,
                            f"{name}_total": size})
            frames[0] += frame
            frames[1] += 1
    result = {
        "accuracy": correct["hit"] / correct["total"],
        "frame": frames[0] / frames[1]}
    if goals[1]:
        result["goal"] = goals[0] / goals[1]
    for name, _, _ in BINS:
        if bins[f"{name}_total"]:
            result[name] = bins[f"{name}_hit"] / bins[f"{name}_total"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--encoder", default="DTAI-KULeuven/robbert-2023-dutch-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=float, default=0.)
    parser.add_argument("--label-smoothing", type=float, default=0.)
    parser.add_argument("--link-weight", type=float, default=0.)
    parser.add_argument("--link-dim", type=int, default=128)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir, out_dir = pathlib.Path(args.data), pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = {
        subset: load(data_dir / f"{subset}.jsonl.gz")[:args.limit]
        for subset in ("train", "dev", "test")}
    if args.link_weight:
        subsets = {
            subset: with_goals(
                rows, load(data_dir / f"links-{subset}.jsonl.gz"))
            for subset, rows in subsets.items()}
    train_counts = Counter()
    for row in subsets["train"]:
        for prefix, size in zip(row["types"], row["sizes"]):
            train_counts[prefix] += size
    vocab = Vocabulary(json.loads((data_dir / "symbols.json").read_text()))
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = (
        Parser(
            AutoModel.from_pretrained(args.encoder), vocab,
            label_smoothing=args.label_smoothing, link_dim=args.link_dim)
        if args.link_weight else Tagger(
            AutoModel.from_pretrained(args.encoder), vocab,
            label_smoothing=args.label_smoothing)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": [
            parameter for name, parameter in model.named_parameters()
            if not name.startswith("encoder.")], "lr": args.head_lr}])
    total = args.epochs * len(batches(subsets["train"], args.batch_size))
    warmup = int(args.warmup * total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, (lambda step: min(
            step / warmup, (total - step) / max(total - warmup, 1)))
        if warmup else (lambda step: 1.))

    best, log = 0., out_dir / "metrics.jsonl"
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_link, n = 0., 0., 0
        for chunk in batches(subsets["train"], args.batch_size, shuffle=True):
            optimizer.zero_grad()
            input_ids, attention_mask, phrase_ids, targets, offsets =\
                tensorise(chunk, tokenizer, vocab, device)
            if args.link_weight:
                tagging, linking = model(
                    input_ids, attention_mask, phrase_ids, targets,
                    link_blocks=blocks(chunk, offsets))
                loss = tagging + args.link_weight * linking
                total_link += linking.item()
            else:
                loss = model(input_ids, attention_mask, phrase_ids, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            total_loss, n = total_loss + loss.item(), n + 1
        metrics = evaluate(
            model, subsets["dev"], tokenizer, vocab, train_counts, device,
            args.batch_size)
        metrics |= {"epoch": epoch, "loss": total_loss / max(n, 1)}
        if args.link_weight:
            metrics |= {"link_loss": total_link / max(n, 1)}
        print(json.dumps(metrics))
        with open(log, "a") as file:
            file.write(json.dumps(metrics) + "\n")
        if metrics["accuracy"] > best:
            best = metrics["accuracy"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    if args.test:
        model.load_state_dict(
            torch.load(out_dir / "best.pt", weights_only=True))
        metrics = {"subset": "test"} | evaluate(
            model, subsets["test"], tokenizer, vocab, train_counts, device,
            args.batch_size)
        print(json.dumps(metrics))
        with open(log, "a") as file:
            file.write(json.dumps(metrics) + "\n")


if __name__ == "__main__":
    main()
