# TODO

Prompt (USER, 🌤️ Daylight session 2026-08-04, verbatim):

> Scale this ACG experiment to get SOTA on the Aethel dataset https://github.com/discopy/discopy/pull/400

Rulings from the same session: staged targets — supertagging first, then full proofs; the
experiment lives in this repo; training runs on Modal (credentials given live, kept out of git).

---

Measurements (2026-08-04): `data/aethel_1.0.0a5.zip` loads under Python 3.11 — 68,763 samples
(56,875 train / 6,118 dev / 5,770 test), 992,385 tokens, 5,762 distinct types. Baselines to beat:
supertagging 94.08 (geometry-aware tagger, thesis rerun) and strict proof exact match 55.63
(SPINDLE, EACL 2023) on 1.0.0a5 — NPN's famous ≈70% is beam search on the filtered 0.4 dataset
and not comparable, see `experiments/BASELINES.md`. USER opened the network policy for
`api.modal.com` (plain HTTPS now passes) but the session proxy does not carry gRPC, which the
Modal client requires — so all Modal runs are driven from GitHub Actions, with
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` as repository secrets, never from a session and never
in git.

## Phase 0 — the bridge (CPU, unblocked)

- [x] Pin the exact SPINDLE and NPN test numbers (supertagging accuracy, proof/term accuracy)
  with table citations in `experiments/BASELINES.md` — done, cross-verified against the
  authors' LaTeX sources; the numbers to beat are 94.08 (tagging) and 55.63 (proofs)
- [x] `experiments/bridge.py`: convert `aethel.mill` types and terms to
  `discopy.grammar.abstract` (discopy#400 — install from its branch until it lands; when
  updating it from `main`, its `closed.py` hunks resolve toward the merged #442)
- [x] Round-trip: convert and type-check all 68,763 proofs — **68,763/68,763 round-trip
  exactly (100.0000%), zero failures, nothing to file** (`experiments/roundtrip.py`,
  discopy at #400's head `25c8493`, Python 3.12)

## Phase 1 — supertagging

- [ ] Data module respecting the official split, with long-tail and unseen-type statistics on
  dev and test
- [ ] Reproduce the SPINDLE tagger evaluation on our split to establish parity
- [ ] Constructive type decoder over a Dutch pretrained encoder (RobBERT-2023 or XLM-R large):
  decode each type as a tree of connectives, never a 5,762-way softmax
- [ ] `experiments/modal_app.py`: GPU training on Modal with checkpoints on a Modal volume,
  launched by a GitHub Actions workflow — `modal-check.yml` is the seed, verified green with
  the repository secrets on 2026-08-04 (run 30917331003)
- [ ] Sweep on dev, report test accuracy against the 94.08 bar

## Phase 2 — full proofs

- [ ] Term construction from predicted types: Sinkhorn-style permutation linking versus discopy
  proof search over `grammar.abstract` — measure both on dev
- [ ] Exact-match term accuracy on test, up to alpha and beta via discopy#442 `normal_form`,
  against SPINDLE's 55.63 (`experiments/BASELINES.md`)
- [ ] Report tables in `experiments/README.md`

Credentials only ever live in `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment variables,
never in git.
