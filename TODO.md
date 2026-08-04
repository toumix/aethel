# TODO

Prompt (USER, 🌤️ Daylight session 2026-08-04, verbatim):

> Scale this ACG experiment to get SOTA on the Aethel dataset https://github.com/discopy/discopy/pull/400

Rulings from the same session: staged targets — supertagging first, then full proofs; the
experiment lives in this repo; training runs on Modal (credentials given live, kept out of git).

---

Measurements (2026-08-04): `data/aethel_1.0.0a5.zip` loads under Python 3.11 — 68,763 samples
(56,875 train / 6,118 dev / 5,770 test), 992,385 tokens, 5,762 distinct types. Baselines to beat:
Neural Proof Nets (CoNLL 2020) ≈70% proof/term accuracy; SPINDLE (EACL 2023), graph-attention
supertagger plus Sinkhorn linking, is the current SOTA. USER opened the network policy for
`api.modal.com` (plain HTTPS now passes) but the session proxy does not carry gRPC, which the
Modal client requires — so all Modal runs are driven from GitHub Actions, with
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` as repository secrets, never from a session and never
in git.

## Phase 0 — the bridge (CPU, unblocked)

- [ ] Pin the exact SPINDLE and NPN test numbers (supertagging accuracy, proof/term accuracy)
  with table citations in `experiments/BASELINES.md`
- [ ] `experiments/bridge.py`: convert `aethel.mill` types and terms to
  `discopy.grammar.abstract` (discopy#400 — install from its branch until it lands; when
  updating it from `main`, its `closed.py` hunks resolve toward the merged #442)
- [ ] Round-trip: convert and type-check all 68,763 proofs, report coverage and file the residue
  as issues

## Phase 1 — supertagging

- [ ] Data module respecting the official split, with long-tail and unseen-type statistics on
  dev and test
- [ ] Reproduce the SPINDLE tagger evaluation on our split to establish parity
- [ ] Constructive type decoder over a Dutch pretrained encoder (RobBERT-2023 or XLM-R large):
  decode each type as a tree of connectives, never a 5,762-way softmax
- [ ] `experiments/modal_app.py`: GPU training on Modal with checkpoints on a Modal volume,
  launched by a GitHub Actions workflow — `modal-check.yml` is the seed, verified green with
  the repository secrets on 2026-08-04 (run 30917331003)
- [ ] Sweep on dev, report test accuracy against SPINDLE

## Phase 2 — full proofs

- [ ] Term construction from predicted types: Sinkhorn-style permutation linking versus discopy
  proof search over `grammar.abstract` — measure both on dev
- [ ] Exact-match term accuracy on test, up to alpha and beta via discopy#442 `normal_form`,
  against NPN ≈70%
- [ ] Report tables in `experiments/README.md`

Credentials only ever live in `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment variables,
never in git.
