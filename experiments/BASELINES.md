# ÆTHEL published state of the art

Compiled 2026-08-04. Direct PDF access to aclanthology.org / arxiv.org is blocked by the
session egress policy, so every number was cross-verified against the ar5iv HTML or the
authors' own LaTeX sources (`konstantinosKokos/presentations`, `konstantinosKokos/phd-thesis`);
per-cell verification flags at the bottom.

| Paper | Venue / year | Task | Metric | Number | ÆTHEL |
|---|---|---|---|---|---|
| Neural Proof Nets | CoNLL 2020 | supertagging | types-correct, greedy / β5 / β7 | 85.5 / 93.2 / 93.4 | 0.4, ≤140-node filter |
| Neural Proof Nets | CoNLL 2020 | frame accuracy | greedy / β5 / β7 | 57.6 / 69.6 / 70.2 | 0.4 |
| Neural Proof Nets | CoNLL 2020 | end-to-end term | untyped exact match, greedy / β5 / β7 | 60.0 / 69.1 / 69.6 | 0.4 |
| Neural Proof Nets | CoNLL 2020 | end-to-end term | typed+deps exact match, greedy / β5 / β7 | 56.9 / 67.1 / 67.6 | 0.4 |
| Geometry-Aware Supertagging | CLASP LSD 2023 | supertagging | overall, greedy | 93.67 | 2022 release |
| Geometry-Aware Supertagging | CLASP LSD 2023 | supertagging | frequent / uncommon / rare / unseen | 94.72 / 73.45 / 53.83 / 15.78 | 2022 release |
| Geometry-Aware (thesis rerun) | LOT thesis 2023 | supertagging | overall, 6-run avg ± σ | **94.08 ± 0.02** | 1.0.0a5 |
| Geometry-Aware (thesis rerun) | LOT thesis 2023 | supertagging | frequent / uncommon / rare / unseen | 95.16 / 75.55 / 58.15 / 18.37 | 1.0.0a5 |
| SPINDLE | EACL 2023 demos | pipeline | parsability / coverage | 86.83 / 84.94 (thesis: 87.35 ± 0.18 / 85.56 ± 0.22) | 1.0.0a5 |
| SPINDLE | EACL 2023 demos | end-to-end proof | frame / strict exact match, greedy | 56.88 / **55.30** (thesis: 57.76 ± 0.55 / 55.63 ± 0.55) | 1.0.0a5 |
| SPINDLE (thesis eval) | LOT thesis 2023 | subproof F1 | strict / modulo types | 89.17 / 92.00 | 1.0.0a5 |

## The numbers to beat

- **Supertagging: 94.08%** overall token accuracy on ÆTHEL 1.0.0a5 (thesis rerun; the
  CLASP-published figure is 93.67 on an earlier release) — beating 94.08 beats every
  published figure. The unseen-type ceiling is 18.37%, the tail is where the headroom is.
- **Proof/term accuracy: 55.63%** strict proof exact match (typed + dependencies, greedy,
  full unfiltered test set; 55.30 in the EACL poster). NPN's famous ≈70% is **not
  comparable**: frame correctness at beam 7 on the filtered ÆTHEL 0.4. Same-protocol
  context: proofs correct given a correct frame are 96.31%, so supertagging is the
  bottleneck; parsability ≈87.4 is the current ceiling on end-to-end accuracy.
- **No published improvement 2023–2026** was found for either task on ÆTHEL; SPINDLE plus
  the geometry-aware tagger are the standing bar.

SPINDLE chains the geometry-aware heterogeneous graph-convolution supertagger, a
Sinkhorn-based parallel axiom-linking module and the `mill` type-checker, greedy only —
the parallel tagger forfeits beam search.

Test-set profile on 1.0.0a5 (matches our local dump): 5,770 sentences, 95,331 tokens
(91,503 frequent / 2,639 uncommon / 826 rare / 363 unseen), 5,762 distinct types
(5,146 in train), splits 56,875 / 6,118 / 5,770.

## Citations

1. K. Kogkalidis, M. Moortgat, R. Moot. *Neural Proof Nets*. CoNLL 2020, pp. 26–40.
   [2020.conll-1.3](https://aclanthology.org/2020.conll-1.3/) ·
   [arXiv:2009.12702](https://arxiv.org/abs/2009.12702)
2. K. Kogkalidis, M. Moortgat. *Geometry-Aware Supertagging with Heterogeneous Dynamic
   Convolutions*. CLASP LSD 2023, pp. 107–119.
   [2023.clasp-1.13](https://aclanthology.org/2023.clasp-1.13/) ·
   [arXiv:2203.12235](https://arxiv.org/abs/2203.12235)
3. K. Kogkalidis, M. Moortgat, R. Moot. *SPINDLE: Spinning Raw Text into Lambda Terms with
   Graph Attention*. EACL 2023 System Demonstrations, pp. 128–135.
   [2023.eacl-demo.15](https://aclanthology.org/2023.eacl-demo.15/) ·
   [arXiv:2302.12050](https://arxiv.org/abs/2302.12050)
4. K. Kogkalidis. *Dependency as Modality, Parsing as Permutation*. PhD thesis, Utrecht
   University, LOT Dissertation Series, 2023.
   [LOT](https://www.lotpublications.nl/dependency-as-modality-parsing-as-permutation) ·
   [source](https://github.com/konstantinosKokos/phd-thesis)
5. K. Kogkalidis, M. Moortgat, R. Moot. *ÆTHEL: Automatically Extracted Typelogical
   Derivations for Dutch*. LREC 2020, pp. 5257–5266.
   [2020.lrec-1.647](https://aclanthology.org/2020.lrec-1.647/) ·
   [arXiv:1912.12635](https://arxiv.org/abs/1912.12635)

## Verification flags

- NPN Table 1 verified twice (ar5iv HTML; author's CoNLL 2020 slides). The ÆTHEL-0.4 /
  ≤140-node filter attribution comes from a thesis footnote, not the paper's own text.
- Geometry-aware CLASP cells confirmed in two author sources; the slides say frequent
  94.72 / unseen 15.78 while the repo README says 94.83 / 15.79 — the CLASP PDF could not
  be opened to arbitrate. The 94.08 row is thesis-only (not a conference venue).
- SPINDLE numbers are from the author's EACL 2023 poster LaTeX presenting the same paper;
  the demo-paper table itself could not be fetched. The thesis reports slightly higher
  3-run averages.
- "No newer results" is a negative claim from August 2026 web searches.
