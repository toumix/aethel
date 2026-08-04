"""
The bridge between æthel's modal intuitionistic linear logic and discopy's
abstract categorial grammars (discopy.grammar.abstract, discopy#400).

Types translate structurally: atoms to atomic types, functors to
exponentials. Modal types have no counterpart in a symmetric closed
category, so ``◇d(A)`` and ``□d(A)`` become atomic types named by their
æthel prefix notation, together with head constants for their four rules:

    ▵d : A → ◇d(A)      ▿d : ◇d(A) → A
    ▴d : A → □d(A)      ▾d : □d(A) → A

``CaseOf(becomes, where, original)`` becomes an explicit head constant
``case : T(becomes) → (T(becomes) → T(original)) → T(original)`` applied to
the translated ``becomes`` and the abstraction of ``where`` in ``original``,
so that decoding is driven by head constants alone and every genuine redex
of the source is preserved.

``encode`` and ``decode`` are exact inverses on every æthel proof term:
``decode(encode(term)) == term`` with equal types, and both directions
type-check on construction — æthel through ``TypeInference``, discopy
through ``Application.__check_dom__``.
"""
from __future__ import annotations

from aethel.mill import terms, types
from discopy.grammar import abstract

DIAMOND_INTRO, DIAMOND_ELIM, BOX_INTRO, BOX_ELIM = "▵", "▿", "▴", "▾"
CASE = "case"


def encode_type(t: types.Type) -> abstract.Ty:
    """Translate an æthel type into an abstract type."""
    match t:
        case types.Atom(sign):
            return abstract.Ty(sign)
        case types.Functor(argument, result):
            return encode_type(argument) >> encode_type(result)
        case types.Modal(_, _):
            return abstract.Ty(t.prefix())
    raise ValueError(f"Unknown æthel type: {t}")


def decode_type(t: abstract.Ty) -> types.Type:
    """Translate an abstract type back into an æthel type."""
    if t.is_exp:
        return types.Functor(decode_type(t.exponent), decode_type(t.base))
    return types.Type.parse_prefix(t.inside[0].name)


def modal(symbol: str, decoration: str, dom: abstract.Ty,
          cod: abstract.Ty) -> abstract.Constant:
    """The head constant of a modal rule, e.g. ``▵det : A → ◇det(A)``."""
    return abstract.Constant(f"{symbol}{decoration}", dom >> cod)


def encode(term: terms.Term) -> abstract.Term:
    """Translate an æthel proof term into an abstract term."""
    match term:
        case terms.Variable(t, index):
            return abstract.Variable(f"x{index}", encode_type(t))
        case terms.Constant(t, index):
            return abstract.Constant(f"c{index}", encode_type(t))
        case terms.ArrowElimination(function, argument):
            return encode(function)(encode(argument))
        case terms.ArrowIntroduction(abstraction, body):
            return abstract.Abstraction(encode(abstraction), encode(body))
        case terms.DiamondIntroduction(decoration, body):
            arg = encode(body)
            return modal(
                DIAMOND_INTRO, decoration, arg.cod,
                encode_type(term.type))(arg)
        case terms.DiamondElimination(decoration, body):
            arg = encode(body)
            return modal(
                DIAMOND_ELIM, decoration, arg.cod,
                encode_type(term.type))(arg)
        case terms.BoxIntroduction(decoration, body):
            arg = encode(body)
            return modal(
                BOX_INTRO, decoration, arg.cod, encode_type(term.type))(arg)
        case terms.BoxElimination(decoration, body):
            arg = encode(body)
            return modal(
                BOX_ELIM, decoration, arg.cod, encode_type(term.type))(arg)
        case terms.CaseOf(becomes, where, original):
            arg, var, body = encode(becomes), encode(where), encode(original)
            branch = abstract.Abstraction(var, body)
            head = abstract.Constant(
                CASE, arg.cod >> (branch.cod >> body.cod))
            return head(arg)(branch)
    raise ValueError(f"Unknown æthel term: {term}")


def decode(term: abstract.Term) -> terms.Term:
    """Translate an abstract term back into an æthel proof term."""
    if isinstance(term, abstract.Variable):
        return terms.Variable(decode_type(term.cod), int(term.name[1:]))
    if isinstance(term, abstract.Abstraction):
        return terms.ArrowIntroduction(
            decode(term.var), decode(term.body))
    if isinstance(term, abstract.Constant):
        return terms.Constant(decode_type(term.cod), int(term.name[1:]))
    if isinstance(term, abstract.Application):
        return decode_application(term)
    raise ValueError(f"Unknown abstract term: {term}")


def decode_application(term: abstract.Application) -> terms.Term:
    """Decode an application, dispatching on its head constant."""
    head, args = term.func, term.args
    if isinstance(head, abstract.Constant):
        symbol, decoration = head.name[:1], head.name[1:]
        if symbol == DIAMOND_INTRO:
            return terms.DiamondIntroduction(decoration, decode(args))
        if symbol == DIAMOND_ELIM:
            return terms.DiamondElimination(decoration, decode(args))
        if symbol == BOX_INTRO:
            return terms.BoxIntroduction(decoration, decode(args))
        if symbol == BOX_ELIM:
            return terms.BoxElimination(decoration, decode(args))
    if isinstance(head, abstract.Application) and isinstance(
            head.func, abstract.Constant) and head.func.name == CASE:
        branch = term.args
        return terms.CaseOf(
            decode(head.args), decode(branch.var), decode(branch.body))
    return terms.ArrowElimination(decode(term.func), decode(term.args))
