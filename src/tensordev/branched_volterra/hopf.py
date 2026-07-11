"""Finite Connes--Kreimer/Butcher Hopf algebra utilities.

The routines are deliberately exact and symbolic: they return formal forests
and integer coefficients.  They provide the algebraic bookkeeping needed by a
future Volterra convolution/renormalisation layer without hiding conventions.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import product

from .trees import RootedTree

Forest = tuple[RootedTree, ...]
Formal = dict[Forest, int]
UNIT: Forest = ()


def forest(*trees: RootedTree) -> Forest:
    """Canonical commutative forest product."""
    return tuple(sorted(trees, key=lambda t: t.key))


def forest_degree(value: Forest) -> int:
    return sum(tree.degree for tree in value)


def tree_factorial(tree: RootedTree) -> int:
    """Tree factorial ``|tau| prod_child child!``."""
    result = tree.degree
    for child in tree.children:
        result *= tree_factorial(child)
    return result


def _add(left: Formal, right: Formal) -> Formal:
    result = defaultdict(int, left)
    for key, value in right.items():
        result[key] += value
    return {key: value for key, value in result.items() if value}


def _scale(value: Formal, coefficient: int) -> Formal:
    return {key: coefficient * amount for key, amount in value.items() if coefficient * amount}


def _mul(left: Formal, right: Formal) -> Formal:
    result = defaultdict(int)
    for left_forest, left_value in left.items():
        for right_forest, right_value in right.items():
            result[forest(*left_forest, *right_forest)] += left_value * right_value
    return {key: value for key, value in result.items() if value}


def _product_formal(values: list[Formal]) -> Formal:
    result: Formal = {UNIT: 1}
    for value in values:
        result = _mul(result, value)
    return result


@lru_cache(maxsize=None)
def _cut_options(tree: RootedTree) -> tuple[tuple[Forest, RootedTree], ...]:
    """Return ``(pruned_forest, trunk)`` for all admissible internal cuts."""
    options: list[list[tuple[Forest, RootedTree | None]]] = []
    for child in tree.children:
        child_options = [(forest(child), None)]
        child_options.extend((pruned, trunk) for pruned, trunk in _cut_options(child))
        options.append(child_options)

    if not options:
        return ((UNIT, tree),)

    result: list[tuple[Forest, RootedTree]] = []
    for choices in product(*options):
        pruned: Forest = forest(*(item for choice in choices for item in choice[0]))
        trunk_children = tuple(choice[1] for choice in choices if choice[1] is not None)
        result.append((pruned, RootedTree(tree.label, trunk_children)))
    return tuple(result)


def coproduct(tree: RootedTree) -> tuple[tuple[Forest, Forest], ...]:
    r"""Connes--Kreimer coproduct using admissible cuts.

    The convention is

    ``Delta(tau) = tau tensor 1 + 1 tensor tau + sum P_c tensor R_c``.
    ``P_c`` is the pruned forest above a non-empty admissible cut and ``R_c``
    is the remaining trunk.
    """
    terms: list[tuple[Forest, Forest]] = [(forest(tree), UNIT), (UNIT, forest(tree))]
    terms.extend((pruned, forest(trunk)) for pruned, trunk in _cut_options(tree) if pruned)
    return tuple(dict.fromkeys(terms))


@lru_cache(maxsize=None)
def antipode_forest(value: Forest) -> tuple[tuple[Forest, int], ...]:
    """Formal antipode of a forest, returned as ``(forest, coefficient)``."""
    if not value:
        return ((UNIT, 1),)
    first, rest = value[0], value[1:]
    result: Formal = {forest(first): -1}
    for pruned, trunk in coproduct(first):
        if not pruned or not trunk:
            continue
        left = dict(antipode_forest(pruned))
        right = dict(antipode_forest(rest))
        result = _add(result, _scale(_mul(left, right), -1))
    # The recursive convolution term is more naturally expressed through the
    # reduced coproduct of the whole forest; multiplicativity handles the rest.
    if rest:
        result = _mul(result, dict(antipode_forest(rest)))
    return tuple(result.items())


def antipode(tree: RootedTree) -> Formal:
    """Return the formal antipode ``S(tree)`` as a forest polynomial."""
    # For a connected rooted-tree Hopf algebra, S(tau)=-tau-sum S(P)R.
    result: Formal = {forest(tree): -1}
    for pruned, trunk in coproduct(tree):
        if not pruned or not trunk:
            continue
        result = _add(
            result,
            _scale(_mul(dict(antipode_forest(pruned)), {forest(trunk): 1}), -1),
        )
    return result
