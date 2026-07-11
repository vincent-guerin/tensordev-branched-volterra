"""Decorated non-planar rooted trees used by branched Volterra features."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement


@dataclass(frozen=True)
class RootedTree:
    """A rooted tree with an integer decoration at its root.

    Children are stored canonically, so two trees that differ only by a
    permutation of children are identified, as required by the Butcher/
    Connes--Kreimer Hopf algebra.
    """

    label: int
    children: tuple["RootedTree", ...] = ()

    def __post_init__(self) -> None:
        if self.label < 0:
            raise ValueError("tree labels must be non-negative")
        object.__setattr__(self, "children", tuple(sorted(self.children, key=lambda t: t.key)))

    @property
    def degree(self) -> int:
        return 1 + sum(child.degree for child in self.children)

    @property
    def key(self) -> tuple:
        return (self.degree, self.label, tuple(child.key for child in self.children))


def decorated_trees(dimension: int, trunc: int) -> tuple[RootedTree, ...]:
    """Deterministically enumerate decorated rooted trees up to ``trunc``."""
    if dimension <= 0 or trunc <= 0:
        raise ValueError("dimension and trunc must be positive")
    by_degree: dict[int, list[RootedTree]] = {1: [RootedTree(i) for i in range(dimension)]}
    previous = list(by_degree[1])
    for degree in range(2, trunc + 1):
        forests: set[tuple[RootedTree, ...]] = set()
        for length in range(1, degree):
            for forest in combinations_with_replacement(previous, length):
                if sum(tree.degree for tree in forest) == degree - 1:
                    forests.add(tuple(sorted(forest, key=lambda t: t.key)))
        by_degree[degree] = sorted(
            [RootedTree(label, forest) for label in range(dimension) for forest in forests],
            key=lambda tree: tree.key,
        )
        previous.extend(by_degree[degree])
    return tuple(tree for degree in range(1, trunc + 1) for tree in by_degree[degree])


def symmetry_factor(tree: RootedTree) -> int:
    """Butcher symmetry factor of a tree."""
    result = 1
    multiplicities: dict[RootedTree, int] = {}
    for child in tree.children:
        result *= symmetry_factor(child)
        multiplicities[child] = multiplicities.get(child, 0) + 1
    for count in multiplicities.values():
        for integer in range(2, count + 1):
            result *= integer
    return result
