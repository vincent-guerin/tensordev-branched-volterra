"""Explicit readout-time wrapper for branched Volterra trajectories."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .signature import BranchedVolterraSignature
from .trees import RootedTree

Array = jnp.ndarray


@dataclass(frozen=True)
class MultitimeBranchedResult:
    """Coefficients indexed by external readout time and rooted tree.

    ``values[b, i, r]`` is the coefficient of tree ``trees[r]`` read at the
    external Volterra time ``times[i]``.  Keeping this axis explicit prevents
    accidentally treating a Volterra signature as an ordinary one-time vector.
    """

    trees: tuple[RootedTree, ...]
    times: Array
    values: Array

    def coefficient(self, tree: RootedTree) -> Array:
        return self.values[..., self.trees.index(tree)]

    @property
    def terminal(self) -> Array:
        return self.values[:, -1, :]


def multitime_branched_vsig(
    X: Array,
    times: Array,
    weights: Array,
    *,
    trunc: int,
    increment_input: bool = False,
    normalize_symmetry: bool = False,
) -> MultitimeBranchedResult:
    """Compute branched Volterra coefficients for every external readout time."""
    times = jnp.asarray(times)
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    transform = BranchedVolterraSignature(
        dimension=int(jnp.shape(X)[-1]),
        trunc=trunc,
        normalize_symmetry=normalize_symmetry,
    )
    result = transform(
        X,
        weights,
        increment_input=increment_input,
        return_trajectory=True,
    )
    if result.values.shape[1] != times.shape[0]:
        raise ValueError("times length must equal the number of readout nodes")
    return MultitimeBranchedResult(transform.trees, times, result.values)
