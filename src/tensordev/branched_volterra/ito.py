"""Finite-grid Ito bracket features for branched Volterra experiments.

This module does not claim to construct a renormalised stochastic Volterra
rough path.  It exposes the additional quadratic-variation information that a
geometric/Stratonovich feature map does not retain by itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .signature import BranchedVolterraSignature
from .trees import RootedTree

Array = jnp.ndarray


def quadratic_variation_increments(dX: Array) -> Array:
    """Per-cell bracket increments ``dX^a dX^b``.

    Input shape is ``(batch, steps, dimension)`` and output shape is
    ``(batch, steps, dimension, dimension)``.
    """
    dX = jnp.asarray(dX)
    if dX.ndim != 3:
        raise ValueError("dX must have shape (batch, steps, dimension)")
    return jnp.einsum("nsa,nsc->nsac", dX, dX)


def volterra_bracket_trajectory(dX: Array, weights: Array) -> Array:
    r"""Compute the kernel-weighted discrete bracket.

    ``bracket[b, i, a, c] = sum_{j<i} W[i,j] dX[b,j,a] dX[b,j,c]``.
    """
    dX = jnp.asarray(dX)
    weights = jnp.asarray(weights, dtype=dX.dtype)
    if weights.shape != (dX.shape[1] + 1, dX.shape[1]):
        raise ValueError("weights must have shape (steps + 1, steps)")
    return jnp.einsum("ij,bjac->biac", weights, quadratic_variation_increments(dX))


@dataclass(frozen=True)
class ItoBranchedVolterraResult:
    """Joint ordinary-tree and bracket features at every readout time."""

    trees: tuple[RootedTree, ...]
    tree_values: Array
    bracket_values: Array

    @property
    def terminal(self) -> Array:
        tree = self.tree_values[:, -1, :]
        bracket = self.bracket_values[:, -1].reshape(self.bracket_values.shape[0], -1)
        return jnp.concatenate((tree, bracket), axis=-1)


def ito_branched_vsig(
    X: Array,
    weights: Array,
    *,
    trunc: int,
    increment_input: bool = False,
    normalize_symmetry: bool = False,
) -> ItoBranchedVolterraResult:
    """Return a finite-grid branched Volterra map augmented by Ito brackets."""
    X = jnp.asarray(X)
    if X.ndim == 2:
        X = X[None, ...]
    dX = X if increment_input else jnp.diff(X, axis=1)
    transform = BranchedVolterraSignature(
        dimension=X.shape[-1],
        trunc=trunc,
        normalize_symmetry=normalize_symmetry,
    )
    tree = transform(X, weights, increment_input=increment_input, return_trajectory=True)
    bracket = volterra_bracket_trajectory(dX, weights)
    return ItoBranchedVolterraResult(tree.trees, tree.values, bracket)
