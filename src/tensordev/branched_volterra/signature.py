r"""Finite-grid, differentiable branched Volterra signatures.

For a tree ``B_a^+(tau_1 ... tau_r)`` the feature is discretised as
``sum_j K(t_i, s_j) product_l Z_{s_j}^{tau_l} Delta X_j^a``.  This is the
left-point product-integration form of the tree-indexed Volterra integral.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from .trees import RootedTree, decorated_trees, symmetry_factor

Array = jax.Array


def flat_causal_weights(steps: int, *, dtype=jnp.float32) -> Array:
    """Cell averages for the flat kernel K=1, with strict causality."""
    return (jnp.arange(steps)[None, :] < jnp.arange(steps + 1)[:, None]).astype(dtype)


def fractional_causal_weights(times: Array, beta: float | Array) -> Array:
    """Analytically cell-averaged fractional kernel weights.

    This integrates the diagonal singularity exactly rather than introducing an
    arbitrary epsilon.  ``beta=1`` recovers :func:`flat_causal_weights`.
    """
    times = jnp.asarray(times)
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times must be a one-dimensional grid with at least two nodes")
    beta = jnp.asarray(beta, dtype=times.dtype)
    if beta.ndim != 0 or bool(beta <= 0):
        raise ValueError("beta must be a positive scalar")
    left, right = times[:-1][None, :], times[1:][None, :]
    target = times[:, None]
    average = (jnp.maximum(target - left, 0) ** beta - jnp.maximum(target - right, 0) ** beta)
    average /= (right - left) * jnp.exp(jax.scipy.special.gammaln(beta + 1))
    return jnp.where(jnp.arange(times.shape[0] - 1)[None, :] < jnp.arange(times.shape[0])[:, None], average, 0)


@dataclass(frozen=True)
class BranchedVolterraResult:
    trees: tuple[RootedTree, ...]
    values: Array

    def coefficient(self, tree: RootedTree) -> Array:
        return self.values[..., self.trees.index(tree)]


@dataclass(frozen=True)
class BranchedVolterraSignature:
    """A tree-indexed Volterra feature map with static tree basis."""

    dimension: int
    trunc: int
    normalize_symmetry: bool = False
    trees: tuple[RootedTree, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trees", decorated_trees(self.dimension, self.trunc))

    def __call__(self, X: Array, weights: Array, *, increment_input: bool = False, return_trajectory: bool = False) -> BranchedVolterraResult:
        X = jnp.asarray(X)
        if X.ndim == 2:
            X = X[None, ...]
        if X.ndim != 3 or X.shape[-1] != self.dimension:
            raise ValueError(f"X must end in dimension {self.dimension}; got {tuple(X.shape)}")
        dX = X if increment_input else jnp.diff(X, axis=1)
        steps = dX.shape[1]
        weights = jnp.asarray(weights, dtype=dX.dtype)
        if weights.shape != (steps + 1, steps):
            raise ValueError(f"weights must have shape {(steps + 1, steps)}, got {tuple(weights.shape)}")

        trajectories: dict[RootedTree, Array] = {}
        for tree in self.trees:
            product = jnp.ones((dX.shape[0], steps), dtype=dX.dtype)
            for child in tree.children:
                product *= trajectories[child][:, :-1]
            integral = jnp.einsum("ij,bj->bi", weights, product * dX[..., tree.label])
            trajectories[tree] = integral / symmetry_factor(tree) if self.normalize_symmetry else integral
        stacked = jnp.stack([trajectories[tree] for tree in self.trees], axis=-1)
        return BranchedVolterraResult(self.trees, stacked if return_trajectory else stacked[:, -1])


def branched_vsig(X: Array, weights: Array, *, trunc: int, increment_input: bool = False, return_trajectory: bool = False) -> BranchedVolterraResult:
    """Functional branched Volterra signature API."""
    transform = BranchedVolterraSignature(int(jnp.shape(X)[-1]), trunc)
    return transform(X, weights, increment_input=increment_input, return_trajectory=return_trajectory)
