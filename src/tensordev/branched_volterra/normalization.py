"""Explicit normalisation conventions for tree-indexed coefficients."""
from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from .hopf import tree_factorial
from .signature import BranchedVolterraResult
from .trees import symmetry_factor

Normalization = Literal["raw", "symmetry", "tree_factorial"]


def normalize(result: BranchedVolterraResult, convention: Normalization = "raw") -> BranchedVolterraResult:
    """Return coefficients under one named and reproducible convention.

    ``raw`` leaves the product-integration recursion unchanged.  ``symmetry``
    divides each tree by its Butcher symmetry factor, while ``tree_factorial``
    uses the B-series tree factorial.  The choice is intentionally explicit;
    compatibility with another library must state its convention.
    """
    if convention == "raw":
        return result
    if convention == "symmetry":
        divisor = jnp.asarray([symmetry_factor(tree) for tree in result.trees], dtype=result.values.dtype)
    elif convention == "tree_factorial":
        divisor = jnp.asarray([tree_factorial(tree) for tree in result.trees], dtype=result.values.dtype)
    else:
        raise ValueError(f"unknown normalization convention: {convention!r}")
    return BranchedVolterraResult(result.trees, result.values / divisor)
