import jax.numpy as jnp
import numpy as np

from tensordev.branched_volterra import BranchedVolterraSignature, RootedTree, flat_causal_weights
from tensordev.branched_volterra.ito import ito_branched_vsig, volterra_bracket_trajectory
from tensordev.branched_volterra.normalization import normalize


def test_flat_volterra_bracket_is_realized_quadratic_variation():
    increments = jnp.array([[[1.0], [2.0], [3.0]]])
    bracket = volterra_bracket_trajectory(increments, flat_causal_weights(3))
    np.testing.assert_allclose(bracket[0, -1, 0, 0], 14.0)


def test_ito_terminal_includes_tree_and_bracket_features():
    increments = jnp.array([[[1.0], [2.0], [3.0]]])
    result = ito_branched_vsig(increments, flat_causal_weights(3), trunc=2, increment_input=True)
    assert result.terminal.shape == (1, len(result.trees) + 1)


def test_tree_factorial_normalization_is_explicit():
    leaf = RootedTree(0)
    tree = RootedTree(0, (leaf,))
    path = jnp.array([[[0.0], [1.0], [3.0]]])
    raw = BranchedVolterraSignature(1, 2)(path, flat_causal_weights(2))
    normalized = normalize(raw, "tree_factorial")
    np.testing.assert_allclose(normalized.coefficient(tree), raw.coefficient(tree) / 2.0)
