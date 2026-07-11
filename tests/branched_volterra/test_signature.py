import jax
import jax.numpy as jnp
import numpy as np

from tensordev.branched_volterra import BranchedVolterraSignature, RootedTree, flat_causal_weights, fractional_causal_weights


def test_leaf_for_flat_kernel_is_path_increment():
    path = jnp.array([[[0.0], [1.0], [3.0], [6.0]]])
    result = BranchedVolterraSignature(1, 3)(path, flat_causal_weights(3))
    np.testing.assert_allclose(result.coefficient(RootedTree(0)), [6.0])


def test_chain_is_discrete_ordered_integral():
    increments = jnp.array([[[1.0], [2.0], [3.0]]])
    result = BranchedVolterraSignature(1, 2)(increments, flat_causal_weights(3), increment_input=True)
    chain = RootedTree(0, (RootedTree(0),))
    np.testing.assert_allclose(result.coefficient(chain), [11.0])


def test_branch_and_chain_are_distinct_degree_three_features():
    increments = jnp.array([[[1.0], [2.0], [3.0]]])
    result = BranchedVolterraSignature(1, 3)(increments, flat_causal_weights(3), increment_input=True)
    leaf = RootedTree(0)
    branch = RootedTree(0, (leaf, leaf))
    chain = RootedTree(0, (RootedTree(0, (leaf,)),))
    np.testing.assert_allclose(result.coefficient(branch), [29.0])
    np.testing.assert_allclose(result.coefficient(chain), [6.0])


def test_fractional_beta_one_recovers_flat_kernel():
    times = jnp.linspace(0.0, 1.0, 6)
    np.testing.assert_allclose(fractional_causal_weights(times, 1.0), flat_causal_weights(5))


def test_transform_is_differentiable():
    transform = BranchedVolterraSignature(1, 3)
    weights = flat_causal_weights(3)
    grad = jax.grad(lambda x: jnp.sum(transform(x, weights, increment_input=True).values))(jnp.ones((1, 3, 1)))
    assert grad.shape == (1, 3, 1)
    assert bool(jnp.all(jnp.isfinite(grad)))
