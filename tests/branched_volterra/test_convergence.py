import jax.numpy as jnp
import numpy as np

from tensordev.branched_volterra import BranchedVolterraSignature, RootedTree, flat_causal_weights


def test_smooth_chain_converges_on_refined_grids():
    errors = []
    chain = RootedTree(0, (RootedTree(0),))
    for steps in (32, 64, 128):
        times = jnp.linspace(0.0, 1.0, steps + 1)
        # x(t)=t, whose second iterated integral is 1/2.
        path = times[None, :, None]
        result = BranchedVolterraSignature(1, 2)(path, flat_causal_weights(steps))
        errors.append(abs(float(result.coefficient(chain)[0]) - 0.5))
    assert errors[-1] < errors[0]
    assert np.isfinite(errors).all()
