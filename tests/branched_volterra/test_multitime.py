import jax.numpy as jnp

from tensordev.branched_volterra import flat_causal_weights
from tensordev.branched_volterra.multitime import multitime_branched_vsig


def test_external_readout_axis_is_preserved():
    times = jnp.linspace(0.0, 1.0, 5)
    path = jnp.stack((times, times**2), axis=-1)[None]
    result = multitime_branched_vsig(path, times, flat_causal_weights(4), trunc=2)
    assert result.values.shape[1] == 5
    assert result.terminal.shape == (1, len(result.trees))
