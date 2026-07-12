import jax.numpy as jnp

from tensordev.volterra import ConvolutionKernel, VolterraRoughPath


def test_fractional_volterra_rough_path_emits_all_readout_times():
    rough_path = VolterraRoughPath(
        kernel=ConvolutionKernel.fractional(beta=jnp.array([0.65]), A=jnp.eye(2)[None]),
        trunc=2,
    )
    X = jnp.array([[[0.0, 0.0], [0.1, 0.2], [0.2, 0.4], [0.3, 0.6]]])
    lift = rough_path.lift(X, dt=1.0 / 3.0)
    assert lift[0].shape == (1, 4)
    assert lift[1].shape == (2, 4)
    assert jnp.allclose(lift[0], 1.0)
