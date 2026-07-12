import jax.numpy as jnp

from tensordev.volterra import ConvolutionKernel, MultitimeVolterraRoughPath


def test_multitime_fractional_lift_has_three_time_axes_and_level_one_chen_relation():
    kernel = ConvolutionKernel.fractional(beta=jnp.array([0.65]), A=jnp.eye(2)[None])
    lift = MultitimeVolterraRoughPath(kernel=kernel, trunc=3).lift(
        jnp.array([[[0.0, 0.0], [0.1, 0.2], [0.2, 0.4], [0.3, 0.6]]]),
        times=jnp.linspace(0.0, 1.0, 4),
    )
    assert lift.levels[0].shape == (1, 4, 4, 4, 2)
    assert lift.levels[1].shape == (1, 4, 4, 4, 4)
    assert lift.levels[2].shape == (1, 4, 4, 4, 8)
    assert jnp.max(jnp.abs(lift.chen_level_one_residual())) < 1e-5
