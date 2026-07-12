import jax.numpy as jnp

from tensordev.volterra import ConvolutionKernel


def test_exponential_mixture_builds_the_expected_scalar_kernel():
    kernel = ConvolutionKernel.exponential_mixture(
        rates=jnp.array([0.2, 2.0, 20.0]), weights=jnp.array([0.15, 0.35, 0.50]), A=jnp.eye(2)[None],
    )
    u = jnp.array([0.0, 0.3])
    readout = jnp.einsum("bij,pj->bpi", kernel.lam.expm(u), kernel.b).sum(axis=-1)[:, 0]
    expected = jnp.sum(
        jnp.array([0.15, 0.35, 0.50]) * jnp.exp(-jnp.array([0.2, 2.0, 20.0])[None, :] * u[:, None]), axis=-1,
    )
    assert jnp.allclose(readout, expected)


def test_damped_oscillatory_builds_the_expected_scalar_kernel():
    kernel = ConvolutionKernel.damped_oscillatory(
        decay=0.7, frequency=2.0, cos_scale=1.2, sin_scale=-0.4, A=jnp.eye(2)[None],
    )
    u = jnp.array([0.0, 0.25])
    readout = jnp.einsum("bij,pj->bpi", kernel.lam.expm(u), kernel.b).sum(axis=-1)[:, 0]
    expected = jnp.exp(-0.7 * u) * (1.2 * jnp.cos(2.0 * u) - 0.4 * jnp.sin(2.0 * u))
    assert jnp.allclose(readout, expected)


def test_rough_fractional_mixture_has_positive_weights_and_runs_vsig():
    kernel = ConvolutionKernel.rough_fractional_mixture(
        beta=0.6, A=jnp.eye(2)[None], rate_min=1e-3, rate_max=1e3, n_factors=12,
    )
    assert jnp.all(kernel.b >= 0.0)


def test_mittag_leffler_mixture_has_positive_weights_and_runs_vsig():
    kernel = ConvolutionKernel.mittag_leffler_mixture(
        alpha=0.7, rate=1.5, A=jnp.eye(2)[None], rate_min=1e-3, rate_max=1e3, n_factors=12,
    )
    assert jnp.all(kernel.b >= 0.0)
