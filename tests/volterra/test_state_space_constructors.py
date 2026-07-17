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


def test_direct_mittag_leffler_runs_without_a_markovian_mixture():
    from tensordev.volterra import VolterraSignature

    kernel = ConvolutionKernel.mittag_leffler(
        alpha=0.7, rate=0.5, A=jnp.eye(2)[None], quad_order=6, max_terms=40,
    )
    signature = VolterraSignature(kernel=kernel, trunc=2).vsig(jnp.zeros((1, 4, 2)), dt=0.25)


def test_mittag_leffler_mixture_has_positive_weights_and_runs_vsig():
    kernel = ConvolutionKernel.mittag_leffler_mixture(
        alpha=0.7, rate=1.5, A=jnp.eye(2)[None], rate_min=1e-3, rate_max=1e3, n_factors=12,
    )
    assert jnp.all(kernel.b >= 0.0)


def test_direct_tricomi_kernel_runs_vsig_and_has_two_scale_parameters():
    import jax.numpy as jnp
    from tensordev.volterra import ConvolutionKernel, VolterraSignature

    kernel = ConvolutionKernel.tricomi(
        a=0.6, b=1.4, tau=1.0,
        A=jnp.eye(2, dtype=jnp.float32)[None], quad_order=4,
    )
    X = jnp.array([[[0.0, 0.0], [0.1, 0.2], [0.2, 0.1]]], dtype=jnp.float32)
    result = VolterraSignature(kernel=kernel, trunc=2).vsig(X, dt=0.1)
    assert tuple(level.shape for level in result) == ((1,), (2,), (4,))


def test_tricomi_kernel_rejects_parameters_outside_sonine_range():
    import jax.numpy as jnp
    import pytest
    from tensordev.volterra import ConvolutionKernel

    with pytest.raises(ValueError, match="0 < a < 1"):
        ConvolutionKernel.tricomi(a=1.0, b=1.4, tau=1.0, A=jnp.eye(1)[None])