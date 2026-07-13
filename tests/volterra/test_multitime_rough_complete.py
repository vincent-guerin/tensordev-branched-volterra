import jax.numpy as jnp

from tensordev.volterra import ConvolutionKernel, MultitimeVolterraRoughPath


def _path(nodes: int):
    t = jnp.linspace(0.0, 1.0, nodes)
    return jnp.stack((t, t ** 2), axis=-1)[None], t


def test_chen_volterra_identity_through_degree_three():
    X, times = _path(3)
    kernel = ConvolutionKernel.fractional(
        beta=jnp.array([0.65]), A=jnp.eye(2)[None]
    )
    result = MultitimeVolterraRoughPath(kernel, trunc=3).lift(X, times=times)
    for degree in (1, 2, 3):
        assert result.chen_report(degree).max_abs < 1e-6


def test_kernel_matrix_A_changes_alphabet_and_increments():
    X, times = _path(3)
    A = jnp.array([[[2.0, -1.0]]])  # (q=1,m=1,d=2)
    kernel = ConvolutionKernel.fractional(beta=jnp.array([0.7]), A=A)
    result = MultitimeVolterraRoughPath(kernel, trunc=2).lift(X, times=times)
    expected = jnp.einsum("...d,md->...m", jnp.diff(X, axis=1), A[0])
    assert result.levels[0].shape[-1] == 1
    assert result.levels[1].shape[-1] == 1
    assert jnp.allclose(result.effective_increments, expected)


def test_zero_rate_gamma_reduces_to_scaled_fractional_weights():
    _, times = _path(4)
    A = jnp.eye(2)[None]
    fractional = ConvolutionKernel.fractional(beta=jnp.array([0.6]), A=A)
    gamma = ConvolutionKernel.gamma(
        beta=jnp.array([0.6]), rate=jnp.array([0.0]),
        scale=jnp.array([2.0]), A=A,
    )
    X, _ = _path(4)
    f = MultitimeVolterraRoughPath(fractional, 1).lift(X, times=times)
    g = MultitimeVolterraRoughPath(gamma, 1).lift(X, times=times)
    assert jnp.allclose(g.weights, 2.0 * f.weights, rtol=1e-5, atol=1e-6)
