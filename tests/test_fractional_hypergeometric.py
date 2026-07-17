import jax
import jax.numpy as jnp

from tensordev.fractional_hypergeometric import (
    ErdelyiKoberLeftIntegral,
    MarichevSaigoMaedaLeftIntegral,
    RiemannLiouvilleLeftIntegral,
    SaigoLeftIntegral,
)


def test_riemann_liouville_constant_is_x_to_alpha_over_gamma_alpha_plus_one():
    alpha = 0.6
    x = jnp.array(2.0)
    value = RiemannLiouvilleLeftIntegral(alpha).power(1.0, x)
    expected = x ** alpha / jnp.exp(jax.scipy.special.gammaln(alpha + 1.0))
    assert jnp.allclose(value, expected)


def test_saigo_reduces_to_riemann_liouville():
    alpha, d, x = 0.7, 1.3, jnp.array(1.4)
    saigo = SaigoLeftIntegral(alpha=alpha, beta=-alpha, delta=0.2)
    rl = RiemannLiouvilleLeftIntegral(alpha)
    assert jnp.allclose(saigo.power(d, x), rl.power(d, x), rtol=1e-5)


def test_erdelyi_kober_is_the_saigo_beta_zero_reduction():
    alpha, delta, d, x = 0.5, 0.3, 1.2, jnp.array(0.8)
    saigo = SaigoLeftIntegral(alpha=alpha, beta=0.0, delta=delta)
    ek = ErdelyiKoberLeftIntegral(alpha=alpha, delta=delta)
    assert jnp.allclose(saigo.power(d, x), ek.power(d, x), rtol=1e-5)


def test_msm_power_series_is_linear():
    msm = MarichevSaigoMaedaLeftIntegral(a=0.1, a_prime=0.1, b=0.05, b_prime=0.2, c=0.8)
    x = jnp.array(1.1)
    coeffs = jnp.array([2.0, -0.5])
    exponents = jnp.array([1.5, 2.0])
    expected = coeffs[0] * msm.power(exponents[0], x) + coeffs[1] * msm.power(exponents[1], x)
    assert jnp.allclose(msm.power_series(coeffs, exponents, x), expected)
