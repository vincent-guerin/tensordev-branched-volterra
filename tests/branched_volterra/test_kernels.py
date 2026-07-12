import jax.numpy as jnp
import numpy as np

from tensordev.branched_volterra import flat_causal_weights, fractional_causal_weights
from tensordev.branched_volterra.kernels import CausalKernel


def test_gamma_rate_zero_is_fractional_limit():
    times = jnp.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(
        CausalKernel.gamma(times, beta=0.7, rate=0.0, scale=1.5),
        1.5 * fractional_causal_weights(times, 0.7),
    )


def test_single_exponential_mixture_matches_gamma_beta_one():
    times = jnp.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(
        CausalKernel.exponential_mixture(times, rates=jnp.array([2.0]), weights=jnp.array([0.4])),
        CausalKernel.gamma(times, beta=1.0, rate=2.0, scale=0.4),
        rtol=1e-6,
        atol=1e-6,
    )


def test_oscillatory_zero_frequency_is_exponential():
    times = jnp.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(
        CausalKernel.oscillatory(times, decay=1.5, frequency=0.0, scale=0.8),
        CausalKernel.exponential_mixture(times, rates=jnp.array([1.5]), weights=jnp.array([0.8])),
        rtol=1e-6,
        atol=1e-6,
    )


def test_rational_and_nonstationary_kernels_are_strictly_causal():
    times = jnp.linspace(0.0, 1.0, 6)
    rational = CausalKernel.rational(times, power=2.0, scale=0.3)
    nonstationary = CausalKernel.nonstationary_exponential(times, rates=jnp.linspace(0.1, 2.0, 6))
    mask = flat_causal_weights(5) == 0
    assert bool(jnp.all(rational[mask] == 0))
    assert bool(jnp.all(nonstationary[mask] == 0))
