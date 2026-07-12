"""Causal Volterra kernel constructors for finite-grid branched signatures.

Every constructor returns a cell-average matrix ``W[i, j]`` of shape
``(nodes, nodes - 1)``.  This makes the convention explicit:
``W[i,j]`` weights the increment on ``[t_j, t_{j+1}]`` when the Volterra
object is read at external time ``t_i``.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from .signature import fractional_causal_weights

Array = jax.Array


def _grid(times: Array) -> tuple[Array, Array, Array, Array, Array]:
    times = jnp.asarray(times)
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times must be a one-dimensional grid with at least two nodes")
    left, right = times[:-1][None, :], times[1:][None, :]
    target = times[:, None]
    source = jnp.arange(times.shape[0] - 1)[None, :]
    readout = jnp.arange(times.shape[0])[:, None]
    return left, right, target, source, readout


def _causal(values: Array, source: Array, readout: Array) -> Array:
    return jnp.where(source < readout, values, jnp.zeros_like(values))


def _exponential_average(lower: Array, upper: Array, dt: Array, rate: Array) -> Array:
    """Average of exp(-rate*u) over [lower, upper], stable at rate=0."""
    rate = jnp.asarray(rate, dtype=dt.dtype)
    safe_rate = jnp.where(rate == 0, jnp.ones_like(rate), rate)
    value = (jnp.exp(-safe_rate * lower) - jnp.exp(-safe_rate * upper)) / (safe_rate * dt)
    return jnp.where(rate == 0, jnp.ones_like(value), value)


class CausalKernel:
    """Factory for analytically averaged causal kernels on a fixed grid."""

    @staticmethod
    def fractional(times: Array, *, beta: float | Array) -> Array:
        return fractional_causal_weights(times, beta)

    @staticmethod
    def gamma(
        times: Array,
        *,
        beta: float | Array,
        rate: float | Array = 1.0,
        scale: float | Array = 1.0,
    ) -> Array:
        r"""Cell averages of ``scale exp(-rate*u) u^(beta-1)/Gamma(beta)``.

        The ``rate=0`` limit is routed exactly to the fractional constructor.
        """
        times = jnp.asarray(times)
        beta = jnp.asarray(beta, dtype=times.dtype)
        rate = jnp.asarray(rate, dtype=times.dtype)
        scale = jnp.asarray(scale, dtype=times.dtype)
        if beta.ndim != 0 or bool(beta <= 0):
            raise ValueError("beta must be a positive scalar")
        if rate.ndim != 0 or bool(rate < 0):
            raise ValueError("rate must be a non-negative scalar")
        if bool(rate == 0):
            return scale * fractional_causal_weights(times, beta)

        left, right, target, source, readout = _grid(times)
        lower = jnp.maximum(target - right, 0)
        upper = jnp.maximum(target - left, 0)
        dt = right - left
        integral = scale * rate ** (-beta) * (
            jax.scipy.special.gammainc(beta, rate * upper)
            - jax.scipy.special.gammainc(beta, rate * lower)
        )
        return _causal(integral / dt, source, readout)

    @staticmethod
    def exponential_mixture(times: Array, *, rates: Array, weights: Array) -> Array:
        r"""Cell averages of ``sum_r weights[r] exp(-rates[r]*u)``."""
        left, right, target, source, readout = _grid(times)
        rates = jnp.asarray(rates, dtype=jnp.asarray(times).dtype)
        weights = jnp.asarray(weights, dtype=jnp.asarray(times).dtype)
        if rates.ndim != 1 or weights.shape != rates.shape:
            raise ValueError("rates and weights must be one-dimensional arrays of equal shape")
        if bool(jnp.any(rates < 0)):
            raise ValueError("rates must be non-negative")
        lower = jnp.maximum(target - right, 0)[..., None]
        upper = jnp.maximum(target - left, 0)[..., None]
        avg = _exponential_average(lower, upper, (right - left)[..., None], rates)
        return _causal(jnp.sum(avg * weights, axis=-1), source, readout)

    @staticmethod
    def oscillatory(
        times: Array,
        *,
        decay: float | Array,
        frequency: float | Array,
        scale: float | Array = 1.0,
    ) -> Array:
        r"""Cell averages of ``scale exp(-decay*u) cos(frequency*u)``."""
        left, right, target, source, readout = _grid(times)
        dtype = jnp.asarray(times).dtype
        decay = jnp.asarray(decay, dtype=dtype)
        frequency = jnp.asarray(frequency, dtype=dtype)
        scale = jnp.asarray(scale, dtype=dtype)
        if decay.ndim != 0 or frequency.ndim != 0 or bool(decay < 0):
            raise ValueError("decay must be non-negative and frequency scalar")
        lower = jnp.maximum(target - right, 0)
        upper = jnp.maximum(target - left, 0)
        z = decay - 1j * frequency
        safe_z = jnp.where(z == 0, 1 + 0j, z)
        average = jnp.real((jnp.exp(-safe_z * lower) - jnp.exp(-safe_z * upper)) / (safe_z * (right - left)))
        average = jnp.where(z == 0, jnp.ones_like(average), average)
        return _causal(scale * average, source, readout)

    @staticmethod
    def rational(
        times: Array,
        *,
        power: float | Array,
        scale: float | Array = 1.0,
    ) -> Array:
        r"""Cell averages of the long-memory kernel ``(1 + u/scale)^(-power)``."""
        left, right, target, source, readout = _grid(times)
        power = jnp.asarray(power, dtype=jnp.asarray(times).dtype)
        scale = jnp.asarray(scale, dtype=jnp.asarray(times).dtype)
        if power.ndim != 0 or bool(power <= 0) or bool(scale <= 0):
            raise ValueError("power and scale must be positive scalars")
        lower = jnp.maximum(target - right, 0)
        upper = jnp.maximum(target - left, 0)
        dt = right - left
        non_unit = scale * ((1 + upper / scale) ** (1 - power) - (1 + lower / scale) ** (1 - power)) / (1 - power)
        unit = scale * jnp.log((1 + upper / scale) / (1 + lower / scale))
        integral = jnp.where(power == 1, unit, non_unit)
        return _causal(integral / dt, source, readout)

    @staticmethod
    def nonstationary_exponential(times: Array, *, rates: Array, scales: Array | float = 1.0) -> Array:
        r"""Readout-dependent exponential kernel ``scale_i exp(-rate_i(t_i-s))``."""
        left, right, target, source, readout = _grid(times)
        rates = jnp.asarray(rates, dtype=jnp.asarray(times).dtype)
        scales = jnp.asarray(scales, dtype=jnp.asarray(times).dtype)
        if rates.shape != (jnp.asarray(times).shape[0],):
            raise ValueError("rates must have one entry per readout time")
        if scales.ndim == 0:
            scales = jnp.broadcast_to(scales, rates.shape)
        if scales.shape != rates.shape or bool(jnp.any(rates < 0)):
            raise ValueError("scales must match rates and rates must be non-negative")
        lower = jnp.maximum(target - right, 0)
        upper = jnp.maximum(target - left, 0)
        average = _exponential_average(lower, upper, right - left, rates[:, None])
        return _causal(scales[:, None] * average, source, readout)

    @staticmethod
    def from_callable(times: Array, kernel: Callable[[Array, Array], Array]) -> Array:
        """Midpoint cell averages for a user-provided causal ``kernel(t, s)``."""
        left, right, target, source, readout = _grid(times)
        midpoint = (left + right) / 2
        values = jnp.asarray(kernel(target, midpoint), dtype=jnp.asarray(times).dtype)
        if values.shape != (jnp.asarray(times).shape[0], jnp.asarray(times).shape[0] - 1):
            raise ValueError("kernel(t, s) must broadcast to shape (nodes, nodes - 1)")
        return _causal(values, source, readout)
