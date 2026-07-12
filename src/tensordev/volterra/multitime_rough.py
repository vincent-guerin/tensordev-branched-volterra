"""Finite-grid multi-time geometric Volterra rough paths.

This module implements the word-indexed (non-branched) Volterra rough-path
lift on a grid.  Level ``n`` stores ``Z[batch, tau, s, t, word]`` for every
external readout time ``tau`` and interval ``[s,t]``.  The recursion is the
discrete counterpart of linearly iterated Volterra integrals.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .kernel import ConvolutionKernel, FractionalKernel, GammaKernel

Array = jax.Array


def _causal_weights(kernel: ConvolutionKernel, times: Array) -> Array:
    """Exact cell-average weights for supported fractional/Gamma kernels."""
    times = jnp.asarray(times)
    left, right = times[:-1][None, :], times[1:][None, :]
    tau = times[:, None]
    dt = right - left
    source = jnp.arange(times.shape[0] - 1)[None, :]
    readout = jnp.arange(times.shape[0])[:, None]

    if isinstance(kernel, FractionalKernel):
        if kernel.q != 1:
            raise ValueError("MultitimeVolterraRoughPath currently requires one kernel component.")
        beta = kernel.beta.reshape(())
        integral = (
            jnp.maximum(tau - left, 0.0) ** beta
            - jnp.maximum(tau - right, 0.0) ** beta
        ) / jnp.exp(jax.scipy.special.gammaln(beta + 1.0))
    elif isinstance(kernel, GammaKernel):
        beta = kernel.beta.reshape(())
        rate = kernel.rate.reshape(())
        scale = kernel.scale.reshape(())
        u_low = jnp.maximum(tau - right, 0.0)
        u_high = jnp.maximum(tau - left, 0.0)
        integral = scale * rate ** (-beta) * (
            jax.scipy.special.gammainc(beta, rate * u_high)
            - jax.scipy.special.gammainc(beta, rate * u_low)
        )
    else:
        raise TypeError("Only ConvolutionKernel.fractional and ConvolutionKernel.gamma are supported.")

    return jnp.where(source < readout, integral / dt, 0.0)


@dataclass(frozen=True)
class MultitimeVolterraRoughPathResult:
    """Levels of a discrete Volterra rough path.

    ``levels[n-1][b, tau, s, t, w]`` is the coefficient associated with the
    word ``w`` of length ``n`` over interval ``[s,t]``, read at external time
    ``tau``.  Values with ``tau < t`` are identically zero.
    """

    times: Array
    levels: tuple[Array, ...]

    @property
    def terminal(self) -> tuple[Array, ...]:
        """Terminal readout ``tau=t=T`` over the full interval ``[0,T]``."""
        last = self.times.shape[0] - 1
        return tuple(level[:, last, 0, last, :] for level in self.levels)

    def chen_level_one_residual(self) -> Array:
        """Residual of additive Chen--Volterra relation at level one.

        For every ``s <= u <= t <= tau`` this verifies
        ``Z^tau_ts = Z^tau_tu + Z^tau_us`` on the grid.
        """
        z = self.levels[0]
        T = self.times.shape[0]
        residuals = []
        for s in range(T):
            for u in range(s, T):
                for t in range(u, T):
                    residuals.append(z[:, t:, s, t] - z[:, t:, u, t] - z[:, t:, s, u])
        return jnp.concatenate([x.reshape((-1, x.shape[-1])) for x in residuals], axis=0)


@dataclass(frozen=True)
class MultitimeVolterraRoughPath:
    """Word-indexed, three-time geometric Volterra rough-path lift."""

    kernel: ConvolutionKernel
    trunc: int

    def __post_init__(self) -> None:
        if self.trunc < 1:
            raise ValueError("trunc must be at least one.")
        if not isinstance(self.kernel, (FractionalKernel, GammaKernel)):
            raise TypeError("Only fractional and Gamma kernels are supported.")

    def lift(self, X: Array, *, times: Array, increment_input: bool = False) -> MultitimeVolterraRoughPathResult:
        X = jnp.asarray(X)
        if X.ndim == 2:
            X = X[None, ...]
        if X.ndim != 3:
            raise ValueError("X must have shape (batch, T, d) or (batch, T-1, d) for increments.")
        times = jnp.asarray(times, dtype=X.dtype)
        dX = X if increment_input else jnp.diff(X, axis=1)
        T = times.shape[0]
        if dX.shape[1] != T - 1:
            raise ValueError("times must have one more node than the number of increments.")

        weights = _causal_weights(self.kernel, times)
        d = dX.shape[-1]
        batch = dX.shape[0]
        starts = jnp.arange(T)
        tau_indices = jnp.arange(T)[:, None]
        levels: list[Array] = []

        # First level: additive Volterra increments on every [s,t].
        current = jnp.zeros((batch, T, T, T, d), dtype=dX.dtype)
        for end in range(1, T):
            j = end - 1
            increment = weights[:, j][None, :, None, None] * dX[:, None, None, j, :]
            increment = increment * (starts <= j)[None, None, :, None]
            value = current[:, :, :, end - 1, :] + increment
            value = value * (tau_indices >= end)[None, :, :, None]
            current = current.at[:, :, :, end, :].set(value)
        levels.append(current)

        # Higher linear Volterra iterates:
        # Z^{(w,a),tau}_{ts} = sum_{j=s}^{t-1} K(tau,j) Z^{w,j}_{js} dX_j^a.
        previous = current
        for degree in range(2, self.trunc + 1):
            width = d ** degree
            current = jnp.zeros((batch, T, T, T, width), dtype=dX.dtype)
            for end in range(1, T):
                j = end - 1
                prefix = previous[:, j, :, j, :]
                outer = jnp.einsum("bsp,ba->bspa", prefix, dX[:, j, :]).reshape(batch, T, width)
                increment = weights[:, j][None, :, None, None] * outer[:, None, :, :]
                value = current[:, :, :, end - 1, :] + increment
                value = value * (tau_indices >= end)[None, :, :, None]
                current = current.at[:, :, :, end, :].set(value)
            levels.append(current)
            previous = current

        return MultitimeVolterraRoughPathResult(times=times, levels=tuple(levels))


__all__ = ["MultitimeVolterraRoughPath", "MultitimeVolterraRoughPathResult"]
