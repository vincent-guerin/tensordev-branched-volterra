"""Mathematically structured finite-grid Volterra rough-path lifts.

This is the word-indexed (linear/tree-ladder) setting.  It keeps all three
times ``(s,t,tau)``, implements the discrete convolution product ``star`` and
checks the convolutional Chen identity at every computed tensor level.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .kernel import ConvolutionKernel, FractionalKernel, GammaKernel

Array = jax.Array


def causal_cell_weights(kernel: ConvolutionKernel, times: Array) -> Array:
    """Cell averages of ``k(tau,s)`` on an arbitrary increasing grid."""
    times = jnp.asarray(times)
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times must be a one-dimensional grid with at least two nodes.")
    dt = jnp.diff(times)
    if kernel.q != 1:
        raise ValueError("The complete word-indexed lift currently requires one scalar kernel component.")

    left, right = times[:-1][None, :], times[1:][None, :]
    tau = times[:, None]
    beta = kernel.beta.reshape(())
    frac_integral = (
        jnp.maximum(tau - left, 0.0) ** beta
        - jnp.maximum(tau - right, 0.0) ** beta
    ) / jnp.exp(jax.scipy.special.gammaln(beta + 1.0))

    if isinstance(kernel, FractionalKernel):
        integral = frac_integral
    elif isinstance(kernel, GammaKernel):
        rate = kernel.rate.reshape(())
        scale = kernel.scale.reshape(())
        safe_rate = jnp.where(rate > 0, rate, jnp.ones_like(rate))
        u_low = jnp.maximum(tau - right, 0.0)
        u_high = jnp.maximum(tau - left, 0.0)
        gamma_integral = scale * safe_rate ** (-beta) * (
            jax.scipy.special.gammainc(beta, safe_rate * u_high)
            - jax.scipy.special.gammainc(beta, safe_rate * u_low)
        )
        integral = jnp.where(rate > 0, gamma_integral, scale * frac_integral)
    else:
        raise TypeError("Only ConvolutionKernel.fractional and ConvolutionKernel.gamma are supported.")

    source = jnp.arange(times.shape[0] - 1)[None, :]
    readout = jnp.arange(times.shape[0])[:, None]
    return jnp.where(source < readout, integral / dt[None, :], 0.0)


def _effective_increments(kernel: ConvolutionKernel, dX: Array) -> Array:
    """Apply the matrix ``A`` in ``K(t,s)=k(t-s)A`` to path increments."""
    A = jnp.asarray(kernel.A[0], dtype=dX.dtype)  # (m,d)
    if dX.shape[-1] != A.shape[-1]:
        raise ValueError(f"Path dimension {dX.shape[-1]} does not match A.shape[-1]={A.shape[-1]}.")
    return jnp.einsum("...d,md->...m", dX, A)


def _seeded_star(
    seed: Array,
    dY: Array,
    weights: Array,
    *,
    start: int,
    end: int,
    suffix_degree: int,
) -> Array:
    r"""Discrete Volterra convolution product.

    ``seed[b,r,p]`` is a past coefficient read at time ``r``.  The result is
    the future suffix iterated over ``[start,end]`` and read at every ``tau``.
    For a suffix of length one this is

    ``sum_{j=start}^{end-1} W[tau,j] seed[j] tensor dY[j]``.
    """
    if suffix_degree < 1:
        raise ValueError("suffix_degree must be positive.")
    batch, T, prefix_width = seed.shape
    alphabet = dY.shape[-1]
    previous_seed = seed
    # Each stage retains its full readout-time axis.  At later stages the
    # coefficient at readout j is used, matching nested Volterra kernels.
    for stage in range(suffix_degree):
        width = prefix_width * alphabet ** (stage + 1)
        trajectory = jnp.zeros((batch, T, T, width), dtype=dY.dtype)
        for stop in range(start + 1, end + 1):
            j = stop - 1
            if stage == 0:
                prefix = previous_seed[:, j, :]
            else:
                prefix = previous[:, j, j, :]
            outer = jnp.einsum("bp,ba->bpa", prefix, dY[:, j, :]).reshape(batch, width)
            increment = weights[:, j][None, :, None] * outer[:, None, :]
            trajectory = trajectory.at[:, :, stop, :].set(
                trajectory[:, :, stop - 1, :] + increment
            )
        previous = trajectory
    return previous[:, :, end, :]


@dataclass(frozen=True)
class ChenVolterraReport:
    degree: int
    max_abs: Array
    rms: Array
    residuals: Array


@dataclass(frozen=True)
class MultitimeVolterraRoughPathResult:
    times: Array
    levels: tuple[Array, ...]
    effective_increments: Array
    weights: Array

    @property
    def terminal(self) -> tuple[Array, ...]:
        last = self.times.shape[0] - 1
        return tuple(level[:, last, 0, last, :] for level in self.levels)

    def coefficient(self, degree: int, *, tau: int, start: int, end: int) -> Array:
        return self.levels[degree - 1][:, tau, start, end, :]

    def chen_report(
        self,
        degree: int,
        *,
        triples: tuple[tuple[int, int, int], ...] | None = None,
    ) -> ChenVolterraReport:
        r"""Check the discrete convolutional Chen identity at one degree.

        ``Z_ts^n = Z_su^n + sum_{m=0}^{n-1} Z_ut^{n-m} star Z_su^m``,
        where ``m=0`` denotes the tensor unit and ``star`` is evaluated by
        :func:`_seeded_star`.
        """
        if not 1 <= degree <= len(self.levels):
            raise ValueError("degree is outside the computed truncation.")
        T = self.times.shape[0]
        if triples is None:
            # This is a proof-oriented diagnostic, not part of the lift.  Its
            # exhaustive form evaluates every s <= u <= t triple and is cubic
            # in the number of grid nodes (with a non-trivial tensor cost).
            if T > 12:
                raise ValueError(
                    "Exhaustive Chen validation is limited to 12 grid nodes; "
                    "pass an explicit tuple of (s, u, t) triples instead."
                )
            triples = tuple(
                (s, u, t)
                for s in range(T)
                for u in range(s, T)
                for t in range(u, T)
            )
        batch = self.effective_increments.shape[0]
        residuals = []
        for s, u, t in triples:
            if not (0 <= s <= u <= t < T):
                raise ValueError("Each Chen triple must satisfy 0 <= s <= u <= t < len(times).")
            lhs = self.levels[degree - 1][:, t:, s, t, :]
            rhs = self.levels[degree - 1][:, t:, s, u, :]
            for past_degree in range(degree):
                if past_degree == 0:
                    seed = jnp.ones((batch, T, 1), dtype=lhs.dtype)
                else:
                    seed = self.levels[past_degree - 1][:, :, s, u, :]
                cross = _seeded_star(
                    seed,
                    self.effective_increments,
                    self.weights,
                    start=u,
                    end=t,
                    suffix_degree=degree - past_degree,
                )[:, t:, :]
                rhs = rhs + cross
            residuals.append((lhs - rhs).reshape((-1, lhs.shape[-1])))
        if not residuals:
            raise ValueError("At least one Chen triple must be supplied.")
        values = jnp.concatenate(residuals, axis=0)
        return ChenVolterraReport(
            degree=degree,
            max_abs=jnp.max(jnp.abs(values)),
            rms=jnp.sqrt(jnp.mean(values ** 2)),
            residuals=values,
        )

    def validate_chen(self, *, atol: float = 1e-5) -> tuple[ChenVolterraReport, ...]:
        reports = tuple(self.chen_report(n) for n in range(1, len(self.levels) + 1))
        # Avoid Python conversion so the reports remain usable with JAX arrays.
        return reports

    def chen_level_one_residual(self) -> Array:
        """Backward-compatible level-one Chen residual array."""
        return self.chen_report(1).residuals


@dataclass(frozen=True)
class MultitimeVolterraRoughPath:
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
            raise ValueError("X must have rank 2 or 3.")
        times = jnp.asarray(times, dtype=X.dtype)
        dX = X if increment_input else jnp.diff(X, axis=1)
        T = times.shape[0]
        if dX.shape[1] != T - 1:
            raise ValueError("times must have one more node than increments.")
        dY = _effective_increments(self.kernel, dX)
        weights = causal_cell_weights(self.kernel, times)
        batch, _, alphabet = dY.shape
        levels: list[Array] = []

        starts = jnp.arange(T)
        tau_indices = jnp.arange(T)[:, None]

        # Dynamic programming form of the same iterated-simplex recursion.
        # Level one is additive on [s,t].
        current = jnp.zeros((batch, T, T, T, alphabet), dtype=dY.dtype)
        for end in range(1, T):
            j = end - 1
            increment = weights[:, j][None, :, None, None] * dY[:, None, None, j, :]
            increment = increment * (starts <= j)[None, None, :, None]
            value = current[:, :, :, end - 1, :] + increment
            value = value * (tau_indices >= end)[None, :, :, None]
            current = current.at[:, :, :, end, :].set(value)
        levels.append(current)

        # At level n, the prefix is read at the new integration time j. This
        # is the nested-kernel rule K(tau,j_n) K(j_n,j_{n-1}) ... .
        previous = current
        for degree in range(2, self.trunc + 1):
            width = alphabet ** degree
            current = jnp.zeros((batch, T, T, T, width), dtype=dY.dtype)
            for end in range(1, T):
                j = end - 1
                prefix = previous[:, j, :, j, :]
                outer = jnp.einsum("bsp,ba->bspa", prefix, dY[:, j, :]).reshape(
                    batch, T, width
                )
                increment = weights[:, j][None, :, None, None] * outer[:, None, :, :]
                value = current[:, :, :, end - 1, :] + increment
                value = value * (tau_indices >= end)[None, :, :, None]
                current = current.at[:, :, :, end, :].set(value)
            levels.append(current)
            previous = current
        return MultitimeVolterraRoughPathResult(
            times=times,
            levels=tuple(levels),
            effective_increments=dY,
            weights=weights,
        )


__all__ = [
    "ChenVolterraReport",
    "MultitimeVolterraRoughPath",
    "MultitimeVolterraRoughPathResult",
    "causal_cell_weights",
]
