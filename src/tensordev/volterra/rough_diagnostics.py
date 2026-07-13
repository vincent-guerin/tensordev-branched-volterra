"""Discrete analytic diagnostics for multi-time Volterra rough paths."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .multitime_rough_complete import MultitimeVolterraRoughPathResult

Array = jax.Array


@dataclass(frozen=True)
class DiscreteVolterraHolderReport:
    degree: int
    alpha: float
    gamma: float
    rho: float
    seminorm: Array
    samples: int


def discrete_volterra_holder_report(
    result: MultitimeVolterraRoughPathResult,
    *,
    degree: int,
    alpha: float,
    gamma: float,
) -> DiscreteVolterraHolderReport:
    r"""Evaluate the first ``V^(alpha,gamma)``-type bound on the grid.

    For ``rho=alpha-gamma>0`` and tree/word degree ``n``, the denominator is

    ``(|tau-t|^-gamma |t-s|^(n*rho+gamma)) min |tau-s|^(n*rho)``.

    Finiteness on one grid is a diagnostic, not a proof of a uniform
    continuous Hölder estimate; convergence requires a bounded sequence of
    reports under mesh refinement.
    """
    if not 1 <= degree <= len(result.levels):
        raise ValueError("degree is outside the lift truncation.")
    if not (0 <= gamma < alpha <= 1):
        raise ValueError("require 0 <= gamma < alpha <= 1.")
    rho = alpha - gamma
    times = result.times
    z = result.levels[degree - 1]
    ratios = []
    T = times.shape[0]
    for s in range(T - 1):
        for t in range(s + 1, T):
            for tau in range(t, T):
                ts = times[t] - times[s]
                tau_s = times[tau] - times[s]
                tau_t = times[tau] - times[t]
                singular = jnp.where(
                    tau_t > 0,
                    tau_t ** (-gamma) * ts ** (degree * rho + gamma),
                    jnp.inf,
                )
                denominator = jnp.minimum(singular, tau_s ** (degree * rho))
                size = jnp.max(jnp.abs(z[:, tau, s, t, :]))
                ratios.append(size / denominator)
    values = jnp.stack(ratios)
    return DiscreteVolterraHolderReport(
        degree=degree,
        alpha=alpha,
        gamma=gamma,
        rho=rho,
        seminorm=jnp.max(values),
        samples=len(ratios),
    )


@dataclass(frozen=True)
class NestedGridConvergenceReport:
    degree: int
    max_abs: Array
    relative_l2: Array


def compare_terminal_nested_grids(
    coarse: MultitimeVolterraRoughPathResult,
    fine: MultitimeVolterraRoughPathResult,
    *,
    degree: int,
) -> NestedGridConvergenceReport:
    """Compare full-interval terminal coefficients on nested grids."""
    c = coarse.terminal[degree - 1]
    f = fine.terminal[degree - 1]
    if c.shape != f.shape:
        raise ValueError("coarse and fine terminal coefficient shapes differ.")
    error = c - f
    return NestedGridConvergenceReport(
        degree=degree,
        max_abs=jnp.max(jnp.abs(error)),
        relative_l2=jnp.linalg.norm(error) / jnp.maximum(jnp.linalg.norm(f), 1e-12),
    )


__all__ = [
    "DiscreteVolterraHolderReport",
    "NestedGridConvergenceReport",
    "compare_terminal_nested_grids",
    "discrete_volterra_holder_report",
]
