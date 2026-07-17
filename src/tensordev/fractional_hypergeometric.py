"""Fractional hypergeometric integral operators.

This module implements the left-sided Marichev--Saigo--Maeda (MSM)
operator on the natural analytic test space of finite power expansions.  The
implementation uses the exact gamma-ratio identities from Jain, Cattani and
Agarwal, *Fractional Hypergeometric Functions*, Symmetry 14 (2022), 714.

For ``f(t) = t**(d-1)`` the MSM integral is evaluated analytically, avoiding
an unsafe numerical evaluation of Appell's ``F3`` outside its defining series
domain.  Saigo, Riemann--Liouville and Erdelyi--Kober operators are included
as reductions of the same formula.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

Array = jax.Array


def _positive_scalar(name: str, value: float) -> None:
    if not value > 0.0:
        raise ValueError(f"{name} must be positive.")


def _gamma_ratio(*numerator: Array, denominator: tuple[Array, ...]) -> Array:
    return jnp.exp(sum(gammaln(x) for x in numerator) - sum(gammaln(x) for x in denominator))


@dataclass(frozen=True)
class MarichevSaigoMaedaLeftIntegral:
    r"""Left-sided MSM fractional integral on monomials.

    The formal kernel is

    ``x**(-a) / Gamma(c) * (x-t)**(c-1) * t**(-a') *
    F3(a,a',b,b';c;1-t/x,1-x/t)``.

    :meth:`power` implements Eq. (29) of the paper for ``t**(d-1)``.
    """

    a: float
    a_prime: float
    b: float
    b_prime: float
    c: float

    def __post_init__(self) -> None:
        _positive_scalar("c", self.c)

    def power(self, d: Array | float, x: Array | float) -> Array:
        """Return ``I[t**(d-1)](x)`` under the Eq. (29) convergence conditions."""
        d_arr = jnp.asarray(d)
        x_arr = jnp.asarray(x, dtype=d_arr.dtype)
        a, ap, b, bp, c = (jnp.asarray(v, dtype=d_arr.dtype) for v in (self.a, self.a_prime, self.b, self.b_prime, self.c))
        coefficient = _gamma_ratio(
            d_arr,
            d_arr + c - a - ap - b,
            d_arr + bp - ap,
            denominator=(d_arr + bp, d_arr + c - a - ap, d_arr + c - ap - b),
        )
        return coefficient * x_arr ** (d_arr + c - a - ap - 1.0)

    def power_series(self, coefficients: Array, exponents: Array, x: Array | float) -> Array:
        """Apply the operator to ``sum coefficients[k] * t**(exponents[k]-1)``."""
        coefficients = jnp.asarray(coefficients)
        exponents = jnp.asarray(exponents, dtype=coefficients.dtype)
        if coefficients.ndim != 1 or exponents.ndim != 1 or coefficients.shape != exponents.shape:
            raise ValueError("coefficients and exponents must be matching one-dimensional arrays.")
        return jnp.sum(coefficients * self.power(exponents, x), axis=0)


@dataclass(frozen=True)
class SaigoLeftIntegral:
    r"""Saigo left fractional integral, a Gauss-hypergeometric MSM reduction."""

    alpha: float
    beta: float
    delta: float

    def __post_init__(self) -> None:
        _positive_scalar("alpha", self.alpha)

    def power(self, d: Array | float, x: Array | float) -> Array:
        r"""Return ``I^{alpha,beta,delta}[t**(d-1)](x)`` exactly.

        The result is
        ``Gamma(d) Gamma(d-beta+delta) /
        (Gamma(d-beta) Gamma(alpha+d+delta)) * x**(d-beta-1)``.
        """
        d_arr = jnp.asarray(d)
        x_arr = jnp.asarray(x, dtype=d_arr.dtype)
        alpha = jnp.asarray(self.alpha, dtype=d_arr.dtype)
        beta = jnp.asarray(self.beta, dtype=d_arr.dtype)
        delta = jnp.asarray(self.delta, dtype=d_arr.dtype)
        coefficient = _gamma_ratio(
            d_arr, d_arr - beta + delta,
            denominator=(d_arr - beta, alpha + d_arr + delta),
        )
        return coefficient * x_arr ** (d_arr - beta - 1.0)


@dataclass(frozen=True)
class RiemannLiouvilleLeftIntegral:
    """Riemann--Liouville left fractional integral on monomials."""

    alpha: float

    def __post_init__(self) -> None:
        _positive_scalar("alpha", self.alpha)

    def power(self, d: Array | float, x: Array | float) -> Array:
        d_arr = jnp.asarray(d)
        x_arr = jnp.asarray(x, dtype=d_arr.dtype)
        alpha = jnp.asarray(self.alpha, dtype=d_arr.dtype)
        return _gamma_ratio(d_arr, denominator=(d_arr + alpha,)) * x_arr ** (d_arr + alpha - 1.0)


@dataclass(frozen=True)
class ErdelyiKoberLeftIntegral:
    """Erdelyi--Kober left fractional integral on monomials."""

    alpha: float
    delta: float

    def __post_init__(self) -> None:
        _positive_scalar("alpha", self.alpha)

    def power(self, d: Array | float, x: Array | float) -> Array:
        d_arr = jnp.asarray(d)
        x_arr = jnp.asarray(x, dtype=d_arr.dtype)
        alpha = jnp.asarray(self.alpha, dtype=d_arr.dtype)
        delta = jnp.asarray(self.delta, dtype=d_arr.dtype)
        return _gamma_ratio(d_arr + delta, denominator=(d_arr + delta + alpha,)) * x_arr ** (d_arr - 1.0)


__all__ = [
    "MarichevSaigoMaedaLeftIntegral",
    "SaigoLeftIntegral",
    "RiemannLiouvilleLeftIntegral",
    "ErdelyiKoberLeftIntegral",
]
