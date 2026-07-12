"""Discrete geometric Volterra rough-path lifts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax

from tensordev.core.universal import DenseElem
from tensordev.volterra.kernel import ConvolutionKernel, FractionalKernel, GammaKernel
from tensordev.volterra.signature import VolterraSignature

Array = jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class VolterraRoughPath:
    """Geometric discrete Volterra rough-path lift.

    Only fractional and Gamma kernels are accepted. ``lift`` emits the
    truncated lift at every grid time under the same piecewise-linear,
    geometric convention as :class:`VolterraSignature`.

    This is not an Itô lift and does not claim the complete continuous
    multitime Chen--Volterra algebra.
    """

    kernel: ConvolutionKernel
    trunc: int = field(metadata={"static": True})

    def __post_init__(self) -> None:
        if not isinstance(self.kernel, (FractionalKernel, GammaKernel)):
            raise TypeError(
                "VolterraRoughPath supports only ConvolutionKernel.fractional(...) "
                "or ConvolutionKernel.gamma(...)."
            )
        if self.trunc < 1:
            raise ValueError("trunc must be at least 1.")

    @property
    def signature(self) -> VolterraSignature:
        return VolterraSignature(kernel=self.kernel, trunc=self.trunc)

    def lift(
        self,
        X: Array,
        *,
        dt: Array | float,
        axis: int = -2,
        increment_input: bool = False,
        dyadic_order: int = 0,
        order: int = 0,
        scheme: Literal["auto", "fft", "quadratic", "adams"] = "auto",
    ) -> DenseElem:
        """Return levels at all readout times, including the tensor unit."""
        return self.signature.vsig(
            X, dt=dt, axis=axis, block_size=1, accumulate=True,
            output_starting_point=True, increment_input=increment_input,
            dyadic_order=dyadic_order, order=order, scheme=scheme,
        )

    def terminal(
        self,
        X: Array,
        *,
        dt: Array | float,
        axis: int = -2,
        increment_input: bool = False,
        dyadic_order: int = 0,
        order: int = 0,
        scheme: Literal["auto", "fft", "quadratic", "adams"] = "auto",
    ) -> DenseElem:
        """Return the terminal element of the same geometric lift."""
        return self.signature.vsig(
            X, dt=dt, axis=axis, increment_input=increment_input,
            dyadic_order=dyadic_order, order=order, scheme=scheme,
        )


__all__ = ["VolterraRoughPath"]
