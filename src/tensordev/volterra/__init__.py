from functools import partial

from .coeffs import VolterraCoefficients
from .kernel import ConvolutionKernel, FractionalKernel, FSSKConvolutionKernel, GammaKernel, MittagLefflerKernel, TricomiKernel
from .rough_path import VolterraRoughPath
from .multitime_rough_complete import (
    ChenVolterraReport,
    MultitimeVolterraRoughPath,
    MultitimeVolterraRoughPathResult,
    causal_cell_weights,
)
from .rough_diagnostics import (
    DiscreteVolterraHolderReport,
    NestedGridConvergenceReport,
    compare_terminal_nested_grids,
    discrete_volterra_holder_report,
)
from .signature import VolterraSignature, vsig
from .iteration_quad import quadratic_iteration
from .iteration_fft import fft_iteration
from .iteration_pc import pc_iteration

vsig_fft = partial(vsig, scheme="fft")
vsig_fft.__doc__ = "Alias for :func:`vsig` with ``scheme='fft'`` pre-set."

__all__ = [
    "VolterraCoefficients",
    "ConvolutionKernel",
    "FractionalKernel",
    "FSSKConvolutionKernel",
    "GammaKernel",
    "MittagLefflerKernel",
    "TricomiKernel",
    "VolterraRoughPath",
    "MultitimeVolterraRoughPath",
    "MultitimeVolterraRoughPathResult",
    "ChenVolterraReport",
    "causal_cell_weights",
    "DiscreteVolterraHolderReport",
    "NestedGridConvergenceReport",
    "compare_terminal_nested_grids",
    "discrete_volterra_holder_report",
    "VolterraSignature",
    "vsig",
    "vsig_fft",
    "quadratic_iteration",
    "fft_iteration",
    "pc_iteration",
]