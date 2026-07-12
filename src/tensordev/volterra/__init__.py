from functools import partial

from .coeffs import VolterraCoefficients
from .kernel import ConvolutionKernel, FractionalKernel, FSSKConvolutionKernel, GammaKernel, MittagLefflerKernel
from .rough_path import VolterraRoughPath
from .multitime_rough import MultitimeVolterraRoughPath, MultitimeVolterraRoughPathResult
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
    "VolterraRoughPath",
    "MultitimeVolterraRoughPath",
    "MultitimeVolterraRoughPathResult",
    "VolterraSignature",
    "vsig",
    "vsig_fft",
    "quadratic_iteration",
    "fft_iteration",
    "pc_iteration",
]