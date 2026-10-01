"""PyTARQUIN - a lightweight Python wrapper for the TARQUIN MRS fitting tool.

TARQUIN itself is a separate GPL-3.0 program by Greg Reynolds and Martin Wilson (see
LICENSE.tarquin and NOTICE). This package only wraps it and contains none of its code.
"""

from ._version import __version__
from .core import PyTARQUIN, TARQUINError
from .results import Report, read_results
from .binaries import resolve_executable, verify_executable
from .io import from_dpt, from_nifti_mrs, load_signals, to_dpt
from . import binaries, container, io, results

__all__ = [
    "PyTARQUIN",
    "TARQUINError",
    "Report",
    "read_results",
    "resolve_executable",
    "verify_executable",
    "from_dpt",
    "from_nifti_mrs",
    "load_signals",
    "to_dpt",
    "binaries",
    "container",
    "io",
    "results",
]
