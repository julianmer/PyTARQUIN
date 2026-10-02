####################################################################################################
#                                             core.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Python wrapper for the TARQUIN spectral fitting tool. Handles binary resolution,        #
#          flexible input (NumPy / NIfTI-MRS / NIfTI-MRS+ / jMRUI / .RAW, time or frequency        #
#          domain), TARQUIN's ".dpt" input and command line, and parsing of its results.           #
#                                                                                                  #
# TARQUIN itself is a separate GPL-3.0 program (see LICENSE.tarquin and NOTICE).                   #
#                                                                                                  #
####################################################################################################

import os
import shutil
import subprocess
import tempfile
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional

import numpy as np

from . import binaries, container, io
from .results import Report, read_results

# What PyTARQUIN writes on the command line itself, and the keyword that sets it instead
_OWNED = {"input": None, "input_w": None, "format": None, "output_txt": None,
          "fs": "bandwidth", "ft": "central_freq", "echo": "echo_time", "ref": "reference",
          "basis_lcm": "path2basis"}

# TARQUIN's .dpt reader takes at most this many points per FID, and reads more as averages
_MAX_POINTS = 16384


#**************************************************************************************************#
#                                     Class TARQUINError                                           #
#**************************************************************************************************#
#                                                                                                  #
# Raised when a TARQUIN run does not produce its results file.                                     #
#                                                                                                  #
#**************************************************************************************************#
class TARQUINError(RuntimeError):
    """Raised when a TARQUIN run does not produce its results file.

    TARQUIN's exit status says little (it ends "--help" with 255, and a fit it gives up on
    may still exit 0), so failure is detected by the absence of the results file and
    reported with the end of whatever TARQUIN printed.
    """


#**************************************************************************************************#
#                                        Class PyTARQUIN                                           #
#**************************************************************************************************#
#                                                                                                  #
# Wrapper around the TARQUIN executable for batch MRS fitting.                                     #
#                                                                                                  #
#**************************************************************************************************#
class PyTARQUIN:
    """Wrapper around the TARQUIN executable for batch MRS fitting.

    Parameters
    ----------
    path2basis : str, optional
        Path to an LCModel ".basis" file. TARQUIN does not resample it: its sampling
        frequency must match the data's. By default TARQUIN rescales an LCModel basis to
        its sIns or Scyllo signal and stops if it has neither; opts={"rescale_lcm_basis":
        False} turns that off. None uses TARQUIN's internal basis ("1h_brain_glth" unless
        opts={"int_basis": ...} picks another), simulated at the data's echo time. Other
        basis formats go through opts, e.g. {"basis_csv": "/path/to/folder"}.
    opts : dict, optional
        TARQUIN's command line options by name, e.g. {"pul_seq": "slaser",
        "max_iters": 50, "auto_phase": False}; booleans become TARQUIN's 'true'/'false'.
        Anything not given is TARQUIN's own default - PRESS, and a start point it picks
        itself, skipping about the first 10 ms of the FID to keep broad baseline out. What
        PyTARQUIN writes itself (input, format, fs, ft, echo, ref, basis_lcm, ...) is set
        through its keywords instead.
    multiprocessing : bool
        Fit the batch in parallel, one TARQUIN process per CPU core at a time.
    conj : bool
        Conjugate the FIDs before fitting. Every loader hands out the orientation that
        NIfTI-MRS gives when indexed (and jMRUI text stores); TARQUIN wants the conjugate.
    save_path : str, optional
        Directory to keep the ".dpt" inputs and ".txt" results (otherwise a temporary one
        is used and removed).
    path2exec : str, optional
        Explicit path to a TARQUIN executable. If omitted it is resolved automatically
        (cache -> release download -> container); the TARQUIN_EXEC environment variable
        is equivalent to passing it.
    domain : {"time", "freq"}
        Domain of the input data (and of the water reference) passed at fit time:
        "time" for FIDs (the default), "freq" for spectra in np.fft.fft order (unshifted).
    bandwidth, central_freq : float, optional
        Hz and MHz; override what the data carries, and are required for data that
        carries neither (NumPy arrays, ".RAW").
    echo_time : float, optional
        Seconds; overrides the data's. TARQUIN simulates its internal basis at it.
    reference : float, optional
        ppm at the centre of the spectrum; overrides the data's, 4.65 when neither says.
    allow_download, allow_docker : bool
        Permit automatic binary download / container use during resolution. With a
        container (docker or podman), TARQUIN runs from an image and only sees the working
        directory and your home directory - keep the basis set and any save_path under one
        of those.
    timeout : float
        Seconds to allow a single TARQUIN run before killing it.
    """

    def __init__(self, path2basis=None, opts=None, multiprocessing=False, conj=True,
                 save_path="", path2exec=None, domain="time", bandwidth=None,
                 central_freq=None, echo_time=None, reference=None, allow_download=True,
                 allow_docker=True, timeout=900):

        if domain not in ("time", "freq"):
            raise ValueError("domain must be 'time' or 'freq'")
        # TARQUIN 4.3.11 crashes on a basis file that is not there, rather than saying so
        if path2basis is not None and not os.path.isfile(path2basis):
            raise FileNotFoundError(f"no basis set at {path2basis}")
        owned = sorted(set(opts or {}) & set(_OWNED))
        if owned:
            raise ValueError(f"PyTARQUIN sets {owned} itself; use its keywords "
                             f"{[_OWNED[k] for k in owned if _OWNED[k]] or 'none'} instead")
        # physical paths throughout, which is how a container sees them (see container.py)
        self.path2basis = None if path2basis is None else os.path.realpath(path2basis)
        self.opts = dict(opts or {})
        self.multiprocessing = multiprocessing
        self.conj = conj
        self.save_path = save_path
        self.domain = domain
        self.bandwidth = bandwidth
        self.central_freq = central_freq
        self.echo_time = echo_time
        self.reference = reference
        self.timeout = timeout
        self.metabolites: List[str] = []   # TARQUIN's signal order, known after a fit

        self.path2exec = binaries.resolve_executable(
            path2exec=path2exec, allow_download=allow_download, allow_docker=allow_docker,
        )
        self._containerised = binaries.is_container_shim(self.path2exec)
        if self.path2basis is not None:
            self._check_visible(self.path2basis, "the basis set")

    #*************#
    #   helpers   #
    #*************#
    def _check_visible(self, path, what):
        if self._containerised and not container.can_see(path):
            raise ValueError(
                f"TARQUIN runs from a container, which only sees the working directory and "
                f"your home directory (on Windows: their drives); {what} is outside both: "
                f"{os.path.abspath(path)}"
            )

    def _workdir(self) -> str:
        """Directory for TARQUIN's input and output files. An empty save_path means a fresh
        temporary folder per call - under the cache in the home directory when TARQUIN
        runs from a container, which sees no system temporary folder."""
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.realpath(self.save_path)
            self._check_visible(path, "save_path")
            return path
        parent = None
        if self._containerised:
            parent = binaries._cache_root() / "runs"
            parent.mkdir(parents=True, exist_ok=True)
            self._check_visible(parent, "the run folder under TARQUIN_CACHE_DIR")
        return os.path.realpath(tempfile.mkdtemp(prefix="tarquin_", dir=parent))

    def _acquisition(self, signals: io.Signals) -> Dict[str, float]:
        """What the ".dpt" header states: given values first, then the data's own."""
        dwell = 1.0 / self.bandwidth if self.bandwidth else signals.dwell
        central = self.central_freq or signals.central_freq
        if not dwell or not central:
            raise ValueError("the data carries no bandwidth or central frequency; pass "
                             "bandwidth= (Hz) and central_freq= (MHz)")
        echo = self.echo_time if self.echo_time is not None else signals.echo_time
        if self.path2basis is None and "basis_csv" not in self.opts \
                and "basis_xml" not in self.opts and echo is None:
            raise ValueError("TARQUIN simulates its internal basis at the data's echo time, "
                             "which the data does not state; pass echo_time= or a basis")
        reference = self.reference if self.reference is not None else signals.reference
        return {"dwell": dwell, "central_freq": central, "echo_time": echo or 0.0,
                "reference": 4.65 if reference is None else reference}

    #**********************#
    #   forward function   #
    #**********************#
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    #*********************#
    #   input to output   #
    #*********************#
    def forward(self, x, x_ref=None):
        """Fit and return (concentrations, %SD), each of shape (batch, signals) in the
        order of "self.metabolites", TARQUIN's own: the single signals, then the combined
        ones (TNAA, TCr, ...).

        See "report" for everything else TARQUIN worked out along the way.
        """
        reports = self.report(x, x_ref)
        concs = [dict(zip(r.metabolites, r.concentrations)) for r in reports]
        sds = [dict(zip(r.metabolites, r.percent_sd)) for r in reports]
        return (np.array([[c[m] for m in self.metabolites] for c in concs]),
                np.array([[s[m] for m in self.metabolites] for s in sds]))

    #*****************#
    #   full report   #
    #*****************#
    def report(self, x, x_ref=None) -> List[Report]:
        """Fit and return everything TARQUIN reported, one Report per spectrum.

        Args:
            x: FIDs (or spectra, with domain="freq") in any form "io.load_signals" reads.
            x_ref: Unsuppressed water reference, one per spectrum, enabling TARQUIN's
                water scaling: the concentrations are then mM rather than arbitrary units.
        """
        signals = io.load_signals(x, domain=self.domain)
        fids = np.conjugate(signals.fids) if self.conj else signals.fids
        _check_points(fids)
        acquisition = self._acquisition(signals)

        water, water_acquisition = None, None
        if x_ref is not None:
            reference = io.load_signals(x_ref, domain=self.domain)
            water = np.conjugate(reference.fids) if self.conj else reference.fids
            _check_points(water)
            # one reference per spectrum, which is how they are acquired; scaling a batch of
            # separate acquisitions by one subject's water would be wrong without being
            # visibly wrong
            if water.shape[0] != fids.shape[0]:
                raise ValueError(f"{water.shape[0]} water references for {fids.shape[0]} "
                                 f"spectra; supply one per spectrum.")
            # the reference's own sampling where it states it, else the data's
            water_acquisition = dict(acquisition, **{
                key: value for key, value in (("dwell", reference.dwell),
                                              ("central_freq", reference.central_freq))
                if value})

        path = self._workdir()
        tasks = [(fid, None if water is None else water[i], acquisition, i, path,
                  water_acquisition) for i, fid in enumerate(fids)]
        try:
            if self.multiprocessing:
                # threads, each waiting on its own TARQUIN process: a process pool would
                # re-run the caller's script in every worker (macOS, Windows, Linux from
                # Python 3.14), which without a "__main__" guard never finishes
                with ThreadPool() as pool:
                    reports = pool.starmap(self.tarquin_forward, tasks)
            else:
                reports = [self.tarquin_forward(*task) for task in tasks]
        finally:
            if not self.save_path:
                shutil.rmtree(path, ignore_errors=True)

        self.metabolites = list(reports[0].metabolites)
        return reports

    #*************#
    #   one fit   #
    #*************#
    def tarquin_forward(self, fid, h2o, acquisition: Dict[str, float], idx: int,
                        path: str, h2o_acquisition: Optional[Dict[str, float]] = None) -> Report:
        """Fit one FID (already in TARQUIN's orientation) and parse its results."""
        stem = os.path.join(path, f"temp{idx}")
        # a kept save_path may hold an earlier run's results, which a failed run must not
        # pass off as its own
        for old in (f"{stem}.txt", f"{stem}_w.dpt"):
            if os.path.exists(old):
                os.remove(old)
        io.to_dpt(fid, f"{stem}.dpt", **acquisition)
        args = ["--input", f"{stem}.dpt", "--format", "dpt", "--output_txt", f"{stem}.txt"]
        if h2o is not None:
            io.to_dpt(h2o, f"{stem}_w.dpt", **(h2o_acquisition or acquisition))
            args += ["--input_w", f"{stem}_w.dpt"]
        if self.path2basis is not None:
            args += ["--basis_lcm", self.path2basis]
        for key, value in self.opts.items():
            args += [f"--{key}", _option(value)]

        output = self.initiate(args)
        if not os.path.exists(f"{stem}.txt"):
            detail = output.strip()[-2000:] or "(TARQUIN produced no output)"
            raise TARQUINError(f"TARQUIN wrote no results for {os.path.basename(stem)}.dpt.\n"
                               f"TARQUIN said:\n{detail}")
        return read_results(f"{stem}.txt")

    #**************#
    #   initiate   #
    #**************#
    def initiate(self, args: List[str]) -> str:
        """Run TARQUIN with these arguments. Returns whatever it printed."""
        with subprocess.Popen([self.path2exec, *args], stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            try:
                out, _ = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                # terminate first: the container launcher passes it on to the container,
                # which a kill of the launcher would leave running
                proc.terminate()
                try:
                    proc.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise TARQUINError(f"TARQUIN did not finish within {self.timeout:g}s.") from None
        return out.decode("utf-8", errors="ignore")


#************#
#   option   #
#************#
def _option(value: Any) -> str:
    """An option value as TARQUIN reads it: only the exact string 'true' is true."""
    return str(value).lower() if isinstance(value, (bool, np.bool_)) else str(value)


#******************#
#   check points   #
#******************#
def _check_points(fids: np.ndarray):
    """FIDs TARQUIN can read from a .dpt: some samples, and no more than it takes."""
    if fids.shape[-1] == 0:
        raise ValueError("the data holds no samples")
    if fids.shape[-1] > _MAX_POINTS:
        raise ValueError(f"{fids.shape[-1]} points; TARQUIN reads at most {_MAX_POINTS} per "
                         f"FID from a .dpt (it takes more for averages), so truncate the FID")
