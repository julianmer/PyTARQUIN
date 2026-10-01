####################################################################################################
#                                              io.py                                               #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Flexible input handling for PyTARQUIN. Accepts NumPy arrays (complex FIDs, or           #
#          real/imag stacked spectra) in the time or frequency domain, NIfTI-MRS files and         #
#          objects (read via the "nifti-mrs" package when available, falling back to nibabel),     #
#          batched wrappers such as NIfTI-MRS+, jMRUI text files and LCModel ".RAW" files. Plus    #
#          the helpers to write/read TARQUIN's plain-text ".dpt" format used to feed the binary.   #
#                                                                                                  #
#          Every loader returns the FIDs in one orientation, the one NIfTI-MRS hands out when      #
#          indexed (and jMRUI text stores); TARQUIN's ".dpt" wants them conjugated, which is       #
#          what PyTARQUIN's "conj=True" does.                                                      #
#                                                                                                  #
####################################################################################################

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


#**************************#
#   loaded-signal bundle   #
#**************************#
@dataclass
class Signals:
    """A batch of time-domain FIDs plus optional acquisition metadata."""

    fids: np.ndarray                      # complex, shape (batch, n_points)
    dwell: Optional[float] = None         # seconds
    central_freq: Optional[float] = None  # MHz
    echo_time: Optional[float] = None     # seconds
    reference: Optional[float] = None     # ppm at the centre of the spectrum


def _stack(sigs: List[Signals], what: str) -> Signals:
    """Concatenate a list of Signals along the batch axis. They must share their points
    and acquisition parameters: one batch is written with one set of them."""
    if not sigs:
        raise ValueError(f"No {what} to load.")
    first = sigs[0]
    for s in sigs:
        if s.fids.shape[-1] != first.fids.shape[-1]:
            raise ValueError(f"{what} have mismatched point counts: "
                             f"{first.fids.shape[-1]} vs {s.fids.shape[-1]}.")
        for name in ("dwell", "central_freq", "echo_time", "reference"):
            a, b = getattr(first, name), getattr(s, name)
            if (a is None) != (b is None) or (a is not None and not np.isclose(a, b)):
                raise ValueError(f"{what} differ in {name} ({a} vs {b}); fit them apart.")
    return Signals(fids=np.concatenate([s.fids for s in sigs], axis=0), dwell=first.dwell,
                   central_freq=first.central_freq, echo_time=first.echo_time,
                   reference=first.reference)


#*******************************#
#   numpy shape normalization   #
#*******************************#
def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Return a complex array of shape (batch, n_points) from a variety of layouts."""
    arr = np.asarray(arr)

    if np.iscomplexobj(arr):
        if arr.ndim == 1:
            return arr[np.newaxis, :]
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Unsupported complex array shape: {arr.shape}")

    # real-valued: interpret a length-2 axis as (real, imag)
    if arr.ndim == 1:
        # purely real signal -> imaginary part 0
        return (arr + 0j)[np.newaxis, :]
    if arr.ndim == 2:
        if arr.shape[0] == 2:                       # (2, n_points)
            return (arr[0] + 1j * arr[1])[np.newaxis, :]
        if arr.shape[1] == 2:                       # (n_points, 2)
            return (arr[:, 0] + 1j * arr[:, 1])[np.newaxis, :]
        return (arr + 0j)                           # (batch, n_points), real only
    if arr.ndim == 3 and arr.shape[1] == 2:         # (batch, 2, n_points)
        return arr[:, 0] + 1j * arr[:, 1]
    if arr.ndim == 3 and arr.shape[2] == 2:         # (batch, n_points, 2)
        return arr[..., 0] + 1j * arr[..., 1]
    raise ValueError(f"Unsupported array shape for MRS data: {arr.shape}")


#**********************#
#   nifti-mrs reader   #
#**********************#
def read_nifti_mrs(path: Union[str, Sequence[str]]) -> Signals:
    """Read one or more NIfTI-MRS files into time-domain FIDs.

    "path" may be a single file path or a list/tuple of paths. When a list is given,
    every file is read and the FIDs are stacked along the batch axis (all files must
    share the same number of points; the metadata comes from the first).

    Reading prefers the dedicated "nifti-mrs" package (correct dwell-time unit handling
    and the NIfTI-MRS -> FSL conjugation convention). If it is not installed, it falls
    back to parsing the file directly with "nibabel".
    """
    if isinstance(path, (list, tuple)):
        return _stack([_read_single_nifti_mrs(p) for p in path], "NIfTI-MRS files")
    return _read_single_nifti_mrs(path)


def _is_nifti_mrs(obj) -> bool:
    """Whether obj is a NIfTI-MRS object, or a batched wrapper of them.

    Recognised by what it exposes rather than by its type: FSL-MRS subclasses NIFTI_MRS
    and NIfTI-MRS+ wraps a list of them, and neither is importable from here.
    """
    if hasattr(obj, "list") and hasattr(obj, "dwelltime"):
        return True
    return all(hasattr(obj, a) for a in
               ("dwelltime", "spectrometer_frequency", "__getitem__", "shape"))


def from_nifti_mrs(nmrs) -> Signals:
    """Build Signals from an already-loaded NIfTI-MRS object.

    Accepts a "nifti_mrs.NIFTI_MRS", anything subclassing it (FSL-MRS extends it), or a
    batched wrapper exposing "list()", such as NIfTI-MRS+. Writing the object out and
    reading it back would lose nothing, but it would send data that is already in
    memory on a round trip through disk.
    """
    if hasattr(nmrs, "list"):                  # a batched wrapper, e.g. NIfTI-MRS+
        return _stack([from_nifti_mrs(one) for one in nmrs.list()], "NIfTI-MRS objects")
    return _signals_from_nifti_mrs(nmrs)


def _read_single_nifti_mrs(path: str) -> Signals:
    try:
        from nifti_mrs.nifti_mrs import NIFTI_MRS
    except Exception:
        return _read_nifti_mrs_nibabel(path)
    return _signals_from_nifti_mrs(NIFTI_MRS(path))


def _fids_from_nifti(data: np.ndarray) -> np.ndarray:
    """NIfTI-MRS keeps the spectral axis at index 3; flatten everything else to a batch."""
    if not np.iscomplexobj(data):
        data = data.astype(np.complex64)
    return np.moveaxis(data, 3, -1).reshape(-1, data.shape[3])


def _signals_from_nifti_mrs(nmrs) -> Signals:
    """Pull the FIDs and acquisition parameters out of a NIfTI-MRS object."""
    dwell = float(nmrs.dwelltime) if nmrs.dwelltime is not None else None
    sf = nmrs.spectrometer_frequency
    central_freq = float(sf[0]) if sf is not None and len(sf) > 0 else None
    hdr = getattr(nmrs, "hdr_ext", None)
    echo_time = float(hdr["EchoTime"]) if hdr is not None and "EchoTime" in hdr else None
    # the centre of the spectrum, where nifti-mrs puts it too: SpecFreqChemShift + RxOffset
    shift = getattr(nmrs, "SpecFreqChemShift", None)
    reference = None if shift is None else float(shift) + float(getattr(nmrs, "RxOffset", 0))
    return Signals(fids=_fids_from_nifti(np.asarray(nmrs[:])), dwell=dwell,
                   central_freq=central_freq, echo_time=echo_time, reference=reference)


def _read_nifti_mrs_nibabel(path: str) -> Signals:
    """Fallback NIfTI-MRS reader using nibabel directly.

    The dwell time is "pixdim[4]", read as-is (assumed seconds), and the data are
    conjugated as the "nifti-mrs" package does on indexing, so both readers agree, except
    that a header without SpecFreqChemShift leaves the reference unset here (4.65 then)
    where nifti-mrs takes the nucleus' usual one; install "nifti-mrs" for standard-compliant
    reading.
    """
    import nibabel as nib

    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    if data.ndim < 4:
        raise ValueError(
            f"NIfTI-MRS data is expected to be >=4D (got {data.ndim}D, shape {data.shape})."
        )
    try:
        dwell = float(img.header["pixdim"][4])
    except Exception:
        dwell = None
    meta = _nifti_header_extension(img)
    freq = meta.get("SpectrometerFrequency")
    if isinstance(freq, (list, tuple)):
        freq = freq[0]
    echo = meta.get("EchoTime")
    shift = meta.get("SpecFreqChemShift")
    return Signals(fids=np.conj(_fids_from_nifti(data)), dwell=dwell,
                   central_freq=None if freq is None else float(freq),
                   echo_time=None if echo is None else float(echo),
                   reference=None if shift is None else
                   float(shift) + float(meta.get("RxOffset", 0)))


def _nifti_header_extension(img) -> dict:
    """The NIfTI-MRS JSON header extension (code 44), or {} when there is none."""
    try:
        for ext in img.header.extensions:
            if getattr(ext, "get_code", lambda: None)() in (44, "44"):
                return json.loads(ext.get_content().decode("utf-8", errors="ignore"))
    except Exception:
        return {}
    return {}


#***********************#
#   jmrui text reader   #
#***********************#
def read_jmrui_txt(path: str) -> Tuple[np.ndarray, dict]:
    """Read a single jMRUI ".txt" file -> (complex FID, metadata dict)."""
    meta = {}
    fid_rows = []
    in_data = False
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if not in_data:
                if ":" in s and not s[0].isdigit() and s[0] != "-":
                    key, _, val = s.partition(":")
                    meta[key.strip()] = val.strip()
                if s.lower().startswith("sig(real)") or "fft(real)" in s.lower():
                    in_data = True
                continue
            if s.lower().startswith(("signal", "name")):
                continue
            parts = s.replace(",", " ").split()
            try:
                re_v = float(parts[0])
                im_v = float(parts[1]) if len(parts) > 1 else 0.0
            except (ValueError, IndexError):
                continue
            fid_rows.append(re_v + 1j * im_v)
    return np.asarray(fid_rows, dtype=np.complex128), meta


def jmrui_metadata(meta: dict) -> Tuple[Optional[float], Optional[float]]:
    """Return (dwell_seconds, central_freq_MHz) from jMRUI header fields, if present."""
    def number(key, scale):
        try:
            return float(meta[key]) * scale
        except (KeyError, ValueError):
            return None
    return number("SamplingInterval", 1e-3), number("TransmitterFrequency", 1e-6)  # ms, Hz


def read_jmrui(path: str) -> Signals:
    """Read a jMRUI ".txt" FID file into a Signals object, one row per dataset in it."""
    fid, meta = read_jmrui_txt(path)
    dwell, central = jmrui_metadata(meta)
    datasets = int(float(meta.get("DatasetsInFile", 1)))
    return Signals(fids=fid.reshape(datasets, -1), dwell=dwell, central_freq=central)


#*****************#
#   main loader   #
#*****************#
def load_signals(data, domain: str = "time", dwell: Optional[float] = None,
                 central_freq: Optional[float] = None) -> Signals:
    """Load MRS data from a NumPy array, NIfTI-MRS file or object, jMRUI ".txt" or ".RAW",
    or a list of such files or objects, stacked along the batch axis (they must share their
    acquisition parameters).

    "domain" describes the domain of the *input*: "time" for FIDs, "freq" for spectra in
    np.fft.fft order (unshifted). The returned signals are always time-domain FIDs.
    """
    if domain not in ("time", "freq"):
        raise ValueError("domain must be 'time' or 'freq'")
    if isinstance(data, os.PathLike):
        data = os.fspath(data)

    if isinstance(data, str):
        lower = data.lower()
        if lower.endswith((".nii", ".nii.gz")):
            sig = read_nifti_mrs(data)
        elif lower.endswith(".txt"):
            sig = read_jmrui(data)
        elif lower.endswith((".raw", ".h2o")):
            # .RAW stores the orientation TARQUIN reads, the conjugate of everything else
            sig = Signals(fids=np.conj(from_raw(data))[np.newaxis, :])
        else:
            raise ValueError(f"Unsupported file type: {data}")
    elif isinstance(data, (list, tuple)) and all(
            isinstance(p, (str, os.PathLike)) or _is_nifti_mrs(p) for p in data):
        sig = _stack([load_signals(p) for p in data], "inputs")
    elif _is_nifti_mrs(data):
        # already loaded: keep its dwell time and central frequency rather than making
        # the caller re-supply what the object already knows
        sig = from_nifti_mrs(data)
    else:
        sig = Signals(fids=_normalize_array(data))

    if domain == "freq":
        sig.fids = np.fft.ifft(sig.fids, axis=-1)

    if dwell is not None:
        sig.dwell = dwell
    if central_freq is not None:
        sig.central_freq = central_freq
    return sig


#************************#
#   lcmodel .raw files   #
#************************#
def from_raw(path) -> np.ndarray:
    """Read the complex points of a ".RAW" file, as stored (LCModel's orientation).

    The samples follow the last namelist end ("$END", "&END" or a lone "/"), which may
    close a line of header values, and run on as the header's FMTDAT says - often several
    (real, imag) pairs per line.
    """
    with open(path, "r", errors="ignore") as fh:
        lines = fh.read().splitlines()
    ends = [i for i, line in enumerate(lines)
            if re.search(r"[$&]END\b", line, re.IGNORECASE) or line.strip() == "/"]
    if not ends:
        raise ValueError(f"{path}: no '$END' closes the header, so this is no LCModel .RAW")
    values = [float(v.replace("D", "E").replace("d", "e"))
              for line in lines[ends[-1] + 1:] for v in line.split()]
    if len(values) % 2:
        raise ValueError(f"{path}: {len(values)} values after the header, not (real, imag) pairs")
    values = np.asarray(values, dtype=np.float64)
    return values[0::2] + 1j * values[1::2]


#************************#
#   tarquin .dpt files   #
#************************#
def to_dpt(fid, file_path, dwell: float, central_freq: float, reference: float = 4.65,
           echo_time: float = 0.0):
    """Write one FID as TARQUIN's plain-text format: 'key<TAB>value' lines, then the
    samples, which TARQUIN reads as they stand (no conjugation).

    Args:
        dwell: Seconds between samples.
        central_freq: Transmitter frequency in MHz.
        reference: ppm at the centre of the spectrum.
        echo_time: Seconds; TARQUIN simulates its internal basis at it.
    """
    fid = np.asarray(fid).reshape(-1)
    lines = ["Dangerplot_version\t1.0",
             f"Number_of_points\t{len(fid)}",
             f"Sampling_frequency\t{1.0 / dwell:8.8e}",
             f"Transmitter_frequency\t{central_freq * 1e6:8.8e}",
             "Phi0\t0", "Phi1\t0",
             f"PPM_reference\t{reference:8.8e}",
             f"Echo_time\t{echo_time:8.8e}",
             "Real_FID\tImag_FID\t"]
    lines += [f"{x.real:8.8e} {x.imag:8.8e}" for x in fid]
    # LF everywhere: a Windows host may hand the file to the Linux binary in the container
    with open(file_path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def from_dpt(path) -> Signals:
    """Read a ".dpt" file back: its samples as stored, and its header as Signals metadata."""
    header, fid = {}, []
    with open(path, "r") as fh:
        for line in fh:
            if "\t" in line and not fid:
                key, _, value = line.partition("\t")
                header[key] = value.strip()
                continue
            parts = line.split()
            if len(parts) == 2:
                fid.append(complex(float(parts[0]), float(parts[1])))
    return Signals(fids=np.array(fid)[np.newaxis, :],
                   dwell=1.0 / float(header["Sampling_frequency"]),
                   central_freq=float(header["Transmitter_frequency"]) * 1e-6,
                   echo_time=float(header["Echo_time"]),
                   reference=float(header["PPM_reference"]))
