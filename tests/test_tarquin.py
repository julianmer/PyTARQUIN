####################################################################################################
#                                         test_tarquin.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: End-to-end test of the PyTARQUIN wrapper by fitting MRS data from the ISMRM 2016        #
#          fitting challenge (jMRUI datasets + .basis) and comparing against the ground truth.     #
#                                                                                                  #
#          The same run doubles as the acceptance test for every TARQUIN binary CI builds: point   #
#          TARQUIN_EXEC at a binary (or at the container launcher) and the fit must reproduce the  #
#          reference error of the official TARQUIN 4.3.11 release. A miscompiled binary either     #
#          writes no results or lands far from it, so this - not a return code - is what proves    #
#          a build works.                                                                          #
#                                                                                                  #
####################################################################################################

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tarquin_wrapper import PyTARQUIN, TARQUINError, binaries, read_results

_REPO = Path(__file__).resolve().parent.parent
_CHALLENGE = _REPO / "example_data" / "2016_fitting_challenge"
_TRUTH = _REPO / "example_data" / "2016_fitting_challenge_gts"
_BASIS = _CHALLENGE / "basisset_LCModel" / "press3T_30ms.BASIS"
_JMRUI = _CHALLENGE / "datasets_JMRUI"
_RAW = _CHALLENGE / "datasets_LCModel"

# Mean absolute concentration error (mM, water-scaled) over the first five datasets, from
# the official TARQUIN 4.3.11 macOS release.
REFERENCE_MAE = 1.859
TOLERANCE = 0.02

needs_challenge = pytest.mark.skipif(
    not _BASIS.is_file() or not any(_TRUTH.glob("*.xlsx")),
    reason="challenge data not checked out (git submodule update --init)")


#******************#
#   ground truth   #
#******************#
def load_truth(path: Path) -> dict:
    """Load ISMRM-2016 ground-truth concentrations -> {metabolite: concentration}."""
    import pandas as pd
    truth = {}
    df = pd.read_excel(str(path), header=17)
    for met, val in zip(df["Metabolites"], df["concentration"]):
        if not isinstance(met, str):
            break
        truth[met] = val
    if "MMBL" in truth:
        truth["Mac"] = truth.pop("MMBL")
    return truth


#***************#
#   challenge   #
#***************#
def challenge_mae(test_size: int = 5) -> float:
    """Fit the first "test_size" challenge datasets and return the MAE against the truth,
    over the basis' own signals (TARQUIN's combined ones, TNAA, TCr, Glx, have no truth)."""
    by_number = {int(re.search(r"dataset(\d+)", p.stem).group(1)): p
                 for p in _TRUTH.iterdir() if p.suffix == ".xlsx"}
    numbers = sorted(by_number)[:test_size]

    tarquin = PyTARQUIN(path2basis=str(_BASIS))
    concs, _ = tarquin([str(_JMRUI / f"dataset{n}_WS.txt") for n in numbers],
                       [str(_JMRUI / f"dataset{n}_nWS.txt") for n in numbers])

    truths = [load_truth(by_number[n]) for n in numbers]
    names = [m for m in tarquin.metabolites if m in truths[0]]
    fitted = concs[:, [tarquin.metabolites.index(m) for m in names]]
    expected = np.array([[t[m] for m in names] for t in truths])
    return float(np.abs(fitted - expected).mean())


# an uninitialised submodule is an empty directory, so look for its content
@needs_challenge
def test_challenge_fit_matches_reference():
    mae = challenge_mae()
    assert abs(mae - REFERENCE_MAE) < TOLERANCE, f"MAE {mae:.4f} vs reference {REFERENCE_MAE}"


#*****************#
#   orientation   #
#*****************#
@needs_challenge
def test_the_wrapper_fits_what_tarquin_reads_itself():
    """The challenge's LCModel .RAW, once read by TARQUIN itself and once by the wrapper
    (loaded, conjugated, written as .dpt), must give the very same fit."""
    raw, h2o = str(_RAW / "dataset1.RAW"), str(_RAW / "dataset1.H2O")
    tarquin = PyTARQUIN(path2basis=str(_BASIS), bandwidth=4000, central_freq=123.261703)
    ours = tarquin.report(raw, h2o)[0]

    # the container sees the home directory, where the wrapper keeps its own runs too
    with tempfile.TemporaryDirectory(dir=binaries._cache_root()) as folder:
        out = os.path.join(folder, "native.txt")
        subprocess.run([tarquin.path2exec, "--input", raw, "--input_w", h2o, "--format", "lcm",
                        "--fs", "4000", "--ft", "123261703", "--echo", "0",
                        "--basis_lcm", str(_BASIS), "--output_txt", out],
                       capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
        native = read_results(out)

    assert ours.metabolites == native.metabolites
    assert ours.concentrations == native.concentrations


@needs_challenge
def test_naa_lands_where_naa_is():
    """Asserted on physiology: a mirrored spectrum - the FID handed over in the wrong
    orientation - still fits, but not with these ratios."""
    tarquin = PyTARQUIN(path2basis=str(_BASIS))
    report = tarquin.report(str(_JMRUI / "dataset1_WS.txt"))[0]
    conc = dict(zip(report.metabolites, report.concentrations))
    assert report.units == "au"
    assert 1.0 < conc["NAA"] / (conc["Cr"] + conc["PCr"]) < 2.0


#*************#
#   options   #
#*************#
@needs_challenge
def test_the_internal_basis_is_simulated_at_the_echo_time():
    """jMRUI text carries no echo time, so the internal basis needs one given."""
    with pytest.raises(ValueError, match="echo time"):
        PyTARQUIN().report(str(_JMRUI / "dataset1_WS.txt"))

    report = PyTARQUIN(echo_time=0.03).report(str(_JMRUI / "dataset1_WS.txt"))[0]
    assert "NAA" in report.metabolites and "Mac" not in report.metabolites


@needs_challenge
def test_a_batch_fits_alike_in_parallel_and_in_turn():
    data = [str(_JMRUI / f"dataset{n}_WS.txt") for n in (1, 2)]
    serial = PyTARQUIN(path2basis=str(_BASIS))(data)
    parallel = PyTARQUIN(path2basis=str(_BASIS), multiprocessing=True)(data)
    assert serial[0].shape == (2, 24)
    assert (serial[0] == parallel[0]).all() and (serial[1] == parallel[1]).all()


#*************#
#   refused   #
#*************#
@needs_challenge
def test_a_fit_without_results_says_what_tarquin_said():
    tarquin = PyTARQUIN(echo_time=0.03, opts={"int_basis": "nonsense"})
    with pytest.raises(TARQUINError, match="unrecognised internal basis 'nonsense'"):
        tarquin.report(str(_JMRUI / "dataset1_WS.txt"))


@needs_challenge
def test_one_water_reference_per_spectrum():
    data = [str(_JMRUI / f"dataset{n}_WS.txt") for n in (1, 2)]
    with pytest.raises(ValueError, match="one per spectrum"):
        PyTARQUIN(path2basis=str(_BASIS)).report(data, str(_JMRUI / "dataset1_nWS.txt"))


if __name__ == "__main__":
    print("binary:", binaries.resolve_executable())
    print("MAE:", challenge_mae())
