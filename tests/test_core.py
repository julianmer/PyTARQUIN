####################################################################################################
#                                          test_core.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: What PyTARQUIN hands TARQUIN - its command line and the ".dpt" header - checked         #
#          without a binary: the run is replaced by one that records its arguments.                #
#                                                                                                  #
####################################################################################################

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, _REPO)

from tarquin_wrapper import PyTARQUIN, TARQUINError, binaries, io
from tarquin_wrapper import core

RESULTS_TXT = """\
Signal     Conc (au)       %SD   SD (au)
----------------------------------------
NAA          8.330     3.000    0.2500
Cr           4.100     5.000    0.2100

TARQUIN version 4.3.11
"""


@pytest.fixture
def recorded(monkeypatch):
    """A PyTARQUIN whose runs record their arguments and the .dpt they were given (the
    water reference's too, or None), and write a results file as TARQUIN would."""
    monkeypatch.setattr(binaries, "resolve_executable", lambda **kw: "/fake/tarquin")
    calls = []

    def initiate(self, args):
        dpt = io.from_dpt(args[args.index("--input") + 1])
        water = io.from_dpt(args[args.index("--input_w") + 1]) if "--input_w" in args else None
        calls.append((args, dpt, water))
        with open(args[args.index("--output_txt") + 1], "w") as fh:
            fh.write(RESULTS_TXT)
        return ""

    monkeypatch.setattr(core.PyTARQUIN, "initiate", initiate)
    return calls


#*************#
#   options   #
#*************#
def test_options_reach_the_command_line_as_tarquin_reads_them(recorded, tmp_path):
    """TARQUIN reads only the exact string 'true' as true; a Python True must arrive so,
    and every other value as it prints."""
    (tmp_path / "x.basis").touch()
    tarquin = PyTARQUIN(path2basis=str(tmp_path / "x.basis"), bandwidth=4000, central_freq=123.2,
                        opts={"rescale_lcm_basis": True, "auto_phase": False, "max_iters": 50})
    tarquin(np.ones(64, complex))

    args, _, water = recorded[0]
    assert args[args.index("--rescale_lcm_basis") + 1] == "true"
    assert args[args.index("--auto_phase") + 1] == "false"
    assert args[args.index("--max_iters") + 1] == "50"
    assert args[args.index("--basis_lcm") + 1] == str(tmp_path / "x.basis")
    assert args[args.index("--format") + 1] == "dpt"
    assert "--input_w" not in args and water is None


def test_no_basis_means_the_internal_one(recorded):
    PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03)(np.ones(64, complex))
    args, dpt, _ = recorded[0]
    assert "--basis_lcm" not in args
    assert dpt.echo_time == pytest.approx(0.03)


#*****************#
#   acquisition   #
#*****************#
def test_given_values_win_over_the_datas(recorded):
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    item = gen_nifti_mrs(np.ones((1, 1, 1, 64), np.complex64), 1 / 2000.0, 297.2)
    item.add_hdr_field("EchoTime", 0.02)

    PyTARQUIN(echo_time=0.03, reference=4.7)(item)
    PyTARQUIN()(item)

    (_, given, _), (_, own, _) = recorded
    assert (given.echo_time, given.reference) == pytest.approx((0.03, 4.7))
    assert (own.echo_time, own.reference) == pytest.approx((0.02, 4.65))
    assert (own.dwell, own.central_freq) == pytest.approx((1 / 2000.0, 297.2))


def test_a_missing_basis_is_refused_before_tarquin_runs(recorded, tmp_path):
    """TARQUIN 4.3.11 crashes on a basis file that is not there."""
    with pytest.raises(FileNotFoundError, match="no basis set"):
        PyTARQUIN(path2basis=str(tmp_path / "missing.basis"))
    assert recorded == []


def test_data_without_acquisition_parameters_is_refused(recorded):
    with pytest.raises(ValueError, match="bandwidth"):
        PyTARQUIN(echo_time=0.03)(np.ones(64, complex))


def test_one_water_reference_per_spectrum_is_written(recorded):
    PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03)(
        np.ones((2, 64), complex), 2 * np.ones((2, 64), complex))
    assert len(recorded) == 2
    for _, data, water in recorded:
        assert data.fids[0, 0] == 1 and water.fids[0, 0] == 2


#*************#
#   results   #
#*************#
def test_forward_orders_by_tarquins_signals(recorded):
    tarquin = PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03)
    concs, sds = tarquin(np.ones((2, 64), complex))
    assert tarquin.metabolites == ["NAA", "Cr"]
    assert concs.tolist() == [[8.33, 4.1], [8.33, 4.1]]
    assert sds.tolist() == [[3.0, 5.0], [3.0, 5.0]]


def test_a_parallel_batch_needs_no_main_guard(tmp_path):
    """A script that fits in parallel at its top level, as scripts and notebooks do. A
    process pool would re-run it in every worker (macOS, Windows, Linux from Python 3.14),
    each of them failing to start, and the batch would never finish."""
    script = tmp_path / "unguarded.py"
    script.write_text(textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {_REPO!r})
        import numpy as np
        from tarquin_wrapper import binaries, core
        binaries.resolve_executable = lambda **kw: "/fake/tarquin"
        def initiate(self, args):
            with open(args[args.index("--output_txt") + 1], "w") as fh:
                fh.write({RESULTS_TXT!r})
            return ""
        core.PyTARQUIN.initiate = initiate
        tarquin = core.PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03,
                                 multiprocessing=True)
        print(tarquin(np.ones((4, 64), complex))[0].shape)
        """))
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=60, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "(4, 2)"


def test_a_run_without_results_is_an_error(monkeypatch):
    monkeypatch.setattr(binaries, "resolve_executable", lambda **kw: "/fake/tarquin")
    monkeypatch.setattr(core.PyTARQUIN, "initiate", lambda self, args: "ERROR: no basis")
    tarquin = PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03)
    with pytest.raises(TARQUINError, match="no basis"):
        tarquin(np.ones(64, complex))


def test_the_working_folder_is_removed_unless_kept(recorded, tmp_path):
    PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03)(np.ones(64, complex))
    assert not os.path.exists(os.path.dirname(recorded[0][0][1]))

    keep = tmp_path / "keep"
    PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03,
              save_path=str(keep))(np.ones(64, complex))
    assert sorted(os.listdir(keep)) == ["temp0.dpt", "temp0.txt"]


def test_a_failed_run_does_not_pass_off_an_earlier_one_as_its_own(monkeypatch, tmp_path):
    """A kept save_path holds the last run's results; a run that writes none must fail."""
    monkeypatch.setattr(binaries, "resolve_executable", lambda **kw: "/fake/tarquin")
    writes = iter([True, False])

    def initiate(self, args):
        if next(writes):
            with open(args[args.index("--output_txt") + 1], "w") as fh:
                fh.write(RESULTS_TXT)
        return "ERROR: unrecognised internal basis"

    monkeypatch.setattr(core.PyTARQUIN, "initiate", initiate)
    tarquin = PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03,
                        save_path=str(tmp_path))
    tarquin(np.ones(64, complex))
    with pytest.raises(TARQUINError, match="unrecognised internal basis"):
        tarquin(np.ones(64, complex))


#*************#
#   refused   #
#*************#
@pytest.mark.parametrize("key, keyword", [("echo", "echo_time"), ("fs", "bandwidth"),
                                          ("basis_lcm", "path2basis"), ("output_txt", "none")])
def test_what_pytarquin_writes_itself_is_refused_in_opts(recorded, key, keyword):
    with pytest.raises(ValueError, match=keyword):
        PyTARQUIN(opts={key: "x"})


def test_numpy_booleans_arrive_as_tarquin_reads_them(recorded):
    PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03,
              opts={"auto_phase": np.bool_(True)})(np.ones(64, complex))
    args, _, _ = recorded[0]
    assert args[args.index("--auto_phase") + 1] == "true"


@pytest.mark.parametrize("n, match", [(0, "no samples"), (16385, "at most 16384")])
def test_fids_tarquin_cannot_read_from_a_dpt_are_refused(recorded, n, match):
    with pytest.raises(ValueError, match=match):
        PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03)(np.ones((1, n), complex))
    assert recorded == []


#***********#
#   water   #
#***********#
def test_water_comes_in_the_datas_domain_with_its_own_sampling(recorded):
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    data = np.fft.fft(np.exp(-np.arange(64) / 10.0))
    water = gen_nifti_mrs(np.ones((1, 1, 1, 64), np.complex64), 1 / 2000.0, 123.2)

    PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03, domain="freq",
              conj=False)(data, np.fft.fft(np.ones(64)))
    _, written, water_written = recorded[0]
    assert np.allclose(written.fids[0], np.exp(-np.arange(64) / 10.0))
    assert np.allclose(water_written.fids[0], 1.0)

    PyTARQUIN(bandwidth=4000, central_freq=123.2, echo_time=0.03)(np.ones(64, complex), water)
    _, written, water_written = recorded[1]
    assert written.dwell == pytest.approx(1 / 4000.0)
    assert water_written.dwell == pytest.approx(1 / 2000.0)


#*****************#
#   orientation   #
#*****************#
def test_a_nifti_singlet_reaches_tarquin_where_it_was_made(recorded):
    """TARQUIN reads a .dpt with a forward FFT and ppm = ref - f/ft: a singlet made at
    2.008 ppm in NIfTI-MRS, fitted with the defaults, must land there."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    n, dwell, f0, ref = 2048, 1 / 2500.0, 127.7, 4.65
    t = np.arange(n) * dwell
    fid = np.exp((2j * np.pi * (2.008 - ref) * f0 - 8.0) * t)
    PyTARQUIN(echo_time=0.03)(gen_nifti_mrs(fid.reshape(1, 1, 1, -1).astype(np.complex64),
                                            dwell, f0))

    _, written, _ = recorded[0]
    spectrum = np.fft.fftshift(np.fft.fft(written.fids[0]))
    f = np.fft.fftshift(np.fft.fftfreq(n, written.dwell))
    ppm = written.reference - f / written.central_freq
    assert ppm[np.argmax(np.abs(spectrum))] == pytest.approx(2.008, abs=0.01)
