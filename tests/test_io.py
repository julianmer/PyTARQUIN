####################################################################################################
#                                           test_io.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Input handling: every loader hands out one orientation, ".RAW" headers as LCModel       #
#          writes them, and ".dpt" files that put a signal where TARQUIN reads it.                 #
#                                                                                                  #
####################################################################################################

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tarquin_wrapper import io

_CHALLENGE = Path(__file__).resolve().parent.parent / "example_data" / "2016_fitting_challenge"
needs_challenge = pytest.mark.skipif(
    not (_CHALLENGE / "datasets_LCModel" / "dataset1.RAW").is_file(),
    reason="challenge data not checked out (git submodule update --init)")


#*********#
#   dpt   #
#*********#
def test_a_dpt_reads_back_as_written(tmp_path):
    fid = np.exp(-np.arange(64) / 20.0) * np.exp(2j * np.pi * np.arange(64) / 7.0)
    io.to_dpt(fid, tmp_path / "x.dpt", dwell=1 / 2000.0, central_freq=123.2, reference=4.7,
              echo_time=0.035)
    back = io.from_dpt(tmp_path / "x.dpt")
    assert np.allclose(back.fids[0], fid, rtol=1e-8)
    assert (back.dwell, back.central_freq, back.reference, back.echo_time) == \
        pytest.approx((1 / 2000.0, 123.2, 4.7, 0.035))


#***************#
#   nifti-mrs   #
#***************#
def test_nifti_mrs_hands_over_its_header(tmp_path):
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    fid = np.exp(-np.arange(128) / 30.0).astype(np.complex64)
    item = gen_nifti_mrs(fid.reshape(1, 1, 1, -1), 1 / 4000.0, 123.2)
    item.add_hdr_field("EchoTime", 0.03)

    got = io.load_signals(item)
    assert got.fids.shape == (1, 128)
    assert (got.dwell, got.central_freq, got.echo_time, got.reference) == \
        pytest.approx((1 / 4000.0, 123.2, 0.03, 4.65))


def test_the_nibabel_fallback_agrees_with_nifti_mrs(tmp_path):
    """Without the nifti-mrs package the file is read with nibabel, which must hand out
    the same orientation and header as the package does."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    fid = (np.exp(-np.arange(128) / 30.0) * np.exp(0.3j * np.arange(128))).astype(np.complex64)
    item = gen_nifti_mrs(fid.reshape(1, 1, 1, -1), 1 / 4000.0, 123.2)
    item.add_hdr_field("EchoTime", 0.03)
    item.add_hdr_field("SpecFreqChemShift", 4.7)
    item.add_hdr_field("RxOffset", -1.65)
    item.save(str(tmp_path / "x.nii.gz"))

    package = io.read_nifti_mrs(str(tmp_path / "x.nii.gz"))
    fallback = io._read_nifti_mrs_nibabel(str(tmp_path / "x.nii.gz"))
    assert np.allclose(fallback.fids, package.fids)
    assert (fallback.dwell, fallback.central_freq, fallback.echo_time, fallback.reference) == \
        pytest.approx((package.dwell, package.central_freq, package.echo_time, package.reference))
    assert package.reference == pytest.approx(4.7 - 1.65)   # the centre, as nifti-mrs has it


def test_a_batched_wrapper_is_stacked():
    """NIfTI-MRS+ (or anything exposing list() and a dwell time) is a batch."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    one = gen_nifti_mrs(np.ones((1, 1, 1, 64), np.complex64), 1 / 4000.0, 123.2)
    two = gen_nifti_mrs(2 * np.ones((1, 1, 1, 64), np.complex64), 1 / 4000.0, 123.2)

    class Batch:
        dwelltime = 1 / 4000.0

        def list(self):
            return [one, two]

    got = io.load_signals(Batch())
    assert got.fids.shape == (2, 64)
    assert np.allclose(got.fids[:, 0], [1, 2])


#*************#
#   lcmodel   #
#*************#
def test_a_raw_header_may_close_on_a_value_line_and_hold_several_namelists(tmp_path):
    """As LCModel and the challenge write them: $SEQPAR before $NMID, '$END' after the
    last header value, and four (real, imag) pairs per line."""
    path = tmp_path / "x.RAW"
    path.write_text(" $SEQPAR\n ECHOT = 30.\n $END\n"
                    " $NMID ID='x', FMTDAT='(8E13.5)'\n TRAMP=1, VOLUME=1 $END\n"
                    "  1.0E+00  2.0E+00  3.0E+00  4.0E+00  5.0E+00  6.0E+00  7.0E+00  8.0E+00\n"
                    " -1.0E+00 -2.0E+00\n")
    assert np.array_equal(io.from_raw(path), [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j, -1 - 2j])


def test_only_what_follows_the_header_is_samples(tmp_path):
    """A header line without '=' is no sample, a lone '/' ends a namelist as '$END' does,
    and Fortran's D exponents read as numbers; an odd count of values is no FID."""
    path = tmp_path / "x.RAW"
    path.write_text(" $SEQPAR\n Echo 30 ms\n $END\n &NMID ID='x'\n /\n 1.0D+00 2.0D+00\n")
    assert np.array_equal(io.from_raw(path), [1 + 2j])
    path.write_text(" $NMID\n $END\n 1.0 2.0 3.0\n")
    with pytest.raises(ValueError, match="pairs"):
        io.from_raw(path)


def test_a_file_without_a_header_is_refused(tmp_path):
    path = tmp_path / "x.RAW"
    path.write_text("1.0 2.0\n")
    with pytest.raises(ValueError, match=r"\$END"):
        io.from_raw(path)


#*****************#
#   orientation   #
#*****************#
@needs_challenge
def test_every_loader_hands_out_one_orientation():
    """jMRUI text stores the conjugate of LCModel's .RAW, and the .RAW loader turns it
    round, so the same challenge spectrum loads alike from either file."""
    jmrui = io.load_signals(str(_CHALLENGE / "datasets_JMRUI" / "dataset1_WS.txt"))
    raw = io.load_signals(str(_CHALLENGE / "datasets_LCModel" / "dataset1.RAW"))
    assert np.allclose(jmrui.fids, raw.fids, rtol=1e-4, atol=1e-3)


@needs_challenge
def test_a_list_of_files_is_a_batch():
    data = _CHALLENGE / "datasets_JMRUI"
    got = io.load_signals([str(data / "dataset1_WS.txt"), str(data / "dataset2_WS.txt")])
    assert got.fids.shape == (2, 2048)
    assert got.dwell == pytest.approx(2.5e-4)


#*************#
#   batches   #
#*************#
def test_a_batch_must_share_its_acquisition():
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    a = gen_nifti_mrs(np.ones((1, 1, 1, 64), np.complex64), 1 / 4000.0, 123.2)
    b = gen_nifti_mrs(np.ones((1, 1, 1, 64), np.complex64), 1 / 2000.0, 297.2)
    assert io.load_signals([a, a]).fids.shape == (2, 64)          # objects stack too
    with pytest.raises(ValueError, match="differ in dwell"):
        io.load_signals([a, b])
    with pytest.raises(ValueError, match="No inputs"):
        io.load_signals([])


def test_a_jmrui_file_with_several_datasets_is_a_batch(tmp_path):
    path = tmp_path / "two.txt"
    rows = "\n".join(f"{v}\t0\t0\t0" for v in range(8))
    path.write_text("jMRUI Data Textfile\n\nPointsInDataset: 4\nDatasetsInFile: 2\n"
                    "SamplingInterval: 2.5E-1\nTransmitterFrequency: 1.2322E8\n\n"
                    "sig(real)\tsig(imag)\tfft(real)\tfft(imag)\n"
                    "Signal 1 out of 2 in file\n" + "\n".join(rows.splitlines()[:4]) + "\n"
                    "Signal 2 out of 2 in file\n" + "\n".join(rows.splitlines()[4:]) + "\n")
    got = io.load_signals(str(path))
    assert got.fids.shape == (2, 4)
    assert got.fids[1, 0] == 4


def test_a_path_object_reads_like_its_string(tmp_path):
    fid = np.exp(-np.arange(16) / 4.0) + 0j
    io.to_dpt(fid, tmp_path / "x.dpt", dwell=1 / 2000.0, central_freq=123.2)
    (tmp_path / "x.RAW").write_text(" $NMID\n $END\n" + "\n".join(f"{v.real} {v.imag}" for v in fid))
    assert np.allclose(io.load_signals(tmp_path / "x.RAW").fids,
                       io.load_signals(str(tmp_path / "x.RAW")).fids)
