####################################################################################################
#                                        test_results.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Parsing of TARQUIN's text results, on a file laid out as TARQUIN 4.3.11 writes it       #
#          (src/common/export_data.cpp): the table, then "key : value" blocks and the version.     #
#                                                                                                  #
####################################################################################################

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tarquin_wrapper import read_results

# a water-scaled fit of challenge dataset 1, cut to three signals and one combination
RESULTS_TXT = """\
Signal     Conc (mM)       %SD   SD (mM)
----------------------------------------
Cr             2.425     18.49    0.4484
GABA           0.000       inf    0.2623
PCr            3.869     12.73    0.4927
TCr            6.294     4.712    0.2966

Fit quality
-----------
Metab FWHM (PPM) : 0.03565
Metab FWHM (Hz)  : 4.395
SNR residual     : 41.43
SNR max          : 52.52
Q                : 1.268

Fitting parameters
------------------
Initial beta : 68.56
Final beta   : 4.769
Max iters    : 75
Start point  : 40
End point    : 1024
Ref (ppm)    : 4.650

Concentration scaling
---------------------
Water scaling    : true
Water amp        : 4.993e+04
Water conc       : 3.588e+04

Data parameters
---------------
Fs (Hz) : 4000.
Ft (Hz) : 1.233e+08
N       : 2048
TE (s)  : 0.03000
Format  : lcm

Optimisation details
--------------------
l2 norm of error at initial p  : 1.184
number of iterations           : 67.00
stopped by small D*p

TARQUIN version 4.3.11
"""


@pytest.fixture
def results(tmp_path):
    path = tmp_path / "result.txt"
    path.write_text(RESULTS_TXT)
    return path


#***********#
#   table   #
#***********#
def test_the_table_is_read_in_tarquins_order(results):
    out = read_results(results)
    assert out.metabolites == ["Cr", "GABA", "PCr", "TCr"]
    assert out.concentrations == [2.425, 0.0, 3.869, 6.294]
    assert out.percent_sd[0] == 18.49 and out.percent_sd[1] == float("inf")
    assert out.sd == [0.4484, 0.2623, 0.4927, 0.2966]


#***********#
#   units   #
#***********#
def test_the_units_follow_the_header(results, tmp_path):
    """With a water reference TARQUIN's table is in mM; without, in arbitrary units."""
    assert read_results(results).units == "mM"
    plain = tmp_path / "plain.txt"
    plain.write_text(RESULTS_TXT.replace("(mM)", "(au)"))
    assert read_results(plain).units == "au"


#**************#
#   sections   #
#**************#
def test_every_block_is_kept_under_snake_case_names(results):
    sections = read_results(results).sections
    assert list(sections) == ["fit_quality", "fitting_parameters", "concentration_scaling",
                              "data_parameters", "optimisation_details"]
    assert sections["fit_quality"] == {"metab_fwhm_ppm": 0.03565, "metab_fwhm_hz": 4.395,
                                       "snr_residual": 41.43, "snr_max": 52.52, "q": 1.268}
    assert sections["concentration_scaling"]["water_scaling"] == "true"
    assert sections["data_parameters"]["format"] == "lcm"
    assert sections["data_parameters"]["te_s"] == 0.03
    # a line without "key : value" is TARQUIN's prose, not a field
    assert sections["optimisation_details"] == {"l2_norm_of_error_at_initial_p": 1.184,
                                                "number_of_iterations": 67.0}


#*************#
#   version   #
#*************#
def test_the_version_is_read(results):
    assert read_results(results).version == "4.3.11"


#*************#
#   refused   #
#*************#
def test_results_without_signals_are_refused(tmp_path):
    path = tmp_path / "result.txt"
    path.write_text(RESULTS_TXT.split("Cr  ")[0])
    with pytest.raises(ValueError, match="no signals"):
        read_results(path)


def test_a_row_run_together_is_refused(tmp_path):
    """TARQUIN's columns are fixed-width: a value that fills its column runs into the
    next, and such a row must not silently drop its metabolite."""
    path = tmp_path / "result.txt"
    path.write_text(RESULTS_TXT.replace("Cr             2.425     18.49    0.4484",
                                        "Ins       -0.0001234-4.052e+04   0.05000"))
    with pytest.raises(ValueError, match="cannot be read"):
        read_results(path)
