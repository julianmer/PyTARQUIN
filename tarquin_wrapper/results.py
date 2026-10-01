####################################################################################################
#                                           results.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Parsing of TARQUIN's text results ("--output_txt"), as its ExportTxtResults writes      #
#          them: a table of signals (name, concentration, %SD, SD) whose header states the         #
#          units, then blocks of "key : value" lines (fit quality, fitting parameters,             #
#          concentration scaling, data parameters, optimisation details) and the version.          #
#                                                                                                  #
####################################################################################################

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


#**************************************************************************************************#
#                                           Class Report                                           #
#**************************************************************************************************#
#                                                                                                  #
# Everything one TARQUIN fit reported.                                                             #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Report:
    """Everything one TARQUIN fit reported.

    Attributes:
        metabolites: Signal names in TARQUIN's order, the combined ones (TNAA, TCr, ...)
            after the single ones.
        concentrations: One per signal, in the table's units.
        percent_sd: TARQUIN's %SD (its CRLB as a percentage); inf for a zero concentration.
        sd: The same as an absolute SD, in the table's units.
        units: 'au' (unreferenced) or 'mM' (water-scaled).
        sections: The "key : value" blocks under snake_case names, e.g.
            sections["fit_quality"]["snr_max"]; numbers as floats, the rest as text.
        version: The TARQUIN release that wrote the file.
    """

    metabolites: List[str]
    concentrations: List[float]
    percent_sd: List[float]
    sd: List[float]
    units: str
    sections: Dict[str, Dict[str, Union[float, str]]] = field(default_factory=dict)
    version: Optional[str] = None


#****************#
#   snake case   #
#****************#
def _snake(text: str) -> str:
    """'Metab FWHM (PPM)' -> 'metab_fwhm_ppm'."""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


#***********#
#   value   #
#***********#
def _value(text: str) -> Union[float, str]:
    try:
        return float(text)
    except ValueError:
        return text.strip()


#******************#
#   read results   #
#******************#
def read_results(path) -> Report:
    """Parse one TARQUIN ".txt" results file.

    Raises:
        ValueError: If a table row cannot be split into its four columns (TARQUIN's
            columns are fixed-width, and a value that fills its column runs into the
            next), or if the table holds no signals.
    """
    with open(path) as fh:
        blocks = [b.splitlines() for b in fh.read().split("\n\n") if b.strip()]

    names, conc, pct, sd = [], [], [], []
    for line in blocks[0][2:]:
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 4:
            raise ValueError(f"TARQUIN's results row cannot be read: {line!r} ({path})")
        names.append(parts[0])
        conc.append(float(parts[1]))
        pct.append(float(parts[2]))
        sd.append(float(parts[3]))
    if not names:
        raise ValueError(f"TARQUIN's results in {path} hold no signals")

    sections, version = {}, None
    for block in blocks[1:]:
        title = block[0].strip()
        if title.startswith("TARQUIN version"):
            version = title.split()[-1]
            continue
        entries = {}
        for line in block[2:]:          # below the title and its underline
            key, sep, value = line.partition(":")
            if sep:
                entries[_snake(key)] = _value(value)
        sections[_snake(title)] = entries

    units = "mM" if "(mM)" in blocks[0][0] else "au"
    return Report(metabolites=names, concentrations=conc, percent_sd=pct, sd=sd, units=units,
                  sections=sections, version=version)
