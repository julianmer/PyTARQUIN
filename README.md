<div align="center">
  <h1 style="margin-bottom: 5px;">PyTARQUIN</h1>
  <p style="margin-top: 0px;"><em>A lightweight Python wrapper for TARQUIN spectral fitting in MR spectroscopy</em></p>

  [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/julianmer/PyTARQUIN/blob/main/LICENSE)
</div>

**PyTARQUIN** is a lightweight Python wrapper that streamlines the use of [TARQUIN](https://github.com/martin3141/tarquin) for MRS fitting. It handles flexible data input, writes TARQUIN's input files, manages the TARQUIN executable for you, and parses its results (with single- and multi-core processing).

---

## Features

- **Zero-setup binaries** — the TARQUIN executable is resolved automatically (download, container image, or your own path); nothing is bundled in the wheel.
- **Flexible input** — NumPy arrays, NIfTI-MRS (files, objects, and batches such as NIfTI-MRS+), jMRUI text, and LCModel `.RAW`, in time or frequency domain.
- **Every TARQUIN option** — any command-line option by name, an LCModel `.basis`, or TARQUIN's internal basis simulated for your echo time.
- **Batch fitting** — single- or multi-core, with full output parsing (concentrations, %SD, fit quality, and every other block TARQUIN reports).

---

## Installation

### From Source
```bash
git clone https://github.com/julianmer/PyTARQUIN.git
cd PyTARQUIN
pip install -e .
```
Add `--recursive` to the clone (or run `git submodule update --init`) to also fetch the
[ISMRM 2016 fitting challenge](https://www.ismrm.org/workshops/Spectroscopy16/mrs_fitting_challenge/)
example data used by the tests.

---

## How the TARQUIN binary is handled

TARQUIN is **not** shipped in the wheel. On first use it is found in this order, and the first one that works is cached:

1. `path2exec` you pass to `PyTARQUIN` (or the `TARQUIN_EXEC` environment variable),
2. a `tarquin` on your `PATH` (a distribution's package, say),
3. the binary built by this repository's CI for the installed version ([releases](https://github.com/julianmer/PyTARQUIN/releases); Linux x86_64/aarch64 and Windows x86_64 statically linked, macOS arm64/x86_64 linking only the OS),
4. the container image `ghcr.io/julianmer/tarquin`, if Docker or podman is running.

Each candidate is run once before it is accepted, so a binary that cannot run on your machine is skipped rather than cached. Useful switches: `allow_download` and `allow_docker` on `PyTARQUIN`, and the environment variables `TARQUIN_EXEC` (a binary of your own), `TARQUIN_CACHE_DIR` (where downloads are kept), `TARQUIN_RELEASE_TAG` and `TARQUIN_DOCKER_IMAGE` (another release or image), and `TARQUIN_NO_DOCKER` (set to anything: never use a container).

The binaries are TARQUIN 4.3.11, built from [martin3141/tarquin](https://github.com/martin3141/tarquin) at commit `47e9b98` with a small build-system patch ([`tarquin/`](https://github.com/julianmer/PyTARQUIN/blob/main/tarquin)); every build must reproduce the official 4.3.11 release on the 2016 challenge in CI, and the macOS and Linux builds fit it identically. To build one yourself: `tarquin/build.sh` (macOS, Linux, or Windows under MSYS2 UCRT64).

With the container, TARQUIN sees your working directory and your home directory; keep the basis set and any `save_path` under one of them. The image also works on its own:
```bash
docker run --rm -v "$PWD:$PWD" -w "$PWD" ghcr.io/julianmer/tarquin --input "$PWD/data.dpt" --format dpt --output_txt "$PWD/result.txt"
```

---

## Getting Started

```python
from tarquin_wrapper import PyTARQUIN

# Initialize the wrapper with your basis set (the TARQUIN binary is resolved automatically)
tarquin = PyTARQUIN(path2basis="/path/to/your/basis_set.basis")

# `data` can be a NumPy array of FIDs, a NIfTI-MRS path or object, a list of files, etc.;
# a water reference (one per spectrum) turns on TARQUIN's water scaling (mM)
concentrations, percent_sd = tarquin(data, water)

print("Signals:", tarquin.metabolites)
print("Concentrations:", concentrations)
```

Everything TARQUIN reported, one `Report` per spectrum:
```python
report = tarquin.report(data)[0]
report.units                                  # 'au', or 'mM' with a water reference
report.sections["fit_quality"]["snr_max"]     # every block of TARQUIN's results
```

TARQUIN's internal basis and options:
```python
tarquin = PyTARQUIN(
    echo_time=0.03,                            # the internal basis is simulated at it
    opts={"pul_seq": "slaser", "int_basis": "1h_brain", "max_iters": 50},
)
```
Data without acquisition parameters (NumPy arrays, `.RAW`) needs `bandwidth=` (Hz) and `central_freq=` (MHz).

---

## Licensing

This wrapper (the Python code) is released under the **Apache License 2.0** (see [LICENSE](https://github.com/julianmer/PyTARQUIN/blob/main/LICENSE)).

**TARQUIN itself is a separate program**, (c) Greg Reynolds and Martin Wilson, distributed under the **GNU General Public License, version 3** (see [LICENSE.tarquin](https://github.com/julianmer/PyTARQUIN/blob/main/LICENSE.tarquin)). This package does not bundle TARQUIN; when it downloads or runs the TARQUIN executable, that licence and the attributions in [NOTICE](https://github.com/julianmer/PyTARQUIN/blob/main/NOTICE) apply, and those of the libraries the binaries link in [THIRD-PARTY-NOTICES](https://github.com/julianmer/PyTARQUIN/blob/main/THIRD-PARTY-NOTICES). The build scripts and patch in [`tarquin/`](https://github.com/julianmer/PyTARQUIN/blob/main/tarquin) are part of the binaries' corresponding source and are GPL-3.0 as well; every release carries that source in full.

---

## Acknowledgements

- TARQUIN: Wilson M, Reynolds G, Kauppinen RA, Arvanitis TN, Peet AC. A constrained least-squares approach to the automated quantitation of in vivo 1H magnetic resonance spectroscopy data. *Magn Reson Med* 2011;65(1):1-12, [doi:10.1002/mrm.22579](https://doi.org/10.1002/mrm.22579). Source: [martin3141/tarquin](https://github.com/martin3141/tarquin)
- NIfTI-MRS: [spec2nii](https://github.com/wtclarke/spec2nii), [NIfTI-MRS Python tools](https://github.com/wtclarke/nifti_mrs_tools) (Will Clarke)
- Example data: [ISMRM 2016 MRS Fitting Challenge](https://www.ismrm.org/workshops/Spectroscopy16/mrs_fitting_challenge/) (Małgorzata Marjańska, Dinesh Deelchand, Roland Kreis), mirrored via [wtclarke/mrs_fitting_challenge](https://github.com/wtclarke/mrs_fitting_challenge)
- Modelled on [PyLCModel](https://github.com/julianmer/PyLCModel)

---

<div align="center">
  <sub>Built with ❤️ for the MRS community</sub>
</div>
