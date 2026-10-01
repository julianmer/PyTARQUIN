####################################################################################################
#                                           binaries.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Resolution of the TARQUIN executable. The wrapper does NOT ship binaries; it resolves   #
#          one at run time in priority order: an explicit "path2exec" (or TARQUIN_EXEC), a         #
#          previously cached download, a download of the CI-built binary attached to the           #
#          PyTARQUIN release of this version, or a container image (docker/podman) behind a        #
#          launcher script that behaves like a native executable (see container.py).               #
#                                                                                                  #
#          Every candidate is exercised before it is accepted (see "verify_executable"), so a      #
#          wrong-architecture download can never be cached and served forever.                     #
#                                                                                                  #
#          There is no upstream download and no build from source, unlike PyLCModel: the           #
#          official binaries are GUI bundles on SourceForge (Linux 32-bit, macOS Intel inside a    #
#          disk image, Windows 4.3.10), and a build needs CMake, Boost, FFTW, LAPACK and Fortran   #
#          (tarquin/build.sh does it, for anyone who wants to).                                    #
#                                                                                                  #
# TARQUIN itself is a separate GPL-3.0 program (see LICENSE.tarquin and NOTICE).                   #
#                                                                                                  #
####################################################################################################

import hashlib
import lzma
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ._version import __version__


#********************************#
#   pytarquin release registry   #
#********************************#
# Binaries compiled in this repository's CI (.github/workflows/tarquin-binaries.yml) from
# the pinned upstream commit and attached to a GitHub release: statically linked on Linux
# and Windows, linking only the OS on macOS.
#
# The release is the one tagged with this package's own version ("v0.1.0"), so a wheel
# always pairs with the binaries built alongside it and old wheels keep working.
_RELEASE_REPO = "julianmer/PyTARQUIN"
_DEFAULT_RELEASE_TAG = f"v{__version__}"

# registry key -> asset name. Every asset is an xz-compressed raw executable with a
# "<asset>.sha256" sidecar produced by the same CI run.
_RELEASE_ASSETS = {
    "linux-x86_64":  "tarquin-linux-x86_64.xz",
    "linux-aarch64": "tarquin-linux-aarch64.xz",
    "darwin-arm64":  "tarquin-macos-arm64.xz",
    "darwin-x86_64": "tarquin-macos-x86_64.xz",
    "windows-amd64": "tarquin-windows-x86_64.xz",
}


def _release_tag() -> str:
    return os.environ.get("TARQUIN_RELEASE_TAG", _DEFAULT_RELEASE_TAG)


def _release_base() -> str:
    return f"https://github.com/{_RELEASE_REPO}/releases/download/{_release_tag()}"


#*********************#
#   container image   #
#*********************#
# The same CI pushes the Linux binary as a multi-arch image (linux/amd64, linux/arm64).
# On a host with docker or podman it runs identically everywhere, because inside the
# container it is always the same statically linked Linux binary.
_DEFAULT_CONTAINER_IMAGE = f"ghcr.io/{_RELEASE_REPO.split('/')[0]}/tarquin"

# Name of the cached launcher script - two lines calling tarquin_wrapper.container.
# Distinct from the native executable name so the two never compete for one cache slot.
_SHIM_NAME = "tarquin-container.cmd" if os.name == "nt" else "tarquin-container"
_SHIM_NAMES = ("tarquin-container", "tarquin-container.cmd")

_PULL_TIMEOUT = 900.0


def _container_image() -> str:
    return os.environ.get("TARQUIN_DOCKER_IMAGE", f"{_DEFAULT_CONTAINER_IMAGE}:{_release_tag()}")


#**********************#
#   platform helpers   #
#**********************#
_MACHINE_ALIASES = {
    "x86_64": "x86_64", "amd64": "x86_64", "x64": "x86_64",
    "i386": "x86", "i686": "x86",
    "aarch64": "arm64", "arm64": "arm64", "armv8b": "arm64", "armv8l": "arm64",
}


def _normalized_machine() -> str:
    """Map the many spellings of a CPU architecture onto a canonical name."""
    machine = platform.machine().lower()
    return _MACHINE_ALIASES.get(machine, machine)


def _platform_keys() -> List[str]:
    """Return an ordered list of candidate registry keys for the current platform.

    An empty list means "no prebuilt binary applies" - resolution then falls through to
    the container rather than downloading something that cannot run here.
    """
    system = platform.system().lower()
    machine = _normalized_machine()

    if system == "linux":
        if machine == "x86_64":
            return ["linux-x86_64"]
        if machine == "arm64":
            return ["linux-aarch64"]
        return []
    if system == "darwin":
        if machine == "arm64":
            # The Intel build is a last resort via Rosetta 2; verification rejects it
            # cleanly if Rosetta is not installed.
            return ["darwin-arm64", "darwin-x86_64"]
        return ["darwin-x86_64"]
    if system == "windows":
        if machine in ("x86_64", "arm64"):   # Windows on ARM emulates x64
            return ["windows-amd64"]
        return []
    return []


def _cache_root() -> Path:
    override = os.environ.get("TARQUIN_CACHE_DIR")
    if override:
        base = Path(override)
    elif platform.system().lower() == "windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "tarquin_wrapper"
    else:
        base = Path.home() / ".cache" / "tarquin_wrapper"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cache_dir() -> Path:
    """Per-architecture cache directory.

    Keyed by architecture so a home directory shared across a mixed-architecture cluster
    (an x86_64 login node and aarch64 compute nodes, say) does not have both fighting
    over one file.
    """
    base = _cache_root() / f"{platform.system().lower()}-{_normalized_machine()}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _make_executable(path: Path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _exec_name() -> str:
    return "tarquin.exe" if platform.system().lower() == "windows" else "tarquin"


def _install(src: Path, cache: Path) -> Path:
    """Move a finished binary into the cache atomically.

    Callers extract into a temporary location first, so an interrupted download can never
    leave a truncated file at the cached path where the next run would pick it up.
    """
    target = cache / _exec_name()
    _make_executable(src)
    os.replace(src, target)
    return target


def _quarantine(path: Path, reason: str = "unknown") -> Optional[Path]:
    """Move a rejected binary aside so the next provider is not shadowed by it.

    Renamed rather than deleted: the artifact stays available for debugging, and a
    concurrent process losing the race must not crash.
    """
    try:
        dest_dir = _cache_root() / "quarantine"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{Path(path).name}-{reason}-{time.time_ns()}"
        os.replace(path, dest)
        print(f"[tarquin_wrapper] Quarantined unusable binary at {dest}")
        return dest
    except (OSError, PermissionError):
        return None


#*************************#
#   health verification   #
#*************************#
# TARQUIN logs "TARQUIN <version> Started" before it reads any argument, so "--help"
# identifies it (and then exits 255).
_HEALTH_MARKER = re.compile(rb"TARQUIN \S+ Started")

_VERIFY_TIMEOUT = 60.0

# Keyed on (resolved path, mtime, size) so repeated PyTARQUIN() construction is free.
_VERIFIED: Dict[tuple, Tuple[bool, str]] = {}

_ELF_MACHINES = {0x03: "x86", 0x3E: "x86-64", 0x28: "ARM", 0xB7: "AArch64", 0xF3: "RISC-V"}
_MACHO_CPUS = {0x01000007: "x86_64", 0x0100000C: "arm64",
               0x00000007: "i386", 0x0000000C: "arm"}


def _describe_binary(path: Path) -> str:
    """Best-effort 'what is this file, really' for error messages only.

    Never used to accept or reject - purely so that a mismatch reports "file is ELF
    x86-64, host is Linux AArch64" instead of a bare "Exec format error".
    """
    host = f"host is {platform.system()} {platform.machine()}"
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
    except OSError:
        return host
    if len(head) < 8:
        return f"file is not an executable ({len(head)} bytes); {host}"
    if head[:4] == b"\x7fELF":
        machine = int.from_bytes(head[18:20], "little")
        return f"file is ELF {_ELF_MACHINES.get(machine, hex(machine))}; {host}"
    if head[:4] in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):
        cpu = int.from_bytes(head[4:8], "little")
        return f"file is Mach-O {_MACHO_CPUS.get(cpu, hex(cpu))}; {host}"
    if head[:4] == b"\xca\xfe\xba\xbe":
        return f"file is a Mach-O universal binary; {host}"
    if head[:2] == b"MZ":
        return f"file is a Windows PE executable; {host}"
    if head[:2] == b"#!":
        return f"file is a script ({head[:20].decode('ascii', 'replace').strip()}); {host}"
    return f"file is not a recognised executable format; {host}"


def verify_executable(path, timeout: Optional[float] = None) -> Tuple[bool, str]:
    """Check that "path" is a runnable TARQUIN binary. Returns (ok, reason).

    Runs the candidate with "--help" and looks for TARQUIN's own start line. Cheap, writes
    nothing, and cannot hang. The exit status is ignored: TARQUIN ends "--help" with 255.
    """
    path = Path(path).expanduser().resolve()   # the probe runs in a temporary directory
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError as err:
        return False, f"cannot stat: {err}"
    if key in _VERIFIED:
        return _VERIFIED[key]

    result = _probe(path, _VERIFY_TIMEOUT if timeout is None else timeout)
    _VERIFIED[key] = result
    return result


def _probe(path: Path, timeout: float) -> Tuple[bool, str]:
    # stdin at EOF, a hard timeout, and a throwaway working directory for stray output
    with tempfile.TemporaryDirectory() as cwd:
        try:
            proc = subprocess.run(
                [str(path), "--help"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            return False, f"did not exit within {timeout:.0f}s"
        except OSError as err:
            # Exec format error, bad CPU type, missing loader, permission denied.
            return False, f"cannot execute ({err}); {_describe_binary(path)}"

    out = proc.stdout or b""
    if _HEALTH_MARKER.search(out):
        return True, "ok"
    if proc.returncode < 0:
        return False, (f"killed by signal {-proc.returncode} (binary may require CPU "
                       f"features this machine lacks); {_describe_binary(path)}")
    return False, (f"ran (exit {proc.returncode}) but did not identify itself as TARQUIN; "
                   f"first output was {out[:120]!r}")


#*******************#
#   download path   #
#*******************#
def _download(url: str, dest: Path):
    print(f"[tarquin_wrapper] Downloading TARQUIN binary from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "tarquin_wrapper"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def _fetch_and_install(url: str, sha256_url: str, cache: Path) -> Path:
    """Download one xz-compressed binary, verify its checksum, extract and install it.

    A release asset is a plain URL anyone with write access could re-upload, so the hash
    from the CI run that built and tested it is what ties the file to that run.
    """
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        tmp = Path(tmp)
        archive = tmp / os.path.basename(url)
        staged = tmp / _exec_name()
        _download(url, archive)

        sidecar = tmp / (archive.name + ".sha256")
        _download(sha256_url, sidecar)
        expected = sidecar.read_text().split()[0].lower()
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"sha256 mismatch for {url}: got {actual}, expected {expected}")

        with lzma.open(archive) as src, open(staged, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return _install(staged, cache)


def _release_assets() -> List[str]:
    """Release asset names applicable to this host, in preference order."""
    return [_RELEASE_ASSETS[key] for key in _platform_keys()]


def _download_release_binary(cache: Path) -> Optional[Path]:
    """Fetch the CI-built binary attached to the PyTARQUIN release of this version."""
    assets = _release_assets()
    if not assets:
        return None
    base = _release_base()
    last_err = None
    for asset in assets:
        url = f"{base}/{asset}"
        try:
            return _fetch_and_install(url, f"{url}.sha256", cache)
        except Exception as err:   # try the next candidate (e.g. the Intel build on arm64)
            last_err = err
            print(f"[tarquin_wrapper] Could not fetch {url}: {err}")

    print(f"[tarquin_wrapper] Release binary download failed: {last_err}")
    return None


#********************#
#   container path   #
#********************#
# The launcher is a two-line script that calls tarquin_wrapper.container with the engine
# and image baked in, and passes its own arguments through; every mount and path decision
# lives in that module. From the wrapper's side the script is just another TARQUIN binary,
# and verify_executable() probes it like one.
def _container_cli() -> Optional[str]:
    for name in ("docker", "podman"):
        if shutil.which(name):
            return name
    return None


def _engine(cli: str, *args: str, timeout: float) -> Tuple[int, str]:
    """Run a container-engine command. Returns (exit code, last stderr line); the code
    is -1 when the command could not run or time out."""
    try:
        proc = subprocess.run([cli, *args], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, f"'{cli} {args[0]}' did not finish within {timeout:.0f}s"
    except OSError as err:
        return -1, f"cannot run '{cli}': {err}"
    lines = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
    return proc.returncode, lines[-1] if lines else f"'{cli} {args[0]}' exited {proc.returncode}"


def _container_ready(cli: str, timeout: float = 30.0) -> Tuple[bool, str]:
    """Is the engine actually usable - daemon running, socket reachable, permissions ok?"""
    code, why = _engine(cli, "info", timeout=timeout)
    return code == 0, "ok" if code == 0 else why


def is_container_shim(path) -> bool:
    """True if "path" is a launcher generated here rather than a native TARQUIN binary."""
    return Path(path).name in _SHIM_NAMES


def _write_shim(cache: Path, cli: str, image: str) -> Path:
    # "-c" rather than "-m tarquin_wrapper.container": the package __init__ already
    # imports that module, and runpy warns when asked to execute an imported module.
    # "--" ends the launcher's own options; everything after it is TARQUIN's.
    entry = "import sys; from tarquin_wrapper.container import main; sys.exit(main())"
    python = sys.executable
    if os.name == "nt":
        script = f'@"{python}" -c "{entry}" --cli {cli} --image {image} -- %*\r\n'
    else:
        script = (f"#!/bin/sh\nexec {shlex.quote(python)} -c {shlex.quote(entry)} "
                  f"--cli {shlex.quote(cli)} --image {shlex.quote(image)} -- \"$@\"\n")

    target = cache / _SHIM_NAME
    fd, tmp = tempfile.mkstemp(dir=cache, prefix=".shim-")
    with os.fdopen(fd, "w", newline="") as fh:
        fh.write(script)
    _make_executable(Path(tmp))
    os.replace(tmp, target)
    return target


def _container_shim(cache: Path) -> Optional[Path]:
    if os.environ.get("TARQUIN_NO_DOCKER"):
        return None
    cli = _container_cli()
    if cli is None:
        print("[tarquin_wrapper] Cannot use a container: neither 'docker' nor 'podman' "
              "found on PATH.")
        return None
    ok, why = _container_ready(cli)
    if not ok:
        print(f"[tarquin_wrapper] Container engine '{cli}' is not usable: {why}")
        return None

    image = _container_image()
    # already in the local store (pulled earlier, or built by hand): no network needed,
    # which keeps this rung working offline
    if _engine(cli, "image", "inspect", image, timeout=30)[0] != 0:
        print(f"[tarquin_wrapper] Pulling TARQUIN container image {image}")
        code, why = _engine(cli, "pull", image, timeout=_PULL_TIMEOUT)
        if code != 0:
            print(f"[tarquin_wrapper] Could not pull {image}: {why}")
            return None

    return _write_shim(cache, cli, image)


#**********************#
#   cache management   #
#**********************#
def _cached_binary(cache: Path) -> Optional[Path]:
    cached = cache / _exec_name()
    if cached.is_file():
        return cached

    # A shim left by an earlier resolution means the native sources all failed on this
    # machine last time; prefer it over re-downloading and re-rejecting them. If the
    # engine is simply not running right now, leave the shim in place and report
    # "unavailable" instead of quarantining a perfectly good script.
    shim = cache / _SHIM_NAME
    if shim.is_file() and not os.environ.get("TARQUIN_NO_DOCKER"):
        cli = _container_cli()
        if cli is None:
            return None
        ok, why = _container_ready(cli)
        if not ok:
            print(f"[tarquin_wrapper] Cached container shim skipped: {why}")
            return None
        return shim
    return None


#*********************#
#   public resolver   #
#*********************#
def resolve_executable(path2exec: Optional[str] = None,
                       allow_download: bool = True,
                       cache_dir: Optional[str] = None,
                       verify: bool = True,
                       allow_docker: bool = True) -> str:
    """Resolve a usable TARQUIN executable and return its absolute path.

    Resolution order: explicit path -> cached -> release download -> container shim.
    Every candidate is run once before being accepted, and anything that fails is
    quarantined so the next source gets a turn instead of being shadowed by a broken file.

    Raises "RuntimeError" if no executable can be obtained.
    """
    # 1. explicit user path - verified, but never silently replaced. The environment
    #    variable is the same thing for code that does not pass the argument (CI
    #    pointing the test suite at a freshly built binary, a cluster-wide install).
    path2exec = path2exec or os.environ.get("TARQUIN_EXEC")
    if path2exec:
        p = Path(path2exec).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"path2exec does not exist: {p}")
        if verify:
            ok, why = verify_executable(p)
            if not ok:
                if why.startswith(("cannot execute", "killed by signal")):
                    raise RuntimeError(
                        f"path2exec is not a runnable TARQUIN binary: {p}\n  {why}"
                    )
                # It ran but did not identify itself - could legitimately be a wrapper
                # script, so defer to the user who named it explicitly.
                print(f"[tarquin_wrapper] Warning: {p} did not identify itself as TARQUIN "
                      f"({why}); using it anyway because it was passed explicitly.")
        return str(p.resolve())

    cache = Path(cache_dir) if cache_dir else _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    providers: List[Tuple[str, Callable[[], Optional[Path]]]] = [
        ("cached", lambda: _cached_binary(cache)),
    ]
    if allow_download:
        providers.append(("release", lambda: _download_release_binary(cache)))
    if allow_docker:
        providers.append(("container", lambda: _container_shim(cache)))

    failures = []
    for name, provide in providers:
        try:
            got = provide()
        except Exception as err:
            failures.append(f"{name}: {err}")
            continue
        if got is None:
            failures.append(f"{name}: unavailable")
            continue
        if verify:
            ok, why = verify_executable(got)
            if not ok:
                print(f"[tarquin_wrapper] Rejected TARQUIN binary from '{name}': {why}")
                _quarantine(Path(got), reason=name)
                failures.append(f"{name}: {why}")
                continue
        return str(Path(got).resolve())

    raise RuntimeError(_resolution_error(failures, allow_download, allow_docker))


def _resolution_error(failures: List[str], allow_download: bool, allow_docker: bool) -> str:
    detail = "\n".join(f"    - {f}" for f in failures) or "    - no sources were tried"
    engine = _container_cli()
    return (
        "Could not resolve a working TARQUIN executable.\n"
        f"  platform: {platform.system()} {platform.machine()}\n"
        f"  cache:    {_cache_dir()}\n"
        "  tried:\n"
        f"{detail}\n"
        "Options:\n"
        "  - pass path2exec='/path/to/tarquin' to PyTARQUIN, or set TARQUIN_EXEC,\n"
        "  - ensure internet access so the matching binary can be downloaded from the\n"
        f"    {_release_tag()} release of https://github.com/{_RELEASE_REPO},\n"
        "  - install and start Docker (or podman) so the TARQUIN container image\n"
        f"    {_container_image()} can be used,\n"
        f"  - or build one with tarquin/build.sh from https://github.com/{_RELEASE_REPO}.\n"
        f"(detail: download={allow_download}, docker={allow_docker}, "
        f"engine={engine or 'no'})"
    )
