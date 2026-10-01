####################################################################################################
#                                          container.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Run TARQUIN from a container image as if it were a native executable.                   #
#                                                                                                  #
#          The wrapper talks to TARQUIN in exactly one way: options on the command line, among     #
#          them the absolute paths of every input and output file. So a launcher that forwards     #
#          its arguments into "docker run" and makes the host paths valid inside the container     #
#          is, from the wrapper's point of view, just another TARQUIN binary - it is even          #
#          health-checked by the same probe. binaries.py caches a two-line script that calls       #
#          this module's main().                                                                   #
#                                                                                                  #
#          Linux/macOS: the working directory and the home directory are bind-mounted at           #
#          identical paths, so the arguments need no translation.                                  #
#          Windows:     a Linux container cannot have a path "C:\...", so each drive is mounted    #
#                       at /host/<LETTER> and every drive-letter argument is rewritten to match    #
#                       (translate_args). Because the wrapper writes every path itself, that is    #
#                       the whole of the translation.                                              #
#                                                                                                  #
####################################################################################################

import argparse
import ntpath
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Where the Windows launcher mounts each drive: C:\Users\me -> /host/C/Users/me
WIN_MOUNT_ROOT = "/host"


#**********************#
#   path translation   #
#**********************#
def windows_to_container(path: str) -> str:
    r"""C:\Users\me\x.basis -> /host/C/Users/me/x.basis. Pure, so it is testable anywhere."""
    drive, rest = ntpath.splitdrive(ntpath.normpath(path))
    if len(drive) != 2 or drive[1] != ":":
        raise ValueError(f"{path}: only drive-letter paths are visible inside the container "
                         f"(UNC paths are not mounted)")
    return f"{WIN_MOUNT_ROOT}/{drive[0].upper()}{rest.replace(chr(92), '/')}"


def to_container(path) -> str:
    """A host path as TARQUIN sees it inside the container: identity on POSIX (identical
    mounts), drive letter mapped under /host on Windows."""
    if os.name == "nt":
        return windows_to_container(os.path.abspath(str(path)))
    return str(path)


def translate_args(args: List[str], mapper: Callable[[str], str] = to_container) -> List[str]:
    """Rewrite every file path among TARQUIN's arguments for the container.

    A path is recognised by a drive letter and a separator ("C:\\..." or "C:/..."), which
    no other value TARQUIN takes starts with; on POSIX the mapper is the identity.
    """
    return [mapper(arg) if re.match(r"[A-Za-z]:[\\/]", arg) else arg for arg in args]


#***********************#
#   mount computation   #
#***********************#
def posix_mounts(cwd: str, home: Optional[str]) -> List[Tuple[str, str]]:
    """(host, container) pairs: the home directory and the working directory, each under
    its own spelling and its physical one (a symlinked home is common on clusters), each
    unless it already lies inside one before it - docker refuses a mount point twice - and
    never "/", which docker refuses outright. The home directory stays visible as a
    whole, since the basis set and the wrapper's own run folder live there."""
    roots: List[str] = []
    for path in (home, home and os.path.realpath(home), cwd, os.path.realpath(cwd)):
        if path and path != os.sep and not _inside(path, roots):
            roots.append(path)
    return [(r, r) for r in roots]


def _inside(path: str, roots: List[str]) -> bool:
    return any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def windows_mounts(cwd: str, home: Optional[str]) -> List[Tuple[str, str]]:
    """The drive of the working directory, plus the drive of the home directory."""
    drives = [ntpath.splitdrive(cwd)[0].upper()]
    if home:
        d = ntpath.splitdrive(home)[0].upper()
        if d and d not in drives:
            drives.append(d)
    return [(f"{d}\\", f"{WIN_MOUNT_ROOT}/{d[0]}") for d in drives if len(d) == 2]


def can_see(path) -> bool:
    """Would the launcher's mounts make "path", as its physical path, visible inside the
    container? The wrapper hands TARQUIN physical paths, so this lets it fail early with a
    clear message instead of TARQUIN reporting a missing file."""
    cwd, home = os.getcwd(), str(Path.home())
    if os.name == "nt":
        drive = ntpath.splitdrive(os.path.abspath(str(path)))[0].upper()
        return any(host.startswith(drive) for host, _ in windows_mounts(cwd, home))
    return _inside(os.path.realpath(path), [host for host, _ in posix_mounts(cwd, home)])


#********************#
#   engine command   #
#********************#
def run_command(cli: str, image: str, args: List[str], cwd: Optional[str] = None,
                home: Optional[str] = None, windows: Optional[bool] = None,
                uid: Optional[Tuple[int, int]] = None) -> List[str]:
    """The full "docker run ... <image> <TARQUIN arguments>" list. Pure given its inputs."""
    cwd = cwd or os.getcwd()
    home = home if home is not None else str(Path.home())
    windows = os.name == "nt" if windows is None else windows

    # --init: TARQUIN as the container's first process would ignore SIGINT and SIGTERM, so
    # neither Ctrl-C nor the wrapper's timeout would stop it
    cmd = [cli, "run", "--rm", "--init"]
    if windows:
        mounts = windows_mounts(cwd, home)
        workdir = windows_to_container(cwd)
        args = translate_args(args, windows_to_container)
    else:
        mounts = posix_mounts(cwd, home)
        workdir = cwd
        # output files must come back owned by the caller, not root
        if cli == "podman":
            cmd.append("--userns=keep-id")
        elif uid is not None:
            cmd += ["-u", f"{uid[0]}:{uid[1]}"]
    for host, inside in mounts:
        cmd += ["-v", f"{host}:{inside}"]
    cmd += ["-w", workdir, image]
    return cmd + list(args)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # the launcher's own options come before "--", TARQUIN's after it
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description="Run TARQUIN from a container image.")
    parser.add_argument("--cli", required=True, help="docker or podman")
    parser.add_argument("--image", required=True)
    own = parser.parse_args(argv[:split])

    image = os.environ.get("TARQUIN_DOCKER_IMAGE", own.image)
    uid = (os.getuid(), os.getgid()) if hasattr(os, "getuid") else None
    cmd = run_command(own.cli, image, argv[split + 1:], uid=uid)
    if os.name == "nt":
        return subprocess.call(cmd)
    os.execvp(cmd[0], cmd)   # stdout and exit status pass straight through
    return 1                 # not reached


if __name__ == "__main__":
    sys.exit(main())
